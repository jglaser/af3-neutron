import functools
import jax
import jax.numpy as jnp
import numpy as np
import haiku as hk
import pathlib
import tokamax

from alphafold3.model import model, features, params, feat_batch
from alphafold3.model.components import utils
from alphafold3.model.network import evoformer as evoformer_network
from alphafold3.model.network import diffusion_head

def make_model_config(num_recycles=10, num_diffusion_samples=5):
    config = model.Model.Config()
    config.global_config.flash_attention_implementation = 'triton'
    
    # Use the passed variables instead of hardcoding 1
    config.heads.diffusion.eval.num_samples = num_diffusion_samples
    config.num_recycles = num_recycles
    
    return config

# -------------------------------------------------------------------------
# HAIKU NAMESPACE WRAPPERS
# These force our detached components to inherit the 'diffuser/' prefix
# so Haiku can find the DeepMind pre-trained weights.
# -------------------------------------------------------------------------
class TrunkWrapper(hk.Module):
    def __init__(self, config, name='diffuser'):
        super().__init__(name=name)
        self.config = config

    def __call__(self, batch):
        embedding_module = evoformer_network.Evoformer(
            self.config.evoformer, self.config.global_config
        )
        target_feat = model.create_target_feat_embedding(
            batch=batch,
            config=embedding_module.config,
            global_config=self.config.global_config,
        )

        def recycle_body(_, args):
            prev, key = args
            key, subkey = jax.random.split(key)
            embeddings = embedding_module(
                batch=batch, prev=prev, target_feat=target_feat, key=subkey
            )
            embeddings['pair'] = embeddings['pair'].astype(jnp.float32)
            embeddings['single'] = embeddings['single'].astype(jnp.float32)
            return embeddings, key

        num_res = batch.num_res
        embeddings = {
            'pair': jnp.zeros([num_res, num_res, self.config.evoformer.pair_channel], dtype=jnp.float32),
            'single': jnp.zeros([num_res, self.config.evoformer.seq_channel], dtype=jnp.float32),
            'target_feat': target_feat,
        }
        
        key = hk.next_rng_key()
        num_iter = self.config.num_recycles + 1
        embeddings, _ = hk.fori_loop(0, num_iter, recycle_body, (embeddings, key))
        
        # Inject target_feat back into the final dict as the diffusion head expects it
        embeddings['target_feat'] = target_feat
        return embeddings


# -------------------------------------------------------------------------
# THE BASELINE WRAPPER (For the 1-step Oracle initialization)
# -------------------------------------------------------------------------
class DiffusionWrapper(hk.Module):
    def __init__(self, config, name='diffuser'):
        super().__init__(name=name)
        self.config = config
        self.diffusion_module = diffusion_head.DiffusionHead(
            self.config.heads.diffusion, self.config.global_config
        )

    def __call__(self, positions_noisy, noise_level, batch, embeddings):
        return self.diffusion_module(
            positions_noisy=positions_noisy,
            noise_level=noise_level,
            batch=batch,
            embeddings=embeddings,
            use_conditioning=True
        )

# -------------------------------------------------------------------------
# 2. THE GUIDED SDE WRAPPER (For the 200-step physics refinement)
# -------------------------------------------------------------------------
class GuidedDiffusionWrapper(hk.Module):
    def __init__(self, config, name='diffuser'):
        super().__init__(name=name)
        self.config = config
        self.diffusion_module = diffusion_head.DiffusionHead(
            self.config.heads.diffusion, self.config.global_config
        )

    def __call__(self, batch, embeddings, grad_fn, sample_key):
        sample_config = self.config.heads.diffusion.eval

        def guided_denoising_step(positions_noisy, t_hat):
            # 1. Native AF3 Vector Field (Evaluates in the Augmented Frame)
            x_0 = self.diffusion_module(
                positions_noisy=positions_noisy,
                noise_level=t_hat,
                batch=batch,
                embeddings=embeddings,
                use_conditioning=True
            )
            
            # 2. Covariant Loss Gradient
            loss_val, grad_x0 = grad_fn(x_0)
            jax.debug.print("SDE Step Loss (Covariant): {loss:.4f}", loss=loss_val)
            
            # 3. Apply physics torque directly to x_0
            lr_heavy = 0.05
            grad_x0_clipped = jnp.clip(grad_x0, -1.0, 1.0)
            x_0_guided = x_0 - (lr_heavy * grad_x0_clipped)
            
            return x_0_guided

        # Run Native Sample Loop entirely inside the Haiku context!
        return diffusion_head.sample(
            denoising_step=guided_denoising_step,
            batch=batch,
            key=sample_key,
            config=sample_config
        )

# -------------------------------------------------------------------------
# MODEL RUNNER
# -------------------------------------------------------------------------
class ModelRunner:
    def __init__(self, config, device, model_dir):
        self._model_config = config
        self._device = device
        self._model_dir = model_dir

    @functools.cached_property
    def model_params(self):
        return params.get_model_haiku_params(model_dir=self._model_dir)

    @functools.cached_property
    def get_conditionings(self):
        """Runs the Evoformer Trunk to get Pair and Single embeddings."""
        @hk.transform
        def forward_trunk(batch_dict):
            batch = feat_batch.Batch.from_data_dict(batch_dict)
            return TrunkWrapper(self._model_config)(batch)

        return functools.partial(jax.jit(forward_trunk.apply, device=self._device), self.model_params)

    @functools.cached_property
    def evaluate_vector_field(self):
        """Evaluates the Diffusion Head to denoise positions."""
        @hk.transform
        def forward_diffusion(positions_noisy, noise_level, batch_dict, embeddings):
            batch = feat_batch.Batch.from_data_dict(batch_dict)
            return DiffusionWrapper(self._model_config)(
                positions_noisy, noise_level, batch, embeddings
            )

        return functools.partial(jax.jit(forward_diffusion.apply, device=self._device), self.model_params)

    @functools.cached_property
    def sample_guided_diffusion(self):
        """Executes the entire Guided Diffusion Loop natively inside Haiku."""
        @hk.transform
        def forward_sample(batch_dict, embeddings, grad_fn, sample_key):
            batch = feat_batch.Batch.from_data_dict(batch_dict)
            return GuidedDiffusionWrapper(self._model_config)(batch, embeddings, grad_fn, sample_key)

        # Note: grad_fn is a Python closure, so it MUST be marked as a static_argnum.
        # Apply signature: (params, rng, batch_dict, embeddings, grad_fn, sample_key)
        # Index 4 is grad_fn.
        return functools.partial(
            jax.jit(forward_sample.apply, static_argnums=(4,), device=self._device),
            self.model_params
        )
