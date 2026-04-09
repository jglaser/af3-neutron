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

    def __call__(self, batch, embeddings, grad_fn, sample_key, initial_chis, num_waters):
        sample_config = self.config.heads.diffusion.eval

        # --- THE PSEUDO-ATOM SMUGGLE ---
        orig_mask = batch.predicted_structure_info.atom_mask
        num_tokens = orig_mask.shape[-2]
        orig_A = orig_mask.shape[-1]

        num_rotors = initial_chis.shape[0]
        num_floats = num_rotors + num_waters * 3

        # FIX: Use standard Python math so JAX doesn't convert the shape into a Tracer!
        if num_floats > 0:
            import math
            N_extra = math.ceil(num_floats / (num_tokens * 3.0))
        else:
            N_extra = 0

        if N_extra > 0:
            pad_shape = orig_mask.shape[:-1] + (N_extra,)
            pad_mask = jnp.zeros(pad_shape, dtype=orig_mask.dtype) # Mask is 0 so augmentations ignore them
            padded_mask = jnp.concatenate([orig_mask, pad_mask], axis=-1)
            
            # FIX: Safely replace the mask anywhere it appears in the PyTree
            def replace_mask(x):
                if id(x) == id(orig_mask):
                    return padded_mask
                return x
            padded_batch = jax.tree_util.tree_map(replace_mask, batch)
        else:
            padded_batch = batch

        def guided_denoising_step(positions_noisy, t_hat):
            if N_extra > 0:
                # SDE provides an unbatched tensor: (num_tokens, orig_A + N_extra, 3)
                real_coords = positions_noisy[..., :orig_A, :]
                angle_trackers = positions_noisy[..., orig_A:, :]
                
                # Flatten the trackers to extract our continuous variables
                flat_angles = angle_trackers.reshape(-1) 
                chi = flat_angles[:num_rotors]
                
                water_flat = flat_angles[num_rotors:num_floats]
                water = water_flat.reshape((num_waters, 3)) if num_waters > 0 else jnp.zeros((0, 3))
            else:
                real_coords = positions_noisy
                chi = initial_chis
                water = jnp.zeros((0, 3))

            # MUST pass the original unpadded batch to avoid neural network shape crashes
            x_0_real = self.diffusion_module(
                positions_noisy=real_coords, noise_level=t_hat,
                batch=batch, embeddings=embeddings, use_conditioning=True
            )
            
            loss_val, (grad_x0, grad_chi, grad_water) = grad_fn(x_0_real, chi, water)
            
            lr_heavy = 0.05
            x_0_guided = x_0_real - (lr_heavy * jnp.clip(grad_x0, -1.0, 1.0))
            
            if N_extra > 0:
                # Pack the physics gradients back into the pseudo-atom shape
                flat_grad = jnp.concatenate([
                    grad_chi.reshape(-1), 
                    grad_water.reshape(-1), 
                    jnp.zeros(num_tokens * N_extra * 3 - num_floats)
                ])
                grad_trackers = flat_grad.reshape(angle_trackers.shape)
                
                # By offsetting the noisy state by our gradient, the SDE step naturally performs SGD!
                denoised_angles = angle_trackers + grad_trackers
                positions_denoised = jnp.concatenate([x_0_guided, denoised_angles], axis=-2)
            else:
                positions_denoised = x_0_guided

            return positions_denoised

        sample_results = diffusion_head.sample(
            denoising_step=guided_denoising_step, batch=padded_batch, 
            key=sample_key, config=sample_config
        )
       
        pos_tensor = sample_results['atom_positions']
        
        if N_extra > 0:
            final_positions = pos_tensor[..., :orig_A, :]
            final_trackers = pos_tensor[..., orig_A:, :]
            
            # final_trackers shape is (num_samples, num_tokens, N_extra, 3)
            # We flatten everything except num_samples
            num_samples = pos_tensor.shape[0]
            flat_angles = final_trackers.reshape(num_samples, -1)
            
            chi_out = flat_angles[:, :num_rotors]
            water_flat = flat_angles[:, num_rotors:num_floats]
            water_out = water_flat.reshape((num_samples, num_waters, 3)) if num_waters > 0 else jnp.zeros((num_samples, 0, 3))
        else:
            final_positions = sample_results
            num_samples = sample_config.num_samples
            chi_out = jnp.tile(initial_chis[None, ...], (num_samples, 1))
            water_out = jnp.zeros((num_samples, 0, 3))

        return {'atom_positions': final_positions, 'chi_angles': chi_out, 'water_rotations': water_out}

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
        @hk.transform
        def forward_sample(batch_dict, embeddings, grad_fn, sample_key, initial_chis, num_waters):
            batch = feat_batch.Batch.from_data_dict(batch_dict)
            return GuidedDiffusionWrapper(self._model_config)(
                batch, embeddings, grad_fn, sample_key, initial_chis, num_waters
            )
            
        def apply_fn(params, rng, batch_dict, embeddings, grad_fn, sample_key, initial_chis, num_waters):
            # No state required!
            out = forward_sample.apply(
                params, rng, batch_dict, embeddings, grad_fn, sample_key, initial_chis, num_waters
            )
            return out

        return functools.partial(
            jax.jit(apply_fn, static_argnums=(4, 7), device=self._device), 
            self.model_params
        )
