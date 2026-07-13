import functools
import pathlib
from typing import Any, Callable, Dict

import haiku as hk
import jax
import jax.numpy as jnp

from alphafold3.model import model, params, feat_batch
from alphafold3.model.network import evoformer as evoformer_network
from alphafold3.model.network import diffusion_head
from alphafold3.model.network import confidence_head
from alphafold3.model import feat_batch

from .types import HostEmbeddings

def make_model_config(
    num_recycles: int = 10, num_diffusion_samples: int = 5
) -> model.Model.Config:
    config = model.Model.Config()
    config.global_config.flash_attention_implementation = "triton"
    config.heads.diffusion.eval.num_samples = num_diffusion_samples
    config.num_recycles = num_recycles
    return config

class _HostModule(hk.Module):
    def __init__(self, config: model.Model.Config, name: str = "diffuser"):
        super().__init__(name=name)
        self.config = config

class _HostTrunkWrapper(_HostModule):
    def __call__(self, batch: feat_batch.Batch) -> HostEmbeddings:
        embedding_module = evoformer_network.Evoformer(self.config.evoformer, self.config.global_config)
        target_feat = model.create_target_feat_embedding(batch, config=embedding_module.config, global_config=self.config.global_config)
        
        def recycle_body(_, args):
            prev, key = args
            key, subkey = jax.random.split(key)
            embs = embedding_module(batch=batch, prev=prev, target_feat=target_feat, key=subkey)
            embs["pair"] = embs["pair"].astype(jnp.float32)
            embs["single"] = embs["single"].astype(jnp.float32)
            return embs, key

        num_res = batch.num_res
        embeddings = {
            "pair": jnp.zeros([num_res, num_res, self.config.evoformer.pair_channel], dtype=jnp.float32),
            "single": jnp.zeros([num_res, self.config.evoformer.seq_channel], dtype=jnp.float32),
            "target_feat": target_feat,
        }

        key = hk.next_rng_key()
        embeddings, _ = hk.fori_loop(0, self.config.num_recycles + 1, recycle_body, (embeddings, key))
        return HostEmbeddings(pair=embeddings["pair"], single=embeddings["single"], target_feat=target_feat)

class _HostDiffusionWrapper(_HostModule):
    def __init__(self, config: model.Model.Config, name: str = "diffuser"):
        super().__init__(config, name=name)
        self.diffusion_module = diffusion_head.DiffusionHead(self.config.heads.diffusion, self.config.global_config)

    def __call__(self, positions_noisy: jnp.ndarray, noise_level: jnp.ndarray, batch: feat_batch.Batch, embeddings: HostEmbeddings) -> jnp.ndarray:
        return self.diffusion_module(
            positions_noisy=positions_noisy, noise_level=noise_level, batch=batch,
            embeddings={"pair": embeddings.pair, "single": embeddings.single, "target_feat": embeddings.target_feat},
            use_conditioning=True,
        )

class _DiffusionHijackWrapper(_HostModule):
    """Evaluates the unguided structural trajectory updates inside the tracking head."""
    def __init__(self, config: model.Model.Config, name: str = "diffuser"):
        super().__init__(config, name=name)
        self.diffusion_module = diffusion_head.DiffusionHead(self.config.heads.diffusion, self.config.global_config)

    def __call__(self, batch: feat_batch.Batch, embeddings: HostEmbeddings, sample_key: jnp.ndarray, proximal_fn: Callable) -> jnp.ndarray:
        sample_config = self.config.heads.diffusion.eval

        def hijacked_denoising_step(positions_noisy: jnp.ndarray, t_hat: jnp.ndarray) -> jnp.ndarray:
            # 1. Evaluate native unguided structural prediction target (\hat{x}_0)
            x_0_real = self.diffusion_module(
                positions_noisy=positions_noisy, noise_level=t_hat, batch=batch,
                embeddings={"pair": embeddings.pair, "single": embeddings.single, "target_feat": embeddings.target_feat},
                use_conditioning=True,
            )
            
            # 2. Evaluate proximal_fn natively on the device
            return proximal_fn(x_0_real, t_hat)

        sample_results = diffusion_head.sample(denoising_step=hijacked_denoising_step, batch=batch, key=sample_key, config=sample_config)
        return sample_results["atom_positions"]

class HostRunner:
    def __init__(self, config: model.Model.Config, device: jax.Device, model_dir: pathlib.Path):
        self._model_config = config
        self._device = device
        self._model_dir = model_dir

    @functools.cached_property
    def model_params(self) -> hk.Params:
        return params.get_model_haiku_params(model_dir=self._model_dir)

    @functools.cached_property
    def get_host_embeddings(self) -> Callable:
        @hk.transform
        def forward_trunk(batch_dict: Dict[str, Any]) -> HostEmbeddings:
            return _HostTrunkWrapper(self._model_config)(feat_batch.Batch.from_data_dict(batch_dict))
        return functools.partial(jax.jit(forward_trunk.apply, device=self._device), self.model_params)

    @functools.cached_property
    def predict_host_vector_field(self) -> Callable:
        @hk.transform
        def forward_diffusion(positions_noisy: jnp.ndarray, noise_level: jnp.ndarray, batch_dict: Dict[str, Any], embeddings: HostEmbeddings) -> jnp.ndarray:
            return _HostDiffusionWrapper(self._model_config)(positions_noisy, noise_level, feat_batch.Batch.from_data_dict(batch_dict), embeddings)
        return functools.partial(jax.jit(forward_diffusion.apply, device=self._device), self.model_params)

    @functools.cached_property
    def sample_guided_diffusion(self) -> Callable:
        @hk.transform
        def forward_sample(batch_dict: Dict[str, Any], embeddings: HostEmbeddings, sample_key: jnp.ndarray, proximal_fn: Callable) -> jnp.ndarray:
            return _DiffusionHijackWrapper(self._model_config)(feat_batch.Batch.from_data_dict(batch_dict), embeddings, sample_key, proximal_fn)

        return functools.partial(jax.jit(forward_sample.apply, static_argnames='proximal_fn', device=self._device), self.model_params)

    def predict_confidence(self, sample_key, batch_dict, embeddings, positions_denoised):
        """
        Evaluates the AF3 Confidence Head using the denoised positions to extract pLDDT.
        """
        # The true positional signature for AF3 ConfidenceHead:
        def forward_confidence(pos, emb, seq_mask_arr, token_to_pseudo, asym_id_arr):
            head = confidence_head.ConfidenceHead(
                self._model_config.heads.confidence,
                self._model_config.global_config
            )
                
            # Convert HostEmbeddings dataclass to a subscriptable dict
            if not isinstance(emb, dict):
                emb = {
                    "pair": emb.pair,
                    "single": emb.single,
                    "target_feat": emb.target_feat
                }
                
            # Call positionally in the EXACT order AF3 expects
            return head(pos, emb, seq_mask_arr, token_to_pseudo, asym_id_arr)

        confidence_fn = hk.transform(forward_confidence)
        
        # Rebuild the Batch object outside to safely extract required topology arrays
        batch_obj = feat_batch.Batch.from_data_dict(batch_dict)
        
        # Extract the specific tensors needed by the network
        token_to_pseudo = batch_obj.pseudo_beta_info.token_atoms_to_pseudo_beta
        asym_id = batch_obj.token_features.asym_id
        seq_mask = batch_obj.token_features.mask

        # DYNAMIC HAIKU RE-SCOPING
        confidence_params = {}
        for mod_name, param_dict in self.model_params.items():
            if "confidence_head" in mod_name:
                idx = mod_name.find("confidence_head")
                new_mod_name = mod_name[idx:]
                confidence_params[new_mod_name] = param_dict

        if not confidence_params:
            raise ValueError("Could not locate 'confidence_head' weights in model_params.")

        # Evaluate the confidence head using the properly scoped params
        confidence_dict = confidence_fn.apply(
            confidence_params, 
            sample_key, 
            positions_denoised,  # 1st: pred_positions
            embeddings,          # 2nd: embeddings
            seq_mask,            # 3rd: seq_mask array
            token_to_pseudo,     # 4th: token_atoms_to_pseudo_beta
            asym_id              # 5th: asym_id
        )
        
        return confidence_dict
