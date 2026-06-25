import functools
import math
import pathlib
from typing import Any, Callable, Dict


import haiku as hk
import jax
import jax.numpy as jnp

from alphafold3.model import model, params, feat_batch
from alphafold3.model.network import evoformer as evoformer_network
from alphafold3.model.network import diffusion_head

from .types import HostEmbeddings

def make_model_config(
    num_recycles: int = 10, num_diffusion_samples: int = 5
) -> model.Model.Config:
    """
    Generates the configuration object for the AlphaFold 3 model.

    Parameters
    ----------
    num_recycles : int, optional
        Number of recycling iterations for the Evoformer trunk, by default 10.
    num_diffusion_samples : int, optional
        Number of independent diffusion trajectories to generate, by default 5.

    Returns
    -------
    model.Model.Config
        The instantiated and populated AlphaFold 3 configuration.
    """
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
    "Evoformer trunk recycling loop"
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
    """Evaluates a single unguided pass of the native Diffusion Head."""
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
    """Hijacks the Host solver trajectory to update non-native physics state."""
    def __init__(self, config: model.Model.Config, name: str = "diffuser"):
        super().__init__(config, name=name)
        self.diffusion_module = diffusion_head.DiffusionHead(self.config.heads.diffusion, self.config.global_config)

    def __call__(self, batch: feat_batch.Batch, embeddings: HostEmbeddings, grad_fn: Callable, sample_key: jnp.ndarray, initial_chis: jnp.ndarray, num_waters: int) -> Dict[str, jnp.ndarray]:
        sample_config = self.config.heads.diffusion.eval
        orig_mask = batch.predicted_structure_info.atom_mask
        num_tokens, orig_A = orig_mask.shape[-2:]
        num_floats = initial_chis.shape[0] + num_waters * 3
        N_extra = math.ceil(num_floats / (num_tokens * 3.0)) if num_floats > 0 else 0

        if N_extra > 0:
            pad_mask = jnp.zeros(orig_mask.shape[:-1] + (N_extra,), dtype=orig_mask.dtype)
            padded_mask = jnp.concatenate([orig_mask, pad_mask], axis=-1)
            padded_batch = jax.tree_util.tree_map(lambda x: padded_mask if id(x) == id(orig_mask) else x, batch)
        else:
            padded_batch = batch

        def hijacked_denoising_step(positions_noisy: jnp.ndarray, t_hat: jnp.ndarray) -> jnp.ndarray:
            if N_extra > 0:
                real_coords = positions_noisy[..., :orig_A, :]
                flat_angles = positions_noisy[..., orig_A:, :].reshape(-1)
                chi = flat_angles[:initial_chis.shape[0]]
                water = flat_angles[initial_chis.shape[0]:num_floats].reshape((num_waters, 3)) if num_waters > 0 else jnp.zeros((0, 3))
            else:
                real_coords, chi, water = positions_noisy, initial_chis, jnp.zeros((0, 3))

            x_0_real = self.diffusion_module(
                positions_noisy=real_coords, noise_level=t_hat, batch=batch,
                embeddings={"pair": embeddings.pair, "single": embeddings.single, "target_feat": embeddings.target_feat},
                use_conditioning=True,
            )

            _, (grad_x0, grad_chi, grad_water) = grad_fn(x_0_real, chi, water)
            x_0_guided = x_0_real - (0.05 * jnp.clip(grad_x0, -1.0, 1.0))

            if N_extra > 0:
                flat_grad = jnp.concatenate([grad_chi.reshape(-1), grad_water.reshape(-1), jnp.zeros(num_tokens * N_extra * 3 - num_floats)])
                positions_denoised = jnp.concatenate([x_0_guided, (positions_noisy[..., orig_A:, :] + flat_grad.reshape(positions_noisy[..., orig_A:, :].shape))], axis=-2)
            else:
                positions_denoised = x_0_guided
            return positions_denoised

        sample_results = diffusion_head.sample(denoising_step=hijacked_denoising_step, batch=padded_batch, key=sample_key, config=sample_config)
        pos_tensor = sample_results["atom_positions"]

        if N_extra > 0:
            final_positions = pos_tensor[..., :orig_A, :]
            flat_angles = pos_tensor[..., orig_A:, :].reshape(pos_tensor.shape[0], -1)
            return {
                "atom_positions": final_positions,
                "chi_angles": flat_angles[:, :initial_chis.shape[0]],
                "water_rotations": flat_angles[:, initial_chis.shape[0]:num_floats].reshape((pos_tensor.shape[0], num_waters, 3)) if num_waters > 0 else jnp.zeros((pos_tensor.shape[0], 0, 3))
            }
        return {"atom_positions": pos_tensor, "chi_angles": jnp.tile(initial_chis[None, ...], (sample_config.num_samples, 1)), "water_rotations": jnp.zeros((sample_config.num_samples, 0, 3))}

class HostRunner:
    """Primary orchestration interface for compiled execution of Host primitives."""
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
        def forward_sample(batch_dict: Dict[str, Any], embeddings: HostEmbeddings, grad_fn: Callable, sample_key: jnp.ndarray, initial_chis: jnp.ndarray, num_waters: int) -> Dict[str, jnp.ndarray]:
            return _DiffusionHijackWrapper(self._model_config)(feat_batch.Batch.from_data_dict(batch_dict), embeddings, grad_fn, sample_key, initial_chis, num_waters)
        return functools.partial(jax.jit(forward_sample.apply, static_argnums=(4, 7), device=self._device), self.model_params)
