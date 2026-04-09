import functools
import pathlib
from typing import Any, Callable, Dict


import haiku as hk
import jax
import jax.numpy as jnp

from alphafold3.model import model, params, feat_batch

from ._wrapper import TrunkWrapper, DiffusionWrapper, GuidedDiffusionWrapper

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

# -------------------------------------------------------------------------
# MODEL RUNNER
# -------------------------------------------------------------------------
class ModelRunner:
    """
    Coordinates the execution of AlphaFold 3 modules by managing JIT compilation
    and state injection via Haiku parameter binding.

    Parameters
    ----------
    config : alphafold3.model.model.Model.Config
        The AlphaFold 3 configuration instance.
    device : jax.Device
        The target hardware accelerator (e.g., GPU).
    model_dir : pathlib.Path
        The directory containing the pre-trained DeepMind weights.
    """

    def __init__(
        self, config: model.Model.Config, device: jax.Device, model_dir: pathlib.Path
    ):
        self._model_config = config
        self._device = device
        self._model_dir = model_dir

    @functools.cached_property
    def model_params(self) -> hk.Params:
        """
        Loads the pre-trained AlphaFold 3 weights into memory.

        Returns
        -------
        hk.Params
            The Haiku parameter dictionary.
        """
        return params.get_model_haiku_params(model_dir=self._model_dir)

    @functools.cached_property
    def get_conditionings(self) -> Callable:
        """
        JIT-compiled function that executes the Evoformer trunk.

        Returns
        -------
        Callable
            A function taking a PRNG key and batch dictionary, returning
            the structural embeddings.
        """

        @hk.transform
        def forward_trunk(batch_dict: Dict[str, Any]) -> Dict[str, jnp.ndarray]:
            batch = feat_batch.Batch.from_data_dict(batch_dict)
            return TrunkWrapper(self._model_config)(batch)

        return functools.partial(
            jax.jit(forward_trunk.apply, device=self._device), self.model_params
        )

    @functools.cached_property
    def evaluate_vector_field(self) -> Callable:
        """
        JIT-compiled function for evaluating the diffusion vector field directly.

        Returns
        -------
        Callable
            A function taking `(rng, positions_noisy, noise_level, batch_dict, embeddings)`
            and returning the denoised positions.
        """

        @hk.transform
        def forward_diffusion(
            positions_noisy: jnp.ndarray,
            noise_level: jnp.ndarray,
            batch_dict: Dict[str, Any],
            embeddings: Dict[str, jnp.ndarray],
        ):
            batch = feat_batch.Batch.from_data_dict(batch_dict)
            return DiffusionWrapper(self._model_config)(
                positions_noisy, noise_level, batch, embeddings
            )

        return functools.partial(
            jax.jit(forward_diffusion.apply, device=self._device), self.model_params
        )

    @functools.cached_property
    def sample_guided_diffusion(self) -> Callable:
        """
        JIT-compiled function that executes the full SDE diffusion loop with
        differentiable crystallographic loss guidance.

        Returns
        -------
        Callable
            A function evaluating the batched multi-sample refinement pipeline.
        """
        @hk.transform
        def forward_sample(
            batch_dict: Dict[str, Any], embeddings: Dict[str, jnp.ndarray], grad_fn: Callable, sample_key: jnp.ndarray, initial_chis: jnp.ndarray, num_waters: int
        ) -> Dict[str, jnp.ndarray]:
            batch = feat_batch.Batch.from_data_dict(batch_dict)
            return GuidedDiffusionWrapper(self._model_config)(
                batch, embeddings, grad_fn, sample_key, initial_chis, num_waters
            )

        def apply_fn(
            params: hk.Params,
            rng: jnp.ndarray,
            batch_dict: Dict[str, Any],
            embeddings: Dict[str, jnp.ndarray],
            grad_fn: Callable,
            sample_key: jnp.ndarray,
            initial_chis: jnp.ndarray,
            num_waters: int,
        ) -> Dict[str, jnp.ndarray]:
            # No state required!
            out = forward_sample.apply(
                params,
                rng,
                batch_dict,
                embeddings,
                grad_fn,
                sample_key,
                initial_chis,
                num_waters,
            )
            return out

        return functools.partial(
            jax.jit(apply_fn, static_argnums=(4, 7), device=self._device),
            self.model_params,
        )
