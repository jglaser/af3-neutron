import functools
import math
import pathlib
from typing import Any, Callable, Dict, Tuple


import haiku as hk
import jax
import jax.numpy as jnp

from alphafold3.model import model, params, feat_batch
from alphafold3.model.network import evoformer as evoformer_network
from alphafold3.model.network import diffusion_head


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
# HAIKU NAMESPACE WRAPPERS
# These force our detached components to inherit the 'diffuser/' prefix
# so Haiku can find the DeepMind pre-trained weights.
# -------------------------------------------------------------------------
class TrunkWrapper(hk.Module):
    """
    Haiku module that executes the AlphaFold 3 Evoformer trunk.

    This wrapper isolates the Pairformer/Evoformer blocks, managing the
    recycling loop to generate the structural conditionings (single and pair
    embeddings) required by the diffusion head.

    Parameters
    ----------
    config : alphafold3.model.model.Model.Config
        The AlphaFold 3 configuration object.
    name : str, optional
        The Haiku variable scope name, by default 'diffuser'.
    """

    def __init__(self, config: model.Model.Config, name: str = "diffuser"):
        super().__init__(name=name)
        self.config = config

    def __call__(self, batch: feat_batch.Batch) -> Dict[str, jnp.ndarray]:
        """
        Executes the Evoformer trunk over the specified number of recycles.

        Parameters
        ----------
        batch : feat_batch.Batch
            The featurized input batch.

        Returns
        -------
        Dict[str, jnp.ndarray]
            A dictionary containing the 'pair', 'single', and 'target_feat'
            embeddings.
        """
        embedding_module = evoformer_network.Evoformer(
            self.config.evoformer, self.config.global_config
        )
        target_feat = model.create_target_feat_embedding(
            batch=batch,
            config=embedding_module.config,
            global_config=self.config.global_config,
        )

        def recycle_body(
            _, args: Tuple[Dict[str, jnp.ndarray], jnp.ndarray]
        ) -> Tuple[Dict[str, jnp.ndarray], jnp.ndarray]:
            prev, key = args
            key, subkey = jax.random.split(key)
            embeddings = embedding_module(
                batch=batch, prev=prev, target_feat=target_feat, key=subkey
            )
            embeddings["pair"] = embeddings["pair"].astype(jnp.float32)
            embeddings["single"] = embeddings["single"].astype(jnp.float32)
            return embeddings, key

        num_res = batch.num_res
        embeddings = {
            "pair": jnp.zeros(
                [num_res, num_res, self.config.evoformer.pair_channel],
                dtype=jnp.float32,
            ),
            "single": jnp.zeros(
                [num_res, self.config.evoformer.seq_channel], dtype=jnp.float32
            ),
            "target_feat": target_feat,
        }

        key = hk.next_rng_key()
        num_iter = self.config.num_recycles + 1
        embeddings, _ = hk.fori_loop(0, num_iter, recycle_body, (embeddings, key))

        # Inject target_feat back into the final dict as the diffusion head expects it
        embeddings["target_feat"] = target_feat
        return embeddings


# -------------------------------------------------------------------------
# THE BASELINE WRAPPER (For the 1-step Oracle initialization)
# -------------------------------------------------------------------------
class DiffusionWrapper(hk.Module):
    """
    Haiku module that provides a single forward pass of the AF3 Diffusion Head.

    Used primarily for extracting baseline unguided coordinate predictions
    (the "Oracle" initialization) prior to running the full SDE loop.

    Parameters
    ----------
    config : alphafold3.model.model.Model.Config
        The AlphaFold 3 configuration object.
    name : str, optional
        The Haiku variable scope name, by default 'diffuser'.
    """

    def __init__(self, config: model.Model.Config, name: str = "diffuser"):
        super().__init__(name=name)
        self.config = config
        self.diffusion_module = diffusion_head.DiffusionHead(
            self.config.heads.diffusion, self.config.global_config
        )

    def __call__(
        self,
        positions_noisy: jnp.ndarray,
        noise_level: jnp.ndarray,
        batch: feat_batch.Batch,
        embeddings: Dict[str, jnp.ndarray],
    ):
        """
        Evaluates the vector field for a given noisy state.

        Parameters
        ----------
        positions_noisy : jnp.ndarray
            The current noisy coordinate state.
        noise_level : jnp.ndarray
            The current noise scale (t_hat).
        batch : feat_batch.Batch
            The input features.
        embeddings : Dict[str, jnp.ndarray]
            The structural embeddings from the Trunk.

        Returns
        -------
        jnp.ndarray
            The predicted denoised positions (x_0).
        """
        return self.diffusion_module(
            positions_noisy=positions_noisy,
            noise_level=noise_level,
            batch=batch,
            embeddings=embeddings,
            use_conditioning=True,
        )


# -------------------------------------------------------------------------
# 2. THE GUIDED SDE WRAPPER (For the 200-step physics refinement)
# -------------------------------------------------------------------------
class GuidedDiffusionWrapper(hk.Module):
    """
    Haiku module that executes the SDE refinement loop with physical guidance.

    This class smuggles custom variables (hydrogen torsion angles and water
    rotations) into the native AF3 diffusion loop by padding the atom mask.
    This forces the native SDE solver to natively refine these auxiliary parameters
    alongside the protein backbone.

    Parameters
    ----------
    config : alphafold3.model.model.Model.Config
        The AlphaFold 3 configuration object.
    name : str, optional
        The Haiku variable scope name, by default 'diffuser'.
    """

    def __init__(self, config: model.Model.Config, name: str = "diffuser"):
        super().__init__(name=name)
        self.config = config
        self.diffusion_module = diffusion_head.DiffusionHead(
            self.config.heads.diffusion, self.config.global_config
        )

    def __call__(
        self,
        batch: feat_batch.Batch,
        embeddings: Dict[str, jnp.ndarray],
        grad_fn: Callable,
        sample_key: jnp.ndarray,
        initial_chis: jnp.ndarray,
        num_waters: int,
    ) -> Dict[str, jnp.ndarray]:
        """
        Executes the guided SDE sampling process.

        Parameters
        ----------
        batch : feat_batch.Batch
            The input feature batch.
        embeddings : Dict[str, jnp.ndarray]
            The structural conditionings from the Evoformer trunk.
        grad_fn : Callable
            The gradient function representing the crystallographic loss.
        sample_key : jnp.ndarray
            The PRNG key for the SDE stochasticity.
        initial_chis : jnp.ndarray
            The starting values for the hydrogen torsion angles.
        num_waters : int
            The number of orientable water molecules.

        Returns
        -------
        Dict[str, jnp.ndarray]
            A dictionary containing the final refined arrays:
            'atom_positions', 'chi_angles', and 'water_rotations'.
        """
        sample_config = self.config.heads.diffusion.eval

        # --- THE PSEUDO-ATOM SMUGGLE ---
        orig_mask = batch.predicted_structure_info.atom_mask
        num_tokens = orig_mask.shape[-2]
        orig_A = orig_mask.shape[-1]

        num_rotors = initial_chis.shape[0]
        num_floats = num_rotors + num_waters * 3

        # FIX: Use standard Python math so JAX doesn't convert the shape into a Tracer!
        if num_floats > 0:
            N_extra = math.ceil(num_floats / (num_tokens * 3.0))
        else:
            N_extra = 0

        if N_extra > 0:
            pad_shape = orig_mask.shape[:-1] + (N_extra,)
            pad_mask = jnp.zeros(
                pad_shape, dtype=orig_mask.dtype
            )  # Mask is 0 so augmentations ignore them
            padded_mask = jnp.concatenate([orig_mask, pad_mask], axis=-1)

            # FIX: Safely replace the mask anywhere it appears in the PyTree
            def replace_mask(x):
                if id(x) == id(orig_mask):
                    return padded_mask
                return x

            padded_batch = jax.tree_util.tree_map(replace_mask, batch)
        else:
            padded_batch = batch

        def guided_denoising_step(
            positions_noisy: jnp.ndarray, t_hat: jnp.ndarray
        ) -> jnp.ndarray:
            if N_extra > 0:
                # SDE provides an unbatched tensor: (num_tokens, orig_A + N_extra, 3)
                real_coords = positions_noisy[..., :orig_A, :]
                angle_trackers = positions_noisy[..., orig_A:, :]

                # Flatten the trackers to extract our continuous variables
                flat_angles = angle_trackers.reshape(-1)
                chi = flat_angles[:num_rotors]

                water_flat = flat_angles[num_rotors:num_floats]
                water = (
                    water_flat.reshape((num_waters, 3))
                    if num_waters > 0
                    else jnp.zeros((0, 3))
                )
            else:
                real_coords = positions_noisy
                chi = initial_chis
                water = jnp.zeros((0, 3))

            # MUST pass the original unpadded batch to avoid neural network shape crashes
            x_0_real = self.diffusion_module(
                positions_noisy=real_coords,
                noise_level=t_hat,
                batch=batch,
                embeddings=embeddings,
                use_conditioning=True,
            )

            loss_val, (grad_x0, grad_chi, grad_water) = grad_fn(x_0_real, chi, water)

            lr_heavy = 0.05
            x_0_guided = x_0_real - (lr_heavy * jnp.clip(grad_x0, -1.0, 1.0))

            if N_extra > 0:
                # Pack the physics gradients back into the pseudo-atom shape
                flat_grad = jnp.concatenate(
                    [
                        grad_chi.reshape(-1),
                        grad_water.reshape(-1),
                        jnp.zeros(num_tokens * N_extra * 3 - num_floats),
                    ]
                )
                grad_trackers = flat_grad.reshape(angle_trackers.shape)

                # By offsetting the noisy state by our gradient, the SDE step naturally performs SGD!
                denoised_angles = angle_trackers + grad_trackers
                positions_denoised = jnp.concatenate(
                    [x_0_guided, denoised_angles], axis=-2
                )
            else:
                positions_denoised = x_0_guided

            return positions_denoised

        sample_results = diffusion_head.sample(
            denoising_step=guided_denoising_step,
            batch=padded_batch,
            key=sample_key,
            config=sample_config,
        )

        pos_tensor = sample_results["atom_positions"]

        if N_extra > 0:
            final_positions = pos_tensor[..., :orig_A, :]
            final_trackers = pos_tensor[..., orig_A:, :]

            # final_trackers shape is (num_samples, num_tokens, N_extra, 3)
            # We flatten everything except num_samples
            num_samples = pos_tensor.shape[0]
            flat_angles = final_trackers.reshape(num_samples, -1)

            chi_out = flat_angles[:, :num_rotors]
            water_flat = flat_angles[:, num_rotors:num_floats]
            water_out = (
                water_flat.reshape((num_samples, num_waters, 3))
                if num_waters > 0
                else jnp.zeros((num_samples, 0, 3))
            )
        else:
            final_positions = sample_results
            num_samples = sample_config.num_samples
            chi_out = jnp.tile(initial_chis[None, ...], (num_samples, 1))
            water_out = jnp.zeros((num_samples, 0, 3))

        return {
            "atom_positions": final_positions,
            "chi_angles": chi_out,
            "water_rotations": water_out,
        }


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
