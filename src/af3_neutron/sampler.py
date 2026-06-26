from typing import Any, Optional, Tuple
import jax
import jax.numpy as jnp

from .kinematics import generalized_nerf_layer, so3_water_layer
from .runner import HostRunner
from .types import HostEmbeddings, OracleMapping


def placeholder_neutron_loss(x_full: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean(x_full**2) * 0.01


def sfc_neutron_loss(x_full: jnp.ndarray, sfc_instance: Any) -> jnp.ndarray:
    """Calculates the experimental structural objective function."""
    f_calc = sfc_instance.Calc_Fprotein(
        atoms_position_tensor=x_full, NO_Bfactor=True, Return=True
    )
    diff = jnp.abs(f_calc) - sfc_instance.Fo
    return jnp.mean((diff**2) / (sfc_instance.SigF**2 + 1e-6)) * 0.01


def hijack_physics_loss(
    positions_denoised: jnp.ndarray,
    chi_angles: jnp.ndarray,
    water_rotations: jnp.ndarray,
    gather_idxs: jnp.ndarray,
    oracle_mapping: OracleMapping,
    sfc_instance: Optional[Any],
) -> jnp.ndarray:
    """Assembles structural primitives across Host frames and returns crystallographic gradients."""
    x_af3_flat = positions_denoised.reshape(-1, 3)[gather_idxs]

    x_full = oracle_mapping.assemble_coordinates(x_af3_flat, chi_angles, water_rotations)

    if sfc_instance is not None:
        return jnp.mean(
            jnp.vectorize(
                lambda x: sfc_neutron_loss(x, sfc_instance), signature="(n,d)->()"
            )(x_full)
        )
    return placeholder_neutron_loss(x_full)


def run_diffusion_hijack(
    model_runner: HostRunner,
    batch_dict: dict,
    embeddings: HostEmbeddings,
    gather_idxs: jnp.ndarray,
    oracle_mapping: OracleMapping,
    sfc_instance: Optional[Any] = None,
    sample_key: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Intercepts and steers Host diffusion trajectories."""

    def single_sample_loss_fn(p_single, c_single, w_single):
        return hijack_physics_loss(
            p_single.reshape((-1, 3)),
            c_single,
            w_single,
            gather_idxs,
            oracle_mapping,
            sfc_instance,
        )

    grad_fn = jax.value_and_grad(single_sample_loss_fn, argnums=(0, 1, 2))
    sample_results = model_runner.sample_guided_diffusion(
        jax.random.PRNGKey(0),
        batch_dict,
        embeddings,
        grad_fn,
        sample_key,
        oracle_mapping.rotor_table.initial_chi,
        oracle_mapping.water_mapping.oxygen_source.shape[0],
    )
    return (
        sample_results["atom_positions"],
        sample_results["chi_angles"],
        sample_results["water_rotations"],
    )


def assemble_hijacked_complex(
    positions_denoised_final: jnp.ndarray,
    chi_angles: jnp.ndarray,
    water_rotations: jnp.ndarray,
    gather_idxs: jnp.ndarray,
    oracle_mapping: OracleMapping,
    reference_coords: jnp.ndarray,
) -> jnp.ndarray:
    """Snaps coordinates back into the crystal's global reference frame via Kabsch alignment."""
    x_af3_flat = positions_denoised_final.reshape((-1, 3))[gather_idxs]

    p = x_af3_flat[oracle_mapping.source_indices] - jnp.mean(x_af3_flat[oracle_mapping.source_indices], axis=0)
    q = reference_coords[oracle_mapping.heavy_indices] - jnp.mean(reference_coords[oracle_mapping.heavy_indices], axis=0)

    U, _, Vt = jnp.linalg.svd(jnp.einsum("ni,nj->ij", p, q))

    thingy = jnp.array([1.0, 1.0, jnp.sign(jnp.linalg.det(U) * jnp.linalg.det(Vt))])
    R = U @ jnp.diag(thingy) @ Vt

    x_af3_aligned = (
        x_af3_flat - jnp.mean(x_af3_flat[oracle_mapping.source_indices], axis=0)
    ) @ R + jnp.mean(reference_coords[oracle_mapping.heavy_indices], axis=0)

    x_full = (
        jnp.zeros((oracle_mapping.num_atoms, 3))
        .at[oracle_mapping.heavy_indices]
        .set(x_af3_aligned[oracle_mapping.source_indices])
    )
    if oracle_mapping.rotor_table.target_idx.shape[0] > 0:
        x_full = (x_full
            .at[oracle_mapping.rotor_table.target_idx]
            .set(
                 generalized_nerf_layer(x_af3_aligned, oracle_mapping.rotor_table, chi_angles)
                .reshape((oracle_mapping.rotor_table.target_idx.shape[0], 3))
            )
        )
    if oracle_mapping.water_mapping.oxygen_source.shape[0] > 0:
        h1, h2 = so3_water_layer(
            x_af3_aligned[oracle_mapping.water_mapping.oxygen_source],
            water_rotations
        )
        x_full = (x_full
            .at[oracle_mapping.water_mapping.h1_target]
            .set(h1)
            .at[oracle_mapping.water_mapping.h2_target]
            .set(h2)
        )
    return x_full
