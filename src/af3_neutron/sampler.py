from typing import Any, Optional, Tuple
import jax
import jax.numpy as jnp

from .kinematics import generalized_nerf_layer, so3_water_layer
from .runner import HostRunner
from .types import RotorTable, AtomMapping, WaterMapping, HostEmbeddings


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
    rotor_table: RotorTable,
    mapping: AtomMapping,
    water_mapping: WaterMapping,
    sfc_instance: Optional[Any],
) -> jnp.ndarray:
    """Assembles structural primitives across Host frames and returns crystallographic gradients."""
    x_af3_flat = positions_denoised.reshape(-1, 3)[gather_idxs]
    x_full = (
        jnp.zeros((mapping.num_oracle_atoms, 3))
        .at[mapping.oracle_heavy]
        .set(x_af3_flat[mapping.af3_source].reshape(mapping.oracle_heavy.shape[0], 3))
    )

    if rotor_table.target_idx.shape[0] > 0:
        rotor_dict = {
            "parent_idx": rotor_table.parent_idx,
            "grandparent_idx": rotor_table.grandparent_idx,
            "greatgrand_idx": rotor_table.greatgrand_idx,
            "ideal_r": rotor_table.ideal_r,
            "ideal_theta": rotor_table.ideal_theta,
        }
        x_full = x_full.at[rotor_table.target_idx].set(
            generalized_nerf_layer(x_af3_flat, rotor_dict, chi_angles).reshape(
                (rotor_table.target_idx.shape[0], 3)
            )
        )

    if water_mapping.oxygen_source.shape[0] > 0:
        h1, h2 = so3_water_layer(
            x_af3_flat[water_mapping.oxygen_source], water_rotations
        )
        x_full = (
            x_full.at[water_mapping.h1_target]
            .set(h1.reshape((water_mapping.h1_target.shape[0], 3)))
            .at[water_mapping.h2_target]
            .set(h2.reshape((water_mapping.h2_target.shape[0], 3)))
        )

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
    rotor_table: RotorTable,
    mapping: AtomMapping,
    water_mapping: WaterMapping,
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
            rotor_table,
            mapping,
            water_mapping,
            sfc_instance,
        )

    grad_fn = jax.value_and_grad(single_sample_loss_fn, argnums=(0, 1, 2))
    sample_results = model_runner.sample_guided_diffusion(
        jax.random.PRNGKey(0),
        batch_dict,
        embeddings,
        grad_fn,
        sample_key,
        rotor_table.initial_chi,
        water_mapping.oxygen_source.shape[0],
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
    rotor_table: RotorTable,
    mapping: AtomMapping,
    water_mapping: WaterMapping,
    reference_coords: jnp.ndarray,
) -> jnp.ndarray:
    """Snaps coordinates back into the crystal's global reference frame via Kabsch alignment."""
    x_af3_flat = positions_denoised_final.reshape((-1, 3))[gather_idxs]
    p, q = (
        x_af3_flat[mapping.af3_source]
        - jnp.mean(x_af3_flat[mapping.af3_source], axis=0),
        reference_coords[mapping.oracle_heavy]
        - jnp.mean(reference_coords[mapping.oracle_heavy], axis=0),
    )

    U, _, Vt = jnp.linalg.svd(jnp.einsum("ni,nj->ij", p, q))
    R = (
        U
        @ jnp.diag(
            jnp.array([1.0, 1.0, jnp.sign(jnp.linalg.det(U) * jnp.linalg.det(Vt))])
        )
        @ Vt
    )
    x_af3_aligned = (
        x_af3_flat - jnp.mean(x_af3_flat[mapping.af3_source], axis=0)
    ) @ R + jnp.mean(reference_coords[mapping.oracle_heavy], axis=0)

    x_full = (
        jnp.zeros((mapping.num_oracle_atoms, 3))
        .at[mapping.oracle_heavy]
        .set(x_af3_aligned[mapping.af3_source])
    )
    if rotor_table.target_idx.shape[0] > 0:
        rotor_dict = {
            "parent_idx": rotor_table.parent_idx,
            "grandparent_idx": rotor_table.grandparent_idx,
            "greatgrand_idx": rotor_table.greatgrand_idx,
            "ideal_r": rotor_table.ideal_r,
            "ideal_theta": rotor_table.ideal_theta,
        }
        x_full = x_full.at[rotor_table.target_idx].set(
            generalized_nerf_layer(x_af3_aligned, rotor_dict, chi_angles)
        )
    if water_mapping.oxygen_source.shape[0] > 0:
        h1, h2 = so3_water_layer(
            x_af3_aligned[water_mapping.oxygen_source], water_rotations
        )
        x_full = (
            x_full.at[water_mapping.h1_target]
            .set(h1)
            .at[water_mapping.h2_target]
            .set(h2)
        )
    return x_full
