from functools import partial
from typing import Any, Optional
import jax
import jax.numpy as jnp

from .types import OracleMapping

@jax.jit
def placeholder_neutron_loss(x_full: jnp.ndarray) -> jnp.ndarray:
    return jnp.array(0, dtype=x_full.dtype)


@partial(jax.jit, static_argnames="sfc_instance")
def sfc_neutron_loss(x_full: jnp.ndarray, sfc_instance: Any) -> jnp.ndarray:
    """Calculates the experimental structural objective function."""
    f_calc = sfc_instance.Calc_Fprotein(
        atoms_position_tensor=x_full, NO_Bfactor=True, Return=True
    )
    diff = jnp.abs(f_calc) - sfc_instance.Fo
    return jnp.mean((diff**2) / (sfc_instance.SigF**2 + 1e-6)) * 0.01


@partial(jax.jit, static_argnames="sfc_instance")
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

    x_drift_heavy = x_af3_flat[oracle_mapping.source_indices]
    x_ref_heavy = oracle_mapping.initial_coordinates[oracle_mapping.heavy_indices]

    avg_drift = jnp.mean(x_drift_heavy, axis=0)
    avg_ref = jnp.mean(x_ref_heavy, axis=0)

    p = x_drift_heavy - avg_drift
    q = x_ref_heavy - avg_ref

    H = jnp.einsum("ni,nj->ij", p, q)
    U, _, Vt = jnp.linalg.svd(H)
    d = jnp.sign(jnp.linalg.det(U) * jnp.linalg.det(Vt))
    R = U @ jnp.diag(jnp.array([1.0, 1.0, d])) @ Vt

    x_af3_aligned = (x_af3_flat - avg_drift) @ R + avg_ref

    x_full = oracle_mapping.assemble_coordinates(x_af3_aligned, chi_angles, water_rotations)

    if sfc_instance is not None:
        return jnp.mean(
            jnp.vectorize(
                lambda x: sfc_neutron_loss(x, sfc_instance), signature="(n,d)->()"
            )(x_full)
        )
    return placeholder_neutron_loss(x_full)
