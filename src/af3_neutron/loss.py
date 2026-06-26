from typing import Any, Optional
import jax.numpy as jnp

from .types import OracleMapping

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
