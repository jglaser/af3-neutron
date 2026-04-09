from typing import Any, Dict, Tuple

import jax.numpy as jnp


def safe_norm(x: jnp.ndarray, axis: int = -1, keepdims: bool = True):
    """Calculates L2 norm safely, preventing NaN gradients when x == 0."""
    return jnp.sqrt(jnp.sum(x**2, axis=axis, keepdims=keepdims) + 1e-8)


def generalized_nerf_layer(heavy_coords: jnp.ndarray, rotor_table: Dict[str, Any], chi_angles: jnp.ndarray) -> jnp.ndarray:
    """
    Places hydrogen atoms using the Natural Extension Reference Frame (NeRF) method.

    This layer constructs a local coordinate system based on three heavy atom
    anchors (parent, grandparent, great-grandparent) and places a hydrogen atom
    using internal coordinates (bond length, angle, and torsion).

    Parameters
    ----------
    heavy_coords : jnp.ndarray
        The 3D coordinates of all heavy atoms in the system.
    rotor_table : Dict[str, Any]
        A mapping containing indices and ideal internal geometries:
        - "parent_idx": Indices of the immediate heavy atom anchors.
        - "grandparent_idx": Indices of the second anchors.
        - "greatgrand_idx": Indices of the third anchors.
        - "ideal_r": Bond lengths between parent and target hydrogen.
        - "ideal_theta": Bond angles (in degrees) at the parent atom.
    chi_angles : jnp.ndarray
        The torsion angles (radians) for the target hydrogen atoms.
        Supports extra batch dimensions for multi-sample diffusion.

    Returns
    -------
    jnp.ndarray
        The 3D coordinates of the placed hydrogen atoms in the global frame.
    """
    p2 = heavy_coords[..., rotor_table["parent_idx"], :]
    p1 = heavy_coords[..., rotor_table["grandparent_idx"], :]
    p0 = heavy_coords[..., rotor_table["greatgrand_idx"], :]

    v1 = p2 - p1
    v2 = p1 - p0

    z_axis = v1 / safe_norm(v1)
    x_axis_raw = jnp.cross(v2, z_axis)
    x_axis = x_axis_raw / safe_norm(x_axis_raw)
    y_axis = jnp.cross(z_axis, x_axis)

    R = jnp.stack([x_axis, y_axis, z_axis], axis=-1)

    # Get the number of extra batch dimensions chi_angles has compared to r
    num_batch_dims = chi_angles.ndim - rotor_table["ideal_r"].ndim

    r = rotor_table["ideal_r"]
    theta_deg = rotor_table["ideal_theta"]
    for _ in range(num_batch_dims):
        r = jnp.expand_dims(r, axis=0)
        theta_deg = jnp.expand_dims(theta_deg, axis=0)

    theta = jnp.radians(theta_deg)

    # ----------------------------------------------------
    # FIX: Explicitly broadcast the Z-coordinate to match
    # the shape of the X and Y coordinates (which inherit
    # their shape from chi_angles).
    # ----------------------------------------------------
    x_coord = r * jnp.sin(theta) * jnp.cos(chi_angles)
    y_coord = r * jnp.sin(theta) * jnp.sin(chi_angles)
    z_coord = -r * jnp.cos(theta)

    z_coord_broadcasted = jnp.broadcast_to(z_coord, chi_angles.shape)

    local_coords = jnp.stack([x_coord, y_coord, z_coord_broadcasted], axis=-1)

    return jnp.einsum("...ij,...j->...i", R, local_coords) + p2


def so3_water_layer(oxygen_coords: jnp.ndarray, rotation_vectors: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Applies SO(3) rigid-body rotations to idealized water hydrogen atoms.

    Uses the Rodrigues' rotation formula to rotate a standard water molecule
    geometry around its oxygen center based on an axis-angle representation.

    Parameters
    ----------
    oxygen_coords : jnp.ndarray
        The 3D coordinates of the water oxygen atoms.
    rotation_vectors : jnp.ndarray
        Axis-angle vectors where the direction is the rotation axis and
        the magnitude is the rotation angle in radians.

    Returns
    -------
    Tuple[jnp.ndarray, jnp.ndarray]
        The 3D global coordinates for hydrogen 1 (h1) and hydrogen 2 (h2).
    """
    ideal_h1 = jnp.array([0.757, 0.586, 0.0])
    ideal_h2 = jnp.array([-0.757, 0.586, 0.0])

    theta = safe_norm(rotation_vectors, axis=-1, keepdims=True)
    u = rotation_vectors / theta

    cos_t = jnp.cos(theta)
    sin_t = jnp.sin(theta)

    ux, uy, uz = u[..., 0], u[..., 1], u[..., 2]
    zero = jnp.zeros_like(ux)

    K = jnp.stack(
        [
            jnp.stack([zero, -uz, uy], axis=-1),
            jnp.stack([uz, zero, -ux], axis=-1),
            jnp.stack([-uy, ux, zero], axis=-1),
        ],
        axis=-2,
    )

    K2 = jnp.einsum("...ij,...jk->...ik", K, K)

    R = jnp.eye(3) + sin_t[..., jnp.newaxis] * K + (1 - cos_t[..., jnp.newaxis]) * K2

    h1_global = jnp.einsum("...ij,j->...i", R, ideal_h1) + oxygen_coords
    h2_global = jnp.einsum("...ij,j->...i", R, ideal_h2) + oxygen_coords

    return h1_global, h2_global
