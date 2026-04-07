import jax.numpy as jnp

def safe_norm(x, axis=-1, keepdims=True):
    """Calculates L2 norm safely, preventing NaN gradients when x == 0."""
    return jnp.sqrt(jnp.sum(x**2, axis=axis, keepdims=keepdims) + 1e-8)

def generalized_nerf_layer(heavy_coords, rotor_table, chi_angles):
    p2 = heavy_coords[rotor_table["parent_idx"]]       
    p1 = heavy_coords[rotor_table["grandparent_idx"]]  
    p0 = heavy_coords[rotor_table["greatgrand_idx"]]   
    
    v1 = p2 - p1
    v2 = p1 - p0
    
    z_axis = v1 / safe_norm(v1)
    x_axis_raw = jnp.cross(v2, z_axis)
    x_axis = x_axis_raw / safe_norm(x_axis_raw)
    y_axis = jnp.cross(z_axis, x_axis)
    
    R = jnp.stack([x_axis, y_axis, z_axis], axis=-1)
    
    r = rotor_table["ideal_r"]
    theta = jnp.radians(rotor_table["ideal_theta"])
    
    local_coords = jnp.stack([
        r * jnp.sin(theta) * jnp.cos(chi_angles),
        r * jnp.sin(theta) * jnp.sin(chi_angles),
        -r * jnp.cos(theta)
    ], axis=-1)
    
    return jnp.einsum('nij,nj->ni', R, local_coords) + p2

def so3_water_layer(oxygen_coords, rotation_vectors):
    """
    Applies SO(3) rigid-body rotations to idealized water hydrogens.
    """
    # Idealized geometry (0.958 Angstroms, 104.5 degrees)
    ideal_h1 = jnp.array([0.757, 0.586, 0.0])
    ideal_h2 = jnp.array([-0.757, 0.586, 0.0])
    
    # Axis-Angle to Rotation Matrix (Rodrigues' Formula)
    theta = safe_norm(rotation_vectors, axis=-1, keepdims=True)
    u = rotation_vectors / theta
    
    cos_t = jnp.cos(theta)
    sin_t = jnp.sin(theta)
    
    ux, uy, uz = u[..., 0], u[..., 1], u[..., 2]
    zero = jnp.zeros_like(ux)
    
    K = jnp.stack([
        jnp.stack([zero, -uz, uy], axis=-1),
        jnp.stack([uz, zero, -ux], axis=-1),
        jnp.stack([-uy, ux, zero], axis=-1)
    ], axis=-2)
    
    I = jnp.eye(3)
    K2 = jnp.einsum('...ij,...jk->...ik', K, K)
    
    R = I + sin_t[..., jnp.newaxis] * K + (1 - cos_t[..., jnp.newaxis]) * K2
    
    # Rotate and translate to Oxygen
    h1_global = jnp.einsum('...ij,j->...i', R, ideal_h1) + oxygen_coords
    h2_global = jnp.einsum('...ij,j->...i', R, ideal_h2) + oxygen_coords
    
    return h1_global, h2_global
