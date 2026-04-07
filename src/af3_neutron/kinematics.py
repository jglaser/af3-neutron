import jax.numpy as jnp

def generalized_nerf_layer(heavy_coords, rotor_table, chi_angles):
    """
    JAX-vectorized NeRF. Calculates ideal 3D coordinates for all hydrogens 
    simultaneously based on heavy-atom anchors and trainable chi_angles.
    """
    p2 = heavy_coords[rotor_table["parent_idx"]]       
    p1 = heavy_coords[rotor_table["grandparent_idx"]]  
    p0 = heavy_coords[rotor_table["greatgrand_idx"]]   
    
    # Gram-Schmidt Basis Vectors
    v1 = p2 - p1
    v2 = p1 - p0
    z_axis = v1 / jnp.linalg.norm(v1, axis=-1, keepdims=True)
    x_axis_raw = jnp.cross(v2, z_axis)
    x_axis = x_axis_raw / jnp.linalg.norm(x_axis_raw, axis=-1, keepdims=True)
    y_axis = jnp.cross(z_axis, x_axis)
    
    R = jnp.stack([x_axis, y_axis, z_axis], axis=-1)
    
    r = rotor_table["ideal_r"]
    theta = jnp.radians(rotor_table["ideal_theta"])
    
    # Local Spherical coordinates
    local_coords = jnp.stack([
        r * jnp.sin(theta) * jnp.cos(chi_angles),
        r * jnp.sin(theta) * jnp.sin(chi_angles),
        -r * jnp.cos(theta)
    ], axis=-1)
    
    # Global Coordinate Transformation
    global_coords = jnp.einsum('nij,nj->ni', R, local_coords) + p2
    return global_coords
