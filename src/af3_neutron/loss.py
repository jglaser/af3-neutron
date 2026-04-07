import jax
import jax.numpy as jnp
from .kinematics import generalized_nerf_layer

def placeholder_neutron_loss(all_coords):
    """
    PLACEHOLDER: Replace this with SFC_Jax likelihood against Careless MTZ.
    """
    return jnp.sum(all_coords ** 2)

def total_crystallographic_loss(heavy_coords, chi_angles, rotor_table, original_coords):
    """
    Merges the AF3 heavy atoms with the NeRF hydrogens, then calculates the loss.
    """
    if rotor_table["target_idx"].shape[0] == 0:
        return placeholder_neutron_loss(original_coords)
        
    h_coords = generalized_nerf_layer(heavy_coords, rotor_table, chi_angles)
    
    all_coords = original_coords.at[rotor_table["target_idx"]].set(h_coords)
    all_coords = all_coords.at[rotor_table["parent_idx"]].set(heavy_coords[rotor_table["parent_idx"]])
    
    return placeholder_neutron_loss(all_coords)

def get_grad_loss_fn():
    """
    Returns the compiled auto-differentiator for the crystallographic loss.
    """
    return jax.value_and_grad(total_crystallographic_loss, argnums=(0, 1))
