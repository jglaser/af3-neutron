import logging
import jax
import jax.numpy as jnp
from alphafold3.model.network import diffusion_head
from .kinematics import generalized_nerf_layer, so3_water_layer

def placeholder_neutron_loss(x_full):
    """Dummy loss used when no MTZ file is provided."""
    return jnp.mean(x_full ** 2) * 0.01

def sfc_neutron_loss(x_full, sfc_instance):
    """Calculates L2 loss against experimental neutron structure factors."""
    f_calc_complex = sfc_instance.Calc_Fprotein(
        atoms_position_tensor=x_full, 
        NO_Bfactor=True,
        Return=True
    )
    f_calc_mag = jnp.abs(f_calc_complex)
    diff = f_calc_mag - sfc_instance.Fo
    loss = jnp.mean((diff ** 2) / (sfc_instance.SigF ** 2 + 1e-6))
    return loss * 0.01

def decoupled_crystallographic_loss(
    x_af3_flat, chi_angles, water_rotations, rotor_table, mapping, water_mapping, sfc_instance=None
):
    x_full = jnp.zeros((mapping["num_oracle_atoms"], 3))
    
    x_full = x_full.at[mapping["oracle_heavy"]].set(x_af3_flat[mapping["af3_source"]])
    
    if rotor_table["target_idx"].shape[0] > 0:
        x_h = generalized_nerf_layer(x_af3_flat, rotor_table, chi_angles)
        x_full = x_full.at[rotor_table["target_idx"]].set(x_h)
        
    if water_mapping["oxygen_source"].shape[0] > 0:
        oxygen_coords = x_af3_flat[water_mapping["oxygen_source"]]
        h1, h2 = so3_water_layer(oxygen_coords, water_rotations)
        x_full = x_full.at[water_mapping["h1_target"]].set(h1)
        x_full = x_full.at[water_mapping["h2_target"]].set(h2)
        
    # --- OPTIONAL SFC ROUTING ---
    if sfc_instance is not None:
        return sfc_neutron_loss(x_full, sfc_instance)
    else:
        return placeholder_neutron_loss(x_full)

grad_loss_fn = jax.value_and_grad(decoupled_crystallographic_loss, argnums=(0, 1, 2))

def run_neutron_guided_diffusion(
    vf_step_fn, batch, embeddings, initial_noise, 
    rotor_table, mapping, water_mapping, sfc_instance=None, n_steps=20
):
    logging.info("Starting Dual-Kinematic ODE Loop...")
    
    positions = initial_noise
    mask = batch['pred_dense_atom_mask']
    
    num_rotors = rotor_table["target_idx"].shape[0]
    num_waters = water_mapping["oxygen_source"].shape[0]
    
    chi_angles = jnp.zeros(num_rotors)
    water_rotations = jnp.zeros((num_waters, 3))
    
    lr_heavy = 0.05
    lr_chi = 0.1
    
    noise_levels = diffusion_head.noise_schedule(jnp.linspace(0, 1, n_steps + 1))
    key = jax.random.PRNGKey(42)
    
    for step in range(n_steps):
        noise_level_prev = noise_levels[step]
        noise_level = noise_levels[step + 1]
        
        key, step_key = jax.random.split(key)
        t_hat = jnp.array([noise_level_prev])
        
        positions_denoised = vf_step_fn(step_key, positions, t_hat, batch, embeddings)
        grad_af3 = (positions - positions_denoised) / t_hat
        
        x_0_flat = positions_denoised.reshape((-1, 3))
        
        loss_val, (grad_heavy_flat, grad_chi, grad_water) = grad_loss_fn(
            x_0_flat, chi_angles, water_rotations, rotor_table, mapping, water_mapping, sfc_instance
        )
        
        grad_heavy_flat = jnp.clip(grad_heavy_flat, -1.0, 1.0)
        grad_chi = jnp.clip(grad_chi, -0.1, 0.1)
        grad_water = jnp.clip(grad_water, -0.1, 0.1)
        
        loss_type = "SFC L2" if sfc_instance else "Dummy"
        logging.info(f"ODE Step {step} | {loss_type} Loss: {loss_val:.4f}")
        
        grad_heavy = grad_heavy_flat.reshape(positions_denoised.shape)
        d_t = noise_level - t_hat
        
        positions = positions + 1.5 * d_t * grad_af3 - (lr_heavy * grad_heavy)
        chi_angles = chi_angles - (lr_chi * grad_chi)
        water_rotations = water_rotations - (lr_chi * grad_water)
        
    return positions * mask[..., None], chi_angles, water_rotations
