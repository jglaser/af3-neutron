import logging
import jax
import jax.numpy as jnp
from alphafold3.model.components import utils
from alphafold3.model.network import diffusion_head
from alphafold3.model.network.diffusion_head import random_rotation

from .kinematics import generalized_nerf_layer, so3_water_layer

def placeholder_neutron_loss(x_full):
    return 0*jnp.mean(x_full ** 2) * 0.01

@jax.jit
def sfc_neutron_loss(x_full, sfc_instance):
    f_calc_complex = sfc_instance.Calc_Fprotein(atoms_position_tensor=x_full, NO_Bfactor=True, Return=True)
    f_calc_mag = jnp.abs(f_calc_complex)
    diff = f_calc_mag - sfc_instance.Fo
    loss = jnp.mean((diff ** 2) / (sfc_instance.SigF ** 2 + 1e-6))
    return loss * 0.01

def decoupled_crystallographic_loss(
    positions_denoised_flat, chi_angles, water_rotations, gather_idxs, 
    rotor_table, mapping, water_mapping, sfc_instance
):
    x_af3_flat = positions_denoised_flat[gather_idxs]
    
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
        
    if sfc_instance is not None:
        return sfc_neutron_loss(x_full, sfc_instance)
    else:
        return placeholder_neutron_loss(x_full)

grad_loss_fn = jax.value_and_grad(decoupled_crystallographic_loss, argnums=(0, 1, 2))

def run_neutron_guided_diffusion(
    vf_step_fn, batch, embeddings, initial_noise, gather_idxs,
    rotor_table, mapping, water_mapping, sfc_instance=None,
    diff_config=None, sample_key=None, physics_start_step=150
):
    n_steps = getattr(diff_config, 'steps', 200) if diff_config else 200
    logging.info(f"Starting FULLY COMPILED Stateless SDE Scan Loop ({n_steps} steps)...")
    logging.info(f"Physics SFC Guidance will activate at step {physics_start_step} to prevent early-noise divergence.")
    
    positions = jnp.expand_dims(initial_noise, axis=0)
    mask = jnp.expand_dims(batch['pred_dense_atom_mask'], axis=0)
    
    num_rotors = rotor_table["target_idx"].shape[0]
    num_waters = water_mapping["oxygen_source"].shape[0]
    
    chi_angles = jnp.zeros(num_rotors)
    water_rotations = jnp.zeros((num_waters, 3))
    
    lr_heavy = 0.05
    lr_chi = 0.1
    
    noise_levels = diffusion_head.noise_schedule(jnp.linspace(0, 1, n_steps + 1))
    
    gamma_0 = getattr(diff_config, 'gamma_0', 0.8) if diff_config else 0.8
    gamma_min = getattr(diff_config, 'gamma_min', 1.0) if diff_config else 1.0
    noise_scale_cfg = getattr(diff_config, 'noise_scale', 1.003) if diff_config else 1.003
    step_scale = getattr(diff_config, 'step_scale', 1.5) if diff_config else 1.5
    
    def sde_step_fn(carry, step):
        positions, chi_angles, water_rotations, key = carry
        
        noise_level_prev = noise_levels[step]
        noise_level = noise_levels[step + 1]

        key, key_noise, key_aug = jax.random.split(key, 3)
        step_key = jax.random.fold_in(jax.random.PRNGKey(0), step)
        
        # position augmentation
        rotation_key, translation_key = jax.random.split(key_aug)
        aug_R = random_rotation(rotation_key)
        translation = jax.random.normal(translation_key, shape=(3,))
        
        center = utils.mask_mean(mask[..., None], positions, axis=(-2, -3), keepdims=True, eps=1e-6)
        
        positions_aug = jnp.einsum('...i,ij->...j', positions - center, aug_R, precision=jax.lax.Precision.HIGHEST) + translation
        positions_aug = positions_aug * mask[..., None]

        # noise injection
        gamma = gamma_0 * (noise_level > gamma_min)
        t_hat = noise_level_prev * (1 + gamma)
        var_diff = jnp.clip(t_hat**2 - noise_level_prev**2, a_min=0.0)
        noise_scale = noise_scale_cfg * jnp.sqrt(var_diff)
        noise = noise_scale * jax.random.normal(key_noise, positions_aug.shape) * mask[..., None]
        
        positions_noisy_aug = positions_aug + noise

        # eval vector field
        t_hat_arr = jnp.array([t_hat])
        positions_denoised_aug = vf_step_fn(step_key, positions_noisy_aug[0], t_hat_arr, batch, embeddings)
        positions_denoised_aug = jnp.expand_dims(positions_denoised_aug, axis=0)
        
        grad_af3_aug = (positions_noisy_aug - positions_denoised_aug) / t_hat
        
        # step
        d_t = noise_level - t_hat
        positions_out_aug = positions_noisy_aug + step_scale * d_t * grad_af3_aug

        # physics reversal (SDE Frame -> Crystal Frame)
        positions_denoised_cryst = jnp.einsum('...i,ij->...j', positions_denoised_aug - translation, aug_R.T, precision=jax.lax.Precision.HIGHEST) + center
        positions_flat = positions_denoised_cryst[0].reshape((-1, 3))
        
        def compute_physics(pos_flat, chi, wat):
            loss_val, (g_pos, g_chi, g_wat) = grad_loss_fn(
                pos_flat, chi, wat, gather_idxs, 
                rotor_table, mapping, water_mapping, sfc_instance
            )

            jax.debug.print("SDE Step {s} | Phase: Physics Active | SFC Loss: {l}", s=step, l=loss_val)
            return jnp.clip(g_pos, -1.0, 1.0), jnp.clip(g_chi, -0.1, 0.1), jnp.clip(g_wat, -0.1, 0.1)

        def skip_physics(pos_flat, chi, wat):
            jax.debug.print("SDE Step {s} | Phase: Pure Denoising", s=step)
            return jnp.zeros_like(pos_flat), jnp.zeros_like(chi), jnp.zeros_like(wat)

        do_physics = step >= physics_start_step
        
        grad_positions, grad_chi, grad_water = jax.lax.cond(
            do_physics,
            compute_physics,
            skip_physics,
            positions_flat, chi_angles, water_rotations
        )
            
        # 6. apply force and reverse updated state 
        positions_out_cryst = jnp.einsum('...i,ij->...j', positions_out_aug - translation, aug_R.T, precision=jax.lax.Precision.HIGHEST) + center
        
        grad_positions = grad_positions.reshape(positions_denoised_cryst[0].shape)
        grad_positions = jnp.expand_dims(grad_positions, axis=0)

        new_positions = positions_out_cryst - (lr_heavy * grad_positions)
        new_positions = new_positions * mask[..., None]
        
        new_chi_angles = chi_angles - (lr_chi * grad_chi)
        new_water_rotations = water_rotations - (lr_chi * grad_water)
        
        # Return new state (carry) and the DENOISED prediction for extraction
        new_carry = (new_positions, new_chi_angles, new_water_rotations, key)
        return new_carry, positions_denoised_cryst[0]
        
    # scan phase
    loop_keys = jax.random.split(sample_key, 1)
    initial_carry = (positions, chi_angles, water_rotations, loop_keys[0])
    steps_array = jnp.arange(n_steps)
   
    final_carry, all_denoised = jax.lax.scan(sde_step_fn, initial_carry, steps_array)
    
    final_chis = final_carry[1]
    final_waters = final_carry[2]
    final_denoised_cryst = all_denoised[-1] # Extract the denoised output of the final step (t=0)
    
    return final_denoised_cryst * mask[0, ..., None], final_chis, final_waters

def generate_final_oracle_coords(positions_denoised_cryst, chi_angles, water_rotations, gather_idxs, rotor_table, mapping, water_mapping):
    positions_flat = positions_denoised_cryst.reshape((-1, 3))
    x_af3_flat = positions_flat[gather_idxs]

    # # Final Kabsch Alignment to experimental MTZ Unit Cell
    # x_drift_heavy = x_af3_flat[mapping["af3_source"]]
    # x_ref_heavy = reference_coords[mapping["af3_source"]]
    
    # com_drift = jnp.mean(x_drift_heavy, axis=0)
    # com_ref = jnp.mean(x_ref_heavy, axis=0)
    
    # p = x_drift_heavy - com_drift
    # q = x_ref_heavy - com_ref
    
    # H = jnp.einsum('ni,nj->ij', p, q)
    # U, S, Vt = jnp.linalg.svd(H)
    # d = jnp.sign(jnp.linalg.det(U) * jnp.linalg.det(Vt))
    # R = U @ jnp.diag(jnp.array([1.0, 1.0, d])) @ Vt
    
    # x_af3_aligned = (x_af3_flat - com_drift) @ R + com_ref
    
    # Assemble full complex in experimental frame
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
        
    return x_full
