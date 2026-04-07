import logging
import jax
import jax.numpy as jnp
from alphafold3.model.network import diffusion_head
from alphafold3.model.network.diffusion_head import random_augmentation, random_rotation
from alphafold3.model.components import utils
from .kinematics import generalized_nerf_layer, so3_water_layer

def placeholder_neutron_loss(x_full):
    return 0*jnp.mean(x_full ** 2) * 0.01

def sfc_neutron_loss(x_full, sfc_instance):
    f_calc_complex = sfc_instance.Calc_Fprotein(atoms_position_tensor=x_full, NO_Bfactor=True, Return=True)
    f_calc_mag = jnp.abs(f_calc_complex)
    diff = f_calc_mag - sfc_instance.Fo
    loss = jnp.mean((diff ** 2) / (sfc_instance.SigF ** 2 + 1e-6))
    return loss * 0.01

def decoupled_crystallographic_loss(
    positions_denoised, chi_angles, water_rotations, gather_idxs, 
    rotor_table, mapping, water_mapping, sfc_instance=None
):
    # FIX: Flatten the (N, 24, 3) tensor to (N*24, 3), then gather the valid atoms
    positions_flat = positions_denoised.reshape((-1, 3))
    x_af3_flat = positions_flat[gather_idxs]
    
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
    rotor_table, mapping, water_mapping, sfc_instance=None, n_steps=200,
    diff_config=None
):
    logging.info("Starting Dual-Kinematic SDE Loop (AF3 Native EDM + Augmentation)...")

    # Add the leading num_samples dimension (usually 1) for DeepMind compatibility
    positions = jnp.expand_dims(initial_noise, axis=0)
    mask = jnp.expand_dims(batch['pred_dense_atom_mask'], axis=0)

    num_rotors = rotor_table["target_idx"].shape[0]
    num_waters = water_mapping["oxygen_source"].shape[0]

    chi_angles = jnp.zeros(num_rotors)
    water_rotations = jnp.zeros((num_waters, 3))

    lr_heavy = 0.05
    lr_chi = 0.1

    noise_levels = diffusion_head.noise_schedule(jnp.linspace(0, 1, n_steps + 1))
    key = jax.random.PRNGKey(42)

    gamma_0 = getattr(diff_config, 'gamma_0', 0.0) if diff_config else 0.0
    gamma_min = getattr(diff_config, 'gamma_min', 0.0) if diff_config else 0.0
    noise_scale_cfg = getattr(diff_config, 'noise_scale', 1.0) if diff_config else 1.0
    step_scale = getattr(diff_config, 'step_scale', 1.0) if diff_config else 1.0

    for step in range(n_steps):
        noise_level_prev = noise_levels[step]
        noise_level = noise_levels[step + 1]

        key, step_key, key_noise, key_aug = jax.random.split(key, 4)

        # ---------------------------------------------------------
        # 1. EXTRACT EXACT NATIVE TRANSFORMS
        # ---------------------------------------------------------
        # We split the key exactly as DeepMind does to capture the actual R and T
        rotation_key, translation_key = jax.random.split(key_aug)
        aug_R = random_rotation(rotation_key)
        translation = jax.random.normal(translation_key, shape=(3,))

        # Calculate the exact center of mass used during augmentation
        center = utils.mask_mean(mask[..., None], positions, axis=(-2, -3), keepdims=True, eps=1e-6)

        # Now safely apply the native AF3 augmentation
        positions_aug = random_augmentation(rng_key=key_aug, positions=positions, mask=mask)

        # ---------------------------------------------------------
        # 2. NATIVE EDM NOISE INJECTION
        # ---------------------------------------------------------
        gamma = gamma_0 * (noise_level > gamma_min)
        t_hat = noise_level_prev * (1 + gamma)

        var_diff = jnp.clip(t_hat**2 - noise_level_prev**2, a_min=0.0)
        noise_scale = noise_scale_cfg * jnp.sqrt(var_diff)
        noise = noise_scale * jax.random.normal(key_noise, positions_aug.shape) * mask[..., None]

        positions_noisy = positions_aug + noise

        # ---------------------------------------------------------
        # 3. EVALUATE VECTOR FIELD (In Augmented Frame)
        # ---------------------------------------------------------
        t_hat_arr = jnp.array([t_hat])

        # Strip batch dim for the custom vf_step_fn, then add it back
        positions_denoised_aug = vf_step_fn(
            step_key, positions_noisy[0], t_hat_arr, batch, embeddings
        )
        positions_denoised_aug = jnp.expand_dims(positions_denoised_aug, axis=0)

        grad_af3_aug = (positions_noisy - positions_denoised_aug) / t_hat

        # ---------------------------------------------------------
        # 4. REVERSE AUGMENTATION TO CRYSTALLOGRAPHIC FRAME
        # ---------------------------------------------------------
        # Forward: x_aug = (x - center) @ R + T
        # Inverse: x     = (x_aug - T) @ R^T + center

        positions_denoised_cryst = jnp.einsum(
            '...i,ij->...j',
            positions_denoised_aug - translation,
            aug_R.T,
            precision=jax.lax.Precision.HIGHEST
        ) + center

        # The velocity gradient just needs the inverse rotation
        grad_af3_cryst = jnp.einsum(
            '...i,ij->...j',
            grad_af3_aug,
            aug_R.T,
            precision=jax.lax.Precision.HIGHEST
        )

        # ---------------------------------------------------------
        # 5. CALCULATE PHYSICS LOSS (In Crystallographic Frame)
        # ---------------------------------------------------------
        positions_flat = positions_denoised_cryst[0].reshape((-1, 3))

        loss_val, (grad_positions, grad_chi, grad_water) = grad_loss_fn(
            positions_flat, chi_angles, water_rotations, gather_idxs,
            rotor_table, mapping, water_mapping, sfc_instance
        )

        grad_positions = jnp.clip(grad_positions, -1.0, 1.0)
        grad_chi = jnp.clip(grad_chi, -0.1, 0.1)
        grad_water = jnp.clip(grad_water, -0.1, 0.1)

        loss_type = "SFC L2" if sfc_instance else "Dummy"
        if step % 10 == 0 or step == n_steps - 1:
            logging.info(f"ODE Step {step} | {loss_type} Loss: {loss_val:.4f}")

        # Reshape gradient back to AF3 padded tensor (with batch dim)
        grad_positions = grad_positions.reshape(positions_denoised_cryst[0].shape)
        grad_positions = jnp.expand_dims(grad_positions, axis=0)

        # ---------------------------------------------------------
        # 6. INTEGRATION UPDATE (In Crystallographic Frame)
        # ---------------------------------------------------------
        d_t = noise_level - t_hat

        # Notice we are stepping the original `positions`, not `positions_noisy`
        positions_out = positions + step_scale * d_t * grad_af3_cryst - (lr_heavy * grad_positions)

        positions = positions_out * mask[..., None]
        chi_angles = chi_angles - (lr_chi * grad_chi)
        water_rotations = water_rotations - (lr_chi * grad_water)

    return positions_denoised_cryst[0] * mask[0, ..., None], chi_angles, water_rotations

def generate_final_oracle_coords(positions_denoised, chi_angles, water_rotations, gather_idxs, rotor_table, mapping, water_mapping):
    # FIX: Flatten and gather here as well
    positions_flat = positions_denoised.reshape((-1, 3))
    x_af3_flat = positions_flat[gather_idxs]
    
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
