import logging
import jax
import jax.numpy as jnp
from alphafold3.model.network import diffusion_head

from .kinematics import generalized_nerf_layer, so3_water_layer

def remove_center_of_mass(positions, mask):
    # positions (N, 24, 3) or (N_flat, 3)
    mask_expanded = mask[..., None]
    
    sum_coords = jnp.sum(positions * mask_expanded, axis=(0, 1) if positions.ndim == 3 else 0)
    num_atoms = jnp.sum(mask) + 1e-8
    com = sum_coords / num_atoms
    
    return positions - com

def placeholder_neutron_loss(x_full):
    return 0*jnp.mean(x_full ** 2) * 0.01

def se3_invariant_neutron_loss(x_full, sfc_instance):
    # STUB: Replace with Kam's theorem / Radial Autocorrelation proxy
    # This dummy implementation guarantees a perfectly covariant gradient.
    # It penalizes the L2 radius from the center of mass to simulate a compacting force.
    com = jnp.mean(x_full, axis=0)
    radii = jnp.linalg.norm(x_full - com, axis=-1)
    return jnp.mean(radii**2) * 0.001

def decoupled_crystallographic_loss_pure(
    positions_denoised_flat, gather_idxs, rotor_table, mapping, water_mapping, sfc_instance, atom_mask_flat
):
    positions_centered =positions_denoised_flat - jnp.mean(positions_denoised_flat[atom_mask_flat], axis=0)
    x_af3_flat = positions_centered[gather_idxs]

    chi_angles = jnp.zeros(rotor_table["target_idx"].shape[0])
    water_rotations = jnp.zeros((water_mapping["oxygen_source"].shape[0], 3))

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
        return se3_invariant_neutron_loss(x_full, sfc_instance)
    else:
        return placeholder_neutron_loss(x_full)

def run_neutron_guided_diffusion(
    model_runner, batch_dict, embeddings, gather_idxs,
    rotor_table, mapping, water_mapping, sfc_instance=None, sample_key=None
):
    logging.info("Delegating to Native AF3 SDE Loop with SE(3) Covariant Physics Hook...")

    atom_mask = batch_dict['pred_dense_atom_mask']
    atom_mask_flat = atom_mask.reshape(-1)

    # Create the pure closure
    def loss_fn(positions_denoised):
        positions_flat = positions_denoised.reshape((-1, 3))
        return decoupled_crystallographic_loss_pure(
            positions_flat, gather_idxs, rotor_table, mapping, water_mapping, sfc_instance, atom_mask_flat
        )

    grad_fn = jax.value_and_grad(loss_fn)

    # Execute the JIT-compiled native sampling loop!
    # jax.random.PRNGKey(0) is passed for the dummy Haiku transform RNG; sample_key drives the SDE.
    sample_results = model_runner.sample_guided_diffusion(
        jax.random.PRNGKey(0), batch_dict, embeddings, grad_fn, sample_key
    )

    final_coords = sample_results['atom_positions'][0]
    final_coords =remove_center_of_mass(final_coords, atom_mask)

    # Return stateless kinematics
    chi_angles = jnp.zeros(rotor_table["target_idx"].shape[0])
    water_rotations = jnp.zeros((water_mapping["oxygen_source"].shape[0], 3))

    return final_coords, chi_angles, water_rotations

def generate_final_oracle_coords(positions_denoised_final, chi_angles, water_rotations, gather_idxs, rotor_table, mapping, water_mapping, reference_coords):
    # 1. Map to oracle heavy atoms
    positions_flat = positions_denoised_final.reshape((-1, 3))
    x_af3_flat = positions_flat[gather_idxs]
    
    # 2. Final Kabsch Alignment to experimental MTZ Unit Cell
    x_drift_heavy = x_af3_flat[mapping["af3_source"]]
    x_ref_heavy = reference_coords[mapping["oracle_heavy"]]
    
    com_drift = jnp.mean(x_drift_heavy, axis=0)
    com_ref = jnp.mean(x_ref_heavy, axis=0)
    
    p = x_drift_heavy - com_drift
    q = x_ref_heavy - com_ref
    
    H = jnp.einsum('ni,nj->ij', p, q)
    U, S, Vt = jnp.linalg.svd(H)
    d = jnp.sign(jnp.linalg.det(U) * jnp.linalg.det(Vt))
    R = U @ jnp.diag(jnp.array([1.0, 1.0, d])) @ Vt
    
    x_af3_aligned = (x_af3_flat - com_drift) @ R + com_ref
    
    # 3. Assemble full complex in experimental frame
    x_full = jnp.zeros((mapping["num_oracle_atoms"], 3))
    x_full = x_full.at[mapping["oracle_heavy"]].set(x_af3_aligned[mapping["af3_source"]])
    
    if rotor_table["target_idx"].shape[0] > 0:
        x_h = generalized_nerf_layer(x_af3_aligned, rotor_table, chi_angles)
        x_full = x_full.at[rotor_table["target_idx"]].set(x_h)
        
    if water_mapping["oxygen_source"].shape[0] > 0:
        oxygen_coords = x_af3_aligned[water_mapping["oxygen_source"]]
        h1, h2 = so3_water_layer(oxygen_coords, water_rotations)
        x_full = x_full.at[water_mapping["h1_target"]].set(h1)
        x_full = x_full.at[water_mapping["h2_target"]].set(h2)
        
    return x_full
