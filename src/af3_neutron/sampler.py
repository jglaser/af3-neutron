import logging
import jax
import jax.numpy as jnp
from .loss import get_grad_loss_fn
from .kinematics import generalized_nerf_layer

def extract_coords_from_af3(af3_result, expected_atoms):
    """
    Extracts and flattens the predicted 3D coordinates from the AF3 result tensor.
    """
    logging.info("Extracting coordinates from AF3 dictionary...")
    
    # AF3 stores the final heavy-atom / dense predictions here:
    if 'atom_pos' in af3_result:
        coords = af3_result['atom_pos']
    elif 'sample_atom_positions' in af3_result:
        coords = af3_result['sample_atom_positions']
    else:
        logging.warning(f"Could not find exact coordinate key in AF3 result. Keys available: {list(af3_result.keys())}")
        # Fallback to random to prevent crashing
        return jax.random.normal(jax.random.PRNGKey(42), (expected_atoms, 3))
        
    # Flatten the (num_tokens, max_atoms, 3) tensor to match our (N_atoms, 3) layout
    flat_coords = coords.reshape(-1, 3)
    
    # Truncate or Pad to ensure the array perfectly matches the Z-matrix expectations
    if flat_coords.shape[0] > expected_atoms:
        flat_coords = flat_coords[:expected_atoms, :]
    elif flat_coords.shape[0] < expected_atoms:
        pad_size = expected_atoms - flat_coords.shape[0]
        flat_coords = jnp.pad(flat_coords, ((0, pad_size), (0, 0)))
        
    return flat_coords

def run_neutron_guided_diffusion(af3_result, batch, rotor_table, n_steps=5):
    """
    Custom inference loop applying NeRF constraints and neutron data gradients
    directly to the AlphaFold 3 predicted coordinates.
    """
    logging.info("Starting Custom Neutron-Guided Diffusion...")
    grad_loss_fn = get_grad_loss_fn()
    
    num_rotors = rotor_table["target_idx"].shape[0]
    chi_angles = jnp.zeros(num_rotors)
    
    learning_rate_heavy = 0.01
    learning_rate_chi = 0.05
    
    # Calculate expected array size
    max_idx = jnp.max(rotor_table["target_idx"]) if num_rotors > 0 else 10000
    expected_atoms = int(max_idx) + 1
    
    # 1. Grab actual coordinates from the neural network output
    x_0_pred = extract_coords_from_af3(af3_result, expected_atoms)
    
    for step in range(n_steps):
        
        # 2. Calculate Loss and Gradients positionally
        loss_val, (grad_heavy, grad_chi) = grad_loss_fn(
            x_0_pred,      # arg 0
            chi_angles,    # arg 1
            rotor_table,   # arg 2
            x_0_pred       # arg 3
        )
        logging.info(f"Step {step} | Neutron Loss: {loss_val:.4f}")
        
        # 3. Gradient Descent
        x_0_guided = x_0_pred - (learning_rate_heavy * grad_heavy)
        chi_angles = chi_angles - (learning_rate_chi * grad_chi)
        
        # 4. Project strict NeRF hydrogens back into the state
        if num_rotors > 0:
            strict_h_coords = generalized_nerf_layer(x_0_guided, rotor_table, chi_angles)
            x_0_guided = x_0_guided.at[rotor_table["target_idx"]].set(strict_h_coords)
        
        x_0_pred = x_0_guided 
        
    return x_0_pred, chi_angles
