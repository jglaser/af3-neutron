import jax
import jax.numpy as jnp
from af3_neutron.sampler import decoupled_crystallographic_loss_pure
from af3_neutron.kinematics import generalized_nerf_layer, so3_water_layer

def test_batched_kinematics_and_loss():
    num_samples = 5
    num_af3_atoms = 4
    num_oracle_atoms = 6
    
    # Simulate the exact shapes generated inside Haiku's sample loop
    x_af3_batched = jax.random.normal(jax.random.PRNGKey(0), (num_samples, num_af3_atoms, 3))
    chi_batched = jnp.zeros((num_samples, 2))
    water_batched = jnp.zeros((num_samples, 1, 3))
    
    mapping = {
        "oracle_heavy": jnp.array([0, 1, 2, 3]), 
        "af3_source": jnp.array([0, 1, 2, 3]),
        "num_oracle_atoms": num_oracle_atoms
    }
    
    rotor_table = {
        "target_idx": jnp.array([4, 5]),
        "parent_idx": jnp.array([3, 2]),
        "grandparent_idx": jnp.array([2, 1]),
        "greatgrand_idx": jnp.array([1, 0]),
        "ideal_r": jnp.array([1.0, 1.0]),
        "ideal_theta": jnp.array([109.5, 109.5]),
    }
    
    water_mapping = {
        "oxygen_source": jnp.array([3], dtype=jnp.int32),
        "h1_target": jnp.array([4], dtype=jnp.int32),
        "h2_target": jnp.array([5], dtype=jnp.int32)
    }
    
    gather_idxs = jnp.arange(num_af3_atoms, dtype=jnp.int32)
    
    # Create a wrapper that captures the static arguments
    def single_sample_loss(x, chi, water):
        return decoupled_crystallographic_loss_pure(
            x, chi, water, gather_idxs, rotor_table, mapping, water_mapping, sfc_instance=None
        )
        
    # VMAP to handle the num_samples dimension, exactly as the real runner does!
    batched_loss_fn = jax.vmap(single_sample_loss, in_axes=(0, 0, 0))
    
    # 1. Test the pure loss function forward pass
    loss = batched_loss_fn(x_af3_batched, chi_batched, water_batched)
    
    # Loss should now return an array of 5 scalars (one for each sample)
    assert loss.shape == (num_samples,)
    assert not jnp.any(jnp.isnan(loss))
    
    # 2. Test Batched Gradients
    batched_grad_fn = jax.vmap(
        jax.value_and_grad(single_sample_loss, argnums=(0, 1, 2)), 
        in_axes=(0, 0, 0)
    )
    
    loss_val, (grad_x0, grad_chi, grad_water) = batched_grad_fn(
        x_af3_batched, chi_batched, water_batched
    )
    
    assert grad_x0.shape == x_af3_batched.shape
    assert grad_chi.shape == chi_batched.shape
    assert grad_water.shape == water_batched.shape
