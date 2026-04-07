import jax
import jax.numpy as jnp
from af3_neutron.sampler import decoupled_crystallographic_loss

def test_loss_mapping_and_gradients():
    # Mock AF3 output (e.g. 5 heavy atoms)
    num_af3_atoms = 5
    x_af3_flat = jax.random.normal(jax.random.PRNGKey(0), (num_af3_atoms, 3))
    
    # Mock Oracle (e.g. 5 heavy + 2 missing hydrogens = 7 total atoms)
    num_oracle_atoms = 7
    
    mapping = {
        "oracle_heavy": jnp.array([0, 1, 2, 3, 4]), 
        "af3_source": jnp.array([0, 1, 2, 3, 4]),
        "num_oracle_atoms": num_oracle_atoms
    }
    
    # We want to NeRF 2 hydrogens
    rotor_table = {
        "target_idx": jnp.array([5, 6]),
        "parent_idx": jnp.array([4, 3]),
        "grandparent_idx": jnp.array([3, 2]),
        "greatgrand_idx": jnp.array([2, 1]),
        "ideal_r": jnp.array([1.0, 1.0]),
        "ideal_theta": jnp.array([109.5, 109.5]),
    }
    
    # Empty water mappings for this test
    water_mapping = {
        "oxygen_source": jnp.array([], dtype=jnp.int32),
        "h1_target": jnp.array([], dtype=jnp.int32),
        "h2_target": jnp.array([], dtype=jnp.int32)
    }
    
    chi_angles = jnp.array([0.0, 0.0])
    water_rotations = jnp.zeros((0, 3))
    
    # 1. Forward Pass Test
    loss = decoupled_crystallographic_loss(
        x_af3_flat, chi_angles, water_rotations, rotor_table, mapping, water_mapping
    )
    assert loss.shape == ()
    assert loss > 0.0
    assert not jnp.isnan(loss)
    
    # 2. Backward Pass Test (argnums=(0,1,2) to catch heavy, chi, and water)
    grad_fn = jax.value_and_grad(decoupled_crystallographic_loss, argnums=(0, 1, 2))
    _, (grad_heavy, grad_chi, grad_water) = grad_fn(
        x_af3_flat, chi_angles, water_rotations, rotor_table, mapping, water_mapping
    )
    
    assert grad_heavy.shape == x_af3_flat.shape
    assert grad_chi.shape == chi_angles.shape
    assert grad_water.shape == water_rotations.shape
    assert jnp.any(jnp.abs(grad_heavy) > 0)

def test_loss_empty_rotors():
    x_af3_flat = jnp.ones((3, 3))
    
    mapping = {
        "oracle_heavy": jnp.array([0, 1, 2]),
        "af3_source": jnp.array([0, 1, 2]),
        "num_oracle_atoms": 3
    }
    
    rotor_table = {
        "target_idx": jnp.array([], dtype=jnp.int32),
        "parent_idx": jnp.array([], dtype=jnp.int32),
        "grandparent_idx": jnp.array([], dtype=jnp.int32),
        "greatgrand_idx": jnp.array([], dtype=jnp.int32),
        "ideal_r": jnp.array([], dtype=jnp.float32),
        "ideal_theta": jnp.array([], dtype=jnp.float32),
    }
    
    water_mapping = {
        "oxygen_source": jnp.array([], dtype=jnp.int32),
        "h1_target": jnp.array([], dtype=jnp.int32),
        "h2_target": jnp.array([], dtype=jnp.int32)
    }
    
    chi_angles = jnp.array([], dtype=jnp.float32)
    water_rotations = jnp.zeros((0, 3))
    
    grad_fn = jax.value_and_grad(decoupled_crystallographic_loss, argnums=(0, 1, 2))
    loss, (grad_heavy, grad_chi, grad_water) = grad_fn(
        x_af3_flat, chi_angles, water_rotations, rotor_table, mapping, water_mapping
    )
    
    assert not jnp.isnan(loss)
    assert grad_heavy.shape == x_af3_flat.shape

def test_water_loss_mapping():
    # Test that SO(3) rotations properly propagate gradients to the water orientation tensor
    x_af3_flat = jnp.array([[0.0, 0.0, 0.0]]) # Just one Oxygen
    num_oracle_atoms = 3 # O, H1, H2
    
    mapping = {
        "oracle_heavy": jnp.array([0]),
        "af3_source": jnp.array([0]),
        "num_oracle_atoms": num_oracle_atoms
    }
    
    rotor_table = {
        "target_idx": jnp.array([], dtype=jnp.int32),
        "parent_idx": jnp.array([], dtype=jnp.int32),
        "grandparent_idx": jnp.array([], dtype=jnp.int32),
        "greatgrand_idx": jnp.array([], dtype=jnp.int32),
        "ideal_r": jnp.array([], dtype=jnp.float32),
        "ideal_theta": jnp.array([], dtype=jnp.float32),
    }
    
    water_mapping = {
        "oxygen_source": jnp.array([0], dtype=jnp.int32),
        "h1_target": jnp.array([1], dtype=jnp.int32),
        "h2_target": jnp.array([2], dtype=jnp.int32)
    }
    
    chi_angles = jnp.array([], dtype=jnp.float32)
    water_rotations = jnp.array([[0.1, 0.2, 0.3]]) # Tiny initial axis-angle rotation
    
    grad_fn = jax.value_and_grad(decoupled_crystallographic_loss, argnums=(0, 1, 2))
    loss, (grad_heavy, grad_chi, grad_water) = grad_fn(
        x_af3_flat, chi_angles, water_rotations, rotor_table, mapping, water_mapping
    )
    
    assert not jnp.isnan(loss)
    assert grad_water.shape == (1, 3)
    # Ensure gradients flowed from the dummy loss back to the axis-angle vector
    assert jnp.any(jnp.abs(grad_water) > 0)
