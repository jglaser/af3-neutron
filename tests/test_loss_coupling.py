import jax
import jax.numpy as jnp
from af3_neutron.sampler import decoupled_crystallographic_loss_pure

def test_loss_mapping_and_gradients():
    num_af3_atoms = 5
    num_oracle_atoms = 7
    
    mapping = {
        "oracle_heavy": jnp.array([0, 1, 2, 3, 4]), 
        "af3_source": jnp.array([0, 1, 2, 3, 4]),
        "num_oracle_atoms": num_oracle_atoms
    }
    
    rotor_table = {
        "target_idx": jnp.array([5, 6]),
        "parent_idx": jnp.array([4, 3]),
        "grandparent_idx": jnp.array([3, 2]),
        "greatgrand_idx": jnp.array([2, 1]),
        "ideal_r": jnp.array([1.0, 1.0]),
        "ideal_theta": jnp.array([109.5, 109.5]),
        "initial_chi": jnp.array([0.0, 0.0]),
    }
    
    water_mapping = {
        "oxygen_source": jnp.array([], dtype=jnp.int32),
        "h1_target": jnp.array([], dtype=jnp.int32),
        "h2_target": jnp.array([], dtype=jnp.int32)
    }
 
    x_af3_flat = jax.random.normal(jax.random.PRNGKey(0), (num_af3_atoms, 3))
    chi_angles = jnp.zeros((2,)) 
    water_rotations = jnp.zeros((0, 3))
    gather_idxs = jnp.arange(num_af3_atoms, dtype=jnp.int32)

    loss = decoupled_crystallographic_loss_pure(
        x_af3_flat, chi_angles, water_rotations, gather_idxs, rotor_table, mapping, water_mapping, None
    )
    assert loss.shape == ()
    assert not jnp.isnan(loss)

    grad_fn = jax.value_and_grad(decoupled_crystallographic_loss_pure, argnums=(0, 1, 2))
    _, (grad_heavy, grad_chi, grad_water) = grad_fn(
        x_af3_flat, chi_angles, water_rotations, gather_idxs, rotor_table, mapping, water_mapping, None
    )

    assert grad_heavy.shape == x_af3_flat.shape
    assert grad_chi.shape == chi_angles.shape
    assert grad_water.shape == water_rotations.shape
    assert jnp.any(jnp.abs(grad_heavy) > 0)

def test_loss_empty_rotors():
    mapping = {"oracle_heavy": jnp.array([0, 1, 2]), "af3_source": jnp.array([0, 1, 2]), "num_oracle_atoms": 3}
    rotor_table = {k: jnp.array([], dtype=jnp.int32 if "idx" in k else jnp.float32) for k in ["target_idx", "parent_idx", "grandparent_idx", "greatgrand_idx", "ideal_r", "ideal_theta"]}
    water_mapping = {k: jnp.array([], dtype=jnp.int32) for k in ["oxygen_source", "h1_target", "h2_target"]}

    x_af3_flat = jnp.ones((3, 3)) 
    chi_angles = jnp.zeros((0,)) 
    water_rotations = jnp.zeros((0, 3))
    gather_idxs = jnp.arange(3, dtype=jnp.int32)
    
    grad_fn = jax.value_and_grad(decoupled_crystallographic_loss_pure, argnums=(0, 1, 2))
    loss, (grad_heavy, grad_chi, grad_water) = grad_fn(
        x_af3_flat, chi_angles, water_rotations, gather_idxs, rotor_table, mapping, water_mapping, None
    )


def test_water_loss_mapping():
    mapping = {"oracle_heavy": jnp.array([0]), "af3_source": jnp.array([0]), "num_oracle_atoms": 3}
    rotor_table = {k: jnp.array([], dtype=jnp.int32 if "idx" in k else jnp.float32) for k in ["target_idx", "parent_idx", "grandparent_idx", "greatgrand_idx", "ideal_r", "ideal_theta"]}
    water_mapping = {"oxygen_source": jnp.array([0], dtype=jnp.int32), "h1_target": jnp.array([1], dtype=jnp.int32), "h2_target": jnp.array([2], dtype=jnp.int32)}
    
    x_af3_flat = jnp.array([[0.0, 0.0, 0.0]]) 
    chi_angles = jnp.zeros((0,))
    water_rotations = jnp.array([[0.1, 0.2, 0.3]])
    gather_idxs = jnp.array([0], dtype=jnp.int32)

    grad_fn = jax.value_and_grad(decoupled_crystallographic_loss_pure, argnums=(0, 1, 2))
    loss, (grad_heavy, grad_chi, grad_water) = grad_fn(
        x_af3_flat, chi_angles, water_rotations, gather_idxs, rotor_table, mapping, water_mapping, None
    )
    assert not jnp.isnan(loss)
    assert grad_water.shape == (1, 3)
    assert jnp.any(jnp.abs(grad_water) > 0)
    assert grad_heavy.shape == (1, 3)

