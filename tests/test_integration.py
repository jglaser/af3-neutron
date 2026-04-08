import os
import tempfile
import numpy as np
import jax
import jax.numpy as jnp

import biotite.structure as struc
import biotite.structure.io.pdb as pdb
import gemmi

# --- MONKEY PATCH GEMMI ---
if not hasattr(gemmi.UnitCell, "fractionalization_matrix"):
    gemmi.UnitCell.fractionalization_matrix = property(lambda self: self.frac.mat)
if not hasattr(gemmi.UnitCell, "orthogonalization_matrix"):
    gemmi.UnitCell.orthogonalization_matrix = property(lambda self: self.orth.mat)
# --------------------------

from SFC_Jax.Fmodel import SFcalculator
from af3_neutron.sampler import run_neutron_guided_diffusion, decoupled_crystallographic_loss_pure

def create_mock_water_oracle():
    """Creates a synthetic PDB structure of a water molecule and a fixed anchor."""
    atoms = struc.AtomArray(4)
    # We add a Carbon anchor to break P1 global translation invariance
    atoms.coord = np.array([
        [2.0, 2.0, 2.0],       # C (Fixed anchor)
        [0.0, 0.0, 0.0],       # O
        [0.757, 0.586, 0.0],   # H1
        [-0.757, 0.586, 0.0]   # H2
    ])
    atoms.element = np.array(["C", "O", "H", "H"])
    atoms.atom_name = np.array(["C", "O", "H1", "H2"])
    atoms.res_name = np.array(["ALA", "HOH", "HOH", "HOH"])
    atoms.chain_id = np.array(["A", "B", "B", "B"])
    atoms.res_id = np.array([1, 2, 2, 2])
    atoms.box = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
    return atoms

class MockModelRunner:
    """Mocks the Haiku-compiled AF3 ModelRunner for integration testing."""
    def __init__(self, initial_positions):
        self.initial_positions = initial_positions
        
    def sample_guided_diffusion(self, rng_key, batch_dict, embeddings, grad_fn, sample_key, initial_chis, num_waters):
        # 1. KEEP the sample dimension (1, 2, 3)
        positions = self.initial_positions
        
        # 2. Add the sample dimension to the kinematic tensors (1, 2) and (1, 1, 3)
        chi = jnp.expand_dims(initial_chis, axis=0)
        water = jnp.zeros((1, num_waters, 3))
        
        lr = 0.05
        for _ in range(15):
            loss_val, (grad_x0, grad_chi, grad_water) = grad_fn(positions, chi, water)
            positions = positions - lr * jnp.clip(grad_x0, -1.0, 1.0)
            chi = chi - 0.1 * jnp.clip(grad_chi, -0.1, 0.1)
            water = water - 0.1 * jnp.clip(grad_water, -0.1, 0.1)
            
        final_state = {
            'diffuser': {'chi_angles': chi, 'water_rotations': water}
        }
        
        # We don't need to expand dims anymore since positions is already batched
        return {'atom_positions': positions}, final_state

def test_full_physics_pipeline():
    oracle = create_mock_water_oracle()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        pdb_path = os.path.join(tmpdir, "oracle.pdb")
        pdb_file = pdb.PDBFile()
        pdb.set_structure(pdb_file, oracle)
        pdb_file.write(pdb_path)
        
        # 1. Initialize SFC
        sfc = SFcalculator(PDBfile_dir=pdb_path, mtzfile_dir=None, dmin=3.0)
        
        # Apply exact Neutron Scattering Lengths
        neutron_fullsf = []
        num_hkls = len(sfc.dr2asu_array)
        for atom_name in sfc.atom_name:
            element = gemmi.Element(atom_name)
            b_c = element.neutron92.calculate_sf(0)
            neutron_fullsf.append(np.full(num_hkls, b_c))
        sfc.fullsf_tensor = jnp.array(neutron_fullsf, dtype=jnp.float32)
        
        # 2. Generate Synthetic Experimental Target (F_obs)
        true_f_complex = sfc.Calc_Fprotein(Return=True, NO_Bfactor=True)
        sfc.Fo = jnp.abs(true_f_complex)
        sfc.SigF = jnp.ones_like(sfc.Fo) * 0.1
        
        # 3. Setup Mappings for 1 Anchor + 1 Water Molecule
        mapping = {
            "oracle_heavy": jnp.array([0, 1], dtype=jnp.int32), 
            "af3_source": jnp.array([0, 1], dtype=jnp.int32),
            "num_oracle_atoms": 4
        }
        rotor_table = {k: jnp.array([], dtype=jnp.int32 if "idx" in k else jnp.float32) 
                       for k in ["target_idx", "parent_idx", "grandparent_idx", "greatgrand_idx", "ideal_r", "ideal_theta"]}
        rotor_table["initial_chi"] = jnp.array([0.0, 0.0])

        water_mapping = {
            "oxygen_source": jnp.array([1], dtype=jnp.int32),
            "h1_target": jnp.array([2], dtype=jnp.int32),
            "h2_target": jnp.array([3], dtype=jnp.int32)
        }
        
        # 4. Initialize Sub-optimal Neural Network State
        initial_positions = jnp.array([[[2.0, 2.0, 2.0], [1.0, 1.0, 1.0]]]) 
        batch = {'pred_dense_atom_mask': jnp.array([[True, True]])}
        gather_idxs = jnp.array([0, 1], dtype=jnp.int32)
        
        initial_loss = decoupled_crystallographic_loss_pure(
            initial_positions.reshape((-1, 3)), 
            jnp.array([]), jnp.array([[0.0, 0.0, 0.0]]),
            gather_idxs, rotor_table, mapping, water_mapping, sfc
        )
        
        # Initialize the mock runner
        mock_runner = MockModelRunner(initial_positions)
        
        # 5. Run the Decoupled ODE Loop via the Mock Runner
        final_coords, final_chis, final_waters = run_neutron_guided_diffusion(
            model_runner=mock_runner,
            batch_dict=batch,
            embeddings=None,
            gather_idxs=gather_idxs,
            rotor_table=rotor_table,
            mapping=mapping,
            water_mapping=water_mapping,
            sfc_instance=sfc,
            sample_key=jax.random.PRNGKey(42)
        )
        
        final_loss = decoupled_crystallographic_loss_pure(
            final_coords.reshape((-1, 3)), 
            final_chis[0], final_waters[0], gather_idxs,
            rotor_table, mapping, water_mapping, sfc
        )

        # --- ASSERTIONS ---
        assert not jnp.isnan(final_loss)

        initial_rel_dist = jnp.linalg.norm(initial_positions[0, 0] - initial_positions[0, 1])
        final_rel_dist = jnp.linalg.norm(final_coords[0, 0] - final_coords[0, 1])

        # 1. Verify that the JAX gradients successfully moved the atoms
        assert jnp.abs(final_rel_dist - initial_rel_dist) > 0.0

        # 2. Verify that the optimizer stepped down the physics gradient
        assert final_loss < initial_loss
