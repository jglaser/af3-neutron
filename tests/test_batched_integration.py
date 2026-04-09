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
from af3_neutron.sampler import run_neutron_guided_diffusion

def create_mock_water_oracle():
    atoms = struc.AtomArray(4)
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

class MockBatchedModelRunner:
    """Mocks the Haiku SDE loop natively operating on a batch of multiple samples."""
    def __init__(self, initial_positions_batched):
        # Shape: (num_samples, num_atoms, 3)
        self.initial_positions = initial_positions_batched
        
    def sample_guided_diffusion(self, rng_key, batch_dict, embeddings, grad_fn, sample_key, initial_chis, num_waters):
        positions = self.initial_positions
        num_samples = positions.shape[0]
        
        chi = jnp.tile(initial_chis[None, :], (num_samples, 1))
        water = jnp.zeros((num_samples, num_waters, 3))
        
        # --- FIX: THE GRADIENT FUNCTION MUST BE VMAPPED NATIVELY ---
        # Because we are mimicking the *outer* Haiku pipeline, 
        # we map the gradient function over the batch axis (0) for the positions, chi, and water.
        batched_grad_fn = jax.vmap(grad_fn, in_axes=(0, 0, 0))
        
        lr = 0.05
        for _ in range(15):
            loss_val, (grad_x0, grad_chi, grad_water) = batched_grad_fn(positions, chi, water)
            
            positions = positions - lr * jnp.clip(grad_x0, -1.0, 1.0)
            chi = chi - 0.1 * jnp.clip(grad_chi, -0.1, 0.1)
            water = water - 0.1 * jnp.clip(grad_water, -0.1, 0.1)
            
        final_state = {
            'diffuser': {'chi_angles': chi, 'water_rotations': water}
        }
        
        return {
            'atom_positions': positions,
            'chi_angles': chi, 
            'water_rotations': water
        }

def test_batched_integration_pipeline():
    oracle = create_mock_water_oracle()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        pdb_path = os.path.join(tmpdir, "oracle.pdb")
        pdb_file = pdb.PDBFile()
        pdb.set_structure(pdb_file, oracle)
        pdb_file.write(pdb_path)
        
        sfc = SFcalculator(PDBfile_dir=pdb_path, mtzfile_dir=None, dmin=3.0)
        
        neutron_fullsf = []
        num_hkls = len(sfc.dr2asu_array)
        for atom_name in sfc.atom_name:
            element = gemmi.Element(atom_name)
            b_c = element.neutron92.calculate_sf(0)
            neutron_fullsf.append(np.full(num_hkls, b_c))
        sfc.fullsf_tensor = jnp.array(neutron_fullsf, dtype=jnp.float32)
        
        true_f_complex = sfc.Calc_Fprotein(Return=True, NO_Bfactor=True)
        sfc.Fo = jnp.abs(true_f_complex)
        sfc.SigF = jnp.ones_like(sfc.Fo) * 0.1
        
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
        
        # --- BATCH CONFIGURATION ---
        num_samples = 3
        # Create 3 slightly different starting configurations
        # Sample 0: Oxygen at [1.0, 1.0, 1.0]
        # Sample 1: Oxygen at [1.2, 0.8, 1.0]
        # Sample 2: Oxygen at [0.8, 1.2, 1.0]
        base_coords = jnp.array([[2.0, 2.0, 2.0], [1.0, 1.0, 1.0]])
        offsets = jnp.array([
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.2, -0.2, 0.0]],
            [[0.0, 0.0, 0.0], [-0.2, 0.2, 0.0]]
        ])
        initial_positions_batched = jnp.tile(base_coords, (num_samples, 1, 1)) + offsets
        
        batch_dict = {'pred_dense_atom_mask': jnp.array([[True, True]])}
        gather_idxs = jnp.array([0, 1], dtype=jnp.int32)
        
        mock_runner = MockBatchedModelRunner(initial_positions_batched)
        
        # --- EXECUTE BATCHED PIPELINE ---
        final_coords_batched, final_chis_batched, final_waters_batched = run_neutron_guided_diffusion(
            model_runner=mock_runner,
            batch_dict=batch_dict,
            embeddings=None,
            gather_idxs=gather_idxs,
            rotor_table=rotor_table,
            mapping=mapping,
            water_mapping=water_mapping,
            sfc_instance=sfc,
            sample_key=jax.random.PRNGKey(42)
        )
        
        # --- ASSERTIONS ---
        # 1. Ensure output tensors preserved the batch dimension
        assert final_coords_batched.shape == (num_samples, 2, 3)
        assert final_chis_batched.shape == (num_samples, 2)
        assert final_waters_batched.shape == (num_samples, 1, 3)
        
        # 2. Verify the physics gradient moved the atoms for every sample in the batch
        ideal_dist = jnp.linalg.norm(jnp.array([2.0, 2.0, 2.0]))
        
        for i in range(num_samples):
            initial_rel_dist = jnp.linalg.norm(initial_positions_batched[i, 0] - initial_positions_batched[i, 1])
            final_rel_dist = jnp.linalg.norm(final_coords_batched[i, 0] - final_coords_batched[i, 1])
            
            # Assert coordinates definitely changed (physics engine is active for this sample)
            assert jnp.abs(final_rel_dist - initial_rel_dist) > 0.0
