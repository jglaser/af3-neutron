import os
import tempfile
import numpy as np
import jax
import jax.numpy as jnp

import biotite.structure as struc
import biotite.structure.io.pdb as pdb
import gemmi

# --- MONKEY PATCH GEMMI FOR SFC_JAX COMPATIBILITY ---
if not hasattr(gemmi.UnitCell, "fractionalization_matrix"):
    gemmi.UnitCell.fractionalization_matrix = property(lambda self: self.frac.mat)
if not hasattr(gemmi.UnitCell, "orthogonalization_matrix"):
    gemmi.UnitCell.orthogonalization_matrix = property(lambda self: self.orth.mat)
# ----------------------------------------------------

from SFC_Jax.Fmodel import SFcalculator

from af3_neutron.sampler import decoupled_crystallographic_loss_pure


def create_mock_water_oracle():
    """Creates a synthetic PDB structure of a single water molecule."""
    atoms = struc.AtomArray(3)
    atoms.coord = np.array([
        [0.0, 0.0, 0.0],       # O
        [0.757, 0.586, 0.0],   # H1
        [-0.757, 0.586, 0.0]   # H2
    ])
    atoms.element = np.array(["O", "H", "H"])
    atoms.atom_name = np.array(["O", "H1", "H2"])
    atoms.res_name = np.array(["HOH", "HOH", "HOH"])
    atoms.chain_id = np.array(["A", "A", "A"])
    atoms.res_id = np.array([1, 1, 1])
    
    # SFC_Jax requires a unit cell box in the PDB (e.g., a 10x10x10 Angstrom box)
    atoms.box = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
    return atoms

def test_neutron_scattering_lengths():
    """Verifies that the adapter correctly extracts Neutron vs X-ray lengths."""
    oracle = create_mock_water_oracle()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        pdb_path = os.path.join(tmpdir, "oracle.pdb")
        pdb_file = pdb.PDBFile()
        pdb.set_structure(pdb_file, oracle)
        pdb_file.write(pdb_path)
        
        # Initialize SFC_Jax with synthetic HKLs (dmin=3.0) instead of an MTZ
        sfc = SFcalculator(PDBfile_dir=pdb_path, mtzfile_dir=None, dmin=3.0)
        
        # Apply our neutron scattering lengths logic
        neutron_fullsf = []
        num_hkls = len(sfc.dr2asu_array)
        for atom_name in sfc.atom_name:
            element = gemmi.Element(atom_name)
            b_c = element.neutron92.calculate_sf(0)
            neutron_fullsf.append(np.full(num_hkls, b_c))
            
        sfc.fullsf_tensor = jnp.array(neutron_fullsf, dtype=jnp.float32)
        
        # Oxygen (~5.8 fm) should be positive
        assert sfc.fullsf_tensor[0, 0] > 0.0 
        
        # Hydrogen (~ -3.74 fm) MUST be negative
        assert sfc.fullsf_tensor[1, 0] < 0.0 
        assert sfc.fullsf_tensor[2, 0] < 0.0

def test_sfc_loss_gradients():
    """Verifies that gradients flow cleanly from SFC_Jax through the SO(3) layer."""
    oracle = create_mock_water_oracle()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        pdb_path = os.path.join(tmpdir, "oracle.pdb")
        pdb_file = pdb.PDBFile()
        pdb.set_structure(pdb_file, oracle)
        pdb_file.write(pdb_path)
        
        sfc = SFcalculator(PDBfile_dir=pdb_path, mtzfile_dir=None, dmin=3.0)
        
        # Mock the neutron lengths
        sfc.fullsf_tensor = jnp.ones((3, len(sfc.dr2asu_array)), dtype=jnp.float32)
        
        # Mock Experimental Data (Fo, SigF)
        sfc.Fo = jnp.ones(len(sfc.dr2asu_array), dtype=jnp.float32) * 10.0
        sfc.SigF = jnp.ones(len(sfc.dr2asu_array), dtype=jnp.float32)

        # Define mappings for a single AF3 Oxygen atom
        mapping = {
            "oracle_heavy": jnp.array([0], dtype=jnp.int32), 
            "af3_source": jnp.array([0], dtype=jnp.int32),
            "num_oracle_atoms": 3
        }
        rotor_table = {k: jnp.array([], dtype=jnp.int32 if "idx" in k else jnp.float32) 
                       for k in ["target_idx", "parent_idx", "grandparent_idx", "greatgrand_idx", "ideal_r", "ideal_theta"]}
        water_mapping = {
            "oxygen_source": jnp.array([0], dtype=jnp.int32),
            "h1_target": jnp.array([1], dtype=jnp.int32),
            "h2_target": jnp.array([2], dtype=jnp.int32)
        }
       
        # Neural Network states (STRICT 2D)
        x_af3_flat = jnp.array([[0.1, 0.2, 0.3]])            # Shape: (1, 3)
        chi_angles = jnp.zeros((0,), dtype=jnp.float32)      # Shape: (0,)
        water_rotations = jnp.array([[0.1, -0.1, 0.05]])     # Shape: (1, 3)

        gather_idxs = jnp.array([0], dtype=jnp.int32)

        # 1. Test Forward Pass
        loss = decoupled_crystallographic_loss_pure(
            x_af3_flat, chi_angles, water_rotations, gather_idxs, rotor_table, mapping, water_mapping, sfc
        )
        assert not jnp.isnan(loss)

        # 2. Test Backward Pass
        grad_fn = jax.value_and_grad(decoupled_crystallographic_loss_pure, argnums=(0, 1, 2))
        _, (grad_heavy, grad_chi, grad_water) = grad_fn(
            x_af3_flat, chi_angles, water_rotations, gather_idxs, rotor_table, mapping, water_mapping, sfc
        )

        # Assert dimensionalities remain intact
        assert grad_heavy.shape == x_af3_flat.shape
        assert grad_water.shape == water_rotations.shape

         # Assert gradients are flowing cleanly without exploding into NaNs
        assert not jnp.any(jnp.isnan(grad_heavy))
        assert not jnp.any(jnp.isnan(grad_water))

        # Assert SFC is actively applying torque to the water orientations
        assert jnp.any(jnp.abs(grad_water) > 0)
