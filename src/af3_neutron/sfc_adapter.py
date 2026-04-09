import logging
import os
import tempfile
from typing import Any

from biotite.structure import AtomArray
import biotite.structure.io.pdb as pdb
import gemmi
import jax.numpy as jnp
import numpy as np
from SFC_Jax.Fmodel import SFcalculator

# --- MONKEY PATCH GEMMI FOR SFC_JAX COMPATIBILITY ---
if not hasattr(gemmi.UnitCell, "fractionalization_matrix"):
    gemmi.UnitCell.fractionalization_matrix = property(lambda self: self.frac.mat)
if not hasattr(gemmi.UnitCell, "orthogonalization_matrix"):
    gemmi.UnitCell.orthogonalization_matrix = property(lambda self: self.orth.mat)
# ----------------------------------------------------

def init_neutron_sfc(oracle_atoms: AtomArray, mtz_path: str) -> SFcalculator:
    """
    Initializes the SFC_Jax crystallographic engine for neutron diffraction.

    This function performs a "physics swap" by initializing a standard X-ray 
    structure factor calculator and replacing its internal atomic form factor 
    tensors with nuclear scattering lengths derived from Gemmi's neutron92 
    tables.

    Parameters
    ----------
    oracle_atoms : biotite.structure.AtomArray
        The molecular structure used to define the unit cell and atomic 
    mtz_path : str
        Path to the experimental MTZ file containing reflections (Fo and SigF).

    Returns
    -------
    SFC_Jax.Fmodel.SFcalculator
        A JAX-compatible structure factor calculator instance configured for 
        neutron scattering.
    """
    logging.info("Initializing SFC_Jax Crystallographic Engine...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        pdb_path = os.path.join(tmpdir, "oracle.pdb")
        pdb_file = pdb.PDBFile()
        pdb.set_structure(pdb_file, oracle_atoms)
        pdb_file.write(pdb_path)
        
        # Initialize the calculator with our experimental data
        sfc = SFcalculator(
            PDBfile_dir=pdb_path,
            mtzfile_dir=mtz_path,
            set_experiment=True # Automatically loads Fo and SigF
        )
        
    logging.info("Querying Gemmi for nuclear scattering lengths...")
    neutron_fullsf = []
    num_hkls = len(sfc.dr2asu_array)
    
    for atom_name in sfc.atom_name:
        # Dynamically fetch the constant bound coherent scattering length in fm
        element = gemmi.Element(atom_name)
        b_c = element.neutron92.calculate_sf(0)
        neutron_fullsf.append(np.full(num_hkls, b_c))
        
    sfc.fullsf_tensor = jnp.array(neutron_fullsf, dtype=jnp.float32)
    
    return sfc
