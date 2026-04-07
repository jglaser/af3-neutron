import logging
import jax.numpy as jnp
import numpy as np
import tempfile
import os
import biotite.structure.io.pdb as pdb
from SFC_Jax.Fmodel import SFcalculator

# Coherent neutron scattering lengths in Fermi (fm)
NEUTRON_LENGTHS = {
    'H': -3.739,
    'D': 6.671,
    'C': 6.646,
    'N': 9.360,
    'O': 5.803,
    'S': 2.847,
    'P': 5.128,
}

def init_neutron_sfc(oracle_atoms, mtz_path):
    """
    Writes the oracle to a temp PDB, initializes SFC_Jax, and hacks the 
    form factor tensor to use constant neutron scattering lengths.
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
        
    logging.info("Converting X-ray form factors to Neutron scattering lengths...")
    neutron_fullsf = []
    
    # Replace the dr2asu_array dimension with constant values
    num_hkls = len(sfc.dr2asu_array)
    
    for atom_name in sfc.atom_name:
        # sfc.atom_name stores the element strings directly (e.g. 'C', 'O', 'H')
        b_c = NEUTRON_LENGTHS.get(atom_name, 0.0) 
        neutron_fullsf.append(np.full(num_hkls, b_c))
        
    sfc.fullsf_tensor = jnp.array(neutron_fullsf, dtype=jnp.float32)
    
    return sfc
