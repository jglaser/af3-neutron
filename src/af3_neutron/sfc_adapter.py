import logging
import os
import sys
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

        # 1. Sanitize residue names to satisfy strict PDB 3-character constraints
        oracle_atoms.res_name = np.array([name[:3] for name in oracle_atoms.res_name])

        pdb.set_structure(pdb_file, oracle_atoms)
        pdb_file.write(pdb_path)

        # 2. Pull unit cell dimensions and space group parameters directly from the MTZ
        mtz = gemmi.read_mtz_file(mtz_path)
        cell = mtz.cell
        sg_name = mtz.spacegroup_name

        # 3. Format a valid PDB CRYST1 record line
        cryst1_line = (
            f"CRYST1{cell.a:9.3f}{cell.b:9.3f}{cell.c:9.3f}"
            f"{cell.alpha:7.2f}{cell.beta:7.2f}{cell.gamma:7.2f} "
            f"{sg_name:<11}\n"
        )

        # 4. Prepend the CRYST1 record line directly into the temporary PDB file
        with open(pdb_path, "r") as f:
            pdb_content = f.read()
        with open(pdb_path, "w") as f:
            f.write(cryst1_line + pdb_content)

        print(f"Injected Symmetry Header: {cryst1_line.strip()}", file=sys.stderr)

        # 5. Initialize the calculator with our fully-qualified experimental data
        sfc = SFcalculator(
            PDBfile_dir=pdb_path,
            mtzfile_dir=mtz_path,
            set_experiment=True
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
