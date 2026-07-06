import os
import sys  
import tempfile
import numpy as np
import jax.numpy as jnp
import gemmi  
from biotite.structure.io import pdb
from SFC_Jax.Fmodel import SFcalculator

# ==============================================================================
# MONKEYPATCH FOR GEMMI VERSION CONFLICT (v0.7.0+)
# SFC_Jax relies on legacy 'fractionalization_matrix' attributes that were 
# completely removed in recent Gemmi releases. We dynamically inject properties 
# to route those legacy lookups to the modern 'frac.mat' and 'orth.mat' objects.
# ==============================================================================
if not hasattr(gemmi.UnitCell, "fractionalization_matrix"):
    gemmi.UnitCell.fractionalization_matrix = property(lambda self: self.frac.mat)
if not hasattr(gemmi.UnitCell, "orthogonalization_matrix"):
    gemmi.UnitCell.orthogonalization_matrix = property(lambda self: self.orth.mat)
# ==============================================================================

class NeutronSFCalculator(SFcalculator):
    """Subclass of SFcalculator that implements a clean, JAX-differentiable 
    structure factor amplitude loss directly on global Cartesian coordinates.
    """
    def compute_loss(self, xyz, t_hat=None):
        # xyz shape: (N, 3) - global Cartesian coordinates from diffusion step
        
        # Execute the forward in-place calculation method
        res = self.Calc_Fprotein(xyz)
        
        # Extract the unique ASU reflection tracer first to match Fo dimensions
        f_calc_complex = getattr(self, "Fprotein_asu", None)
        if f_calc_complex is None:
            f_calc_complex = getattr(self, "Fprotein_HKL", None)
        if f_calc_complex is None:
            f_calc_complex = res
            
        if f_calc_complex is None:
            available_attrs = [a for a in dir(self) if not a.startswith("__")]
            raise AttributeError(
                f"Calc_Fprotein returned None and no populated structure factor attribute "
                f"('Fprotein_asu' or 'Fprotein_HKL') was found. Available attributes: {available_attrs}"
            )
            
        f_calc_mag = jnp.abs(f_calc_complex)
        
        # Safely extract experimental amplitudes using explicit 'is not None' checks
        f_obs_attr = getattr(self, "Fo", None)
        if f_obs_attr is None:
            f_obs_attr = getattr(self, "Fobs", None)
        if f_obs_attr is None:
            f_obs_attr = getattr(self, "fo", None)
            
        if f_obs_attr is None:
            raise AttributeError("Could not locate experimental amplitude array (Fo/Fobs) on the calculator instance.")
        f_obs = jnp.array(f_obs_attr)
        
        # Calculate an analytical scale factor k to align calculated and observed intensities
        scale_factor = jnp.sum(f_obs * f_calc_mag) / (jnp.sum(f_calc_mag ** 2) + 1e-8)
        
        # Evaluate scale-invariant least-squares crystallographic residuals
        residuals_sq = (f_obs - scale_factor * f_calc_mag) ** 2
        
        # Apply continuous adaptive resolution shielding if t_hat is passed from the outer loop
        if t_hat is not None:
            # self.dHKL maps the explicit resolution d-spacing (in Angstroms) per reflection
            d_hkl = jnp.array(self.dHKL)
            # Damps high-frequency phases under heavy noise (t_hat -> 1), filters down cleanly as t_hat -> 0
            weight = jnp.exp(-50.0 * t_hat * (1.0 / (d_hkl ** 2)))
            residuals_sq = residuals_sq * weight
            
        return jnp.sum(residuals_sq)

def init_neutron_sfc(oracle_atoms, mtz_path):
    with tempfile.TemporaryDirectory() as tmpdir:
        pdb_path = os.path.join(tmpdir, "oracle.pdb")
        pdb_file = pdb.PDBFile()
        
        # Sanitize residue names to satisfy strict PDB 3-character constraints
        oracle_atoms.res_name = np.array([name[:3] for name in oracle_atoms.res_name])
        
        pdb.set_structure(pdb_file, oracle_atoms)
        pdb_file.write(pdb_path)
        
        # Pull unit cell dimensions and space group parameters directly from the MTZ
        mtz = gemmi.read_mtz_file(mtz_path)
        cell = mtz.cell
        sg_name = mtz.spacegroup_name
        dmin_val = mtz.resolution_high()
        
        # Format a valid PDB CRYST1 record line
        cryst1_line = (
            f"CRYST1{cell.a:9.3f}{cell.b:9.3f}{cell.c:9.3f}"
            f"{cell.alpha:7.2f}{cell.beta:7.2f}{cell.gamma:7.2f} "
            f"{sg_name:<11}\n"
        )
        
        # Prepend the CRYST1 record line into the scratch file
        with open(pdb_path, "r") as f:
            pdb_content = f.read()
        with open(pdb_path, "w") as f:
            f.write(cryst1_line + pdb_content)
            
        print(f"Injected Symmetry Header: {cryst1_line.strip()} | Dmin Limit: {dmin_val}A", file=sys.stderr)
            
        # Initialize our custom wrapped structural calculator enforcing full experimental limits
        sfc = NeutronSFCalculator(
            PDBfile_dir=pdb_path,
            mtzfile_dir=mtz_path,
            dmin=dmin_val,
            set_experiment=True
        )
        return sfc
