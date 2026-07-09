import os
import sys  
import tempfile
import numpy as np
import jax
import jax.numpy as jnp
import gemmi  
from biotite.structure.io import pdb
from SFC_Jax.Fmodel import SFcalculator

# ==============================================================================
# MONKEYPATCH FOR GEMMI VERSION CONFLICT (v0.7.0+)
# ==============================================================================
if not hasattr(gemmi.UnitCell, "fractionalization_matrix"):
    gemmi.UnitCell.fractionalization_matrix = property(lambda self: self.frac.mat)
if not hasattr(gemmi.UnitCell, "orthogonalization_matrix"):
    gemmi.UnitCell.orthogonalization_matrix = property(lambda self: self.orth.mat)
# ==============================================================================

class NeutronSFCalculator(SFcalculator):
    """Subclass of SFcalculator that implements a clean, scale-invariant, 
    JAX-differentiable structure factor amplitude loss, utilizing the native
    Calc_Ftotal pipeline and precomputed bulk solvent masks.
    """
    def compute_loss(self, xyz, t_hat=None):
        # 1. Update the internal Fprotein_HKL using the current diffusion coordinates
        self.Calc_Fprotein(atoms_position_tensor=xyz)
        
        # 2. Use the library's native total structure factor calculation.
        # CRITICAL: We must explicitly pass kaniso=jnp.zeros(6). If omitted, 
        # SFC_Jax injects random noise (jax.random.normal) for Aniso-B initialization!
        f_calc_complex = self.Calc_Ftotal(
            kall=jnp.array(1.0),
            kaniso=jnp.zeros(6), 
            ksol=jnp.array(0.35), 
            bsol=jnp.array(50.0)
        )
        
        f_calc_mag = jnp.abs(f_calc_complex)
        
        # 3. Safely extract experimental amplitudes
        f_obs_attr = getattr(self, "Fo", None)
        if f_obs_attr is None:
            f_obs_attr = getattr(self, "Fobs", None)
        if f_obs_attr is None:
            f_obs_attr = getattr(self, "fo", None)
            
        if f_obs_attr is None:
            raise AttributeError("Could not locate experimental amplitude array on the calculator instance.")
        
        f_obs = jnp.array(f_obs_attr)
        
        # 4. Enforce Cross-Validation Partitions
        mask_valid = (f_obs > 0.0) & (~jnp.isnan(f_obs))
        mask_free = mask_valid & (self.freer_flags == 0)
        mask_work = mask_valid & (self.freer_flags != 0)
        
        f_obs = jnp.where(mask_valid, f_obs, 0.0)
        f_calc_mag = jnp.where(mask_valid, f_calc_mag, 0.0)
        
        # 5. Dynamic Linear Scaling (Evaluated strictly on the Working Set)
        num = jnp.sum(jnp.where(mask_work, f_obs * f_calc_mag, 0.0))
        den = jnp.sum(jnp.where(mask_work, f_calc_mag ** 2, 0.0)) + 1e-8
        scale_factor = num / den
        
        # 6. Monitor R-Factors
        diff = jnp.abs(f_obs - scale_factor * f_calc_mag)
        r_work = jnp.sum(jnp.where(mask_work, diff, 0.0)) / (jnp.sum(jnp.where(mask_work, f_obs, 0.0)) + 1e-8)
        r_free = jnp.sum(jnp.where(mask_free, diff, 0.0)) / (jnp.sum(jnp.where(mask_free, f_obs, 0.0)) + 1e-8)
        
        # 7. Normalize Loss to prevent gradient clipping saturation
        residuals_sq = jnp.where(mask_work, (f_obs - scale_factor * f_calc_mag) ** 2, 0.0)
        normalization = jnp.sum(jnp.where(mask_work, f_obs ** 2, 0.0)) + 1e-8
        normalized_loss = jnp.sum(residuals_sq) / normalization
        
        return normalized_loss, (r_work, r_free)

def init_neutron_sfc(oracle_atoms, mtz_path):
    with tempfile.TemporaryDirectory() as tmpdir:
        pdb_path = os.path.join(tmpdir, "oracle.pdb")
        pdb_file = pdb.PDBFile()
        
        oracle_atoms.res_name = np.array([name[:3] for name in oracle_atoms.res_name])
        pdb.set_structure(pdb_file, oracle_atoms)
        pdb_file.write(pdb_path)
        
        mtz = gemmi.read_mtz_file(mtz_path)
        
        free_r_col = None
        for col in mtz.columns:
            if col.type == 'I' and 'free' in col.label.lower():
                free_r_col = col.label
                break
                
        working_mtz_path = mtz_path
        if free_r_col is None:
            print("No Free-R flags found! Injecting 5% holdout set...", file=sys.stderr)
            mtz.add_free_r_flags(fraction=0.05)
            free_r_col = "FreeR_flag"
            working_mtz_path = os.path.join(tmpdir, "working_data.mtz")
            mtz.write_to_file(working_mtz_path)
        else:
            print(f"Detected existing Free-R flags in column: {free_r_col}", file=sys.stderr)
        
        cell = mtz.cell
        sg_name = mtz.spacegroup_name
        dmin_val = mtz.resolution_high()
        
        cryst1_line = (
            f"CRYST1{cell.a:9.3f}{cell.b:9.3f}{cell.c:9.3f}"
            f"{cell.alpha:7.2f}{cell.beta:7.2f}{cell.gamma:7.2f} "
            f"{sg_name:<11}\n"
        )
        
        with open(pdb_path, "r") as f:
            pdb_content = f.read()
        with open(pdb_path, "w") as f:
            f.write(cryst1_line + pdb_content)
            
        print(f"Injected Symmetry Header: {cryst1_line.strip()} | Dmin Limit: {dmin_val}A", file=sys.stderr)
            
        sfc = NeutronSFCalculator(
            PDBfile_dir=pdb_path,
            mtzfile_dir=working_mtz_path,
            dmin=dmin_val,
            set_experiment=True,
            mode="neutron"
        )

        # ==============================================================================
        # BASELINE SOLVENT INITIALIZATION
        # Precompute the bulk solvent mask Fmask_HKL once using the unrefined template 
        # coords so we don't have to rebuild grids during JAX diffusion.
        # ==============================================================================
        print("Initializing Baseline Bulk Solvent Mask...", file=sys.stderr)
        sfc.inspect_data()
        sfc.Calc_Fprotein(jnp.array(oracle_atoms.coord))
        sfc.Calc_Fsolvent()
        
        # ==============================================================================
        # FLAG ALIGNMENT
        # ==============================================================================
        hk_col = np.array(mtz.column_with_label("H").array, dtype=int)
        kk_col = np.array(mtz.column_with_label("K").array, dtype=int)
        ll_col = np.array(mtz.column_with_label("L").array, dtype=int)
        flag_col = np.array(mtz.column_with_label(free_r_col).array, dtype=int)
        
        flag_dict = {(h, k, l): f for h, k, l, f in zip(hk_col, kk_col, ll_col, flag_col)}
        
        sfc_hkl = getattr(sfc, "HKL_array", None)
        if sfc_hkl is None:
            sfc_hkl = getattr(sfc, "Hasu_array", None)
            
        aligned_flags = np.zeros(len(sfc_hkl), dtype=int)
        for i, hkl in enumerate(sfc_hkl):
            h, k, l = int(hkl[0]), int(hkl[1]), int(hkl[2])
            aligned_flags[i] = flag_dict.get((h, k, l), 1)
            
        sfc.freer_flags = jnp.array(aligned_flags)
        
        num_free = np.sum(aligned_flags == 0)
        num_work = np.sum(aligned_flags != 0)
        print(f"Cross-Validation Split | Work: {num_work} reflections | Free: {num_free} reflections", file=sys.stderr)
        
        return sfc
