import os
import sys  
import tempfile
import numpy as np
import jax
import jax.numpy as jnp
import gemmi  
from biotite.structure.io import pdb
from SFC_Jax.Fmodel import SFcalculator, F_protein # <-- IMPORT PURE TENSOR FUNCTION

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
    JAX-differentiable structure factor amplitude loss, utilizing a pure
    functional pipeline to guarantee gradient tracking inside XLA loops.
    """
    def compute_loss(self, xyz, t_hat=None):
        # 1. PURE JAX COMPUTATION: Completely bypass self.Calc_Fprotein 
        # to prevent state-mutation bugs inside JAX fori_loops.
        atom_pos_frac = jnp.tensordot(xyz, self.orth2frac_tensor.T, 1)
        
        f_calc_protein_asu = F_protein(
            self.Hasu_array, 
            self.dr2asu_array,
            self.fullsf_tensor,
            self.reciprocal_cell_paras,
            self.R_G_tensor_stack, 
            self.T_G_tensor_stack,
            atom_pos_frac,
            self.atom_b_iso, 
            self.atom_b_aniso, 
            self.atom_occ
        )
        
        f_calc_protein = f_calc_protein_asu[self.asu2HKL_index]
        
        # 2. Apply Bulk Solvent Correction (Pure)
        dr2_tensor = jnp.array(self.dr2HKL_array)
        scaled_fmask = 0.35 * self.Fmask_HKL * jnp.exp(-50.0 * dr2_tensor / 4.0)
        
        f_calc_complex = f_calc_protein + scaled_fmask
        f_calc_mag = jnp.abs(f_calc_complex)
        
        # 3. Safely extract experimental amplitudes
        f_obs_attr = getattr(self, "Fo", None)
        if f_obs_attr is None:
            f_obs_attr = getattr(self, "Fobs", None)
        if f_obs_attr is None:
            f_obs_attr = getattr(self, "fo", None)
            
        f_obs = jnp.array(f_obs_attr)
        
        # 4. Enforce Cross-Validation Partitions
        mask_valid = (f_obs > 0.0) & (~jnp.isnan(f_obs))
        mask_free = mask_valid & (self.freer_flags == 0)
        mask_work = mask_valid & (self.freer_flags != 0)
        
        f_obs = jnp.where(mask_valid, f_obs, 0.0)
        f_calc_mag = jnp.where(mask_valid, f_calc_mag, 0.0)
        
        # 5. Dynamic Linear Scaling
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
        
        # Enforce realistic B-factors to prevent high-resolution noise amplification
        b_factors = np.full(oracle_atoms.array_length(), 30.0, dtype=np.float32)
        oracle_atoms.set_annotation("b_factor", b_factors)
        
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
