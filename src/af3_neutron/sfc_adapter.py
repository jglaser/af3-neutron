import os
import sys  
import json
import tempfile
import dataclasses
import numpy as np
import jax
import jax.numpy as jnp
import gemmi  
import reciprocalspaceship as rs
from biotite.structure.io import pdb
import biotite.structure.io.pdbx as pdbx
import biotite.structure as struc
from SFC_Jax.Fmodel import SFcalculator, F_protein

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
    def __init__(self, *args, b_factors=None, occupancies=None, **kwargs):
        # 1. Execute parent initialization (parses PDB and sets Gemmi scattering lengths)
        super().__init__(*args, **kwargs)

        # 2. Safely overwrite the parent's parsed B-factors/occupancies
        # using the exact attribute names expected by F_protein
        if b_factors is not None:
            self.atom_b_iso = jnp.array(b_factors, dtype=jnp.float32)
            
        if occupancies is not None:
            self.atom_occ = jnp.array(occupancies, dtype=jnp.float32)

    def compute_loss(self, xyz, t_hat=None):
        # 1. PURE JAX COMPUTATION
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
        dr2_tensor = jnp.array(self.dr2HKL_array)
        
        # 2. Reactivate Bulk Solvent Mask
        # Fetch SFC_Jax solvent parameters (with underscores), using safe defaults
        k_sol = getattr(self, "k_sol", 0.35)
        b_sol = getattr(self, "b_sol", 50.0)
        Fmask_HKL = getattr(self, "Fmask_HKL")
        scaled_fmask = k_sol * jnp.exp(-b_sol * dr2_tensor / 4.0) * Fmask_HKL
        
        f_calc_complex = f_calc_protein + scaled_fmask
        f_calc_mag = jnp.abs(f_calc_complex)
        
        # 3. Safely extract experimental amplitudes
        f_obs_attr = getattr(self, "Fo", None)
        if f_obs_attr is None:
            f_obs_attr = getattr(self, "Fobs", None)
        if f_obs_attr is None:
            f_obs_attr = getattr(self, "fo", None)
            
        f_obs = jnp.array(f_obs_attr)
        
        # 4. Enforce Cross-Validation
        # NOTE: The low-resolution cutoff has been removed so the model 
        # can fit the newly activated solvent envelope at low angles.
        mask_valid = (f_obs > 0.0) & (~jnp.isnan(f_obs))

        # Resolution window for guidance.  Set `sfc.guidance_d_high = 8.0` to
        # restrict the target to d >= 8 A.  This is not an optimisation, it is
        # what makes the potential informative: a model 9.9 A from truth has no
        # signal past ~4 A, and for this cell only 1.6% of the 37,877 unique
        # reflections lie beyond 8 A -- so summing to 2.0 A drowns the usable
        # shells in noise.  It also cuts the (n_atoms x n_hkl) intermediate that
        # dominates peak memory by ~50x.
        # The window applies to the GUIDED loss only.  R_work/R_free are always
        # reported on the full resolution range, because a windowed R is (a) not
        # comparable across runs and (b) statistically useless as R_free: beyond
        # 8 A this cell has ~316 unique reflections, so a 5% free set is ~16
        # of them.
        d_high = getattr(self, "guidance_d_high", None)
        if d_high is None:
            mask_guide_res = jnp.ones_like(mask_valid)
        else:
            d_spacing = 1.0 / jnp.sqrt(jnp.maximum(dr2_tensor, 1e-12))
            mask_guide_res = d_spacing >= d_high

        mask_free = mask_valid & self.freer_mask
        mask_work = mask_valid & (~self.freer_mask)
        mask_loss = mask_work & mask_guide_res
        
        f_obs = jnp.where(mask_valid, f_obs, 0.0)
        f_calc_mag = jnp.where(mask_valid, f_calc_mag, 0.0)
        
        # 5. Dynamic Linear Scaling -- fitted separately for each purpose.
        #
        # The scale MUST be fitted on the same reflections the loss is evaluated
        # on.  A single global scale fitted to 2.0 A data and then applied to the
        # d >= 8 A subset is badly wrong, because bulk solvent makes the
        # effective scale strongly resolution dependent.  Symptom: the normalised
        # loss came out at 1.85-2.18, which is impossible for an LSQ-optimal
        # scale -- that quantity is 1 - CC^2 and so bounded by 1.  Values above 1
        # mean the scale is mis-fitted for the subset, which inflates the loss and
        # injects a gradient that chases the scale error rather than the
        # structure.
        def _lsq_scale(mask):
            num = jnp.sum(jnp.where(mask, f_obs * f_calc_mag, 0.0))
            den = jnp.sum(jnp.where(mask, f_calc_mag ** 2, 0.0)) + 1e-8
            return num / den

        scale_factor = _lsq_scale(mask_work)   # for the reported R-factors
        scale_loss = _lsq_scale(mask_loss)     # for the guided loss

        # 6. Monitor R-Factors (full resolution range, own scale)
        diff = jnp.abs(f_obs - scale_factor * f_calc_mag)
        r_work = jnp.sum(jnp.where(mask_work, diff, 0.0)) / (jnp.sum(jnp.where(mask_work, f_obs, 0.0)) + 1e-8)
        r_free = jnp.sum(jnp.where(mask_free, diff, 0.0)) / (jnp.sum(jnp.where(mask_free, f_obs, 0.0)) + 1e-8)
        
        # 7. Normalize Loss (windowed)
        residuals_sq = jnp.where(mask_loss, (f_obs - scale_loss * f_calc_mag) ** 2, 0.0)
        normalization = jnp.sum(jnp.where(mask_loss, f_obs ** 2, 0.0)) + 1e-8
        normalized_loss = jnp.sum(residuals_sq) / normalization
        
        return normalized_loss, (r_work, r_free)


def make_low_resolution_sfc(sfc, d_high: float, verbose: bool = True):
    """A copy of ``sfc`` whose *reflection list* is truncated to ``d >= d_high``.

    ``guidance_d_high`` masks reflections after ``F_protein`` has run, so it buys
    the statistical benefit but none of the memory: the (n_atoms x n_hkl) phase
    array -- which is what dominates peak memory, and is held live for the
    backward pass -- is still built at full size.  Truncating the list itself is
    what pays: for a 73x73x99 cell at 2.0 A there are ~37,900 unique reflections
    but only ~316 beyond 8 A, a factor of ~120 on that term.

    Use this for the guided loss and keep the full ``sfc`` for reporting
    R_work/R_free.  That is also the way to raise ``num_diffusion_samples``
    without hitting the ceiling: the per-particle cost of the guidance gradient
    is what scales with the particle count, and this is the term to shrink.

    Attributes are sliced by their leading axis, so this is robust to SFC_Jax
    gaining or losing per-reflection fields; anything whose first dimension does
    not match the HKL count is passed through untouched.
    """
    import copy as _copy

    import numpy as _np

    dr2 = _np.asarray(sfc.dr2HKL_array)
    n_hkl = dr2.shape[0]
    d = _np.where(dr2 > 0, 1.0 / _np.sqrt(_np.maximum(dr2, 1e-12)), _np.inf)
    keep = d >= float(d_high)
    if keep.sum() == 0:
        raise ValueError(f"no reflections with d >= {d_high} A")

    out = _copy.copy(sfc)
    n_sliced = []
    for name in dir(sfc):
        if name.startswith("__"):
            continue
        try:
            val = getattr(sfc, name)
        except Exception:
            continue
        arr = getattr(val, "shape", None)
        if arr is None or len(val.shape) == 0 or val.shape[0] != n_hkl:
            continue
        try:
            setattr(out, name, val[keep])
            n_sliced.append(name)
        except Exception:
            pass

    # asu2HKL_index maps HKL rows into the ASU list; it is sliced above, and the
    # ASU-side arrays are left whole, which is correct -- only the HKL side shrinks.
    if verbose:
        print(
            f"Low-resolution guidance SFC: {int(keep.sum())}/{n_hkl} reflections "
            f"(d >= {d_high} A); sliced {sorted(n_sliced)}",
            file=sys.stderr,
        )
    return out


def align_oracle_to_reference(oracle, reference_path):
    """
    Aligns the unanchored AF3 oracle coordinates to an explicit absolute crystal 
    lattice frame (e.g., the exact deposited neutron structure) rather than the AF3 template.
    """
    if not os.path.exists(reference_path):
        print(f"WARNING: Reference file '{reference_path}' not found. Skipping lattice alignment.", file=sys.stderr)
        return oracle

    print(f"Aligning Oracle coordinates to explicit crystal reference: {reference_path}", file=sys.stderr)
    
    # Handle both CIF and PDB reference files
    if reference_path.endswith('.cif') or reference_path.endswith('.mmcif'):
        ref_file = pdbx.CIFFile.read(reference_path)
        try:
            ref_atoms = pdbx.get_structure(ref_file, model=1, extra_fields=["altloc_id"])
            has_altloc = True
        except KeyError:
            ref_atoms = pdbx.get_structure(ref_file, model=1)
            has_altloc = False
    else:
        ref_file = pdb.PDBFile.read(reference_path)
        ref_atoms = pdb.get_structure(ref_file, model=1)
        has_altloc = "altloc_id" in ref_atoms.get_annotation_categories()

    # Filter for CA atoms
    ca_mask = (ref_atoms.atom_name == "CA")
    
    # Safely avoid alternate locations if present
    if has_altloc:
        altloc_mask = (ref_atoms.altloc_id == "") | (ref_atoms.altloc_id == ".") | (ref_atoms.altloc_id == "A")
        ca_mask = ca_mask & altloc_mask
        
    ref_ca = ref_atoms[ca_mask]
    oracle_ca = oracle.atoms[oracle.atoms.atom_name == "CA"]
    
    # Pair CA atoms sequentially (ignores mismatched residue numbering)
    min_len = min(len(ref_ca), len(oracle_ca))
    
    if min_len < 10:
        print("WARNING: Not enough CA atoms for Kabsch alignment. Skipping.", file=sys.stderr)
        return oracle
        
    matched_ref = ref_ca.coord[:min_len]
    matched_oracle = oracle_ca.coord[:min_len]
    
    # Biotite's superimpose returns the fitted coordinates and the AffineTransformation object
    fitted_coords, transform = struc.superimpose(matched_ref, matched_oracle)
    rmsd = struc.rmsd(matched_ref, fitted_coords)
    
    # Apply the AffineTransformation directly to the mobile Oracle coordinates
    aligned_coords = transform.apply(oracle.atoms.coord)
    oracle.atoms.coord = aligned_coords
    
    # Update the JAX mapping coordinates so the diffusion target is aligned
    oracle.mapping = dataclasses.replace(
        oracle.mapping, 
        initial_coordinates=jnp.array(aligned_coords, dtype=jnp.float32)
    )
    
    print(f"Successfully aligned Oracle to reference. (CA RMSD: {rmsd:.3f}A)", file=sys.stderr)
    return oracle


def init_neutron_sfc(oracle_atoms, mtz_path, deuterate=False, perdeuterate=False):
    with tempfile.TemporaryDirectory() as tmpdir:
        pdb_path = os.path.join(tmpdir, "oracle.pdb")
        pdb_file = pdb.PDBFile()
        
        # Clone oracle atoms so Hydride's internal "H" reliance isn't broken
        sfc_atoms = oracle_atoms.copy()
        
        # ==============================================================================
        # SANITIZE B-FACTORS FOR PDB FORMAT COMPATIBILITY
        # ==============================================================================
        # De-novo AF3 predictions with low pLDDT can generate B-factors >= 1000.0 A^2.
        # Clip to [0.0, 999.0] to fit Biotite's 3-pre-decimal digit PDB limit (F6.2).
        safe_b_factors = np.nan_to_num(sfc_atoms.b_factor, nan=30.0)
        sfc_atoms.b_factor = np.clip(safe_b_factors, 0.0, 999.0)
        
        # ==============================================================================
        # SIMULATE NEUTRON ISOTOPIC COMPOSITION (H/D EXCHANGE VS PERDEUTERATION)
        # ==============================================================================
        if perdeuterate:
            print("Enforcing PERDEUTERATION (All H -> D) for structure factor evaluation...", file=sys.stderr)
            h_mask = (sfc_atoms.element == "H")
            sfc_atoms.element[h_mask] = "D"
            for i in np.where(h_mask)[0]:
                sfc_atoms.atom_name[i] = "D" + sfc_atoms.atom_name[i][1:]

        elif deuterate:
            print("Simulating D2O H/D exchange (labile N/O/S-H -> D) for structure factor evaluation...", file=sys.stderr)
            h_mask = (sfc_atoms.element == "H")
            for i in np.where(h_mask)[0]:
                bonded_indices = sfc_atoms.bonds.get_bonds(i)[0]
                for neighbor in bonded_indices:
                    if sfc_atoms.element[neighbor] in ["N", "O", "S"]:
                        sfc_atoms.element[i] = "D"
                        sfc_atoms.atom_name[i] = "D" + sfc_atoms.atom_name[i][1:]
                        break
        
        sfc_atoms.res_name = np.array([name[:3] for name in sfc_atoms.res_name])
        pdb.set_structure(pdb_file, sfc_atoms)
        pdb_file.write(pdb_path)

        # ==============================================================================
        # FORCE ROBUST FLAGS USING RECIPROCALSPACESHIP
        # ==============================================================================
        mtz_rs = rs.read_mtz(mtz_path)
        
        print("Forcing robust 5% Free-R holdout set...", file=sys.stderr)
        
        # Generate reproducible random flags: 0 for Free (5%), 1 for Work (95%)
        np.random.seed(42)
        fresh_flags = np.random.choice([0, 1], size=len(mtz_rs), p=[0.05, 0.95])
        mtz_rs["FreeR_flag"] = rs.DataSeries(fresh_flags, dtype="I")
        
        working_mtz_path = os.path.join(tmpdir, "working_data.mtz")
        mtz_rs.write_mtz(working_mtz_path)
        
        # Read headers safely via Gemmi for the PDB CRYST1 line
        mtz_gemmi = gemmi.read_mtz_file(mtz_path)
        cell = mtz_gemmi.cell
        sg_name = mtz_gemmi.spacegroup_name
        dmin_val = mtz_gemmi.resolution_high()
        
        cryst1_line = (
            f"CRYST1{cell.a:9.3f}{cell.b:9.3f}{cell.c:9.3f}"
            f"{cell.alpha:7.2f}{cell.beta:7.2f}{cell.gamma:7.2f} "
            f"{sg_name:<11}\n"
        )
        
        with open(pdb_path, "r") as f:
            pdb_content = f.read()
        with open(pdb_path, "w") as f:
            f.write(cryst1_line + pdb_content)
            
        print(f"Injected Symmetry Header: {cryst1_line.strip()} | Dmin Limit: {dmin_val:.3f}A", file=sys.stderr)
            
        # Explicitly thread the pLDDT-derived B-factors into the constructor
        sfc = NeutronSFCalculator(
            PDBfile_dir=pdb_path,
            mtzfile_dir=working_mtz_path,
            dmin=dmin_val,
            set_experiment=True,
            freeflag="FreeR_flag",
            mode="neutron",
            b_factors=sfc_atoms.b_factor,
            occupancies=np.ones(sfc_atoms.array_length(), dtype=np.float32)
        )
        
        # EXPLICITLY OVERRIDE EXPERIMENTAL DATA
        sfc.Fo = jnp.array(mtz_rs["FP"].to_numpy(), dtype=jnp.float32)
        sfc.SigF = jnp.array(mtz_rs["SIGFP"].to_numpy(), dtype=jnp.float32)
        sfc.freer_mask = jnp.array(fresh_flags == 0, dtype=bool)

        print("Initializing Baseline Bulk Solvent Mask...", file=sys.stderr)
        sfc.inspect_data()
        sfc.Calc_Fprotein(jnp.array(sfc_atoms.coord))
        
        # Generate the solvent mask grid dynamically
        sfc.Calc_Fsolvent()
        sfc.deuterated_solvent = (deuterate or perdeuterate)
        
        num_free = int(np.sum(fresh_flags == 0))
        num_work = len(fresh_flags) - num_free
        print(f"Cross-Validation Split | Work: {num_work} | Free: {num_free}", file=sys.stderr)
        
        return sfc
