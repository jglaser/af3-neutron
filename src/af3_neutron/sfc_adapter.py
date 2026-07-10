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
        
        # 2. Add Bulk Solvent Mask
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
        
        # 4. Enforce Cross-Validation Partitions safely via precomputed mask
        mask_valid = (f_obs > 0.0) & (~jnp.isnan(f_obs))
        mask_free = mask_valid & self.freer_mask
        mask_work = mask_valid & (~self.freer_mask)
        
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
        
        jax.debug.print("SF mean: {m}", m=jnp.mean(self.fullsf_tensor))
        return normalized_loss, (r_work, r_free)

def align_oracle_to_template_from_json(oracle, json_path):
    """
    Parses the AF3 input JSON, extracts the template mmCIF path, and aligns
    the unanchored AF3 oracle coordinates to the absolute crystal lattice frame.
    """
    with open(json_path, 'r') as f:
        af3_input = json.load(f)

    template_cif_path = None

    # Traverse the AF3 JSON schema to locate the first available template
    for seq in af3_input.get("sequences", []):
        if "protein" in seq and "templates" in seq["protein"]:
            templates = seq["protein"]["templates"]
            if templates and "mmcifPath" in templates[0]:
                template_cif_path = templates[0]["mmcifPath"]
                break

    if template_cif_path is None:
        print("WARNING: No template mmcifPath found in JSON. Skipping lattice alignment.", file=sys.stderr)
        return oracle

    # Resolve the template path relative to the JSON file's directory
    json_dir = os.path.dirname(os.path.abspath(json_path))
    full_template_path = os.path.join(json_dir, template_cif_path)

    if not os.path.exists(full_template_path):
        print(f"WARNING: Template file '{full_template_path}' not found. Skipping lattice alignment.", file=sys.stderr)
        return oracle

    template_file = pdbx.CIFFile.read(full_template_path)

    # Safely attempt to parse altloc_id if the CIF file contains it
    try:
        template_atoms = pdbx.get_structure(template_file, model=1, extra_fields=["altloc_id"])
        has_altloc = True
    except KeyError:
        template_atoms = pdbx.get_structure(template_file, model=1)
        has_altloc = False

    # Filter for CA atoms
    ca_mask = (template_atoms.atom_name == "CA")

    # Safely avoid alternate locations if the template file specifies them
    if has_altloc and "altloc_id" in template_atoms.get_annotation_categories():
        altloc_mask = (template_atoms.altloc_id == "") | (template_atoms.altloc_id == ".") | (template_atoms.altloc_id == "A")
        ca_mask = ca_mask & altloc_mask

    temp_ca = template_atoms[ca_mask]
    oracle_ca = oracle.atoms[oracle.atoms.atom_name == "CA"]

    # Because PDB residue IDs often do not match AF3 1-indexed IDs,
    # we pair the CA atoms sequentially.
    min_len = min(len(temp_ca), len(oracle_ca))

    if min_len < 10:
        print("WARNING: Not enough CA atoms for Kabsch alignment. Skipping.", file=sys.stderr)
        return oracle

    matched_temp = temp_ca.coord[:min_len]
    matched_oracle = oracle_ca.coord[:min_len]

    # Biotite's superimpose returns the fitted coordinates and the AffineTransformation object
    fitted_coords, transform = struc.superimpose(matched_temp, matched_oracle)
    rmsd = struc.rmsd(matched_temp, fitted_coords)

    # Apply the AffineTransformation directly to the mobile Oracle coordinates
    aligned_coords = transform.apply(oracle.atoms.coord)
    oracle.atoms.coord = aligned_coords

    # Safely replace the frozen dataclass field so the JAX diffusion target is updated
    oracle.mapping = dataclasses.replace(
        oracle.mapping,
        initial_coordinates=jnp.array(aligned_coords, dtype=jnp.float32)
    )

    print(f"Aligned Oracle to crystal template: {template_cif_path} (CA RMSD: {rmsd:.3f}A)", file=sys.stderr)
    return oracle

def init_neutron_sfc(oracle_atoms, mtz_path, deuterate=False):
    with tempfile.TemporaryDirectory() as tmpdir:
        pdb_path = os.path.join(tmpdir, "oracle.pdb")
        pdb_file = pdb.PDBFile()
        
        # Clone oracle atoms so Hydride's internal "H" reliance isn't broken
        sfc_atoms = oracle_atoms.copy()
        
        # Enforce realistic B-factors to prevent high-resolution noise amplification
        b_factors = np.full(sfc_atoms.array_length(), 30.0, dtype=np.float32)
        sfc_atoms.set_annotation("b_factor", b_factors)
        
        # ==============================================================================
        # SIMULATE H/D EXCHANGE STRICTLY FOR SFC MAP EVALUATION
        # ==============================================================================
        if deuterate:
            print("Simulating D2O H/D exchange for structure factor evaluation...", file=sys.stderr)
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
            
        sfc = NeutronSFCalculator(
            PDBfile_dir=pdb_path,
            mtzfile_dir=working_mtz_path,
            dmin=dmin_val,
            set_experiment=True,
            freeflag="FreeR_flag",
            mode="neutron"
        )
        
        # EXPLICITLY OVERRIDE EXPERIMENTAL DATA
        sfc.Fo = jnp.array(mtz_rs["FP"].to_numpy(), dtype=jnp.float32)
        sfc.SigF = jnp.array(mtz_rs["SIGFP"].to_numpy(), dtype=jnp.float32)
        sfc.freer_mask = jnp.array(fresh_flags == 0, dtype=bool)

        print("Initializing Baseline Bulk Solvent Mask...", file=sys.stderr)
        sfc.inspect_data()
        sfc.Calc_Fprotein(jnp.array(sfc_atoms.coord))
        
        # CRITICAL: Force Fmask to zero to avoid X-ray solvent artifacts
        sfc.Fmask_HKL = jnp.zeros_like(sfc.Fprotein_HKL)
        
        num_free = int(np.sum(fresh_flags == 0))
        num_work = len(fresh_flags) - num_free
        print(f"Cross-Validation Split | Work: {num_work} | Free: {num_free}", file=sys.stderr)
        
        return sfc
