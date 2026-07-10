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
                    # If bonded to Nitrogen, Oxygen, or Sulfur, it exchanges with D2O
                    if sfc_atoms.element[neighbor] in ["N", "O", "S"]:
                        sfc_atoms.element[i] = "D"
                        sfc_atoms.atom_name[i] = "D" + sfc_atoms.atom_name[i][1:]
                        break
        
        sfc_atoms.res_name = np.array([name[:3] for name in sfc_atoms.res_name])
        pdb.set_structure(pdb_file, sfc_atoms)
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

        print("Initializing Baseline Bulk Solvent Mask...", file=sys.stderr)
        sfc.inspect_data()
        sfc.Calc_Fprotein(jnp.array(sfc_atoms.coord))
        sfc.Calc_Fsolvent()
        
        # ==============================================================================
        # ROBUST FLAG ALIGNMENT (Bypassing SFC_Jax's broken internal parser)
        # ==============================================================================
        mtz_rs = rs.read_mtz(working_mtz_path)
        
        if free_r_col not in mtz_rs.columns:
            raise ValueError(f"Free-R column '{free_r_col}' not found in MTZ!")
            
        rfree_series = mtz_rs[free_r_col]
        
        # Safely identify the minority class (Free set) using pandas to handle pd.NA
        valid_data = rfree_series.dropna().to_numpy()
        
        if len(valid_data) > 0:
            unique, counts = np.unique(valid_data, return_counts=True)
            free_val = unique[np.argmin(counts)]
            work_val = unique[np.argmax(counts)]
        else:
            free_val, work_val = 0, 1
            
        # Fill missing flags (pd.NA) with the working set value, then cast safely
        flag_array = rfree_series.fillna(work_val).to_numpy()
        hkls = mtz_rs.get_hkls()
        
        flag_dict = {}
        for h, k, l, f in zip(hkls[:,0], hkls[:,1], hkls[:,2], flag_array):
            flag_dict[(h, k, l)] = (f == free_val)
            
        sfc_hkl = getattr(sfc, "HKL_array", None)
        if sfc_hkl is None:
            sfc_hkl = getattr(sfc, "Hasu_array", None)
            
        aligned_masks = np.zeros(len(sfc_hkl), dtype=bool)
        for i, hkl in enumerate(sfc_hkl):
            h, k, l = int(hkl[0]), int(hkl[1]), int(hkl[2])
            aligned_masks[i] = flag_dict.get((h, k, l), False)
            
        sfc.freer_mask = jnp.array(aligned_masks)
        
        num_free = np.sum(aligned_masks)
        num_work = len(aligned_masks) - num_free
        print(f"Cross-Validation Split | Work: {num_work} reflections | Free: {num_free} reflections", file=sys.stderr)
        
        return sfc
