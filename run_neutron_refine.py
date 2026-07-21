import logging
import pathlib
import os
import json
import numpy as np
import jax
import jax.numpy as jnp
import biotite.structure.io.pdbx as pdbx
import gemmi
from scipy.interpolate import RegularGridInterpolator
from absl import app, flags

from alphafold3.common import folding_input
from alphafold3.data import featurisation
from alphafold3.constants import chemical_components
from alphafold3.model.pipeline import structure_cleaning
from alphafold3.model.components import utils
from alphafold3.model.network import diffusion_head
from alphafold3.model import feat_batch
from alphafold3.model.atom_layout import atom_layout

from af3_neutron import make_model_config, HostRunner, Hijacker
from af3_neutron.sfc_adapter import init_neutron_sfc, align_oracle_to_reference
from af3_neutron.hijack import optimize_solvent_grid

import jax
import jax.numpy as jnp
import haiku as hk
import logging
from alphafold3.model.network.evoformer import Evoformer
from alphafold3.model.scoring import scoring

# Import your custom modules
from af3_neutron.diffraction_embedding import compute_debye_features, sample_patterson_map, compute_diffraction_delta_z
from af3_neutron.patterson import extract_patterson_grid_from_mtz
from alphafold3.constants import chemical_components

def get_local_ccd():
    """Retrieve AF3's local CCD instance by inspecting the constants module."""
    # Check standard attribute getter names
    for attr in ["CCD", "DEFAULT_CCD", "get_ccd", "read_ccd", "load_ccd_data"]:
        if hasattr(chemical_components, attr):
            val = getattr(chemical_components, attr)
            return val() if callable(val) else val

    # Dynamic fallback: inspect module attributes for a CCD instance or getter
    for name in dir(chemical_components):
        if "ccd" in name.lower():
            val = getattr(chemical_components, name)
            if callable(val):
                try:
                    res = val()
                    if res is not None:
                        return res
                except Exception:
                    continue
            elif val is not None:
                return val
    return None

# Load the local CCD once at script startup
_LOCAL_CCD = get_local_ccd()

# Save the original AF3 method
_original_embed_template_pair = Evoformer._embed_template_pair
CUSTOM_DIFFRACTION_DATA = {}

def patched_embed_template_pair(self, batch, pair_activations, pair_mask, key):
    updated_pair, next_key = _original_embed_template_pair(self, batch, pair_activations, pair_mask, key)

    if batch.templates.aatype.shape[0] > 0 and "patterson_grid" in CUSTOM_DIFFRACTION_DATA:
        # 1. Fetch true 1D protein token mask from AF3's top-level batch features
        is_protein = batch.token_features.is_protein
        is_protein_1d = (jnp.asarray(is_protein, dtype=jnp.float32) > 0.5)  # [N_tokens]

        # Diagnostic output to verify ligand token exclusion
        num_protein_tokens = jnp.sum(is_protein_1d)
        total_tokens = is_protein_1d.shape[0]
        jax.debug.print(
            "[Diffraction Patch Mask] Valid Protein Tokens: {p}/{t}",
            p=num_protein_tokens,
            t=total_tokens,
        )

        # 2. Construct 2D Protein-Protein Pair Mask [N_tokens, N_tokens]
        # Evaluates to 1.0 ONLY for (Protein, Protein) pairs; 0.0 if either token is ligand
        protein_pair_mask = is_protein_1d[:, None] * is_protein_1d[None, :]

        # 3. Combine with AF3 spatial pair_mask
        effective_mask = pair_mask * protein_pair_mask

        # 4. Compute ΔZ strictly for protein-protein pair channels
        delta_z = compute_diffraction_delta_z(
            aatype=batch.templates.aatype[0],
            atom_positions=batch.templates.atom_positions[0],
            atom_mask=batch.templates.atom_mask[0],
            patterson_grid=CUSTOM_DIFFRACTION_DATA["patterson_grid"],
            grid_origin=CUSTOM_DIFFRACTION_DATA["grid_origin"],
            grid_spacing=CUSTOM_DIFFRACTION_DATA["grid_spacing"],
            pair_mask=effective_mask,
            out_channels=pair_activations.shape[-1],
            gamma=1.0,
        )

        norm_z = jnp.linalg.norm(updated_pair)
        norm_delta = jnp.linalg.norm(delta_z)
        ratio = norm_delta / (norm_z + 1e-8)
        jax.debug.print("[Diffraction Patch] ||Z||: {norm_z} | ||ΔZ||: {norm_delta} | Ratio: {ratio}",
                        norm_z=norm_z, norm_delta=norm_delta, ratio=ratio)

        updated_pair = updated_pair + delta_z

    return updated_pair, next_key

# Apply the patch to the AF3 source class
_original_embed_template_pair = Evoformer._embed_template_pair
Evoformer._embed_template_pair = patched_embed_template_pair

from jax.experimental.compilation_cache import compilation_cache as cc
cc.set_cache_dir(os.path.expanduser('./.jax_cache'))

flags.DEFINE_string(
    "json_path",
    "../forward_model/betalac_tetramer_refinement_input.json",
    "Path to input.",
)
flags.DEFINE_string("model_dir", "../af3_model_parameters/", "Path to weights.")
flags.DEFINE_integer("gpu_device", 0, "GPU ID.")
flags.DEFINE_string("mtz_path", "", "Optional data path.")
flags.DEFINE_string("output_path", "neutron_refined_output.cif", "Output path.")
flags.DEFINE_integer("num_recycles", 10, "Recycles.", lower_bound=1)
flags.DEFINE_integer("num_diffusion_samples", 5, "Samples.", lower_bound=1)
flags.DEFINE_bool("deuterate", False, "Simulate H/D exchange (swap H for D on N, O, S) for neutron scattering.")
flags.DEFINE_bool("perdeuterate", False, "Simulate perdeuterated system (swap all H for D) for neutron scattering.")
flags.DEFINE_string("reference_cif", None, "Path to the explicit crystal structure (e.g., 4BD1.cif) to align the AF3 model into the correct unit cell frame.")

FLAGS = flags.FLAGS

import pickle

def inject_adapter_weights(af3_params: hk.Params, filepath: str = "diffraction_adapter_weights.pkl") -> hk.Params:
    """Injects saved adapter weights into AF3's parameter tree at the expected scope path."""
    with open(filepath, "rb") as f:
        loaded_dict = pickle.load(f)

    # Convert to mutable dictionary
    merged_params = dict(af3_params)

    # Exact module scope path expected by Haiku inside Evoformer
    target_prefix = "diffuser/evoformer/diffraction_pair_adapter"

    for loaded_name, module_dict in loaded_dict.items():
        # Extract submodule suffix (e.g., 'adapter_mlp_1', 'adapter_mlp_2')
        suffix = loaded_name.split("diffraction_pair_adapter/")[-1]
        target_key = f"{target_prefix}/{suffix}"

        # Inject converted JAX array weights directly into the param tree
        merged_params[target_key] = jax.tree_util.tree_map(jnp.array, module_dict)
        logging.info(f"Adapter weights successfully injected into: {target_key}")

    return merged_params

def resolve_ligand_smiles(ligand_entry: dict) -> str:
    """Extract or resolve SMILES string for a ligand using AF3's local CCD database."""
    # 1. Check for direct SMILES in input JSON
    if "smiles" in ligand_entry:
        return ligand_entry["smiles"]
    
    # 2. Resolve CCD code using AF3's internal CCD database
    if "ccdCodes" in ligand_entry and len(ligand_entry["ccdCodes"]) > 0:
        ccd_code = ligand_entry["ccdCodes"][0].upper()
        
        if _LOCAL_CCD is not None:
            # Query local CCD via AF3's component_name_to_info utility
            info = chemical_components.component_name_to_info(ccd=_LOCAL_CCD, res_name=ccd_code)
            if info and getattr(info, "pdbx_smiles", None):
                smiles = info.pdbx_smiles
                logging.info(f"Resolved CCD code '{ccd_code}' from local AF3 database: {smiles}")
                return smiles
                
            # Direct mapping fallback if _LOCAL_CCD is a dict
            if hasattr(_LOCAL_CCD, "get"):
                entry = _LOCAL_CCD.get(ccd_code)
                if entry:
                    smiles = None
                    if isinstance(entry, dict):
                        smiles = entry.get('_chem_comp.pdbx_smiles') or entry.get('pdbx_smiles')
                    elif hasattr(entry, 'pdbx_smiles'):
                        smiles = entry.pdbx_smiles
                    
                    if isinstance(smiles, list):
                        smiles = smiles[0]
                    if smiles and smiles not in ('?', '.', None):
                        logging.info(f"Resolved CCD code '{ccd_code}' directly from CCD entry: {smiles}")
                        return smiles

    raise ValueError(
        f"Unable to resolve SMILES definition for ligand {ligand_entry} "
        "using local AF3 CCD database. Please provide 'smiles' explicitly in the JSON."
    )

def main(argv):
    del argv
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ccd = chemical_components.Ccd()

    # featurization
    fold_input = next(
        folding_input.load_fold_inputs_from_path(pathlib.Path(FLAGS.json_path))
    )
    cleaned_struc, _ = structure_cleaning.clean_structure(
        fold_input.to_structure(ccd=ccd),
        ccd=ccd,
        drop_non_standard_atoms=True,
        drop_missing_sequence=True,
        filter_clashes=False,
        filter_waters=False,
        filter_hydrogens=True,
        covalent_bonds_only=True,
        filter_leaving_atoms=True,
        only_glycan_ligands_for_leaving_atoms=True,
        filter_crystal_aids=True,
        remove_polymer_polymer_bonds=True,
        remove_bad_bonds=True,
        remove_nonsymmetric_bonds=False
    )

    MIN_EMBEDDING_SIZE = 128
    MAX_EMBEDDING_SIZE = 8192
    featurised = featurisation.featurise_input(
        fold_input=fold_input,
        buckets=list(range(MIN_EMBEDDING_SIZE, MAX_EMBEDDING_SIZE + 1, 128)),
        ccd=ccd,
        verbose=False,
    )
    batch_obj = feat_batch.Batch.from_data_dict(featurised[0])
    batch = jax.tree.map(jnp.asarray, utils.remove_invalidly_typed_feats(featurised[0]))

    model_output_to_flat = atom_layout.compute_gather_idxs(
            source_layout=batch_obj.convert_model_output.token_atoms_layout,
            target_layout=batch_obj.convert_model_output.flat_output_layout,
    )

    gather_idxs = jnp.array(model_output_to_flat.gather_idxs)

    # runner
    device = jax.local_devices(backend="gpu")[FLAGS.gpu_device]
    runner = HostRunner(
        config=make_model_config(FLAGS.num_recycles, FLAGS.num_diffusion_samples),
        device=device,
        model_dir=pathlib.Path(FLAGS.model_dir),
    )
    # make embeddings and oracle
    rng = jax.random.split(
        jax.random.PRNGKey(fold_input.rng_seeds[0] if fold_input.rng_seeds else 1)
    )[0]

    # 1. First: Extract Patterson map and load adapter weights into runner
    if FLAGS.mtz_path and os.path.exists(FLAGS.mtz_path):
        logging.info(f"Extracting Patterson grid directly from {FLAGS.mtz_path}...")
        p_grid, origin, spacing = extract_patterson_grid_from_mtz(
            FLAGS.mtz_path, 
            fobs_col="FP"
        )
        
        # Populate registry BEFORE running host embeddings
        CUSTOM_DIFFRACTION_DATA["patterson_grid"] = jnp.array(p_grid)
        CUSTOM_DIFFRACTION_DATA["grid_origin"] = jnp.array(origin)
        CUSTOM_DIFFRACTION_DATA["grid_spacing"] = jnp.array(spacing)

        # Inject trained adapter weights into AF3 model parameters
        runner.model_params = inject_adapter_weights(runner.model_params, "diffraction_adapter_weights.pkl")

    # 2. Second: Run host embeddings WITH active diffraction patching
    rng, rng_emb = jax.random.split(rng)
    logging.info("Computing diffraction-informed host embeddings...")
    embeddings = runner.get_host_embeddings(
        rng_emb,
        batch,
    )
    n_steps = getattr(runner._model_config.heads.diffusion.eval, "steps", 200)
    noise_schedule = diffusion_head.noise_schedule(jnp.linspace(0, 1, n_steps + 1))

    initial_noise = (
        jax.random.normal(
            jax.random.split(jax.random.PRNGKey(42))[1],
            batch["pred_dense_atom_mask"].shape + (3,),
        )
        * noise_schedule[0]
    )
    positions_denoised = runner.predict_host_vector_field(
        jax.random.PRNGKey(0), initial_noise, jnp.array([noise_schedule[0]]), batch, embeddings
    )

    # ==============================================================================
    # EVALUATE CONFIDENCE HEAD
    # ==============================================================================
    logging.info("Running Confidence Head to extract pLDDT...")
    confidence_dict = runner.predict_confidence(
        jax.random.PRNGKey(0), batch, embeddings, positions_denoised
    )

    plddt_per_atom = atom_layout.convert(
        gather_info=model_output_to_flat,
        arr=confidence_dict['predicted_lddt'],
        layout_axes=(-2, -1),  # Mapping residue dim to atom dim
    )
    plddt_heavy_only = np.array(plddt_per_atom.reshape(-1))
    # ==============================================================================

    logging.info("Parsing ligand definitions from input JSON...")
    with open(FLAGS.json_path, "r") as f:
        input_json_data = json.load(f)

    ligand_smiles_dict = {}
    for seq in input_json_data.get("sequences", []):
        if "ligand" in seq:
            lig_dict = seq["ligand"]
            l_id = lig_dict["id"]
            if isinstance(l_id, list):
                l_id = l_id[0]
            
            l_smiles = resolve_ligand_smiles(lig_dict)
            ligand_smiles_dict[l_id] = l_smiles
            logging.info(f"Ligand ID '{l_id}' mapped to SMILES: {l_smiles}")

    # Parse covalent bond definitions
    bonded_atom_pairs = input_json_data.get("bondedAtomPairs", [])
    if bonded_atom_pairs:
        logging.info(f"Found {len(bonded_atom_pairs)} covalent bond(s) in input JSON: {bonded_atom_pairs}")

    oracle = Hijacker.build_oracle(
        batch_obj.convert_model_output.flat_output_layout,
        np.array(positions_denoised.reshape((-1, 3))[gather_idxs]),
        ligand_smiles_dict=ligand_smiles_dict,
        bonded_atom_pairs=bonded_atom_pairs,
        ph=7.4,
    )

    # 3. Apply annotations to heavy atoms

    # Get the total number of atoms (Heavy + Hydrogens)
    num_oracle_atoms = oracle.atoms.array_length()

    # Create a full-size array initialized to 0.0
    # (or a reasonable default, like the mean pLDDT)
    full_b_factors = np.zeros(num_oracle_atoms)

    # Configurable exponential scaling parameters
    variance_base = 0.1   # Minimum variance floor for perfect predictions
    variance_scale = 0.05 # Scaling multiplier
    variance_decay = 10.0 # How fast the variance explodes as pLDDT drops

    # Map pLDDT to Positional Variance (sigma^2)
    # Low pLDDT inflates the denominator, dynamically dropping the penalty to near-zero
    sigma_sq = variance_base + variance_scale * jnp.exp((100.0 - plddt_heavy_only) / variance_decay)

    # Inject the gathered pLDDT values into the heavy atom indices
    # oracle.mapping.heavy_indices holds the correct indices for heavy atoms
    full_b_factors[oracle.mapping.heavy_indices] = sigma_sq

    # Use the precomputed map to broadcast B-factors to Hydrogens
    final_b_factors = 26.3 * full_b_factors[np.array(oracle.mapping.hydrogen_to_heavy_map)]
    print(f"B-factor stats: min={final_b_factors.min()}, max={final_b_factors.max()}, mean={final_b_factors.mean()}")

    #  Apply to the oracle
    oracle.atoms.set_annotation("b_factor", final_b_factors)

    # Use explicit reference alignment if provided
    if FLAGS.reference_cif:
        oracle = align_oracle_to_reference(oracle, FLAGS.reference_cif)
    else:
        logging.warning("No --reference_cif provided. The model will remain at the AF3 origin, which may cause high R-factors.")

    sfc = init_neutron_sfc(oracle.atoms, FLAGS.mtz_path, deuterate=FLAGS.deuterate, perdeuterate=FLAGS.perdeuterate) if FLAGS.mtz_path else None

    # Get baseline coordinates
    xyz_baseline = oracle.mapping.initial_coordinates

    # Run the grid search to find the perfect neutron solvent parameters
    if sfc is not None:
        best_k_sol, best_b_sol = optimize_solvent_grid(sfc, xyz_baseline)
        print(f"Optimal Solvent Found -> k_sol: {best_k_sol:.3f}, b_sol: {best_b_sol:.1f}")

        # Lock them in for the diffusion loop
        sfc.k_sol = best_k_sol
        sfc.b_sol = best_b_sol

    # hijack loop using the generalized proximal-based implementation
    sfc_weight=1000
    conformations = Hijacker.hijack_diffusion(
        runner,
        batch,
        embeddings,
        gather_idxs,
        oracle,
        sfc,
        jax.random.PRNGKey(0),
        sfc_weight=sfc_weight
    )

    logging.info("Assembling final atomic coordinates...")
    oracle.atoms.coord = Hijacker.assemble_coordinates(
        conformations[0],
        gather_idxs,
        oracle,
        sfc_weight=sfc_weight
    )

    # hydride append hydrogen to array end, sort by chain and res for viz of ss
    contiguous_indices = np.lexsort((oracle.atoms.res_id, oracle.atoms.chain_id))
    oracle.atoms = oracle.atoms[contiguous_indices]

    if FLAGS.perdeuterate:
        h_mask = (oracle.atoms.element == "H")
        oracle.atoms.element[h_mask] = "D"
        for i in np.where(h_mask)[0]:
            oracle.atoms.atom_name[i] = "D" + oracle.atoms.atom_name[i][1:]
    elif FLAGS.deuterate:
        h_mask = (oracle.atoms.element == "H")
        for i in np.where(h_mask)[0]:
            bonded_indices = oracle.atoms.bonds.get_bonds(i)[0]
            for neighbor in bonded_indices:
                if oracle.atoms.element[neighbor] in ["N", "O", "S"]:
                    oracle.atoms.element[i] = "D"
                    oracle.atoms.atom_name[i] = "D" + oracle.atoms.atom_name[i][1:]
                    break

    output_path = pathlib.Path(FLAGS.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cif_file = pdbx.CIFFile()
    pdbx.set_structure(cif_file, oracle.atoms, data_block="neutron_refined")
    cif_file.write(output_path)
    logging.info("Refinement pipeline finished successfully.")

if __name__ == "__main__":
    app.run(main)
