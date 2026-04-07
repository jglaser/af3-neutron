import logging
import jax
import jax.numpy as jnp
import pathlib # Add this import!

from alphafold3.common import folding_input
from alphafold3.data import pipeline
from alphafold3.model import features
from alphafold3.constants import chemical_components
from alphafold3.model.pipeline import structure_cleaning

from af3_neutron.topology import inject_superset_topology_and_get_rotors
from af3_neutron.sampler import run_neutron_guided_diffusion

def main():
    logging.basicConfig(level=logging.INFO)
    
    json_path = "betalac_tetramer_refinement_input.json"
    logging.info(f"Processing {json_path}")
    
    # THE FIX: Use pathlib and extract the first job
    fold_inputs = folding_input.load_fold_inputs_from_path(pathlib.Path(json_path))
    fold_input = next(fold_inputs)
    
    ccd = chemical_components.Ccd()
    
    # AF3 Pipeline Step 1: Clean Structure
    # AF3 Pipeline Step 1: Clean Structure
    struct = fold_input.to_structure(ccd=ccd)
    cleaned_struc, _ = structure_cleaning.clean_structure(
        struct,
        ccd=ccd,
        drop_non_standard_atoms=True,
        drop_missing_sequence=True,
        # --- Required Keyword Arguments ---
        filter_clashes=False,
        filter_crystal_aids=False,
        filter_waters=True,
        filter_hydrogens=False, # <-- CRITICAL FOR NEUTRON PIPELINE: Keep H's!
        filter_leaving_atoms=True,
        only_glycan_ligands_for_leaving_atoms=True,
        covalent_bonds_only=True,
        remove_polymer_polymer_bonds=True,
        remove_bad_bonds=True,
        remove_nonsymmetric_bonds=False
    )

    # AF3 Pipeline Step 2: Intercept to build Super-Set Topology & Z-Matrix
    flat_output_layout, rotor_table = inject_superset_topology_and_get_rotors(cleaned_struc, ccd)
   
    # AF3 Pipeline Step 3: Tokenization
    all_tokens, all_token_atoms_layout, standard_token_idxs = features.tokenizer(
        flat_output_layout,
        ccd=ccd,
        max_atoms_per_token=48,  # <-- INCREASED TO 48 to fit all Arginine/Tryptophan hydrogens
        flatten_non_standard_residues=True,
        logging_name="neutron_refinement",
    )
    logging.info(f"Tokenized {len(all_tokens.atom_name)} atoms.")
    
    # AF3 Pipeline Step 4: Pad Tensors
    # Bypass the import error with a simple inline bucket calculator
    num_tokens = len(all_tokens.atom_name)
    buckets = [256, 512, 1024, 2048, 4096, 5120]
    padded_token_length = next((b for b in buckets if b >= num_tokens), num_tokens)
    
    padding_shapes = features.PaddingShapes(
        num_tokens=padded_token_length, 
        msa_size=128, 
        num_chains=1000, 
        num_templates=4, 
        num_atoms=padded_token_length * 48  # <-- MUST MATCH max_atoms_per_token
    )
    
    batch_token_features = features.TokenFeatures.compute_features(all_tokens, padding_shapes)
    
    batch = batch_token_features.as_data_dict()
    jax_batch = jax.tree.map(jnp.asarray, batch)
    
    # Run Refinement
    final_coords, final_chis = run_neutron_guided_diffusion(
        af3_step_fn=None, # Dummy fn for testing without weights
        batch=jax_batch, 
        rotor_table=rotor_table
    )
    
    logging.info("Refinement Complete.")

if __name__ == "__main__":
    main()
