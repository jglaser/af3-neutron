import logging
import pathlib
import jax
import jax.numpy as jnp
import haiku as hk
from absl import app
from absl import flags

from alphafold3.common import folding_input
from alphafold3.data import pipeline
from alphafold3.model import features, model_config, model, params
from alphafold3.constants import chemical_components
from alphafold3.model.pipeline import structure_cleaning

from af3_neutron.topology import inject_superset_topology_and_get_rotors
from af3_neutron.sampler import run_neutron_guided_diffusion

# --- COMMAND LINE ARGUMENTS ---
FLAGS = flags.FLAGS
flags.DEFINE_string('json_path', 'betalac_tetramer_refinement_input.json', 'Path to the input JSON file.')
flags.DEFINE_string('model_dir', '../af3_model_parameters/', 'Path to the AF3 weights directory.')

def main(argv):
    del argv  # Unused
    logging.basicConfig(level=logging.INFO)
    
    json_path = pathlib.Path(FLAGS.json_path)
    model_dir = pathlib.Path(FLAGS.model_dir)
    
    logging.info(f"Processing {json_path}")
    fold_inputs = folding_input.load_fold_inputs_from_path(json_path)
    fold_input = next(fold_inputs)
    
    ccd = chemical_components.Ccd()
    
    # AF3 Pipeline Step 1: Clean Structure
    struct = fold_input.to_structure(ccd=ccd)
    cleaned_struc, _ = structure_cleaning.clean_structure(
        struct, 
        ccd=ccd, 
        drop_non_standard_atoms=True, 
        drop_missing_sequence=True,
        filter_clashes=False,
        filter_crystal_aids=False,
        filter_waters=True,
        filter_hydrogens=False, 
        filter_leaving_atoms=True,
        only_glycan_ligands_for_leaving_atoms=True,
        covalent_bonds_only=True,
        remove_polymer_polymer_bonds=True,
        remove_bad_bonds=True,
        remove_nonsymmetric_bonds=False
    )
    
    # AF3 Pipeline Step 2: Topology
    flat_output_layout, rotor_table = inject_superset_topology_and_get_rotors(
        cleaned_struc, ccd, fold_input
    )
    
    # AF3 Pipeline Step 3: Tokenization
    all_tokens, all_token_atoms_layout, standard_token_idxs = features.tokenizer(
        flat_output_layout,
        ccd=ccd,
        max_atoms_per_token=48,
        flatten_non_standard_residues=True,
        logging_name="neutron_refinement",
    )
    logging.info(f"Tokenized {len(all_tokens.atom_name)} atoms.")
    
    # AF3 Pipeline Step 4: Pad Tensors
    num_tokens = len(all_tokens.atom_name)
    buckets = [256, 512, 1024, 2048, 4096, 5120]
    padded_token_length = next((b for b in buckets if b >= num_tokens), num_tokens)
    
    padding_shapes = features.PaddingShapes(
        num_tokens=padded_token_length, msa_size=128, num_chains=1000, 
        num_templates=4, num_atoms=padded_token_length*48
    )
    
    batch_token_features = features.TokenFeatures.compute_features(all_tokens, padding_shapes)
    batch = batch_token_features.as_data_dict()
    jax_batch = jax.tree.map(jnp.asarray, batch)
    
    # =================================================================
    # RUN ALPHAFOLD 3 FORWARD PASS
    # =================================================================
    logging.info(f"Loading AF3 Weights from {model_dir}...")
    config = model_config.get_model_config("alphafold3")
    haiku_params = params.get_model_haiku_params(model_dir=model_dir)
    
    @hk.transform
    def forward_fn(b):
        return model.Model(config)(b)
    
    device = jax.local_devices()[0]
    af3_apply_jitted = jax.jit(forward_fn.apply, device=device)
    
    logging.info("Executing AlphaFold 3 Neural Network...")
    rng_key = jax.random.PRNGKey(42)
    af3_result = af3_apply_jitted(haiku_params, rng_key, jax_batch)
    logging.info("AF3 Forward Pass Complete.")
    
    # Run Refinement on Actual Predicted Coordinates
    final_coords, final_chis = run_neutron_guided_diffusion(
        af3_result=af3_result, 
        batch=jax_batch, 
        rotor_table=rotor_table
    )
    
    logging.info("Refinement Complete.")

if __name__ == "__main__":
    app.run(main)
