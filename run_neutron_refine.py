import logging
import pathlib
import os
import numpy as np
import jax
import jax.numpy as jnp
import biotite.structure.io.pdbx as pdbx
from absl import app
from absl import flags

from alphafold3.common import folding_input
from alphafold3.data import featurisation
from alphafold3.constants import chemical_components
from alphafold3.model.pipeline import structure_cleaning
from alphafold3.model.components import utils
from alphafold3.model.network import diffusion_head
from alphafold3.model import feat_batch
from alphafold3.model.atom_layout import atom_layout

from af3_neutron.runner import ModelRunner, make_model_config
from af3_neutron.topology import build_decoupled_topology
from af3_neutron.sampler import run_neutron_guided_diffusion, generate_final_oracle_coords
from af3_neutron.sfc_adapter import init_neutron_sfc

from jax.experimental.compilation_cache import compilation_cache as cc
cc.set_cache_dir(os.path.expanduser('./jax_cache'))

FLAGS = flags.FLAGS
flags.DEFINE_string('json_path', 'betalac_tetramer_refinement_input.json', 'Path to JSON.')
flags.DEFINE_string('model_dir', '../af3_model_parameters/', 'Path to weights.')
flags.DEFINE_integer('gpu_device', 0, 'GPU to use.')
flags.DEFINE_string('mtz_path', '', 'Optional path to MTZ file for neutron refinement.')
flags.DEFINE_string('output_path', 'neutron_refined_output.cif', 'Path to save the final mmCIF.')

_NUM_RECYCLES = flags.DEFINE_integer(
    'num_recycles',
    10,
    'Number of recycles to use during inference.',
    lower_bound=1,
)
_NUM_DIFFUSION_SAMPLES = flags.DEFINE_integer(
    'num_diffusion_samples',
    5,
    'Number of diffusion samples to generate.',
    lower_bound=1,
)

def main(argv):
    del argv
    logging.basicConfig(level=logging.INFO)
    
    json_path = pathlib.Path(FLAGS.json_path)
    model_dir = pathlib.Path(FLAGS.model_dir)
    fold_input = next(folding_input.load_fold_inputs_from_path(json_path))
    ccd = chemical_components.Ccd()
    
    struct = fold_input.to_structure(ccd=ccd)
    cleaned_struc, _ = structure_cleaning.clean_structure(
        struct, ccd=ccd, drop_non_standard_atoms=True, drop_missing_sequence=True,
        filter_clashes=False, filter_crystal_aids=False, filter_waters=False,
        filter_hydrogens=True, filter_leaving_atoms=True, # filter_hydrogens MUST BE TRUE for accurate VRAM loading
        only_glycan_ligands_for_leaving_atoms=True, covalent_bonds_only=True,
        remove_polymer_polymer_bonds=True, remove_bad_bonds=True, remove_nonsymmetric_bonds=False
    )
    
    logging.info("Running native AF3 featurisation...")
    buckets = list(range(128, 8192+1, 128))
    featurised_examples = featurisation.featurise_input(
        fold_input=fold_input, buckets=buckets, ccd=ccd, verbose=False
    )
   
    batch_dict = featurised_examples[0]
    batch_obj_raw = feat_batch.Batch.from_data_dict(batch_dict)
    flat_layout = batch_obj_raw.convert_model_output.flat_output_layout
    token_layout = batch_obj_raw.convert_model_output.token_atoms_layout
    batch = jax.tree.map(jnp.asarray, utils.remove_invalidly_typed_feats(batch_dict))
    
    gather_info = atom_layout.compute_gather_idxs(
        source_layout=token_layout, 
        target_layout=flat_layout
    )
    gather_idxs = jnp.array(gather_info.gather_idxs) 

    logging.info("Loading AF3 Weights...")
    device = jax.local_devices(backend='gpu')[FLAGS.gpu_device]

    # Pass the flag values down into the config builder
    model_config = make_model_config(
        num_recycles=_NUM_RECYCLES.value,
        num_diffusion_samples=_NUM_DIFFUSION_SAMPLES.value
    )

    model_runner = ModelRunner(
        config=model_config,
        device=device,
        model_dir=model_dir,
    )
   
    # 1. Extract Seed from JSON to perfectly match reference run
    model_seed = fold_input.rng_seeds[0] if fold_input.rng_seeds else 1

    logging.info("Executing AF3 Trunk (Pairformer)...")
    rng_key = jax.random.PRNGKey(model_seed)
    trunk_key, diffusion_key = jax.random.split(rng_key)
    
    embeddings = model_runner.get_conditionings(trunk_key, batch)
    
    # Extract the diffusion config
    diff_config = model_runner._model_config.heads.diffusion.eval
    n_steps = getattr(diff_config, 'steps', 200) if diff_config else 200

    mask_shape = batch['pred_dense_atom_mask'].shape
    noise_levels = diffusion_head.noise_schedule(jnp.linspace(0, 1, n_steps + 1))
    sample_key, noise_key = jax.random.split(diffusion_key)
    initial_noise = jax.random.normal(noise_key, mask_shape + (3,)) * noise_levels[0]
   
    # --- NEW: Generate 1-step baseline prediction for Oracle initialization ---
    logging.info("Generating 1-step baseline prediction for full-complex Oracle initialization...")
    t_hat = jnp.array([noise_levels[0]])
    positions_denoised = model_runner.evaluate_vector_field(
        jax.random.PRNGKey(0), initial_noise, t_hat, batch, embeddings
    )
    
    positions_flat = positions_denoised.reshape((-1, 3))
    x_af3_flat_baseline = np.array(positions_flat[gather_idxs])
    
    # We pass DeepMind's flat_layout and the baseline coordinates to build the full tetramer anchors
    rotor_table, mapping, water_mapping, oracle_atoms = build_decoupled_topology(
        flat_layout, x_af3_flat_baseline
    )
    # -------------------------------------------------------------------------

    if FLAGS.mtz_path:
        sfc_instance = init_neutron_sfc(oracle_atoms, FLAGS.mtz_path)
    else:
        logging.info("No MTZ file provided. Running with dummy physics loss.")
        sfc_instance = None
    
   
    # Run the Guided Diffusion Loop via the batched JIT wrapper
    final_coords_batched, final_chis_batched, final_waters_batched = run_neutron_guided_diffusion(
        model_runner=model_runner,
        batch_dict=batch,
        embeddings=embeddings,
        gather_idxs=gather_idxs,
        rotor_table=rotor_table,
        mapping=mapping,
        water_mapping=water_mapping,
        sfc_instance=sfc_instance,
        sample_key=sample_key
    )
    
    # Since DeepMind's sample_results returns a dictionary designed for multi-sample extraction,
    # we need to overwrite the output structure in the script to handle them independently.
    
    logging.info("Assembling final atomic coordinates for all samples...")
    reference_coords = jnp.array(oracle_atoms.coord, dtype=jnp.float32)
    num_samples = final_coords_batched.shape[0]
    
    all_final_structures = []
    for i in range(num_samples):
        final_x_full = generate_final_oracle_coords(
            final_coords_batched[i], final_chis_batched[i], final_waters_batched[i], 
            gather_idxs, rotor_table, mapping, water_mapping, reference_coords
        )
        all_final_structures.append(final_x_full)
        
    # Example logic for writing the best one, or saving all to list:
    # (Adapt this block based on how your post_processing module expects to save the CIFs)
    oracle_atoms.coord = np.array(all_final_structures[0]) # Storing sample 0 as the reference    

    output_cif_path = pathlib.Path(FLAGS.output_path)
    output_cif_path.parent.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Writing refined structure to {output_cif_path}...")
    cif_file = pdbx.CIFFile()
    pdbx.set_structure(cif_file, oracle_atoms, data_block="neutron_refined")
    cif_file.write(output_cif_path)
    logging.info("Done.")

if __name__ == "__main__":
    app.run(main)
