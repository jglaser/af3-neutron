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
from alphafold3.model import feat_batch # <--- NEW IMPORT
from alphafold3.model.atom_layout import atom_layout # <--- NEW IMPORT

from af3_neutron.af3_runner import ModelRunner, make_model_config
from af3_neutron.topology import build_decoupled_topology
from af3_neutron.sampler import run_neutron_guided_diffusion, generate_final_oracle_coords
from af3_neutron.sfc_adapter import init_neutron_sfc

FLAGS = flags.FLAGS
flags.DEFINE_string('json_path', 'betalac_tetramer_refinement_input.json', 'Path to JSON.')
flags.DEFINE_string('model_dir', '../af3_model_parameters/', 'Path to weights.')
flags.DEFINE_integer('gpu_device', 0, 'GPU to use.')
flags.DEFINE_string('mtz_path', '', 'Optional path to MTZ file for neutron refinement.')
flags.DEFINE_string('output_path', 'neutron_refined_output.cif', 'Path to save the final mmCIF.')

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
    featurised_examples = featurisation.featurise_input(
        fold_input=fold_input, buckets=[256, 512, 1024, 2048, 4096], ccd=ccd, verbose=False
    )
   
    batch_dict = featurised_examples[0]
    batch_obj_raw = feat_batch.Batch.from_data_dict(batch_dict)
    flat_layout = batch_obj_raw.convert_model_output.flat_output_layout
    token_layout = batch_obj_raw.convert_model_output.token_atoms_layout
    batch = jax.tree.map(jnp.asarray, utils.remove_invalidly_typed_feats(batch_dict))
    batch_obj = feat_batch.Batch.from_data_dict(batch)
    
    gather_info = atom_layout.compute_gather_idxs(
        source_layout=token_layout, 
        target_layout=flat_layout
    )
    gather_idxs = jnp.array(gather_info.gather_idxs) 

    logging.info("Loading AF3 Weights...")
    device = jax.local_devices(backend='gpu')[FLAGS.gpu_device]
    model_runner = ModelRunner(config=make_model_config(), device=device, model_dir=model_dir)
    
    logging.info("Executing AF3 Trunk (Pairformer)...")
    rng_key = jax.random.PRNGKey(42)
    trunk_key, noise_key = jax.random.split(rng_key)
    
    embeddings = model_runner.get_conditionings(trunk_key, batch)
    
    mask_shape = batch['pred_dense_atom_mask'].shape
    noise_levels = diffusion_head.noise_schedule(jnp.linspace(0, 1, 21))
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
    
    # Extract the diffusion config
    diff_config = model_runner._model_config.heads.diffusion.eval
   
    final_coords, final_chis, final_waters = run_neutron_guided_diffusion(
        vf_step_fn=model_runner.evaluate_vector_field,
        batch=batch,
        embeddings=embeddings,
        initial_noise=initial_noise,
        gather_idxs=gather_idxs,
        rotor_table=rotor_table,
        mapping=mapping,
        water_mapping=water_mapping,
        sfc_instance=sfc_instance,
        n_steps=200,
        diff_config=diff_config
    )
     
    logging.info("Assembling final atomic coordinates...")
    final_x_full = generate_final_oracle_coords(
        final_coords, final_chis, final_waters, gather_idxs, 
        rotor_table, mapping, water_mapping
    )

    oracle_atoms.coord = np.array(final_x_full)
    
    output_cif_path = pathlib.Path(FLAGS.output_path)
    output_cif_path.parent.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Writing refined structure to {output_cif_path}...")
    cif_file = pdbx.CIFFile()
    pdbx.set_structure(cif_file, oracle_atoms, data_block="neutron_refined")
    cif_file.write(output_cif_path)
    logging.info("Done.")

if __name__ == "__main__":
    app.run(main)
