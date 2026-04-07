import logging
import pathlib
import jax
import jax.numpy as jnp
from absl import app
from absl import flags

from alphafold3.common import folding_input
from alphafold3.data import featurisation
from alphafold3.constants import chemical_components
from alphafold3.model.pipeline import structure_cleaning
from alphafold3.model.components import utils
from alphafold3.model.network import diffusion_head

from af3_neutron.af3_runner import ModelRunner, make_model_config
from af3_neutron.topology import build_decoupled_topology_from_struct
from af3_neutron.sampler import run_neutron_guided_diffusion, generate_final_oracle_coords

FLAGS = flags.FLAGS
flags.DEFINE_string('mtz_path', '', 'Optional path to MTZ file for neutron refinement.')
flags.DEFINE_string('json_path', 'betalac_tetramer_refinement_input.json', 'Path to JSON.')
flags.DEFINE_string('model_dir', '../af3_model_parameters/', 'Path to weights.')
flags.DEFINE_integer('gpu_device', 0, 'GPU to use.')

def main(argv):
    del argv
    logging.basicConfig(level=logging.INFO)
    
    json_path = pathlib.Path(FLAGS.json_path)
    model_dir = pathlib.Path(FLAGS.model_dir)
    fold_input = next(folding_input.load_fold_inputs_from_path(json_path))
    ccd = chemical_components.Ccd()
    
    struct = fold_input.to_structure(ccd=ccd)
    # Ensure waters are NOT filtered
    cleaned_struc, _ = structure_cleaning.clean_structure(
        struct, ccd=ccd, drop_non_standard_atoms=True, drop_missing_sequence=True,
        filter_clashes=False, filter_crystal_aids=False, filter_waters=False,  # <--- MUST BE FALSE
        filter_hydrogens=False, filter_leaving_atoms=True,
        only_glycan_ligands_for_leaving_atoms=True, covalent_bonds_only=True,
        remove_polymer_polymer_bonds=True, remove_bad_bonds=True, remove_nonsymmetric_bonds=False
    )

    from af3_neutron.sfc_adapter import init_neutron_sfc

    rotor_table, mapping, water_mapping, oracle_atoms = build_decoupled_topology_from_struct(cleaned_struc, ccd, fold_input)

    if FLAGS.mtz_path:
        sfc_instance = init_neutron_sfc(oracle_atoms, FLAGS.mtz_path)
    else:
        logging.info("No MTZ file provided. Running with dummy physics loss.")
        sfc_instance = None 

    logging.info("Running native AF3 featurisation...")
    featurised_examples = featurisation.featurise_input(
        fold_input=fold_input, buckets=[256, 512, 1024, 2048, 4096], ccd=ccd, verbose=False
    )
    batch = jax.tree.map(
        jnp.asarray, utils.remove_invalidly_typed_feats(featurised_examples[0])
    )
    
    logging.info("Loading AF3 Weights...")
    device = jax.local_devices(backend='gpu')[FLAGS.gpu_device]
    model_runner = ModelRunner(config=make_model_config(), device=device, model_dir=model_dir)
    
    logging.info("Executing AF3 Trunk (Pairformer)...")
    rng_key = jax.random.PRNGKey(42)
    trunk_key, noise_key = jax.random.split(rng_key)
    
    # Pass trunk_key FIRST, batch SECOND
    embeddings = model_runner.get_conditionings(trunk_key, batch) 

    # Generate initial flow matching noise
    # (num_tokens, max_atoms_per_token, 3)
    mask_shape = batch['pred_dense_atom_mask'].shape
    noise_levels = diffusion_head.noise_schedule(jnp.linspace(0, 1, 21))
    initial_noise = jax.random.normal(noise_key, mask_shape + (3,)) * noise_levels[0]
  
    # 6. Phase 2: The Hijacked ODE Loop
    final_coords, final_chis, final_waters = run_neutron_guided_diffusion(
        vf_step_fn=model_runner.evaluate_vector_field,
        batch=batch,
        embeddings=embeddings,
        initial_noise=initial_noise,
        rotor_table=rotor_table,
        mapping=mapping,
        water_mapping=water_mapping,
        sfc_instance=sfc_instance,
        n_steps=20
    )
    logging.info("Neutron ODE Refinement Complete!")
    
    # 7. Construct Final Structure
    logging.info("Assembling final atomic coordinates...")
    
    # final_coords has shape (num_tokens, max_atoms_per_token, 3). Flatten it.
    final_af3_flat = final_coords.reshape((-1, 3))
    
    # Generate the full coordinate array including optimized protons
    final_x_full = generate_final_oracle_coords(
        final_af3_flat, final_chis, final_waters, 
        rotor_table, mapping, water_mapping
    )
    
    # Apply coordinates to the Biotite Oracle
    oracle_atoms.coord = np.array(final_x_full)
    
    # 8. Write to mmCIF
    output_cif_path = pathlib.Path(FLAGS.output_path)
    output_cif_path.parent.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Writing refined structure to {output_cif_path}...")
    cif_file = pdbx.CIFFile()
    pdbx.set_structure(cif_file, oracle_atoms, data_block="neutron_refined")
    cif_file.write(output_cif_path)
    
    logging.info("Done.")

if __name__ == "__main__":
    app.run(main)
