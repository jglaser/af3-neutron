import logging
import pathlib
import os
import numpy as np
import jax
import jax.numpy as jnp
import biotite.structure.io.pdbx as pdbx
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
from af3_neutron.sfc_adapter import init_neutron_sfc

from jax.experimental.compilation_cache import compilation_cache as cc
cc.set_cache_dir(os.path.expanduser('./.jax_cache'))

import json

FLAGS = flags.FLAGS
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
    gather_idxs = jnp.array(
        atom_layout.compute_gather_idxs(
            source_layout=batch_obj.convert_model_output.token_atoms_layout,
            target_layout=batch_obj.convert_model_output.flat_output_layout,
        ).gather_idxs
    )

    # runner
    device = jax.local_devices(backend="gpu")[FLAGS.gpu_device]
    runner = HostRunner(
        config=make_model_config(FLAGS.num_recycles, FLAGS.num_diffusion_samples),
        device=device,
        model_dir=pathlib.Path(FLAGS.model_dir),
    )
    # make embeddings and oracle
    embeddings = runner.get_host_embeddings(
        jax.random.split(
            jax.random.PRNGKey(fold_input.rng_seeds[0] if fold_input.rng_seeds else 1)
        )[0],
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

    logging.info("Parsing ligand SMILES definitions from input JSON...")
    with open(FLAGS.json_path, "r") as f:
        input_json_data = json.load(f)

    ligand_smiles_dict = {}
    for seq in input_json_data.get("sequences", []):
        if "ligand" in seq:
            l_id = seq["ligand"]["id"][0]
            l_smiles = seq["ligand"]["smiles"]
            ligand_smiles_dict[l_id] = l_smiles

    oracle = Hijacker.build_oracle(
        batch_obj.convert_model_output.flat_output_layout,
        np.array(positions_denoised.reshape((-1, 3))[gather_idxs]),
        ligand_smiles_dict=ligand_smiles_dict
    )
    
    sfc = init_neutron_sfc(oracle.atoms, FLAGS.mtz_path) if FLAGS.mtz_path else None
    
    # hijack loop using the generalized proximal-based implementation
    conformations = Hijacker.hijack_diffusion(
        runner,
        batch,
        embeddings,
        gather_idxs,
        oracle,
        sfc,
        jax.random.PRNGKey(0),
    )

    logging.info("Assembling final atomic coordinates...")
    oracle.atoms.coord = Hijacker.assemble_coordinates(
        conformations[0],
        gather_idxs,
        oracle,
    )

    # hydride append hydrogen to array end, sort by chain and res for viz of ss
    contiguous_indices = np.lexsort((oracle.atoms.res_id, oracle.atoms.chain_id))
    oracle.atoms = oracle.atoms[contiguous_indices]

    # Define standard residues locally to differentiate ATOM vs HETATM records
    STANDARD_AMINO_ACIDS = {
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
        "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"
    }
    
    # Identify non-standard entities (ligands like BZB, ions, and waters)
    is_hetero = np.array([res.strip().upper() not in STANDARD_AMINO_ACIDS for res in oracle.atoms.res_name])
    oracle.atoms.set_annotation("hetero", is_hetero)

    output_path = pathlib.Path(FLAGS.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cif_file = pdbx.CIFFile()
    pdbx.set_structure(cif_file, oracle.atoms, data_block="neutron_refined")
    cif_file.write(output_path)
    logging.info("Refinement pipeline finished successfully.")

if __name__ == "__main__":
    app.run(main)
