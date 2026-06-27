import logging
from typing import Any, Optional

import hydride
import biotite.structure as struc
import jax
import jax.numpy as jnp
import numpy as np
from SFC_Jax.Fmodel import SFcalculator as SFC

from .loss import hijack_physics_loss # expose into this file later
from .runner import HostRunner
from .types import (
    HostEmbeddings,
    RotorTable,
    WaterMapping,
    Oracle,
    OracleMapping,
    Conformation,
    Conformations
)

def _build_oracle_from_baseline_af3_prediction(
    flat_layout: Any, x_af3_flat_baseline: jnp.ndarray
) -> Oracle:
    """Builds a full complex topological oracle from an unguided baseline prediction."""
    logging.info(
        "Building full-complex Hydride Oracle from Host baseline prediction..."
    )

    num_atoms = flat_layout.shape[0]
    atoms = struc.AtomArray(num_atoms)
    atoms.coord = np.array(x_af3_flat_baseline)

    atoms.atom_name = np.array(flat_layout.atom_name, dtype="U")
    atoms.res_name = np.array(flat_layout.res_name, dtype="U")
    atoms.chain_id = np.array(flat_layout.chain_id, dtype="U")
    atoms.res_id = np.array(flat_layout.res_id, dtype=int)
    atoms.element = np.array(
        [str(e).strip().upper() for e in flat_layout.atom_element], dtype="U2"
    )

    oracle_atoms = atoms[(atoms.element != "H") & (atoms.element != "D")]
    oracle_atoms.bonds = struc.connect_via_residue_names(oracle_atoms)
    if "charge" not in oracle_atoms.get_annotation_categories():
        oracle_atoms.add_annotation("charge", dtype=int)
        oracle_atoms.charge[:] = 0

    oracle_atoms, _ = hydride.add_hydrogen(oracle_atoms)
    oracle_atoms.coord = hydride.relax_hydrogen(oracle_atoms)
    num_oracle_atoms = oracle_atoms.array_length()

    af3_lookup = {
        (flat_layout.chain_id[i], flat_layout.res_id[i], flat_layout.atom_name[i]): i
        for i in range(num_atoms)
    }
    bonds, _ = oracle_atoms.bonds.get_all_bonds()

    water_o_source, water_h1_target, water_h2_target = [], [], []
    for i in range(num_oracle_atoms):
        if (
            oracle_atoms.res_name[i] not in ["HOH", "WAT", "H2O"]
            or oracle_atoms.element[i] != "O"
        ):
            continue
        o_key = (
            oracle_atoms.chain_id[i],
            oracle_atoms.res_id[i],
            oracle_atoms.atom_name[i],
        )
        if o_key not in af3_lookup:
            continue
        h_idx = [
            idx for idx in bonds[i][bonds[i] != -1] if oracle_atoms.element[idx] == "H"
        ]
        if len(h_idx) == 2:
            water_o_source.append(af3_lookup[o_key])
            water_h1_target.append(h_idx[0])
            water_h2_target.append(h_idx[1])

    rotor_table = {
        k: []
        for k in [
            "target_idx",
            "parent_idx",
            "grandparent_idx",
            "greatgrand_idx",
            "ideal_r",
            "ideal_theta",
            "initial_chi",
        ]
    }
    oracle_heavy_indices, af3_source_indices = [], []

    for i in range(num_oracle_atoms):
        h_key = (
            oracle_atoms.chain_id[i],
            oracle_atoms.res_id[i],
            oracle_atoms.atom_name[i],
        )
        if oracle_atoms.element[i] != "H":
            if h_key in af3_lookup:
                oracle_heavy_indices.append(i)
                af3_source_indices.append(af3_lookup[h_key])
            continue
        if i in water_h1_target or i in water_h2_target:
            continue

        p_idx = bonds[i][bonds[i] != -1]
        if len(p_idx) == 0:
            continue
        p_i = p_idx[0]

        gp_idx = bonds[p_i][(bonds[p_i] != -1) & (bonds[p_i] != i)]
        if len(gp_idx) == 0:
            continue
        gp_i = gp_idx[0]

        ggp_idx = bonds[gp_i][(bonds[gp_i] != -1) & (bonds[gp_i] != p_i)]
        if len(ggp_idx) == 0:
            continue
        ggp_i = ggp_idx[0]

        p_key = (
            oracle_atoms.chain_id[p_i],
            oracle_atoms.res_id[p_i],
            oracle_atoms.atom_name[p_i],
        )
        gp_key = (
            oracle_atoms.chain_id[gp_i],
            oracle_atoms.res_id[gp_i],
            oracle_atoms.atom_name[gp_i],
        )
        ggp_key = (
            oracle_atoms.chain_id[ggp_i],
            oracle_atoms.res_id[ggp_i],
            oracle_atoms.atom_name[ggp_i],
        )

        if not (p_key in af3_lookup and gp_key in af3_lookup and ggp_key in af3_lookup):
            continue

        c_h, c_p, c_gp, c_ggp = (
            oracle_atoms.coord[i],
            oracle_atoms.coord[p_i],
            oracle_atoms.coord[gp_i],
            oracle_atoms.coord[ggp_i],
        )
        v_hp, v_gpp = c_h - c_p, c_gp - c_p

        r_ideal = np.linalg.norm(v_hp)
        cos_theta = np.dot(v_hp, v_gpp) / (r_ideal * np.linalg.norm(v_gpp) + 1e-8)
        theta_ideal = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

        z_axis = (c_p - c_gp) / (np.linalg.norm(c_p - c_gp) + 1e-8)
        x_axis_raw = np.cross(c_gp - c_ggp, z_axis)
        x_axis = x_axis_raw / (np.linalg.norm(x_axis_raw) + 1e-8)
        y_axis = np.cross(z_axis, x_axis)

        chi_initial = np.arctan2(np.dot(v_hp, y_axis), np.dot(v_hp, x_axis))

        rotor_table["target_idx"].append(i)
        rotor_table["parent_idx"].append(af3_lookup[p_key])
        rotor_table["grandparent_idx"].append(af3_lookup[gp_key])
        rotor_table["greatgrand_idx"].append(af3_lookup[ggp_key])
        rotor_table["ideal_r"].append(r_ideal)
        rotor_table["ideal_theta"].append(theta_ideal)
        rotor_table["initial_chi"].append(chi_initial)

    return Oracle(
        mapping=OracleMapping(
            num_atoms=num_oracle_atoms,
            heavy_indices=jnp.array(oracle_heavy_indices, dtype=jnp.int32),
            source_indices=jnp.array(af3_source_indices, dtype=jnp.int32),
            initial_coordinates=jnp.array(oracle_atoms.coord, dtype=jnp.float32),
            rotor_table = RotorTable(
                **{ k: jnp.array(v, dtype=jnp.float32 if ("ideal" in k or "chi" in k) else jnp.int32)
                    for k, v in rotor_table.items()
                }),
            water_mapping=WaterMapping(
                oxygen_source=jnp.array(water_o_source, dtype=jnp.int32),
                h1_target=jnp.array(water_h1_target, dtype=jnp.int32),
                h2_target=jnp.array(water_h2_target, dtype=jnp.int32),
            )
        ),
        atoms=oracle_atoms,
    )

# TODO(vivek): let user handle desired loss (or default) to pass and directly run guided diffusion
def _hijack_diffusion_with_custom_loss(
    model_runner: HostRunner,
    batch_dict: dict,
    embeddings: HostEmbeddings,
    gather_idxs: jnp.ndarray,
    oracle_mapping: OracleMapping,
    sfc_instance: Optional[SFC] = None,
    sample_key: Optional[jnp.ndarray] = None,
) -> Conformations:
    """Intercepts and steers Host diffusion trajectories."""

    def single_sample_loss_fn(p_single, c_single, w_single):
        return hijack_physics_loss(
            p_single.reshape((-1, 3)),
            c_single,
            w_single,
            gather_idxs,
            oracle_mapping,
            sfc_instance,
        )

    val_and_grad_fn = jax.value_and_grad(single_sample_loss_fn, argnums=(0, 1, 2))
    sample_results = model_runner.sample_guided_diffusion(
        jax.random.PRNGKey(0),
        batch_dict,
        embeddings,
        val_and_grad_fn,
        sample_key,
        oracle_mapping.rotor_table.initial_chi,
        oracle_mapping.water_mapping.oxygen_source.shape[0],
    )
    return Conformations(
        sample_results["atom_positions"],
        sample_results["chi_angles"],
        sample_results["water_rotations"],
    )


def _assemble_coordinates_from_conformation(
    conformation: Conformation,
    gather_idxs: jnp.ndarray,
    oracle_mapping: OracleMapping,
    reference_coords: jnp.ndarray,
) -> jnp.ndarray:
    """Snaps coordinates back into the crystal's global reference frame via Kabsch alignment."""
    x_af3_flat = conformation.atom_positions.reshape((-1, 3))[gather_idxs]

    x_drift_heavy = x_af3_flat[oracle_mapping.source_indices]
    x_ref_heavy = oracle_mapping.initial_coordinates[oracle_mapping.heavy_indices]

    avg_drift = jnp.mean(x_drift_heavy, axis=0)
    avg_ref = jnp.mean(x_ref_heavy, axis=0)

    p = x_drift_heavy - avg_drift
    q = x_ref_heavy - avg_ref

    H = jnp.einsum("ni,nj->ij", p, q)
    U, _, Vt = jnp.linalg.svd(H)
    d = jnp.sign(jnp.linalg.det(U) * jnp.linalg.det(Vt))
    R = U @ jnp.diag(jnp.array([1.0, 1.0, d])) @ Vt

    x_af3_aligned = (x_af3_flat - avg_drift) @ R + avg_ref

    return oracle_mapping.assemble_coordinates(x_af3_aligned, conformation.chi_angles, conformation.water_rotations)

# TODO: need descriptive comments and type annotation
class Hijacker:
    @staticmethod
    def build_oracle(layout, denoised_vector_field_positions) -> Oracle:
        return _build_oracle_from_baseline_af3_prediction(layout, denoised_vector_field_positions)

    @staticmethod
    def hijack_diffusion(runner: HostRunner, batch_dict: dict, embeddings: HostEmbeddings, gather_idxs: jnp.ndarray, oracle_mapping: OracleMapping, sfc: Optional[SFC] = None, key: Optional[jnp.ndarray] = None) -> Conformations:
        return _hijack_diffusion_with_custom_loss(runner, batch_dict, embeddings, gather_idxs, oracle_mapping, sfc, key)

    @staticmethod
    def assemble_coordinates(conformation: Conformation, gather_idxs: jnp.ndarray, oracle: Oracle) -> np.ndarray:
        "Assemble AtomArray compatible coordinates from conformations and oracle"
        oracle_atoms_coord = jnp.array(oracle.atoms.coord, dtype=jnp.float32)
        complex = _assemble_coordinates_from_conformation(conformation, gather_idxs, oracle.mapping, oracle_atoms_coord)
        return np.array(complex)


