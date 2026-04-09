import logging
from typing import Any, Dict, Tuple

import numpy as np
import jax.numpy as jnp

import biotite.structure as struc
import hydride


def build_decoupled_topology(
    flat_layout: Any, x_af3_flat_baseline: jnp.ndarray
) -> Tuple[
    Dict[str, jnp.ndarray], Dict[str, Any], Dict[str, jnp.ndarray], struc.AtomArray
]:
    """
    Constructs a full-complex topology from an AlphaFold 3 baseline prediction.

    This function bridges the gap between AF3's heavy-atom-centric output and a
    complete molecular model by adding hydrogens via Hydride and mapping them
    into a decoupled coordinate system for differentiable refinement.

    Parameters
    ----------
    flat_layout : Any
        The AlphaFold 3 flat output layout containing atom/residue metadata.
    x_af3_flat_baseline : jnp.ndarray
        The baseline coordinate prediction (x_0) from the AF3 diffusion head.

    Returns
    -------
    rotor_table : Dict[str, jnp.ndarray]
        Topology and ideal geometries for protons placed via JAX NeRF.
    mapping : Dict[str, Any]
        Index mappings between the AF3 token layout and the full "Oracle" complex.
    water_mapping : Dict[str, jnp.ndarray]
        Index mappings for SO(3) orientable water molecules.
    oracle_atoms : biotite.structure.AtomArray
        The complete molecular structure including added hydrogens.
    """
    logging.info("Building full-complex Hydride Oracle from AF3 baseline prediction...")

    num_atoms = flat_layout.shape[0]
    atoms = struc.AtomArray(num_atoms)
    atoms.coord = np.array(x_af3_flat_baseline)

    # Enforce standard datatypes
    atoms.atom_name = np.array(flat_layout.atom_name, dtype="U")
    atoms.res_name = np.array(flat_layout.res_name, dtype="U")
    atoms.chain_id = np.array(flat_layout.chain_id, dtype="U")
    atoms.res_id = np.array(flat_layout.res_id, dtype=int)

    # Force the sequence into standard uppercase strings, then cast to U2 NumPy array
    elements = [str(e).strip().upper() for e in flat_layout.atom_element]
    atoms.element = np.array(elements, dtype="U2")

    # 2. Start Oracle with Heavy Atoms only (The "Riding Proton" paradigm)
    heavy_mask = (atoms.element != "H") & (atoms.element != "D")
    oracle_atoms = atoms[heavy_mask]

    # 3. Add hydrogens dynamically to the entire complex
    oracle_atoms.bonds = struc.connect_via_residue_names(oracle_atoms)
    if "charge" not in oracle_atoms.get_annotation_categories():
        oracle_atoms.add_annotation("charge", dtype=int)
        oracle_atoms.charge[:] = 0

    oracle_atoms, _ = hydride.add_hydrogen(oracle_atoms)
    oracle_atoms.coord = hydride.relax_hydrogen(oracle_atoms)
    num_oracle_atoms = oracle_atoms.array_length()

    # The flat_layout matches the exact atom order DeepMind expects
    af3_lookup = {}

    for i in range(flat_layout.shape[0]):
        key = (flat_layout.chain_id[i], flat_layout.res_id[i], flat_layout.atom_name[i])
        af3_lookup[key] = i

    bonds, _ = oracle_atoms.bonds.get_all_bonds()

    water_o_source, water_h1_target, water_h2_target = [], [], []
    for i in range(num_oracle_atoms):
        is_water = oracle_atoms.res_name[i] in ["HOH", "WAT", "H2O"]
        if is_water and oracle_atoms.element[i] == "O":
            o_key = (
                oracle_atoms.chain_id[i],
                oracle_atoms.res_id[i],
                oracle_atoms.atom_name[i],
            )
            if o_key in af3_lookup:
                bonded_idx = bonds[i][bonds[i] != -1]
                h_idx = [idx for idx in bonded_idx if oracle_atoms.element[idx] == "H"]
                if len(h_idx) == 2:
                    water_o_source.append(af3_lookup[o_key])
                    water_h1_target.append(h_idx[0])
                    water_h2_target.append(h_idx[1])

    rotor_table = {
        "target_idx": [],
        "parent_idx": [],
        "grandparent_idx": [],
        "greatgrand_idx": [],
        "ideal_r": [],
        "ideal_theta": [],
        "initial_chi": [],
    }
    oracle_heavy_indices, af3_source_indices = [], []

    for i in range(num_oracle_atoms):
        is_hydrogen = oracle_atoms.element[i] == "H"
        h_key = (
            oracle_atoms.chain_id[i],
            oracle_atoms.res_id[i],
            oracle_atoms.atom_name[i],
        )

        if not is_hydrogen:
            if h_key in af3_lookup:
                oracle_heavy_indices.append(i)
                af3_source_indices.append(af3_lookup[h_key])
            continue

        if i in water_h1_target or i in water_h2_target:
            continue

        p_indices = bonds[i][bonds[i] != -1]
        if len(p_indices) == 0:
            continue
        p_i = p_indices[0]

        gp_indices = bonds[p_i][bonds[p_i] != -1]
        gp_indices = gp_indices[gp_indices != i]
        if len(gp_indices) == 0:
            continue
        gp_i = gp_indices[0]

        ggp_indices = bonds[gp_i][bonds[gp_i] != -1]
        ggp_indices = ggp_indices[ggp_indices != p_i]
        if len(ggp_indices) == 0:
            continue
        ggp_i = ggp_indices[0]

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

        c_h, c_p, c_gp = (
            oracle_atoms.coord[i],
            oracle_atoms.coord[p_i],
            oracle_atoms.coord[gp_i],
        )
        c_ggp = oracle_atoms.coord[
            ggp_i
        ]  # We need the great-grandparent coord to define the local X-axis
        v_hp, v_gpp = c_h - c_p, c_gp - c_p

        r_ideal = np.linalg.norm(v_hp)
        cos_theta = np.dot(v_hp, v_gpp) / (r_ideal * np.linalg.norm(v_gpp) + 1e-8)
        theta_ideal = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

        # Extract the hydride-optimized Chi Angle
        # Match the generalized_nerf_layer axes exactly
        v1 = c_p - c_gp
        v2 = c_gp - c_ggp

        z_axis = v1 / (np.linalg.norm(v1) + 1e-8)
        x_axis_raw = np.cross(v2, z_axis)
        x_axis = x_axis_raw / (np.linalg.norm(x_axis_raw) + 1e-8)
        y_axis = np.cross(z_axis, x_axis)

        # Project the Hydrogen vector onto the local X and Y axes
        x_proj = np.dot(v_hp, x_axis)
        y_proj = np.dot(v_hp, y_axis)
        chi_initial = np.arctan2(y_proj, x_proj)

        rotor_table["target_idx"].append(i)
        rotor_table["parent_idx"].append(af3_lookup[p_key])
        rotor_table["grandparent_idx"].append(af3_lookup[gp_key])
        rotor_table["greatgrand_idx"].append(af3_lookup[ggp_key])
        rotor_table["ideal_r"].append(r_ideal)
        rotor_table["ideal_theta"].append(theta_ideal)
        rotor_table["initial_chi"].append(chi_initial)

    logging.info(
        f"Mapped {len(oracle_heavy_indices)} AF3 natively tracked heavy atoms."
    )
    logging.info(f"Mapped {len(water_o_source)} SO(3) orientable water molecules.")
    logging.info(
        f"Delegated {len(rotor_table['target_idx'])} dynamically generated protons to JAX NeRF."
    )

    return (
        {
            k: jnp.array(
                v, dtype=jnp.float32 if ("ideal" in k or "chi" in k) else jnp.int32
            )
            for k, v in rotor_table.items()
        },
        {
            "oracle_heavy": jnp.array(oracle_heavy_indices, dtype=jnp.int32),
            "af3_source": jnp.array(af3_source_indices, dtype=jnp.int32),
            "num_oracle_atoms": num_oracle_atoms,
        },
        {
            "oxygen_source": jnp.array(water_o_source, dtype=jnp.int32),
            "h1_target": jnp.array(water_h1_target, dtype=jnp.int32),
            "h2_target": jnp.array(water_h2_target, dtype=jnp.int32),
        },
        oracle_atoms,
    )
