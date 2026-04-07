import io
import logging
import numpy as np
import jax.numpy as jnp

import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
import hydride

from alphafold3.model.atom_layout import atom_layout

def build_decoupled_topology_from_struct(cleaned_struc, ccd, fold_input):
    """
    Builds the Hydride Oracle using the pre-tokenized cleaned structure and 
    templates, establishing the mappings before the neural network even runs.
    """
    logging.info("Building Hydride Crystallographic Oracle from template...")
    
    # 1. Get standard AF3 layout for mapping
    residues = atom_layout.residues_from_structure(cleaned_struc)
    flat_layout = atom_layout.make_flat_atom_layout(
        residues, ccd, with_hydrogens=True, skip_unk_residues=True
    )
    
    # 2. Parse Template for Oracle Base
    template_cif_string = None
    for chain in fold_input.chains:
        if hasattr(chain, 'templates') and chain.templates:
            for t in chain.templates:
                if t.mmcif: template_cif_string = t.mmcif; break
        if template_cif_string: break

    cif_file = pdbx.CIFFile.read(io.StringIO(template_cif_string))
    oracle_atoms = pdbx.get_structure(cif_file, model=1)
    oracle_atoms = oracle_atoms[oracle_atoms.element != "H"] 
    
    # 3. Build Full Hydrogen Topology
    oracle_atoms.bonds = struc.connect_via_residue_names(oracle_atoms)
    if "charge" not in oracle_atoms.get_annotation_categories():
        oracle_atoms.add_annotation("charge", dtype=int)
        oracle_atoms.charge[:] = 0 
        
    oracle_atoms, _ = hydride.add_hydrogen(oracle_atoms)
    oracle_atoms.coord = hydride.relax_hydrogen(oracle_atoms)
    num_oracle_atoms = oracle_atoms.array_length()

    # 4. Map AF3 flat_layout to fast lookup dictionary
    af3_lookup = {}
    for i in range(flat_layout.shape[0]):
        key = (flat_layout.chain_id[i], flat_layout.res_id[i], flat_layout.atom_name[i])
        af3_lookup[key] = i

    # 5. Extract the Z-Matrix
    bonds, _ = oracle_atoms.bonds.get_all_bonds()

    rotor_table = {
        "target_idx": [], "parent_idx": [], 
        "grandparent_idx": [], "greatgrand_idx": [],
        "ideal_r": [], "ideal_theta": []
    }
    
    oracle_heavy_indices = []
    af3_source_indices = []

    for i in range(num_oracle_atoms):
        is_hydrogen = (oracle_atoms.element[i] == "H")
        h_key = (oracle_atoms.chain_id[i], oracle_atoms.res_id[i], oracle_atoms.atom_name[i])
        
        if h_key in af3_lookup:
            oracle_heavy_indices.append(i)
            af3_source_indices.append(af3_lookup[h_key])
            continue
            
        if not is_hydrogen: continue
            
        p_indices = bonds[i][bonds[i] != -1]
        if len(p_indices) == 0: continue
        p_i = p_indices[0]
        
        gp_indices = bonds[p_i][bonds[p_i] != -1]
        gp_indices = gp_indices[gp_indices != i]
        if len(gp_indices) == 0: continue 
        gp_i = gp_indices[0]
        
        ggp_indices = bonds[gp_i][bonds[gp_i] != -1]
        ggp_indices = ggp_indices[ggp_indices != p_i]
        if len(ggp_indices) == 0: continue
        ggp_i = ggp_indices[0]
        
        p_key = (oracle_atoms.chain_id[p_i], oracle_atoms.res_id[p_i], oracle_atoms.atom_name[p_i])
        gp_key = (oracle_atoms.chain_id[gp_i], oracle_atoms.res_id[gp_i], oracle_atoms.atom_name[gp_i])
        ggp_key = (oracle_atoms.chain_id[ggp_i], oracle_atoms.res_id[ggp_i], oracle_atoms.atom_name[ggp_i])
        
        if not (p_key in af3_lookup and gp_key in af3_lookup and ggp_key in af3_lookup):
            continue
            
        c_h, c_p, c_gp = oracle_atoms.coord[i], oracle_atoms.coord[p_i], oracle_atoms.coord[gp_i]
        v_hp, v_gpp = c_h - c_p, c_gp - c_p
        
        r_ideal = np.linalg.norm(v_hp)
        cos_theta = np.dot(v_hp, v_gpp) / (r_ideal * np.linalg.norm(v_gpp) + 1e-8)
        theta_ideal = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
        
        rotor_table["target_idx"].append(i) 
        rotor_table["parent_idx"].append(af3_lookup[p_key]) 
        rotor_table["grandparent_idx"].append(af3_lookup[gp_key])
        rotor_table["greatgrand_idx"].append(af3_lookup[ggp_key])
        rotor_table["ideal_r"].append(r_ideal)
        rotor_table["ideal_theta"].append(theta_ideal)

    logging.info(f"Pre-mapped {len(oracle_heavy_indices)} AF3 targets to Oracle space.")
    logging.info(f"Delegated {len(rotor_table['target_idx'])} protons to JAX NeRF.")

    return (
        {k: jnp.array(v, dtype=jnp.float32 if "ideal" in k else jnp.int32) for k, v in rotor_table.items()},
        {
            "oracle_heavy": jnp.array(oracle_heavy_indices, dtype=jnp.int32),
            "af3_source": jnp.array(af3_source_indices, dtype=jnp.int32),
            "num_oracle_atoms": num_oracle_atoms
        }
    )
