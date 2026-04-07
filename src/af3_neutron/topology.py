import dataclasses
import logging
import numpy as np
import jax.numpy as jnp
from rdkit import Chem
from rdkit.Chem import rdMolTransforms

from alphafold3.constants import chemical_components
from alphafold3.model.atom_layout import atom_layout

def inject_superset_topology_and_get_rotors(cleaned_struc, ccd):
    """
    Hacks AF3's internal layout to prevent it from deleting titratable protons,
    builds the flat layout, and extracts the kinematic Z-matrix.
    """
    logging.info("Injecting Super-Set Topology...")
    residues = atom_layout.residues_from_structure(cleaned_struc)
    
    # 1. Override the deprotonation rules with empty sets to keep ALL hydrogens
    num_res = residues.shape[0]
    super_set_deprotonation = np.array([set() for _ in range(num_res)], dtype=object)
    residues_super_set = dataclasses.replace(residues, deprotonation=super_set_deprotonation)
    
    # 2. Generate the flat layout WITH hydrogens
    flat_layout = atom_layout.make_flat_atom_layout(
        residues_super_set, ccd, with_hydrogens=True, skip_unk_residues=True
    )
    
    logging.info("Building Kinematic Rotor Table...")
    # NOTE: In a full implementation, you would generate an RDKit Mol from the 
    # fully hydrogenated template (e.g., using hydride to get 3D coordinates) 
    # to measure the ideal geometry. 
    # dummy_rdkit_mol = Chem.MolFromSmiles("C(CO)N") 
    # rotor_table = _build_kinematic_rotor_table_from_mol(dummy_rdkit_mol)
    
    # Placeholder returning an empty valid JAX dict to allow compilation
    rotor_table = {
        "target_idx": jnp.array([], dtype=jnp.int32),
        "parent_idx": jnp.array([], dtype=jnp.int32),
        "grandparent_idx": jnp.array([], dtype=jnp.int32),
        "greatgrand_idx": jnp.array([], dtype=jnp.int32),
        "ideal_r": jnp.array([], dtype=jnp.float32),
        "ideal_theta": jnp.array([], dtype=jnp.float32)
    }
    
    return flat_layout, rotor_table

def _build_kinematic_rotor_table_from_mol(mol_3d):
    """
    Uses RDKit to traverse the 3D target structure and build the Z-matrix
    for all rotatable hydrogens.
    """
    conf = mol_3d.GetConformer()
    rotor_table = {
        "target_idx": [], "parent_idx": [], 
        "grandparent_idx": [], "greatgrand_idx": [],
        "ideal_r": [], "ideal_theta": []
    }
    
    polar_h_pattern = Chem.MolFromSmarts("[!#6]-[H]")
    polar_h_matches = mol_3d.GetSubstructMatches(polar_h_pattern)
    target_h_indices = {match[1] for match in polar_h_matches}

    for atom in mol_3d.GetAtoms():
        target_idx = atom.GetIdx()
        if target_idx not in target_h_indices:
            continue
            
        neighbors = atom.GetNeighbors()
        if not neighbors: continue
        parent = neighbors[0]
        
        gp_candidates = [n for n in parent.GetNeighbors() if n.GetIdx() != target_idx]
        if not gp_candidates: continue 
        grandparent = gp_candidates[0]
        
        ggp_candidates = [n for n in grandparent.GetNeighbors() if n.GetIdx() != parent.GetIdx()]
        if not ggp_candidates: continue
        greatgrand = ggp_candidates[0]
        
        r_ideal = rdMolTransforms.GetBondLength(conf, target_idx, parent.GetIdx())
        theta_ideal = rdMolTransforms.GetAngleDeg(conf, target_idx, parent.GetIdx(), grandparent.GetIdx())
        
        rotor_table["target_idx"].append(target_idx)
        rotor_table["parent_idx"].append(parent.GetIdx())
        rotor_table["grandparent_idx"].append(grandparent.GetIdx())
        rotor_table["greatgrand_idx"].append(greatgrand.GetIdx())
        rotor_table["ideal_r"].append(r_ideal)
        rotor_table["ideal_theta"].append(theta_ideal)

    return {k: jnp.array(v) for k, v in rotor_table.items()}
