import functools
import logging
import sys
from typing import Any, Optional, Dict

import hydride
import biotite.structure as struc
import jax
import jax.numpy as jnp
import numpy as np
from SFC_Jax.Fmodel import SFcalculator as SFC
from rdkit import Chem

from .runner import HostRunner
from .types import (
    HostEmbeddings,
    Oracle,
    OracleMapping,
    Conformations
)

def build_custom_bond_dict(atoms: struc.AtomArray, ligand_smiles_dict: Dict[str, str]) -> Dict[str, Dict[tuple, Any]]:
    from biotite.structure import BondType
    from biotite.structure.info import bonds_in_residue
    
    custom_bond_dict = {}
    
    unique_res_names = np.unique(atoms.res_name)
    for res_name in unique_res_names:
        standard_bonds = bonds_in_residue(res_name)
        if standard_bonds is not None:
            custom_bond_dict[res_name] = dict(standard_bonds)
        else:
            custom_bond_dict[res_name] = {}

    for chain_id, smiles in ligand_smiles_dict.items():
        chain_mask = (atoms.chain_id == chain_id)
        res_names = np.unique(atoms.res_name[chain_mask])
        if len(res_names) == 0:
            continue
        res_name = res_names[0]
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        Chem.Kekulize(mol, clearAromaticFlags=True)
        
        element_counters = {}
        rdkit_atom_names = []
        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol().upper()
            element_counters[symbol] = element_counters.get(symbol, 0) + 1
            atom_name = f"{symbol}{element_counters[symbol]}"
            rdkit_atom_names.append(atom_name)
            
        res_bonds = custom_bond_dict.get(res_name, {})
        for bond in mol.GetBonds():
            idx1 = bond.GetBeginAtomIdx()
            idx2 = bond.GetEndAtomIdx()
            
            name1 = rdkit_atom_names[idx1]
            name2 = rdkit_atom_names[idx2]
            
            rdkit_btype = bond.GetBondType()
            if rdkit_btype == Chem.BondType.SINGLE:
                btype = int(BondType.SINGLE)
            elif rdkit_btype == Chem.BondType.DOUBLE:
                btype = int(BondType.DOUBLE)
            elif rdkit_btype == Chem.BondType.TRIPLE:
                btype = int(BondType.TRIPLE)
            else:
                btype = int(BondType.SINGLE)
                
            res_bonds[(name1, name2)] = btype
            
        custom_bond_dict[res_name] = res_bonds
        
    return custom_bond_dict


def _build_oracle_from_baseline_af3_prediction(
    flat_layout: Any, x_af3_flat_baseline: jnp.ndarray, ligand_smiles_dict: Dict[str, str], ph: float = 7.4
) -> Oracle:
    logging.info(
        f"Building full-complex Hydride Oracle from Host baseline prediction at pH {ph}..."
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
    
    custom_bonds = build_custom_bond_dict(oracle_atoms, ligand_smiles_dict)
    
    oracle_atoms.bonds = struc.connect_via_residue_names(oracle_atoms, inter_residue=True, custom_bond_dict=custom_bonds)

    charges_array = hydride.estimate_amino_acid_charges(oracle_atoms, ph)
    for chain_id, smiles in ligand_smiles_dict.items():
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        try:
            Chem.ComputeGasteigerCharges(mol)
            rdkit_charges = [float(a.GetProp("_GasteigerCharge")) for a in mol.GetAtoms()]
            rdkit_charges = [0.0 if np.isnan(c) or np.isinf(c) else c for c in rdkit_charges]
        except Exception:
            rdkit_charges = [0.0] * mol.GetNumAtoms()
            
        element_counters = {}
        for idx, atom in enumerate(mol.GetAtoms()):
            symbol = atom.GetSymbol().upper()
            element_counters[symbol] = element_counters.get(symbol, 0) + 1
            atom_name = f"{symbol}{element_counters[symbol]}"
            
            matching_indices = np.where((oracle_atoms.chain_id == chain_id) & (oracle_atoms.atom_name == atom_name))[0]
            for g_idx in matching_indices:
                charges_array[g_idx] = rdkit_charges[idx]
                
    oracle_atoms.set_annotation("charge", charges_array)

    oracle_atoms, _ = hydride.add_hydrogen(oracle_atoms)
    oracle_atoms.coord = hydride.relax_hydrogen(oracle_atoms)
    num_oracle_atoms = oracle_atoms.array_length()

    is_heavy = (oracle_atoms.element != "H") & (oracle_atoms.element != "D")
    parent_map = np.arange(num_oracle_atoms)
    bonds = oracle_atoms.bonds.as_array()

    for i in range(bonds.shape[0]):
        a1, a2, _ = bonds[i]
        # Map H to Heavy
        if not is_heavy[a1] and is_heavy[a2]:
            parent_map[a1] = a2
        elif not is_heavy[a2] and is_heavy[a1]:
            parent_map[a2] = a1

    af3_lookup = {
        (flat_layout.chain_id[i], flat_layout.res_id[i], flat_layout.atom_name[i]): i
        for i in range(num_atoms)
    }

    oracle_heavy_indices, af3_source_indices = [], []
    for i in range(num_oracle_atoms):
        h_key = (
            oracle_atoms.chain_id[i],
            oracle_atoms.res_id[i],
            oracle_atoms.atom_name[i],
        )
        if oracle_atoms.element[i] != "H" and h_key in af3_lookup:
            oracle_heavy_indices.append(i)
            af3_source_indices.append(af3_lookup[h_key])

    return Oracle(
        mapping=OracleMapping(
            num_atoms=num_oracle_atoms,
            heavy_indices=jnp.array(oracle_heavy_indices, dtype=jnp.int32),
            source_indices=jnp.array(af3_source_indices, dtype=jnp.int32),
            initial_coordinates=jnp.array(oracle_atoms.coord, dtype=jnp.float32),
            hydrogen_to_heavy_map=jnp.array(parent_map, dtype=jnp.int32)
        ),
        atoms=oracle_atoms,
    )


def _hijack_diffusion_with_custom_loss(
    model_runner: HostRunner,
    batch_dict: dict,
    embeddings: HostEmbeddings,
    gather_idxs: jnp.ndarray,
    oracle: Oracle,
    sfc_instance: Optional[SFC] = None,
    sample_key: Optional[jnp.ndarray] = None,
    prox_steps: int = 3,
    prox_lr: float = 5e-3,
    eta_init: float = 1e-2,
    sfc_weight: float = 500.0,
) -> Conformations:
    oracle_mapping = oracle.mapping
    params = hydride.get_relaxation_params(oracle.atoms)

    @functools.partial(jax.jit, inline=False)
    def proximal_operator_fn(x_0_real: jnp.ndarray, t_hat: jnp.ndarray) -> jnp.ndarray:
        x_0_flat = x_0_real.reshape(-1, 3)
        current_eta = eta_init * (t_hat ** 2)
        
        x_af3_flat = x_0_flat[gather_idxs]
        x_0_heavy_mapped = x_af3_flat[oracle_mapping.source_indices]

        def step_body(i, r_current):
            # Extract pLDDT from the oracle's raw B-factor column
            plddt_heavy = jnp.array(oracle.atoms.b_factor[oracle_mapping.heavy_indices], dtype=jnp.float32)

            def local_loss_fn(R_heavy):
                # 1. JAX-Differentiable Kabsch Alignment
                R_ref = oracle_mapping.initial_coordinates[oracle_mapping.heavy_indices]
                
                avg_heavy = jnp.mean(R_heavy, axis=0)
                avg_ref = jnp.mean(R_ref, axis=0)
                
                p = R_heavy - avg_heavy
                q = R_ref - avg_ref
                
                H = jnp.einsum("ni,nj->ij", p, q)
                U, _, Vt = jnp.linalg.svd(H, full_matrices=False)
                d = jnp.sign(jnp.linalg.det(U) * jnp.linalg.det(Vt))
                R_mat = U @ jnp.diag(jnp.array([1.0, 1.0, d])) @ Vt
                
                R_aligned = (R_heavy - avg_heavy) @ R_mat + avg_ref
                
                # 2. Evaluate Structure Factors on the density-aligned coordinates
                X_base = oracle_mapping.initial_coordinates.at[oracle_mapping.heavy_indices].set(R_aligned)
                X_relaxed, _, _ = hydride.relax_hydrogen_jit(X_base, *params, iterations=5)
                
                e_exp = 0.0
                if sfc_instance is not None:
                    e_exp, _ = sfc_instance.compute_loss(X_relaxed)

                # 3. BAYESIAN GAUSSIAN NLL RESTRAINT
                sq_diff = jnp.sum((R_heavy - x_0_heavy_mapped) ** 2, axis=-1)
                
                # Configurable exponential scaling parameters
                variance_base = 0.1   # Minimum variance floor for perfect predictions
                variance_scale = 0.05 # Scaling multiplier
                variance_decay = 10.0 # How fast the variance explodes as pLDDT drops
                
                # Map pLDDT to Positional Variance (sigma^2)
                # Low pLDDT inflates the denominator, dynamically dropping the penalty to near-zero
                sigma_sq = variance_base + variance_scale * jnp.exp((100.0 - plddt_heavy) / variance_decay)
                
                # NLL = (x - mu)^2 / (2 * sigma^2)
                # Current_eta scales the overall influence based on the diffusion timestep.
                bayesian_restraint = (0.5 / current_eta) * jnp.sum(sq_diff / sigma_sq)
                
                return (sfc_weight * e_exp) + bayesian_restraint  

            # 1. Calculate raw gradients
            grads = jax.grad(local_loss_fn)(r_current)
            
            # 2. Scale-Invariant Gradients
            N_atoms = r_current.shape[0]
            scaled_grads = grads * N_atoms
            
            # 3. Natural Learning Rate
            lr = sfc_weight * jnp.sqrt(t_hat) * 0.01
            raw_step = scaled_grads * lr
            
            # 4. ATOM-WISE SPEED LIMIT
            max_allowed_step = 0.05
            
            # Calculate the norm for EACH atom independently (keepdims=True is critical for broadcasting)
            atom_step_norms = jnp.linalg.norm(raw_step, axis=-1, keepdims=True) + 1e-8
            
            # If an atom wants to move > 0.05, slow IT down. If it wants to move < 0.05, leave IT alone.
            clip_factor = jnp.where(atom_step_norms > max_allowed_step, 
                                    max_allowed_step / atom_step_norms, 
                                    1.0)
            
            # 5. Apply the safe, organic step
            r_next = r_current - (raw_step * clip_factor)
            
            return r_next

        R_current = x_0_heavy_mapped
        R_optimized = jax.lax.fori_loop(0, prox_steps, step_body, R_current)
        
        # Log final metrics once per timestep outside the gradient loop
        if sfc_instance is not None:
            # We must re-align R_optimized one last time purely for accurate terminal logging
            R_ref = oracle_mapping.initial_coordinates[oracle_mapping.heavy_indices]
            avg_opt = jnp.mean(R_optimized, axis=0)
            avg_ref = jnp.mean(R_ref, axis=0)
            p = R_optimized - avg_opt
            q = R_ref - avg_ref
            H = jnp.einsum("ni,nj->ij", p, q)
            U, _, Vt = jnp.linalg.svd(H, full_matrices=False)
            R_mat = U @ jnp.diag(jnp.array([1.0, 1.0, jnp.sign(jnp.linalg.det(U) * jnp.linalg.det(Vt))])) @ Vt
            R_log_aligned = (R_optimized - avg_opt) @ R_mat + avg_ref

            X_final = oracle_mapping.initial_coordinates.at[oracle_mapping.heavy_indices].set(R_log_aligned)
            X_rel, _, _ = hydride.relax_hydrogen_jit(X_final, *params, iterations=5)
            
            _, (rw, rf) = sfc_instance.compute_loss(X_rel)
            jax.debug.print("t_hat: {t:.3f} | R_work: {rw:.4f} | R_free: {rf:.4f}", t=t_hat, rw=rw, rf=rf)

        x_af3_updated = x_af3_flat.at[oracle_mapping.source_indices].set(R_optimized)
        return x_0_flat.at[gather_idxs].set(x_af3_updated).reshape(x_0_real.shape)

    rng_key = jax.random.PRNGKey(0) if sample_key is None else sample_key
    atom_positions = model_runner.sample_guided_diffusion(
        rng_key,
        batch_dict,
        embeddings,
        rng_key,
        proximal_operator_fn
    )

    return Conformations(atom_positions=atom_positions)

def _assemble_coordinates_from_conformation(
    atom_positions: jnp.ndarray,
    gather_idxs: jnp.ndarray,
    oracle_mapping: OracleMapping,
    oracle_atoms: Any,
    sfc_instance: Optional[SFC] = None,
    sfc_weight: float = 1000.0
) -> jnp.ndarray:
    params = hydride.get_relaxation_params(oracle_atoms)

    x_af3_flat = atom_positions.reshape((-1, 3))[gather_idxs]

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
    
    X_final = oracle_mapping.initial_coordinates.at[oracle_mapping.heavy_indices].set(x_af3_aligned[oracle_mapping.source_indices])
    
    if sfc_instance is None:
        X_relaxed, _, _ = hydride.relax_hydrogen_jit(X_final, *params, iterations=200)
        return X_relaxed

    pairs, elec_param, eps, r_6, r_12 = params[3], params[4], params[5], params[6], params[7]
    box, box_inv = params[9], params[10]
    reduction_indices, reduction_signs, reduction_pair_map = params[11], params[12], params[13]

    @jax.jit
    def final_refinement_step(carry, i):
        R_heavy, m, v = carry
        
        def loss_fn(R):
            X_complex = oracle_mapping.initial_coordinates.at[oracle_mapping.heavy_indices].set(R)
            X_rel, _, _ = hydride.relax_hydrogen_jit(X_complex, *params, iterations=10)
            
            e_phys = hydride.relax.compute_energy(X_rel, pairs, elec_param, eps, r_6, r_12, box, box_inv, reduction_indices, reduction_signs, reduction_pair_map)
            e_exp, _ = sfc_instance.compute_loss(X_rel)
            
            return (sfc_weight * e_exp) + (0.05 * e_phys)
            
        grads = jax.grad(loss_fn)(R_heavy)
        grads = jnp.clip(grads, -1.0, 1.0)
        
        m_next = 0.9 * m + 0.1 * grads
        v_next = 0.999 * v + 0.001 * (grads ** 2)
        
        m_hat = m_next / (1.0 - 0.9 ** (i + 1))
        v_hat = v_next / (1.0 - 0.999 ** (i + 1))
        
        R_next = R_heavy - 5e-3 * m_hat / (jnp.sqrt(v_hat) + 1e-8)
        return (R_next, m_next, v_next), None

    R_init = X_final[oracle_mapping.heavy_indices]
    (R_opt, _, _), _ = jax.lax.scan(
        final_refinement_step, 
        (R_init, jnp.zeros_like(R_init), jnp.zeros_like(R_init)), 
        jnp.arange(150)
    )
    
    X_refined = oracle_mapping.initial_coordinates.at[oracle_mapping.heavy_indices].set(R_opt)
    X_relaxed, _, _ = hydride.relax_hydrogen_jit(X_refined, *params, iterations=200)
    
    _, (rw, rf) = sfc_instance.compute_loss(X_relaxed)
    jax.debug.print("Final Assembly | R_work: {rw:.4f} | R_free: {rf:.4f}", rw=rw, rf=rf)
    
    return X_relaxed


class Hijacker:
    @staticmethod
    def build_oracle(layout, denoised_vector_field_positions, ligand_smiles_dict: Dict[str, str], ph: float = 7.4) -> Oracle:
        return _build_oracle_from_baseline_af3_prediction(layout, denoised_vector_field_positions, ligand_smiles_dict, ph=ph)

    @staticmethod
    def hijack_diffusion(
        runner: HostRunner, 
        batch_dict: dict, 
        embeddings: HostEmbeddings, 
        gather_idxs: jnp.ndarray, 
        oracle: Oracle, 
        sfc: Optional[SFC] = None, 
        key: Optional[jnp.ndarray] = None,
        sfc_weight: float = 1000.0
    ) -> jnp.ndarray:
        return _hijack_diffusion_with_custom_loss(runner, batch_dict, embeddings, gather_idxs, oracle, sfc, key, sfc_weight=sfc_weight)

    @staticmethod
    def assemble_coordinates(atom_positions: jnp.ndarray, gather_idxs: jnp.ndarray, oracle: Oracle, sfc: Optional[SFC] = None, sfc_weight: float = 1000.0) -> np.ndarray:
        complex_coords = _assemble_coordinates_from_conformation(atom_positions, gather_idxs, oracle.mapping, oracle.atoms, sfc, sfc_weight)
        return np.array(complex_coords)
