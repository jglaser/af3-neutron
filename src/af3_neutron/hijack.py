import functools
import logging
from typing import Any, Optional

import hydride
import biotite.structure as struc
import jax
import jax.numpy as jnp
import numpy as np
from SFC_Jax.Fmodel import SFcalculator as SFC

from .runner import HostRunner
from .types import (
    HostEmbeddings,
    Oracle,
    OracleMapping,
    Conformations
)

def _build_oracle_from_baseline_af3_prediction(
    flat_layout: Any, x_af3_flat_baseline: jnp.ndarray, ph: float = 7.4
) -> Oracle:
    """Builds a full complex topological oracle from an unguided baseline prediction."""
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
    oracle_atoms.bonds = struc.connect_via_residue_names(oracle_atoms)
    
    # Calculate pKa-dependent protonation states
    oracle_atoms.set_annotation("charge", hydride.estimate_amino_acid_charges(oracle_atoms, ph))

    oracle_atoms, _ = hydride.add_hydrogen(oracle_atoms)
    oracle_atoms.coord = hydride.relax_hydrogen(oracle_atoms)
    num_oracle_atoms = oracle_atoms.array_length()

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
            initial_coordinates=jnp.array(oracle_atoms.coord, dtype=jnp.float32)
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
) -> Conformations:
    """Intercepts and steers Host diffusion trajectories using an Envelope-Theorem optimized proximal operator."""
    oracle_mapping = oracle.mapping
    
    params = hydride.get_relaxation_params(oracle.atoms)
    if params is None:
        raise ValueError("No rotatable bonds found in the structure configuration.")

    (center_indices, axis_indices, is_free_mask, pairs, elec_param, eps, r_6, r_12, atom_to_bond_idx,
        box, box_inv, reduction_indices, reduction_signs, reduction_pair_map) = params

    @functools.partial(jax.jit, inline=False)
    def proximal_operator_fn(x_0_real: jnp.ndarray, t_hat: jnp.ndarray) -> jnp.ndarray:
        x_0_flat = x_0_real.reshape(-1, 3)
        current_eta = eta_init * (t_hat ** 2)
        
        x_af3_flat = x_0_flat[gather_idxs]
        x_0_heavy_mapped = x_af3_flat[oracle_mapping.source_indices]

        def step_body(i, r_val):
            X = oracle_mapping.initial_coordinates
            X = X.at[oracle_mapping.heavy_indices].set(r_val)

            X_relaxed, _, _ = hydride.relax_hydrogen_jit(
                X, center_indices, axis_indices, is_free_mask,
                pairs, elec_param, eps, r_6, r_12, atom_to_bond_idx, box=box, box_inv=box_inv,
                reduction_indices=reduction_indices, reduction_signs=reduction_signs,
                reduction_pair_map=reduction_pair_map,
                iterations=40
            )
            
            X_frozen = jax.lax.stop_gradient(X_relaxed)
            X_final = X_frozen.at[oracle_mapping.heavy_indices].set(r_val)

            def local_loss_fn(R_heavy):
                X_local = X_final.at[oracle_mapping.heavy_indices].set(R_heavy)
                e_physics = hydride.relax.compute_energy(X_local, pairs, elec_param, eps, r_6, r_12,
                    box=box, box_inv=box_inv, reduction_indices=reduction_indices,
                    reduction_signs=reduction_signs, reduction_pair_map=reduction_pair_map)
                
                e_exp = 0.0
                if sfc_instance is not None:
                    e_exp = sfc_instance.compute_loss(X_local)

                restraint = (0.5 / current_eta) * jnp.sum((R_heavy - x_0_heavy_mapped) ** 2)
                return e_physics + e_exp + restraint

            grads = jax.grad(local_loss_fn)(r_val)
            return r_val - prox_lr * jnp.clip(grads, -1.0, 1.0)

        R_current = x_0_heavy_mapped
        R_optimized = jax.lax.fori_loop(0, prox_steps, step_body, R_current)
        x_af3_updated = x_af3_flat.at[oracle_mapping.source_indices].set(R_optimized)
        
        return x_0_flat.at[gather_idxs].set(x_af3_updated).reshape(x_0_real.shape)

    rng_key = jax.random.PRNGKey(0) if sample_key is None else sample_key
    atom_positions = model_runner.sample_guided_diffusion(
        rng_key,
        batch_dict,
        embeddings,
        rng_key,
        proximal_operator_fn,
    )

    return Conformations(atom_positions=atom_positions)


def _assemble_coordinates_from_conformation(
    atom_positions: jnp.ndarray,
    gather_idxs: jnp.ndarray,
    oracle_mapping: OracleMapping,
    oracle_atoms: Any
) -> jnp.ndarray:
    """Snaps coordinates back into the crystal's global reference frame via Kabsch alignment."""
    params = hydride.get_relaxation_params(oracle_atoms)
    (center_indices, axis_indices, is_free_mask, pairs, elec_param, eps, r_6, r_12, atom_to_bond_idx,
        box, box_inv, reduction_indices, reduction_signs, reduction_pair_map) = params


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
    
    X_final = oracle_mapping.initial_coordinates
    X_final = X_final.at[oracle_mapping.heavy_indices].set(x_af3_aligned[oracle_mapping.source_indices])
    
    X_relaxed, _, _ = hydride.relax_hydrogen_jit(
        X_final, center_indices, axis_indices, is_free_mask,
        pairs, elec_param, eps, r_6, r_12, atom_to_bond_idx, box=box, box_inv=box_inv,
        reduction_indices=reduction_indices,
        reduction_signs=reduction_signs,
        reduction_pair_map=reduction_pair_map,
        iterations=200
    )
    return X_relaxed


class Hijacker:
    @staticmethod
    def build_oracle(layout, denoised_vector_field_positions, ph: float = 7.4) -> Oracle:
        return _build_oracle_from_baseline_af3_prediction(layout, denoised_vector_field_positions, ph=ph)

    @staticmethod
    def hijack_diffusion(
        runner: HostRunner, 
        batch_dict: dict, 
        embeddings: HostEmbeddings, 
        gather_idxs: jnp.ndarray, 
        oracle: Oracle, 
        sfc: Optional[SFC] = None, 
        key: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        return _hijack_diffusion_with_custom_loss(runner, batch_dict, embeddings, gather_idxs, oracle, sfc, key)

    @staticmethod
    def assemble_coordinates(atom_positions: jnp.ndarray, gather_idxs: jnp.ndarray, oracle: Oracle) -> np.ndarray:
        complex_coords = _assemble_coordinates_from_conformation(atom_positions, gather_idxs, oracle.mapping, oracle.atoms)
        return np.array(complex_coords)
