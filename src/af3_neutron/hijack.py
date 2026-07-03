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
    RotorTable,
    WaterMapping,
    Oracle,
    OracleMapping,
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

    dummy_rotor = RotorTable(
        target_idx=jnp.zeros(0, dtype=jnp.int32), parent_idx=jnp.zeros(0, dtype=jnp.int32),
        grandparent_idx=jnp.zeros(0, dtype=jnp.int32), greatgrand_idx=jnp.zeros(0, dtype=jnp.int32),
        ideal_r=jnp.zeros(0, dtype=jnp.float32), ideal_theta=jnp.zeros(0, dtype=jnp.float32),
        initial_chi=jnp.zeros(0, dtype=jnp.float32)
    )
    dummy_water = WaterMapping(
        oxygen_source=jnp.zeros(0, dtype=jnp.int32), h1_target=jnp.zeros(0, dtype=jnp.int32),
        h2_target=jnp.zeros(0, dtype=jnp.int32)
    )

    return Oracle(
        mapping=OracleMapping(
            num_atoms=num_oracle_atoms,
            heavy_indices=jnp.array(oracle_heavy_indices, dtype=jnp.int32),
            source_indices=jnp.array(af3_source_indices, dtype=jnp.int32),
            initial_coordinates=jnp.array(oracle_atoms.coord, dtype=jnp.float32),
            rotor_table=dummy_rotor,
            water_mapping=dummy_water
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
) -> Conformations:
    """Intercepts and steers Host diffusion trajectories using an Envelope-Theorem optimized proximal operator."""
    oracle_mapping = oracle.mapping

    # Extract force-field parameters cleanly using the fully-bonded structure
    params = hydride.get_relaxation_params(oracle.atoms)

    if params is None:
        raise ValueError("No rotatable bonds found in the structure configuration.")

    center_indices, axis_indices, is_free_mask, pairs, elec_param, eps, r_6, r_12, atom_to_bond_idx, box = params

    def proximal_operator_fn(x_0_real: jnp.ndarray, t_hat: jnp.ndarray) -> jnp.ndarray:
        x_0_flat = x_0_real.reshape((x_0_real.shape[0], -1, 3))
        current_eta = 1e-2 * (t_hat ** 2)
        lambda_exp = 1.0

        def single_sample_prox(x_0_single):
            x_af3_flat = x_0_single[gather_idxs]
            x_0_heavy_mapped = x_af3_flat[oracle_mapping.source_indices]

            def joint_objective(R_heavy):
                # 1. Isolate the forward relaxation coordinates to prune the backprop tape
                X_static = oracle_mapping.initial_coordinates
                X_static = X_static.at[oracle_mapping.heavy_indices].set(R_heavy)
                X_static = jax.lax.stop_gradient(X_static)

                # 2. Run hydrogen optimization as a pure forward pass (Zero memory footprint)
                X_relaxed, _, _ = hydride.relax_hydrogen_jit(
                    X_static, center_indices, axis_indices, is_free_mask,
                    pairs, elec_param, eps, r_6, r_12, atom_to_bond_idx, box=box, iterations=40
                )
                
                # 3. Disconnect the optimization trajectory entirely 
                X_final = jax.lax.stop_gradient(X_relaxed)
                
                # 4. Graft the active heavy atom tracers back into the system to compute strict partial forces
                X_final = X_final.at[oracle_mapping.heavy_indices].set(R_heavy)

                # 5. Evaluate the active potential fields
                e_physics = hydride.relax.compute_energy(X_final, pairs, elec_param, eps, r_6, r_12, box=box)
                e_exp = 0.0
                if sfc_instance is not None:
                    e_exp = sfc_instance.compute_loss(X_final)

                restraint = (0.5 / current_eta) * jnp.sum((R_heavy - x_0_heavy_mapped) ** 2)
                return e_physics + (lambda_exp * e_exp) + restraint

            R_current = x_0_heavy_mapped
            def step_body(i, r_val):
                grads = jax.grad(joint_objective)(r_val)
                return r_val - prox_lr * jnp.clip(grads, -1.0, 1.0)

            R_optimized = jax.lax.fori_loop(0, prox_steps, step_body, R_current)
            x_af3_updated = x_af3_flat.at[oracle_mapping.source_indices].set(R_optimized)
            return x_0_single.at[gather_idxs].set(x_af3_updated)

        return jax.vmap(single_sample_prox)(x_0_flat).reshape(x_0_real.shape)

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
    oracle_atoms: Any
) -> jnp.ndarray:
    """Snaps coordinates back into the crystal's global reference frame via Kabsch alignment."""
    params = hydride.get_relaxation_params(oracle_atoms)
    center_indices, axis_indices, is_free_mask, pairs, elec_param, eps, r_6, r_12, atom_to_bond_idx, box = params

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
        pairs, elec_param, eps, r_6, r_12, atom_to_bond_idx, box=box, iterations=200
    )
    return X_relaxed


class Hijacker:
    @staticmethod
    def build_oracle(layout, denoised_vector_field_positions) -> Oracle:
        return _build_oracle_from_baseline_af3_prediction(layout, denoised_vector_field_positions)

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
