"""
diffraction_embedding.py
========================
Dynamic-length (N <= N_max) diffraction-informed template embedding module.
"""

import pickle
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Any, Optional

from alphafold3.model import features, model_config
from alphafold3.model.scoring import scoring
from alphafold3.model.network.template_modules import (
    TemplateEmbedding,
    SingleTemplateEmbedding,
)


# ==============================================================================
# 1. Mask-Aware Diffraction Feature Calculators
# ==============================================================================

def sample_patterson_map(
    positions: jnp.ndarray,       # [N_max, 3]
    patterson_grid: jnp.ndarray,  # [D, H, W]
    grid_origin: jnp.ndarray,     # [3]
    grid_spacing: jnp.ndarray,    # [3]
    mask_2d: jnp.ndarray,         # [N_max, N_max]
) -> jnp.ndarray:
    """Queries 3D Patterson map density at u_ij = r_i - r_j for valid pairs."""
    u_ij = positions[:, None, :] - positions[None, :, :]
    grid_coords = (u_ij - grid_origin) / grid_spacing
    
    sampled = jax.scipy.ndimage.map_coordinates(
        patterson_grid,
        jnp.moveaxis(grid_coords, -1, 0),
        order=1,
        mode="nearest",
    )[..., None]
    
    # Zero out padded residue pairs
    return sampled * mask_2d[..., None]

def compute_debye_features(
    positions: jnp.ndarray, # [N_max, 3]
    q_bins: jnp.ndarray,    # [num_q]
    mask_2d: jnp.ndarray,   # [N_max, N_max]
) -> jnp.ndarray:
    diff = positions[:, None, :] - positions[None, :, :]
    d_ij = jnp.sqrt(jnp.sum(jnp.square(diff), axis=-1, keepdims=True) + 1e-8)
    
    q_d = d_ij * q_bins[None, None, :]
    # Compute sinc wave
    debye_raw = jnp.where(q_d < 1e-4, 1.0, jnp.sin(q_d) / q_d)
    
    # STRICT MASKING: Zero out padded residue pairs entirely
    return debye_raw * mask_2d[..., None]

# ==============================================================================
# 2. Haiku Adapter & Dynamic Augmented Embedder
# ==============================================================================

class DiffractionPairAdapter(hk.Module):
    """2-Layer MLP adapter predicting dynamic residual updates ΔZ."""

    def __init__(self, out_channels: int, name: str = "diffraction_pair_adapter"):
        super().__init__(name=name)
        self.out_channels = out_channels

    def __call__(
        self,
        debye_feats: jnp.ndarray,
        patterson_feats: jnp.ndarray,
        mask_2d: jnp.ndarray,
    ) -> jnp.ndarray:
        x = jnp.concatenate([debye_feats, patterson_feats], axis=-1)
        x = hk.Linear(128, name="adapter_mlp_1")(x)
        x = jax.nn.relu(x)
        
        delta_z = hk.Linear(
            self.out_channels,
            w_init=hk.initializers.Constant(0.0),
            name="adapter_mlp_2",
        )(x)
        
        # Multiply by 2D mask to guarantee padding tokens remain 0.0
        return delta_z * mask_2d[..., None]


class DiffractionAugmentedTemplateEmbedding(hk.Module):
    """Wraps native AF3 SingleTemplateEmbedding with dynamic length support."""

    def __init__(
        self,
        config: TemplateEmbedding.Config,
        global_config: model_config.GlobalConfig,
        q_bins: jnp.ndarray,
        name: str = "diffraction_augmented_template_embedding",
    ):
        super().__init__(name=name)
        self.config = config
        self.global_config = global_config
        self.q_bins = q_bins

        self.native_embedder = SingleTemplateEmbedding(
            config=config,
            global_config=global_config,
            name="native_single_template_embedding",
        )

    def __call__(
        self,
        query_embedding: jnp.ndarray,  # [N_max, N_max, C]
        templates: features.Templates,  # [N_max, 24, 3]
        padding_mask_2d: jnp.ndarray,  # [N_max, N_max] (1.0 for valid, 0.0 for pad)
        multichain_mask_2d: jnp.ndarray,  # [N_max, N_max]
        patterson_grid: jnp.ndarray,
        grid_origin: jnp.ndarray,
        grid_spacing: jnp.ndarray,
        key: jnp.ndarray,
        use_adapter: bool = True,
    ) -> jnp.ndarray:

        # 1. Native AF3 Base Forward Pass
        native_z = self.native_embedder(
            query_embedding=query_embedding,
            templates=templates,
            padding_mask_2d=padding_mask_2d,
            multichain_mask_2d=multichain_mask_2d,
            key=key,
        )

        if not use_adapter:
            return native_z * padding_mask_2d[..., None]

        # 2. Extract Pseudo-Beta/CA positions across all residues [N_max, 3]
        cb_positions, _ = scoring.pseudo_beta_fn(
            templates.aatype, templates.atom_positions, templates.atom_mask
        )

        # 3. Compute Debye & Patterson features with explicit mask
        debye_feats = compute_debye_features(cb_positions, self.q_bins, padding_mask_2d)
        patterson_feats = sample_patterson_map(
            cb_positions, patterson_grid, grid_origin, grid_spacing, padding_mask_2d
        )

        # 4. Predict & Inject Delta Z
        adapter = DiffractionPairAdapter(out_channels=native_z.shape[-1])
        delta_z = adapter(debye_feats, patterson_feats, padding_mask_2d)

        # Return masked augmented pair representation
        return (native_z + delta_z) * padding_mask_2d[..., None]


# ==============================================================================
# 3. Weight Serialization Utilities
# ==============================================================================

def save_adapter_weights(params: hk.Params, filepath: str = "diffraction_adapter_weights.pkl"):
    adapter_params = {}
    for module_name, module_dict in params.items():
        if "diffraction_pair_adapter" in module_name:
            adapter_params[module_name] = jax.tree_util.tree_map(np.array, module_dict)

    with open(filepath, "wb") as f:
        pickle.dump(adapter_params, f)
    print(f"Successfully saved adapter weights to '{filepath}'.")


def load_adapter_weights(params: hk.Params, filepath: str = "diffraction_adapter_weights.pkl") -> hk.Params:
    with open(filepath, "rb") as f:
        loaded_dict = pickle.load(f)

    merged_params = dict(params)
    for module_name, module_dict in loaded_dict.items():
        if module_name in merged_params:
            merged_params[module_name] = jax.tree_util.tree_map(jnp.array, module_dict)

    return merged_params
