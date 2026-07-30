"""Dynamic-length (N <= N_max) diffraction-informed template embedding."""

import pickle

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from alphafold3.model import features, model_config
from alphafold3.model.network.template_modules import (
    SingleTemplateEmbedding,
    TemplateEmbedding,
)
from alphafold3.model.scoring import scoring

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

        # Use standard VarianceScaling initialization so weights start non-zero
        delta_z = hk.Linear(
            self.out_channels,
            w_init=hk.initializers.VarianceScaling(scale=0.1),
            name="adapter_mlp_2",
        )(x)

        return delta_z * mask_2d[..., None]

def compute_diffraction_delta_z(
    aatype: jnp.ndarray,
    atom_positions: jnp.ndarray,
    atom_mask: jnp.ndarray,
    patterson_grid: jnp.ndarray,
    grid_origin: jnp.ndarray,
    grid_spacing: jnp.ndarray,
    pair_mask: jnp.ndarray,
    out_channels: int,
    q_bins: jnp.ndarray = None,
    gamma: float = 1.0,
) -> jnp.ndarray:
    """Shared core pipeline: extracts pseudo-beta, computes features, and predicts residual ΔZ."""
    if q_bins is None:
        q_bins = jnp.linspace(0.05, 0.5, 16)

    # 1. Pseudo-Beta / CA positions
    cb_positions, _ = scoring.pseudo_beta_fn(aatype, atom_positions, atom_mask)

    # 2. Extract physical features
    debye_feats = compute_debye_features(cb_positions, q_bins, pair_mask)
    patterson_feats = sample_patterson_map(
        cb_positions, patterson_grid, grid_origin, grid_spacing, pair_mask
    )

    # 3. Predict adapter residual update
    adapter = DiffractionPairAdapter(out_channels=out_channels)
    delta_z = adapter(debye_feats, patterson_feats, pair_mask)

    return gamma * delta_z

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
        query_embedding: jnp.ndarray,
        templates: features.Templates,
        padding_mask_2d: jnp.ndarray,
        multichain_mask_2d: jnp.ndarray,
        patterson_grid: jnp.ndarray,
        grid_origin: jnp.ndarray,
        grid_spacing: jnp.ndarray,
        key: jnp.ndarray,
        use_adapter: bool = True,
    ) -> jnp.ndarray:

        native_z = self.native_embedder(
            query_embedding=query_embedding,
            templates=templates,
            padding_mask_2d=padding_mask_2d,
            multichain_mask_2d=multichain_mask_2d,
            key=key,
        )

        if not use_adapter:
            return native_z * padding_mask_2d[..., None]

        delta_z = compute_diffraction_delta_z(
            aatype=templates.aatype,
            atom_positions=templates.atom_positions,
            atom_mask=templates.atom_mask,
            patterson_grid=patterson_grid,
            grid_origin=grid_origin,
            grid_spacing=grid_spacing,
            pair_mask=padding_mask_2d,
            out_channels=native_z.shape[-1],
            q_bins=self.q_bins,
        )

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
