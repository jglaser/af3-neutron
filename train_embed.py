import jax
import jax.numpy as jnp
import haiku as hk
import optax
import numpy as np
import gemmi
from typing import Dict, Any, Tuple

# Native AlphaFold 3 imports
from alphafold3.model import features, model_config
from alphafold3.model.network.template_modules import TemplateEmbedding

# Custom diffraction embedding imports
from af3_neutron.diffraction_embedding import (
    DiffractionAugmentedTemplateEmbedding,
    save_adapter_weights,
)


# ==============================================================================
# 1. Gemmi Forward Physics Simulator
# ==============================================================================

def generate_patterson_map_gemmi(
    coords: np.ndarray, 
    elements: list[str], 
    box_size: float = 60.0,
    d_min: float = 2.5,
    is_neutron: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generates a 3D Patterson map from a 3D atomic point cloud using Gemmi + FFT."""
    structure = gemmi.Structure()
    cell = gemmi.UnitCell(box_size, box_size, box_size, 90, 90, 90)
    structure.cell = cell
    
    model = gemmi.Model("1")
    chain = gemmi.Chain("A")
    
    for i, (pos, elem_str) in enumerate(zip(coords, elements)):
        res = gemmi.Residue()
        res.name = "UNK"
        res.seqid = gemmi.SeqId(str(i + 1))  # String conversion for SeqId
        
        atom = gemmi.Atom()
        atom.name = f"P{i}"
        atom.element = gemmi.Element(elem_str)
        atom.pos = gemmi.Position(float(pos[0]), float(pos[1]), float(pos[2]))
        atom.occ = 1.0
        atom.b_iso = 15.0
        
        res.add_atom(atom)
        chain.add_residue(res)
        
    model.add_chain(chain)
    structure.add_model(model)
    
    dencalc = gemmi.DensityCalculatorN() if is_neutron else gemmi.DensityCalculatorX()
    dencalc.d_min = d_min
    dencalc.grid.spacegroup = gemmi.find_spacegroup_by_name("P1")
    dencalc.grid.set_unit_cell(cell)
    dencalc.put_model_density_on_grid(model)
    
    density_map = np.array(dencalc.grid, copy=True)
    
    # 3D FFT Autocorrelation
    f_k = np.fft.fftn(density_map)
    intensities = np.abs(f_k) ** 2
    patterson_map = np.real(np.fft.ifftn(intensities)).astype(np.float32)
    
    grid_origin = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    grid_spacing = np.array([
        box_size / patterson_map.shape[0],
        box_size / patterson_map.shape[1],
        box_size / patterson_map.shape[2]
    ], dtype=np.float32)
    
    return patterson_map, grid_origin, grid_spacing


# ==============================================================================
# 2. Variable-Length Synthetic Batch Generator (Full N, CA, C, CB Geometry)
# ==============================================================================

def generate_synthetic_batch(
    n_max: int = 1024,
    min_res: int = 16,
    max_res: int = 128,
    noise_scale: float = 2.5,
    seed: int = None
) -> Dict[str, Any]:
    """Generates variable-length point clouds with complete backbone geometry padded to N_max."""
    if seed is not None:
        np.random.seed(seed)
        
    n_valid = np.random.randint(min_res, min(max_res, n_max) + 1)
    
    # 1. Generate CA backbone trajectory
    t = np.linspace(0, 4 * np.pi, n_valid)
    x = 12.0 * np.cos(t) + 20.0
    y = 12.0 * np.sin(t) + 20.0
    z = 2.5 * t + 10.0
    ca_true_valid = np.stack([x, y, z], axis=-1)
    
    # Compute Patterson map using true coordinates
    elements = ["C"] * n_valid
    p_grid, origin, spacing = generate_patterson_map_gemmi(ca_true_valid, elements)
    
    # Perturb backbone for reference template
    ca_ref_valid = ca_true_valid + np.random.normal(0.0, noise_scale, size=ca_true_valid.shape)

    # 2. Construct complete N, CA, C, CB backbone positions
    def make_padded_af3_atoms(ca_positions):
        atom_pos = np.zeros((n_max, 24, 3), dtype=np.float32)
        
        # Standard ideal backbone offsets relative to CA
        n_offset  = np.array([-1.2,  0.8, 0.0], dtype=np.float32)
        c_offset  = np.array([ 1.2,  0.8, 0.0], dtype=np.float32)
        cb_offset = np.array([ 0.0, -1.2, 0.8], dtype=np.float32)
        
        for i in range(n_valid):
            ca = ca_positions[i]
            atom_pos[i, 0, :] = ca + n_offset   # Atom 0: N
            atom_pos[i, 1, :] = ca              # Atom 1: CA
            atom_pos[i, 2, :] = ca + c_offset   # Atom 2: C
            atom_pos[i, 3, :] = ca + cb_offset  # Atom 3: CB
            
        return atom_pos

    # 3. Create 1D and 2D Padding Masks
    mask_1d = np.zeros((n_max,), dtype=np.float32)
    mask_1d[:n_valid] = 1.0
    mask_2d = mask_1d[:, None] * mask_1d[None, :]

    # Mask valid backbone atoms (0: N, 1: CA, 2: C, 3: CB)
    atom_mask = np.zeros((n_max, 24), dtype=np.float32)
    atom_mask[:n_valid, :4] = 1.0

    return {
        "templates_true": features.Templates(
            aatype=jnp.zeros((n_max,), dtype=jnp.int32),
            atom_positions=jnp.array(make_padded_af3_atoms(ca_true_valid)),
            atom_mask=jnp.array(atom_mask),
        ),
        "templates_ref": features.Templates(
            aatype=jnp.zeros((n_max,), dtype=jnp.int32),
            atom_positions=jnp.array(make_padded_af3_atoms(ca_ref_valid)),
            atom_mask=jnp.array(atom_mask),
        ),
        "patterson_grid": jnp.array(p_grid),
        "grid_origin": jnp.array(origin),
        "grid_spacing": jnp.array(spacing),
        "padding_mask_2d": jnp.array(mask_2d),
        "multichain_mask_2d": jnp.array(mask_2d),
        "n_valid": n_valid,
    }


# ==============================================================================
# 3. Training Loop with Pre-Flight Diagnostics
# ==============================================================================

def main():
    n_max = 512         # Allocation size for spatial pair tensors
    num_channels = 128
    learning_rate = 1e-3
    num_steps = 50
    val_interval = 5

    template_config = TemplateEmbedding.Config(num_channels=num_channels)
    global_config = model_config.GlobalConfig()
    q_bins = jnp.linspace(0.05, 0.5, 16)

    # Haiku Model Transformation
    def forward_fn(
        query_embedding, templates, padding_mask_2d, multichain_mask_2d,
        patterson_grid, grid_origin, grid_spacing, rng_key, use_adapter=True
    ):
        model = DiffractionAugmentedTemplateEmbedding(
            config=template_config, global_config=global_config, q_bins=q_bins
        )
        return model(
            query_embedding, templates, padding_mask_2d, multichain_mask_2d,
            patterson_grid, grid_origin, grid_spacing, rng_key, use_adapter
        )

    transformed_model = hk.without_apply_rng(hk.transform(forward_fn))

    # PRNG & Synthetic Batch Setup
    key = jax.random.PRNGKey(42)
    print(f"Generating synthetic batches (padded to N_max={n_max})...")
    val_batch = generate_synthetic_batch(n_max=n_max, min_res=32, max_res=64, noise_scale=2.5, seed=999)
    dummy_batch = generate_synthetic_batch(n_max=n_max, min_res=16, max_res=128, noise_scale=2.5, seed=123)

    # Construct masked query_embedding
    query_embed = jax.random.normal(key, (n_max, n_max, num_channels))
    query_embed = query_embed * dummy_batch["padding_mask_2d"][..., None]

    # Initialize Parameters
    init_params = transformed_model.init(
        key, query_embed, dummy_batch["templates_ref"], dummy_batch["padding_mask_2d"],
        dummy_batch["multichain_mask_2d"], dummy_batch["patterson_grid"],
        dummy_batch["grid_origin"], dummy_batch["grid_spacing"], key, True
    )

    # --- Pre-Flight Diagnostic Suite ---
    print("\n=== Running Pre-Flight Diagnostics ===")
    
    # Diagnostic 1: Verify Optax Mask Targets
    adapter_keys = [k for k in init_params.keys() if "diffraction_pair_adapter" in k]
    print(f"1. Optax Adapter Target Keys: {adapter_keys}")
    assert len(adapter_keys) > 0, "DIAGNOSTIC FAILURE: Optax adapter module key mismatch!"

    # Diagnostic 2: Check Ground-Truth Latent Signal Strength
    z_true_check = transformed_model.apply(
        init_params, query_embed, dummy_batch["templates_true"], dummy_batch["padding_mask_2d"],
        dummy_batch["multichain_mask_2d"], dummy_batch["patterson_grid"],
        dummy_batch["grid_origin"], dummy_batch["grid_spacing"], key, False
    )
    max_z_val = float(jnp.max(jnp.abs(z_true_check)))
    print(f"2. Ground-Truth Max Absolute |Z_true|: {max_z_val:.6f}")
    assert max_z_val > 1e-4, "DIAGNOSTIC FAILURE: Z_true evaluates to zero! Check backbone atoms."

    print("=== All Diagnostics Passed! ===\n")

    # Mask Optimizer Setup (Freeze native AF3 weights, update only adapter)
    def make_adapter_mask(params):
        return jax.tree_util.tree_map_with_path(
            lambda path, _: "diffraction_pair_adapter" in jax.tree_util.keystr(path),
            params
        )

    optimizer = optax.masked(optax.adam(learning_rate), make_adapter_mask)
    opt_state = optimizer.init(init_params)

    # JIT-Compiled Training Step
    @jax.jit
    def train_step(params, opt_state, query_embed, batch, rng_key):
        mask_2d = batch["padding_mask_2d"]
        num_valid_pairs = jnp.sum(mask_2d) + 1e-8

        z_true = transformed_model.apply(
            params, query_embed, batch["templates_true"], mask_2d,
            batch["multichain_mask_2d"], batch["patterson_grid"], batch["grid_origin"],
            batch["grid_spacing"], rng_key, False
        )

        def loss_fn(trainable_params):
            z_pred = transformed_model.apply(
                trainable_params, query_embed, batch["templates_ref"], mask_2d,
                batch["multichain_mask_2d"], batch["patterson_grid"], batch["grid_origin"],
                batch["grid_spacing"], rng_key, True
            )
            
            diff_sq = jnp.square(z_pred - jax.lax.stop_gradient(z_true))
            return jnp.sum(diff_sq * mask_2d[..., None]) / (num_valid_pairs * z_pred.shape[-1])

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        return new_params, new_opt_state, loss

    # JIT-Compiled Validation Step
    @jax.jit
    def eval_step(params, query_embed, batch, rng_key):
        mask_2d = batch["padding_mask_2d"]
        num_valid_pairs = jnp.sum(mask_2d) + 1e-8

        z_true = transformed_model.apply(
            params, query_embed, batch["templates_true"], mask_2d,
            batch["multichain_mask_2d"], batch["patterson_grid"], batch["grid_origin"],
            batch["grid_spacing"], rng_key, False
        )
        z_pred = transformed_model.apply(
            params, query_embed, batch["templates_ref"], mask_2d,
            batch["multichain_mask_2d"], batch["patterson_grid"], batch["grid_origin"],
            batch["grid_spacing"], rng_key, True
        )
        
        diff_sq = jnp.square(z_pred - jax.lax.stop_gradient(z_true))
        return jnp.sum(diff_sq * mask_2d[..., None]) / (num_valid_pairs * z_pred.shape[-1])

    # Execution Loop
    print("Starting Variable-Length Training Loop...")
    params = init_params
    best_val_loss = float("inf")

    for step in range(1, num_steps + 1):
        key, subkey = jax.random.split(key)
        train_batch = generate_synthetic_batch(n_max=n_max, min_res=16, max_res=128, noise_scale=2.5)
        
        params, opt_state, train_loss = train_step(params, opt_state, query_embed, train_batch, subkey)

        if step % val_interval == 0 or step == 1:
            val_loss = eval_step(params, query_embed, val_batch, subkey)
            
            checkpoint_str = ""
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_adapter_weights(params, "diffraction_adapter_weights.pkl")
                checkpoint_str = " [★ Checkpoint Saved]"

            print(
                f"Step {step:02d}/{num_steps} | N_valid={train_batch['n_valid']:03d} | "
                f"Train MSE: {train_loss:.6f} | "
                f"Val MSE: {val_loss:.6f}{checkpoint_str}"
            )

    print(f"\nTraining Complete! Best Validation Score: {best_val_loss:.6f}")


if __name__ == "__main__":
    main()
