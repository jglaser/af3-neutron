import gemmi
import jax.numpy as jnp
import numpy as np
from scipy.interpolate import RegularGridInterpolator


def generate_patterson_pair_rep(mtz_path: str, coords_xyz: np.ndarray, f_obs_label: str = "FOBS") -> np.ndarray:
    """Patterson map from MTZ amplitudes, evaluated at all pairwise interatomic vectors."""
    import gemmi
    import numpy as np

    mtz = gemmi.read_mtz_file(mtz_path)
    labels = mtz.column_labels()

    # 1. Robust column name resolution
    plausible_labels = [f_obs_label, 'FP', 'FOBS', 'F', 'F-obs', 'F_obs']
    actual_f_label = None
    for label in plausible_labels:
        if label in labels:
            actual_f_label = label
            break

    if actual_f_label is None:
        raise ValueError(f"Could not find amplitude column. Available MTZ columns: {labels}")

    # 2. Add a dummy Phase column to trick the FFT engine
    if 'PATT_PHI' not in labels:
        mtz.add_column('PATT_PHI', 'P')

    # Re-fetch the underlying numpy array because adding a column reallocates memory
    mtz_data = np.array(mtz, copy=False)
    labels = mtz.column_labels()

    f_idx = labels.index(actual_f_label)
    phi_idx = labels.index('PATT_PHI')

    # 3. Extract original amplitudes and square them in-place
    f_obs = mtz_data[:, f_idx]
    mtz_data[:, f_idx] = np.where(np.isnan(f_obs), np.nan, f_obs**2)

    # 4. Set phases to exactly 0.0
    mtz_data[:, phi_idx] = np.where(np.isnan(f_obs), np.nan, 0.0)

    # 5. Perform Spacegroup-aware FFT to real space
    # sample_rate=2.0 ensures a sufficiently fine grid (Nyquist/2)
    grid = mtz.transform_f_phi_to_map(actual_f_label, 'PATT_PHI', sample_rate=2.0)
    grid.normalize() # Mean 0, Std 1

    # Extract the raw 3D real-space array
    patterson_array = np.array(grid, copy=False)
    nx, ny, nz = patterson_array.shape

    # 6. Set up 3D interpolator across fractional unit cell [0, 1)
    x = np.linspace(0, 1, nx, endpoint=False)
    y = np.linspace(0, 1, ny, endpoint=False)
    z = np.linspace(0, 1, nz, endpoint=False)

    interpolator = RegularGridInterpolator((x, y, z), patterson_array, bounds_error=False, fill_value=0.0)

    # 7. Compute all pairwise vectors
    diff_xyz = coords_xyz[:, None, :] - coords_xyz[None, :, :]
    flat_diff_xyz = diff_xyz.reshape(-1, 3)

    # Convert Cartesian to fractional coordinates
    cell = mtz.cell
    flat_diff_frac = np.array([cell.fractionalize(gemmi.Position(*v)).tolist() for v in flat_diff_xyz])

    # Wrap fractional coordinates to [0, 1) unit cell bounds
    flat_diff_frac = flat_diff_frac % 1.0

    # 8. Evaluate Patterson map at these pairwise vectors
    patterson_intensities = interpolator(flat_diff_frac)
    pair_rep_patterson = patterson_intensities.reshape(coords_xyz.shape[0], coords_xyz.shape[0])

    return pair_rep_patterson

def inject_patterson_into_first_bin(batch_dict: dict, patterson_map: jnp.ndarray) -> dict:
    """Inject a Patterson map into AF3's template distogram, zero-shot.

    Normalised intensity goes entirely into the 0-th (shortest-distance) bin.
    """
    p_max = jnp.max(patterson_map)
    p_norm = patterson_map / (p_max + 1e-8)

    N = p_norm.shape[0]

    if "template_distogram" in batch_dict:
        distogram = batch_dict["template_distogram"]
        mod_template = jnp.zeros_like(distogram[0])
        mod_template = mod_template.at[:, :, 0].set(p_norm)
        batch_dict["template_distogram"] = distogram.at[0].set(mod_template)
    elif "template_pair_feat" in batch_dict:
        pair_feat = batch_dict["template_pair_feat"]
        mod_template = jnp.zeros_like(pair_feat[0])
        mod_template = mod_template.at[:, :, 0].set(p_norm)
        batch_dict["template_pair_feat"] = pair_feat.at[0].set(mod_template)
    else:
        mod_template = jnp.zeros((1, N, N, 39))
        mod_template = mod_template.at[0, :, :, 0].set(p_norm)
        batch_dict["template_distogram"] = mod_template

    if "template_mask" in batch_dict:
        batch_dict["template_mask"] = batch_dict["template_mask"].at[0].set(1.0)
    else:
        batch_dict["template_mask"] = jnp.ones((1,))

    return batch_dict

def extract_patterson_grid_from_mtz(mtz_path: str, fobs_col: str = "FOBS", d_min: float = 2.0) -> tuple:
    """Computes a 3D Patterson grid directly from an MTZ file via FFT."""
    mtz = gemmi.read_mtz_file(mtz_path)

    # 1. Extract Miller indices and experimental amplitudes
    data = np.array(mtz, copy=False)
    hkl = data[:, :3].astype(int)
    fobs = mtz.column_with_label(fobs_col).array

    # 2. Patterson maps use Intensities (F^2) with strictly zero phase
    intensities = fobs ** 2

    # 3. Define a dense grid based on the unit cell and Nyquist sampling limit
    cell = mtz.cell
    na = int(cell.a / (d_min / 3.0))
    nb = int(cell.b / (d_min / 3.0))
    nc = int(cell.c / (d_min / 3.0))

    # 4. Populate reciprocal space grid and enforce Friedel symmetry
    fft_grid = np.zeros((na, nb, nc), dtype=np.complex64)
    for (h, k, ell), i_val in zip(hkl, intensities, strict=True):
        fft_grid[h % na, k % nb, ell % nc] = i_val
        fft_grid[-h % na, -k % nb, -ell % nc] = i_val

    # 5. Inverse FFT to real space (Patterson vector space)
    patterson_grid = np.real(np.fft.ifftn(fft_grid))

    # 6. Extract spatial metadata for JAX map_coordinates
    grid_origin = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    grid_spacing = np.array([
        cell.a / na,
        cell.b / nb,
        cell.c / nc
    ], dtype=np.float32)

    return patterson_grid, grid_origin, grid_spacing
