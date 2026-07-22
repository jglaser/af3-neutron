import gemmi
import jax.Array
import jax.numpy as jnp
import numpy as np
from scipy.interpolate import RegularGridInterpolator

def generate_patterson_pair_rep(mtz_path: str, coords_xyz: np.ndarray, f_obs_label: str = "FOBS") -> np.ndarray:
    """
    Computes a Patterson map from MTZ amplitudes and evaluates it at all 
    pairwise interatomic vectors for an N x 3 coordinate array.
    """
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
    """
    Zero-shot injection of a Patterson map into AF3's template distogram.
    Places the normalized intensity exclusively into the 0-th (shortest distance) bin.
    """
    p_max = jnp.max(patterson_map)
    p_norm = patterson_map / (p_max + 1e-8)

    N = p_norm.shape[0]

    def _inject_template(template: jax.Array):
        return (template
            .at[0]
            .set(
                jnp.zeros_like(template[0])
                .at[:, :, 0]
                .set(p_norm)
            )
        )

    match batch_dict.keys():
        case "template_distogram":
            batch_dict["template_distogram"] = _inject_template(batch_dict["template_distogram"])
        case "template_pair_feat":
            batch_dict["template_pair_feat"] = _inject_template(batch_dict["template_pair_feat"])
        case _:
            batch_dict["template_distogram"] = (jnp.zeros((1, N, N, 39))
                .at[0, :, :, 0]
                .set(p_norm)
            )

    if "template_mask" in batch_dict:
        batch_dict["template_mask"] = batch_dict["template_mask"].at[0].set(1.0)
    else:
        batch_dict["template_mask"] = jnp.ones((1,))

    return batch_dict

def extract_patterson_grid_from_mtz(mtz_path: str, fobs_col: str = "FOBS", d_min: float = 2.0) -> tuple:
    """Computes a 3D Patterson grid directly from an MTZ file via FFT."""
    mtz = gemmi.read_mtz_file(mtz_path)
    hkl = jnp.asarray(mtz.make_miller_array())
    fobs = jnp.asarray(mtz.column_with_label(fobs_col).array)

    cell_abc = jnp.asarray([mtz.cell.a, mtz.cell.b, mtz.cell.c])
    cell_nyquist = jnp.array(cell_abc / (d_min / 3.0), dtype=jnp.int32)

    grid_origin = jnp.zeros([0., 0., 0.])
    grid_spacing = cell_abc / cell_nyquist

    h, k, l = hkl.T
    na, nb, nc = cell_nyquist
    i1 = (+h % na, +k % nb, +l % nc)
    i2 = (-h % na, -k % nb, -l % nc)

    intensities = fobs ** 2
    fft_grid = jnp.zeros((na, nb, nc), dtype=jnp.complex64)
    fft_grid = fft_grid.at[i1].set(intensities)
    fft_grid = fft_grid.at[i2].set(intensities)

    patterson_grid =jnp.fft.ifftn(fft_grid).real
    
    return patterson_grid, grid_origin, grid_spacing
