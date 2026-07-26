import os
import sys  
import json
import tempfile
import dataclasses
import numpy as np
import jax
import jax.numpy as jnp
import gemmi  
import reciprocalspaceship as rs
from biotite.structure.io import pdb
import biotite.structure.io.pdbx as pdbx
import biotite.structure as struc
from SFC_Jax.Fmodel import SFcalculator, F_protein

# ==============================================================================
# MONKEYPATCH FOR GEMMI VERSION CONFLICT (v0.7.0+)
# ==============================================================================
if not hasattr(gemmi.UnitCell, "fractionalization_matrix"):
    gemmi.UnitCell.fractionalization_matrix = property(lambda self: self.frac.mat)
if not hasattr(gemmi.UnitCell, "orthogonalization_matrix"):
    gemmi.UnitCell.orthogonalization_matrix = property(lambda self: self.orth.mat)
# ==============================================================================

class NeutronSFCalculator(SFcalculator):
    """Subclass of SFcalculator that implements a clean, scale-invariant, 
    JAX-differentiable structure factor amplitude loss, utilizing a pure
    functional pipeline to guarantee gradient tracking inside XLA loops.
    """
    def __init__(self, *args, b_factors=None, occupancies=None, **kwargs):
        # 1. Execute parent initialization (parses PDB and sets Gemmi scattering lengths)
        super().__init__(*args, **kwargs)

        # 2. Safely overwrite the parent's parsed B-factors/occupancies
        # using the exact attribute names expected by F_protein
        if b_factors is not None:
            self.atom_b_iso = jnp.array(b_factors, dtype=jnp.float32)
            
        if occupancies is not None:
            self.atom_occ = jnp.array(occupancies, dtype=jnp.float32)

    def compute_loss(self, xyz, t_hat=None):
        # 1. PURE JAX COMPUTATION
        atom_pos_frac = jnp.tensordot(xyz, self.orth2frac_tensor.T, 1)
        
        f_calc_protein_asu = F_protein(
            self.Hasu_array, 
            self.dr2asu_array,
            self.fullsf_tensor,
            self.reciprocal_cell_paras,
            self.R_G_tensor_stack, 
            self.T_G_tensor_stack,
            atom_pos_frac,
            self.atom_b_iso, 
            self.atom_b_aniso, 
            self.atom_occ
        )
        
        f_calc_protein = f_calc_protein_asu[self.asu2HKL_index]
        dr2_tensor = jnp.array(self.dr2HKL_array)
        
        # 2. Reactivate Bulk Solvent Mask
        # Fetch SFC_Jax solvent parameters (with underscores), using safe defaults
        k_sol = getattr(self, "k_sol", 0.35)
        b_sol = getattr(self, "b_sol", 50.0)

        # Two bulk-solvent models.
        #
        # MASK (default, current behaviour): Fmask_HKL is precomputed from the
        # baseline coordinates and held constant, so d(F_bulk)/d(xyz) = 0.  That
        # is fine when guiding on high-resolution data, where solvent is
        # negligible -- but it is actively harmful when guiding on d >= 8 A,
        # because bulk solvent is a large fraction of F_calc there.  With
        # k_sol ~ 0.53 and b_sol = 10, exp(-b_sol*s^2/4) ~ 0.96 at 8 A, so
        # roughly half of the low-resolution F_calc is a frozen constant
        # computed from a placement that gave 0.849 solvent.  The gradient sees
        # only the protein half while the residual is dominated by the wrong,
        # non-differentiable half.
        #
        # BABINET: F_bulk = -k_sol * exp(-b_sol*s^2/4) * F_protein.  Mask-free,
        # so it can never go stale; differentiable, so the solvent term responds
        # to the coordinates; and least accurate at high resolution, which is
        # exactly where it is not being used.  Set `sfc.solvent_model = "babinet"`.
        if getattr(self, "solvent_model", "mask") == "babinet":
            f_bulk = -k_sol * jnp.exp(-b_sol * dr2_tensor / 4.0) * f_calc_protein
        else:
            Fmask_HKL = getattr(self, "Fmask_HKL")
            f_bulk = k_sol * jnp.exp(-b_sol * dr2_tensor / 4.0) * Fmask_HKL

        f_calc_complex = f_calc_protein + f_bulk
        f_calc_mag = jnp.abs(f_calc_complex)
        
        # 3. Safely extract experimental amplitudes
        f_obs_attr = getattr(self, "Fo", None)
        if f_obs_attr is None:
            f_obs_attr = getattr(self, "Fobs", None)
        if f_obs_attr is None:
            f_obs_attr = getattr(self, "fo", None)
            
        f_obs = jnp.array(f_obs_attr)
        
        # 4. Enforce Cross-Validation
        # NOTE: The low-resolution cutoff has been removed so the model 
        # can fit the newly activated solvent envelope at low angles.
        mask_valid = (f_obs > 0.0) & (~jnp.isnan(f_obs))

        # Resolution window for guidance.  Set `sfc.guidance_d_high = 8.0` to
        # restrict the target to d >= 8 A.  This is not an optimisation, it is
        # what makes the potential informative: a model 9.9 A from truth has no
        # signal past ~4 A, and for this cell only 1.6% of the 37,877 unique
        # reflections lie beyond 8 A -- so summing to 2.0 A drowns the usable
        # shells in noise.  It also cuts the (n_atoms x n_hkl) intermediate that
        # dominates peak memory by ~50x.
        # The window applies to the GUIDED loss only.  R_work/R_free are always
        # reported on the full resolution range, because a windowed R is (a) not
        # comparable across runs and (b) statistically useless as R_free: beyond
        # 8 A this cell has ~316 unique reflections, so a 5% free set is ~16
        # of them.
        d_high = getattr(self, "guidance_d_high", None)
        if d_high is None:
            mask_guide_res = jnp.ones_like(mask_valid)
        else:
            d_spacing = 1.0 / jnp.sqrt(jnp.maximum(dr2_tensor, 1e-12))
            mask_guide_res = d_spacing >= d_high

        mask_free = mask_valid & self.freer_mask
        mask_work = mask_valid & (~self.freer_mask)
        mask_loss = mask_work & mask_guide_res
        
        f_obs = jnp.where(mask_valid, f_obs, 0.0)
        f_calc_mag = jnp.where(mask_valid, f_calc_mag, 0.0)
        
        # 5. Dynamic Linear Scaling -- fitted separately for each purpose.
        #
        # The scale MUST be fitted on the same reflections the loss is evaluated
        # on.  A single global scale fitted to 2.0 A data and then applied to the
        # d >= 8 A subset is badly wrong, because bulk solvent makes the
        # effective scale strongly resolution dependent.  Symptom: the normalised
        # loss came out at 1.85-2.18, which is impossible for an LSQ-optimal
        # scale -- that quantity is 1 - CC^2 and so bounded by 1.  Values above 1
        # mean the scale is mis-fitted for the subset, which inflates the loss and
        # injects a gradient that chases the scale error rather than the
        # structure.
        # An overall B belongs in the scaling.  A single linear k has no
        # resolution dependence, so any mismatch between the model's B-factors
        # and the data's falls straight through into R.  With a mean B of ~11 A^2
        # against 2.0 A data (20-40 would be typical), a dB of 15 A^2 costs a
        # factor exp(-15*s^2/4) = 0.39 at 2.0 A that k cannot absorb.  It also
        # explains k_sol pinning at its grid boundary run after run: with no B in
        # the scale, the solvent parameters are the only place a resolution-
        # dependent error can go, so the fit abuses them.
        fit_b = bool(getattr(self, "fit_overall_b", True))
        b_grid = jnp.asarray(getattr(self, "overall_b_grid", None)
                             if getattr(self, "overall_b_grid", None) is not None
                             else jnp.linspace(-30.0, 150.0, 61))

        def _lsq_scale_b(mask):
            """LSQ-optimal (k, B) for f_obs ~ k * exp(-B s^2/4) * f_calc_mag."""
            if not fit_b:
                num = jnp.sum(jnp.where(mask, f_obs * f_calc_mag, 0.0))
                den = jnp.sum(jnp.where(mask, f_calc_mag ** 2, 0.0)) + 1e-8
                return num / den, jnp.zeros(())

            def resid_for(b):
                fc = f_calc_mag * jnp.exp(-b * dr2_tensor / 4.0)
                num = jnp.sum(jnp.where(mask, f_obs * fc, 0.0))
                den = jnp.sum(jnp.where(mask, fc ** 2, 0.0)) + 1e-8
                k = num / den
                r = jnp.sum(jnp.where(mask, (f_obs - k * fc) ** 2, 0.0))
                return r, k

            resid, ks = jax.vmap(resid_for)(b_grid)
            i = jnp.argmin(resid)
            return ks[i], b_grid[i]

        k_report, b_report = _lsq_scale_b(mask_work)   # for the reported R
        k_loss, b_loss = _lsq_scale_b(mask_loss)       # for the guided loss
        scale_factor = k_report * jnp.exp(-b_report * dr2_tensor / 4.0)
        scale_loss = k_loss * jnp.exp(-b_loss * dr2_tensor / 4.0)

        # 6. Monitor R-Factors (full resolution range, own scale)
        diff = jnp.abs(f_obs - scale_factor * f_calc_mag)
        r_work = jnp.sum(jnp.where(mask_work, diff, 0.0)) / (jnp.sum(jnp.where(mask_work, f_obs, 0.0)) + 1e-8)
        r_free = jnp.sum(jnp.where(mask_free, diff, 0.0)) / (jnp.sum(jnp.where(mask_free, f_obs, 0.0)) + 1e-8)
        
        # 7. Normalize Loss (windowed)
        residuals_sq = jnp.where(mask_loss, (f_obs - scale_loss * f_calc_mag) ** 2, 0.0)
        normalization = jnp.sum(jnp.where(mask_loss, f_obs ** 2, 0.0)) + 1e-8
        normalized_loss = jnp.sum(residuals_sq) / normalization
        
        return normalized_loss, (r_work, r_free)



def _rodrigues(omega):
    """so(3) -> SO(3), written analytic in u = |omega|^2.

    The textbook form normalises by |omega|, whose derivative is NaN at the
    origin, and jnp.where does not save you because JAX evaluates both branches.
    Since the pose is refined from the identity, that NaN would land on the very
    first step.  Here A(u) = sin(sqrt u)/sqrt u and B(u) = (1-cos sqrt u)/u are
    both even analytic functions of the angle, so the small-angle branch is a
    Taylor series in u and the large-angle branch never sees u = 0.
    """
    u = jnp.sum(omega * omega)
    tol = 1e-8
    u_safe = jnp.where(u < tol, jnp.ones_like(u), u)
    th = jnp.sqrt(u_safe)
    A = jnp.where(u < tol, 1.0 - u / 6.0 + u * u / 120.0, jnp.sin(th) / th)
    B = jnp.where(u < tol, 0.5 - u / 24.0 + u * u / 720.0, (1.0 - jnp.cos(th)) / u_safe)
    wx, wy, wz = omega[0], omega[1], omega[2]
    W = jnp.array([[0.0, -wz, wy], [wz, 0.0, -wx], [-wy, wx, 0.0]])
    return jnp.eye(3) + A * W + B * (W @ W)


def refine_rigid_pose(sfc, xyz, n_steps: int = 40, lr_rot: float = 2e-3,
                      lr_trans: float = 2e-2):
    """Refine a 6-DOF rigid pose of ``xyz`` against F_obs and return moved coords.

    Why this is needed: the guidance operator recovers the crystal frame by
    Kabsch-superposing the current x_0 onto ``initial_coordinates`` at every
    step.  The round trip is exact, but it means any *collective rigid* component
    of the correction is removed again before F_calc is evaluated -- the gradient
    can rearrange atoms within the frame, but the frame itself is pinned to the
    initial model forever.  So a starting pose that is rigidly wrong stays wrong
    no matter how long you sample.

    Measured: aligning the same 0.44 A fold to 4BD0 (X-ray) rather than 4BD1
    (neutron, same crystal as the data) costs R_work 0.339 -> 0.489.  That is
    ~0.15 in R sitting in 6 parameters the sampler currently cannot touch -- for
    comparison, fitting an overall B is worth ~0.01.  There is plenty of signal;
    the operator simply had no way to express the correction.

    CAPTURE RANGE -- read this before using it.  Adam moves about ``lr`` per step,
    so this reaches at most ``lr_rot * n_steps`` in rotation and
    ``lr_trans * n_steps`` in translation: with the defaults, 0.08 rad (4.6 deg)
    and 0.8 A.  That is a *local polish*, nothing more.  It cannot cross an origin
    mismatch: in P3(2)21 the allowed origin shifts along c are multiples of
    c/3 = 33 A, and the 3-fold relates positions ~42 A apart in the ab plane.  If
    two references disagree about the origin, use
    :func:`search_rigid_placement` first -- a local gradient will never find it,
    and running this per step instead just makes the trajectory slow without
    moving the frame.

    Stateless by design: it re-refines from the identity on each call, which is
    fine because the frame error is a fixed offset, and it avoids threading pose
    state through the scan carry.  It is also expensive -- one ``compute_loss``
    per inner step, per proximal step, per particle, per level -- so keep
    ``n_steps`` small and pair it with :func:`make_low_resolution_sfc`.

    Insert into ``proximal_operator_fn`` immediately before ``compute_loss``::

        X_rel = refine_rigid_pose(sfc_instance, X_rel)
        e_exp, (rw, rf) = sfc_instance.compute_loss(X_rel)
    """
    centre = jnp.mean(xyz, axis=0)

    def moved(params):
        omega, tau = params[:3], params[3:]
        return (xyz - centre) @ _rodrigues(omega).T + centre + tau

    def loss(params):
        return sfc.compute_loss(moved(params))[0]

    grad_fn = jax.value_and_grad(loss)
    lrs = jnp.concatenate([jnp.full((3,), lr_rot), jnp.full((3,), lr_trans)])

    def body(carry, i):
        p, m, v = carry
        _, g = grad_fn(p)
        i1 = i + 1.0
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g**2
        p = p - lrs * (m / (1 - 0.9**i1)) / (jnp.sqrt(v / (1 - 0.999**i1)) + 1e-8)
        return (p, m, v), None

    z = jnp.zeros((6,))
    (p, _, _), _ = jax.lax.scan(body, (z, z, z), jnp.arange(n_steps, dtype=jnp.float32))
    return moved(p)



def search_rigid_placement(sfc, xyz, n_grid=(12, 12, 16), use_symops=False,
                           chunk=64, refine_steps=2, min_z=6.0, verbose=True):
    """Brute-force search for the rigid placement that best explains F_obs.

    This is the piece a gradient cannot supply.  Measured on this system:
    aligning the same 0.44 A fold to 4BD0 rather than 4BD1 costs R_work
    0.339 -> 0.489, i.e. ~0.15 of R living in six parameters -- but the offset
    between two origin conventions is tens of Angstrom, whereas a local
    refinement reaches under 1 A.  A search is the only thing that closes that
    gap, and it is cheap because it needs the likelihood *value* only, no
    gradient, and only the low-resolution shells.

    Run it ONCE after alignment, not inside the sampling loop.  Pair it with
    ``make_low_resolution_sfc(sfc, 8.0)``: at 544 reflections a 6 x 12 x 12 x 16
    search is a few thousand cheap evaluations, against one full-resolution
    gradient.

    The coarse grid is followed by ``refine_steps`` local passes, each a fine grid
    over one previous cell width.  This is not a refinement of convenience: a
    12-division grid on this cell has 6.1 A spacing, so it localises the answer to
    only ~3 A, while :func:`refine_rigid_pose` captures 0.8 A.  Without the local
    passes there is a gap between the two stages that neither can cross.  Two
    passes take 6.1 -> 0.5 -> 0.04 A, comfortably inside the gradient's reach.

    ``use_symops`` defaults to False, and should normally stay there.
    ``compute_loss`` already expands the ASU by the full space group, so
    pre-applying one operator to the ASU generates the same crystal: the set
    {s.(op.x + t)} equals {s'.x + s'.(op.t)} with s' = s.op, so the operator is
    absorbed into a relabelling of the translation.  As long as the translation
    grid spans the cell, searching operators separately only multiplies the cost.

    ACCEPTANCE IS BY Z-SCORE, not by improvement.  Taking the best of ~2300 grid
    points from a flat landscape produces an apparent improvement even when there
    is no signal, and the low-resolution translation landscape *is* fairly flat
    because low-resolution amplitudes constrain the molecular envelope rather than
    its position.  Observed failure: an 8% improvement (0.4151 -> 0.3821) selected
    a 61 A shift in x on a structure whose solvent fraction was already healthy at
    0.573, i.e. a placement that was not wrong.  ``info["z"]`` is
    ``(mean - best) / std`` over the coarse grid; molecular replacement convention
    (Phaser TFZ) wants > 8 for a confident solution, and ``min_z`` defaults to 6.
    Below that the caller should reject.

    Returns ``(xyz_best, info)`` with the symop index, fractional shift, loss,
    identity loss, the grid score statistics, ``z``, and ``accept``.
    """
    import itertools

    import numpy as _np

    frac_shifts = _np.array(
        list(
            itertools.product(
                _np.arange(n_grid[0]) / n_grid[0],
                _np.arange(n_grid[1]) / n_grid[1],
                _np.arange(n_grid[2]) / n_grid[2],
            )
        )
    )

    R_stack = _np.asarray(sfc.R_G_tensor_stack)
    T_stack = _np.asarray(sfc.T_G_tensor_stack)
    n_ops = R_stack.shape[0] if use_symops else 1

    orth2frac = jnp.asarray(sfc.orth2frac_tensor)
    frac2orth = jnp.linalg.inv(orth2frac)
    xyz_j = jnp.asarray(xyz)
    frac0 = xyz_j @ orth2frac.T

    # The symmetry operator must NOT be a traced argument: indexing a numpy
    # array with a tracer raises TracerArrayConversionError.  Apply the operator
    # outside jit (it does not depend on the shift) and pass the resulting
    # fractional coordinates in as an array.  Shape is fixed across operators, so
    # this compiles once and is reused for all of them.
    def score(frac_op, shift):
        return sfc.compute_loss((frac_op + shift) @ frac2orth.T)[0]

    # Sequential, NOT vmapped.  A vmap over `chunk` shifts multiplies the whole
    # structure-factor computation by `chunk`: with chunk=64 that requested
    # 122 GiB (1.9 GiB per shift).  lax.map traces once and runs one shift at a
    # time, so peak memory is that of a single evaluation.  The search is a
    # one-off setup step, so paying for it in time is the right trade.
    def scored(frac_op, shifts):
        return jax.lax.map(lambda sh: score(frac_op, sh), shifts)

    scored = jax.jit(scored)

    best = (0, jnp.zeros(3), float("inf"))
    ident = float(sfc.compute_loss(xyz_j)[0])
    all_scores = []
    for op in range(n_ops):
        frac_op = frac0 @ jnp.asarray(R_stack[op]).T + jnp.asarray(T_stack[op])
        vals = []
        for i in range(0, len(frac_shifts), chunk):
            vals.append(scored(frac_op, jnp.asarray(frac_shifts[i : i + chunk])))
        vals = jnp.concatenate(vals)
        all_scores.append(_np.asarray(vals))
        i = int(jnp.argmin(vals))
        if float(vals[i]) < best[2]:
            best = (op, jnp.asarray(frac_shifts[i]), float(vals[i]))
        if verbose:
            print(f"  symop {op}: best loss {float(vals[i]):.4f} at frac "
                  f"{frac_shifts[i].round(3)}", file=sys.stderr)

    grid = _np.concatenate(all_scores)
    g_mean, g_std = float(grid.mean()), float(grid.std())
    z = (g_mean - best[2]) / max(g_std, 1e-12)

    # Local passes: re-grid over one previous cell width around the winner.
    op, shift, loss = best
    span = _np.array([1.0 / n_grid[0], 1.0 / n_grid[1], 1.0 / n_grid[2]])
    frac_op = frac0 @ jnp.asarray(R_stack[op]).T + jnp.asarray(T_stack[op])
    for _ in range(max(int(refine_steps), 0)):
        axes = [_np.linspace(-span[k], span[k], 7) + float(shift[k]) for k in range(3)]
        local = _np.array(list(itertools.product(*axes)))
        vals = []
        for i in range(0, len(local), chunk):
            vals.append(scored(frac_op, jnp.asarray(local[i : i + chunk])))
        vals = jnp.concatenate(vals)
        i = int(jnp.argmin(vals))
        if float(vals[i]) < loss:
            shift, loss = jnp.asarray(local[i]), float(vals[i])
        span = span / 6.0
        if verbose:
            print(f"  local pass: loss {loss:.4f} at frac "
                  f"{_np.asarray(shift).round(4)}", file=sys.stderr)

    if verbose:
        print(
            f"Rigid placement search: identity loss {ident:.4f} -> {loss:.4f} "
            f"(symop {op}, fractional shift {_np.asarray(shift).round(4)})",
            file=sys.stderr,
        )
        print(
            f"  grid scores: mean {g_mean:.4f} std {g_std:.4f} -> Z = {z:.2f} "
            f"({'ACCEPT' if z >= min_z else 'REJECT'}, threshold {min_z})",
            file=sys.stderr,
        )
        if z < min_z:
            print(
                "  Z below threshold: the best placement is not distinguishable "
                "from the best of many random ones.  Rejecting -- either the "
                "placement was already right, or the low-resolution landscape is "
                "too flat to localise it.",
                file=sys.stderr,
            )

    frac_best = frac_op + shift
    return (frac_best @ frac2orth.T), {
        "symop": op,
        "frac_shift": _np.asarray(shift),
        "loss": loss,
        "identity_loss": ident,
        "grid_mean": g_mean,
        "grid_std": g_std,
        "z": z,
        "accept": bool(z >= min_z),
    }


def make_low_resolution_sfc(sfc, d_high: float, verbose: bool = True):
    """A copy of ``sfc`` whose *reflection list* is truncated to ``d >= d_high``.

    ``guidance_d_high`` masks reflections after ``F_protein`` has run, so it buys
    the statistical benefit but none of the memory: the (n_atoms x n_hkl) phase
    array -- which is what dominates peak memory, and is held live for the
    backward pass -- is still built at full size.  Truncating the list itself is
    what pays: for a 73x73x99 cell at 2.0 A there are ~37,900 unique reflections
    but only 544 beyond 8 A (measured), a factor of ~70 on that term.

    Use this for the guided loss and keep the full ``sfc`` for reporting
    R_work/R_free.  That is also the way to raise ``num_diffusion_samples``
    without hitting the ceiling: the per-particle cost of the guidance gradient
    is what scales with the particle count, and this is the term to shrink.

    LIMITATION -- this truncates the HKL side only.  ``Calc_Fprotein`` runs over
    ``Hasu_array``/``dr2asu_array`` and then maps into the HKL list via
    ``asu2HKL_index``, and those ASU-side arrays have a different leading
    dimension, so the leading-axis heuristic below does not match them.  The
    result is a correct but *not cheaper* calculation: the expensive
    [N_asu x N_atoms] intermediates are still full size.  Truncating the ASU side
    as well requires remapping ``asu2HKL_index`` onto the reduced ASU list, which
    is the real memory lever and is not implemented here.  Until it is, do not
    rely on this for memory -- use it for the statistical benefit and control
    memory with the particle count and ``unroll``.

    Attributes are sliced by their leading axis, so this is robust to SFC_Jax
    gaining or losing per-reflection fields; anything whose first dimension does
    not match the HKL count is passed through untouched.
    """
    import copy as _copy

    import numpy as _np

    dr2 = _np.asarray(sfc.dr2HKL_array)
    n_hkl = dr2.shape[0]
    d = _np.where(dr2 > 0, 1.0 / _np.sqrt(_np.maximum(dr2, 1e-12)), _np.inf)
    keep = d >= float(d_high)
    if keep.sum() == 0:
        raise ValueError(f"no reflections with d >= {d_high} A")

    out = _copy.copy(sfc)
    n_sliced = []
    for name in dir(sfc):
        if name.startswith("__"):
            continue
        try:
            val = getattr(sfc, name)
        except Exception:
            continue
        arr = getattr(val, "shape", None)
        if arr is None or len(val.shape) == 0 or val.shape[0] != n_hkl:
            continue
        try:
            setattr(out, name, val[keep])
            n_sliced.append(name)
        except Exception:
            pass

    # asu2HKL_index maps HKL rows into the ASU list; it is sliced above, and the
    # ASU-side arrays are left whole, which is correct -- only the HKL side shrinks.
    if verbose:
        print(
            f"Low-resolution guidance SFC: {int(keep.sum())}/{n_hkl} reflections "
            f"(d >= {d_high} A); sliced {sorted(n_sliced)}",
            file=sys.stderr,
        )
    return out


def align_oracle_to_reference(oracle, reference_path):
    """
    Aligns the unanchored AF3 oracle coordinates to an explicit absolute crystal 
    lattice frame (e.g., the exact deposited neutron structure) rather than the AF3 template.
    """
    if not os.path.exists(reference_path):
        print(f"WARNING: Reference file '{reference_path}' not found. Skipping lattice alignment.", file=sys.stderr)
        return oracle

    print(f"Aligning Oracle coordinates to explicit crystal reference: {reference_path}", file=sys.stderr)
    
    # Handle both CIF and PDB reference files
    if reference_path.endswith('.cif') or reference_path.endswith('.mmcif'):
        ref_file = pdbx.CIFFile.read(reference_path)
        try:
            ref_atoms = pdbx.get_structure(ref_file, model=1, extra_fields=["altloc_id"])
            has_altloc = True
        except KeyError:
            ref_atoms = pdbx.get_structure(ref_file, model=1)
            has_altloc = False
    else:
        ref_file = pdb.PDBFile.read(reference_path)
        ref_atoms = pdb.get_structure(ref_file, model=1)
        has_altloc = "altloc_id" in ref_atoms.get_annotation_categories()

    # Filter for CA atoms
    ca_mask = (ref_atoms.atom_name == "CA")
    
    # Safely avoid alternate locations if present
    if has_altloc:
        altloc_mask = (ref_atoms.altloc_id == "") | (ref_atoms.altloc_id == ".") | (ref_atoms.altloc_id == "A")
        ca_mask = ca_mask & altloc_mask
        
    ref_ca = ref_atoms[ca_mask]
    oracle_ca = oracle.atoms[oracle.atoms.atom_name == "CA"]
    
    # Pair CA atoms sequentially (ignores mismatched residue numbering)
    min_len = min(len(ref_ca), len(oracle_ca))
    
    if min_len < 10:
        print("WARNING: Not enough CA atoms for Kabsch alignment. Skipping.", file=sys.stderr)
        return oracle
        
    matched_ref = ref_ca.coord[:min_len]
    matched_oracle = oracle_ca.coord[:min_len]
    
    # Biotite's superimpose returns the fitted coordinates and the AffineTransformation object
    fitted_coords, transform = struc.superimpose(matched_ref, matched_oracle)
    rmsd = struc.rmsd(matched_ref, fitted_coords)
    
    # Apply the AffineTransformation directly to the mobile Oracle coordinates
    aligned_coords = transform.apply(oracle.atoms.coord)
    oracle.atoms.coord = aligned_coords
    
    # Update the JAX mapping coordinates so the diffusion target is aligned
    oracle.mapping = dataclasses.replace(
        oracle.mapping, 
        initial_coordinates=jnp.array(aligned_coords, dtype=jnp.float32)
    )
    
    print(f"Successfully aligned Oracle to reference. (CA RMSD: {rmsd:.3f}A)", file=sys.stderr)
    return oracle


def init_neutron_sfc(oracle_atoms, mtz_path, deuterate=False, perdeuterate=False):
    with tempfile.TemporaryDirectory() as tmpdir:
        pdb_path = os.path.join(tmpdir, "oracle.pdb")
        pdb_file = pdb.PDBFile()
        
        # Clone oracle atoms so Hydride's internal "H" reliance isn't broken
        sfc_atoms = oracle_atoms.copy()
        
        # ==============================================================================
        # SANITIZE B-FACTORS FOR PDB FORMAT COMPATIBILITY
        # ==============================================================================
        # De-novo AF3 predictions with low pLDDT can generate B-factors >= 1000.0 A^2.
        # Clip to [0.0, 999.0] to fit Biotite's 3-pre-decimal digit PDB limit (F6.2).
        safe_b_factors = np.nan_to_num(sfc_atoms.b_factor, nan=30.0)
        sfc_atoms.b_factor = np.clip(safe_b_factors, 0.0, 999.0)
        
        # ==============================================================================
        # SIMULATE NEUTRON ISOTOPIC COMPOSITION (H/D EXCHANGE VS PERDEUTERATION)
        # ==============================================================================
        if perdeuterate:
            print("Enforcing PERDEUTERATION (All H -> D) for structure factor evaluation...", file=sys.stderr)
            h_mask = (sfc_atoms.element == "H")
            sfc_atoms.element[h_mask] = "D"
            for i in np.where(h_mask)[0]:
                sfc_atoms.atom_name[i] = "D" + sfc_atoms.atom_name[i][1:]

        elif deuterate:
            print("Simulating D2O H/D exchange (labile N/O/S-H -> D) for structure factor evaluation...", file=sys.stderr)
            h_mask = (sfc_atoms.element == "H")
            for i in np.where(h_mask)[0]:
                bonded_indices = sfc_atoms.bonds.get_bonds(i)[0]
                for neighbor in bonded_indices:
                    if sfc_atoms.element[neighbor] in ["N", "O", "S"]:
                        sfc_atoms.element[i] = "D"
                        sfc_atoms.atom_name[i] = "D" + sfc_atoms.atom_name[i][1:]
                        break
        
        sfc_atoms.res_name = np.array([name[:3] for name in sfc_atoms.res_name])
        pdb.set_structure(pdb_file, sfc_atoms)
        pdb_file.write(pdb_path)

        # ==============================================================================
        # FORCE ROBUST FLAGS USING RECIPROCALSPACESHIP
        # ==============================================================================
        mtz_rs = rs.read_mtz(mtz_path)
        
        print("Forcing robust 5% Free-R holdout set...", file=sys.stderr)
        
        # Generate reproducible random flags: 0 for Free (5%), 1 for Work (95%)
        np.random.seed(42)
        fresh_flags = np.random.choice([0, 1], size=len(mtz_rs), p=[0.05, 0.95])
        mtz_rs["FreeR_flag"] = rs.DataSeries(fresh_flags, dtype="I")
        
        working_mtz_path = os.path.join(tmpdir, "working_data.mtz")
        mtz_rs.write_mtz(working_mtz_path)
        
        # Read headers safely via Gemmi for the PDB CRYST1 line
        mtz_gemmi = gemmi.read_mtz_file(mtz_path)
        cell = mtz_gemmi.cell
        sg_name = mtz_gemmi.spacegroup_name
        dmin_val = mtz_gemmi.resolution_high()
        
        cryst1_line = (
            f"CRYST1{cell.a:9.3f}{cell.b:9.3f}{cell.c:9.3f}"
            f"{cell.alpha:7.2f}{cell.beta:7.2f}{cell.gamma:7.2f} "
            f"{sg_name:<11}\n"
        )
        
        with open(pdb_path, "r") as f:
            pdb_content = f.read()
        with open(pdb_path, "w") as f:
            f.write(cryst1_line + pdb_content)
            
        print(f"Injected Symmetry Header: {cryst1_line.strip()} | Dmin Limit: {dmin_val:.3f}A", file=sys.stderr)
            
        # Explicitly thread the pLDDT-derived B-factors into the constructor
        sfc = NeutronSFCalculator(
            PDBfile_dir=pdb_path,
            mtzfile_dir=working_mtz_path,
            dmin=dmin_val,
            set_experiment=True,
            freeflag="FreeR_flag",
            mode="neutron",
            b_factors=sfc_atoms.b_factor,
            occupancies=np.ones(sfc_atoms.array_length(), dtype=np.float32)
        )
        
        # EXPLICITLY OVERRIDE EXPERIMENTAL DATA
        sfc.Fo = jnp.array(mtz_rs["FP"].to_numpy(), dtype=jnp.float32)
        sfc.SigF = jnp.array(mtz_rs["SIGFP"].to_numpy(), dtype=jnp.float32)
        sfc.freer_mask = jnp.array(fresh_flags == 0, dtype=bool)

        print("Initializing Baseline Bulk Solvent Mask...", file=sys.stderr)
        sfc.inspect_data()
        sfc.Calc_Fprotein(jnp.array(sfc_atoms.coord))
        
        # Generate the solvent mask grid dynamically
        sfc.Calc_Fsolvent()
        sfc.deuterated_solvent = (deuterate or perdeuterate)
        
        num_free = int(np.sum(fresh_flags == 0))
        num_work = len(fresh_flags) - num_free
        print(f"Cross-Validation Split | Work: {num_work} | Free: {num_free}", file=sys.stderr)
        
        return sfc
