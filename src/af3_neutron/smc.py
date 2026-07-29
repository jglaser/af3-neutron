"""Drop-in replacement for ``alphafold3...diffusion_head.sample`` adding SMC.

Same signature, same defaults, same numerics.  With ``smc_config=None`` (or
``lambda_max=0``) this is byte-for-byte the stock AF3 sampler: the inner
``apply_denoising_step`` is copied verbatim, and ``random_augmentation`` /
``noise_schedule`` are imported from AF3 rather than reimplemented, so there is
nothing to drift out of sync.

Why selection rather than a bigger gradient
-------------------------------------------
DPS-style guidance is a local gradient in x_0 space; it does not tunnel.  The
barrier crossing comes from the *prior*, which at high sigma visits different
basins on different noise draws.  The data's realistic job is to **select among
the basins the prior offers** -- and selection needs only the likelihood
*value*, not its gradient.  That matters because at sigma ~ 10 A the crystal
frame recovered by superposition is noisy enough that any gradient is
meaningless, while the likelihood value on the denoised estimate still is not.

Two things make this cheap to bolt on here:

* ``config.num_samples`` is already the particle axis -- AF3 vmaps the step over
  it and scans.  Set ``num_samples`` to 16-64 and you have an SMC ensemble.
* the guidance operator already evaluates ``sfc_instance.compute_loss`` once per
  step for its ``R_free`` log line.  Returning that scalar alongside x_0 is the
  only interface change needed.

So ``denoising_step`` may now return either ``x_0`` (stock behaviour) or
``(x_0, V)`` where ``V`` is a scalar crystallographic negative log likelihood.

Scaling caveat worth knowing before tuning
------------------------------------------
The log-likelihood ratio between two basins of the prior is *extensive* in the
number of atoms that differ, going as ``N d^2 / sigma^2``.  So the noise level
at which the prior will consider a different fold scales as ``d sqrt(N)``.  For
a whole 2000-atom protein and a 3 A error that is tens of Angstrom -- past the
point where the pose survives.  For a 40-atom loop it is a few Angstrom.
Selection helps most when the alternative folds differ over a *localised*
region; it will not rescue a globally wrong model.
"""

from __future__ import annotations

import dataclasses
import inspect
from typing import Any, Callable, Optional, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

# AF3 is imported lazily so that the pure SMC helpers (SMCConfig, lambda_ramp,
# systematic_resample, effective_sample_size) can be imported and unit-tested in
# an environment without alphafold3 installed.  There is deliberately NO
# fallback implementation: `sample` needs AF3's own `random_augmentation` and
# `noise_schedule`, and reimplementing them here -- or in a test double -- would
# mean the parity test compares a copy against a copy.  If AF3 is missing,
# `sample` raises.
_af3_import_error = None
try:
    from alphafold3.model.network import diffusion_head as _diffusion_head
except ImportError as exc:  # pragma: no cover
    # NB: bind to a module-level name explicitly -- Python deletes the
    # `except ... as exc` target at the end of the block.
    _diffusion_head = None
    _af3_import_error = exc


def _af3():
    """The real AF3 diffusion_head module, or a clear error."""
    if _diffusion_head is None:
        raise ImportError(
            "af3_neutron.smc.sample requires alphafold3; the pure SMC helpers "
            "are importable without it."
        ) from _af3_import_error
    return _diffusion_head


def random_augmentation(*args, **kwargs):
    """AF3's own `random_augmentation`; never reimplemented here."""
    return _af3().random_augmentation(*args, **kwargs)


def noise_schedule(*args, **kwargs):
    """AF3's own `noise_schedule`; never reimplemented here."""
    return _af3().noise_schedule(*args, **kwargs)


@dataclasses.dataclass(frozen=True)
class SMCConfig:
    """Selection parameters.  ``lambda_max=0`` disables SMC entirely.

    Attributes
    ----------
    lambda_max
        Inverse temperature on the potential.  Start around 1-10 and tune by
        watching ESS: if it collapses to ~1 immediately, lambda is too high and
        the ensemble degenerates to a single particle; if it never drops below
        the threshold, the potential is not discriminating and lambda is too low
        (or the data cannot tell the folds apart at all -- see the module note
        on sqrt(N) scaling).
    sigma_on, sigma_width
        Logistic ramp on sigma: ``lambda(sigma) = lambda_max *
        sigmoid((sigma_on - sigma) / sigma_width)``.  Defaults turn selection on
        around 12 A, which is roughly where a localised fold error melts.  Note
        this is deliberately *earlier* than the existing gradient weight
        ``sfc_weight * exp(-t_hat)``, which is numerically dead until sigma < 3 A
        (8e-9 at 18.6 A, 6e-3 at 5.1 A) and therefore only ever acts after the
        topology has committed.  The two are independent knobs.
    ess_threshold
        Resample when ``ESS/num_samples`` falls below this.
    lambda_mode
        ``"fixed"`` uses ``lambda_max`` with the sigma ramp.  ``"adaptive_ess"``
        instead solves, at every level, for the lambda that puts ESS at
        ``ess_target * num_samples``.

        Use adaptive mode only once the potential is known to carry signal.  A
        fixed lambda has to be matched to the *inter-particle spread* of V, not
        to V itself, and that spread is not knowable in advance: in a run with
        ``lambda_max=1`` on a normalised LSQ residual, V_mean was 0.9395 and
        V_best 0.9371, so the log-weight spread was 0.88 * 0.0024 = 0.002 and ESS
        sat at exactly 8.0/8 for all 200 levels -- zero resampling, SMC a no-op.
        Reaching an ESS of 4/8 there would have needed lambda ~ 400.

        But cranking lambda to 400 would have been wrong, and so would adaptive
        mode: if the spread in V is noise rather than signal, adaptive tempering
        will faithfully amplify the noise until ESS hits the target and then
        resample on it.  Make V discriminating first (see ``guidance_d_high`` in
        the sfc adapter), then let adaptive mode set the scale.
    sigma_end
        Stop the schedule at this sigma instead of running to zero.  Together
        with ``sigma_start`` this lets the trajectory be run in *segments* from
        the host, recomputing the bulk-solvent mask (and rescaling k_sol/b_sol)
        between them -- which is the practical answer to a stale mask, since
        ``Calc_Fsolvent`` is host-side numpy/gemmi and cannot run inside the
        jitted loop.  Four segments of 50 levels is usually plenty: the mask is
        insensitive to sub-Angstrom conformational change, and only needs
        revisiting when the model has moved appreciably.
    unroll
        ``hk.scan`` unroll factor.  AF3 uses 4, which is fine when the step is
        just the network.  Here each step also carries the guidance operator and
        its gradient, so 4 live copies multiply peak memory by ~4x.  Default 1.
    lambda_floor
        Never resample while ``lambda(sigma)`` is below this.  Resampling on an
        uninformative potential is not merely useless, it is harmful: the
        ensemble locks onto whichever particle was transiently lucky, and since
        the only later source of divergence is the churn noise (which is small
        at low sigma) the collapse is permanent.  In a single-basin test where V
        carries no signal, enabling selection early made the final potential
        *worse* than unguided (196.6 vs a best-of-ensemble 108.7).  This is why
        the ESS log line matters and why the melting/likelihood-gap diagnostic
        should be run before turning selection on at all.
    sigma_start
        If set, the schedule is *reparametrised* over ``[t0, 1]`` so that it
        begins at this sigma, and ``x_start`` is noised to it (SDEdit).  Without
        a warm start the trajectory begins at sigma_max = 16 * 160 = 2560 A,
        where the crystal frame recovered by superposition is an arbitrary
        rotation -- so F_c, its gradient, and the potential are all noise, and
        selection has nothing to select on.  Reparametrising rather than
        truncating matters: the EDM update propagates x_0 with weight
        ``a = step_scale * |dsigma| / sigma``, and ``a > 1`` over-relaxes and
        diverges.  Truncating a coarse schedule at 2 A gives jumps of 8 -> 0.5 A,
        i.e. ``a = 1.4``.  Reparametrising keeps ``a`` at 0.24-0.38.
    augment
        Leave True for AF3.  Only set False for analytic/mock denoisers that are
        not equivariant -- AF3's test-time augmentation is harmless only because
        the network is approximately equivariant, having been trained with it.
    verbose
        Print sigma / lambda / V / ESS per level via ``jax.debug.print``,
        matching the existing logging style.
    """

    lambda_max: float = 0.0
    sigma_on: float = 12.0
    sigma_width: float = 6.0
    ess_threshold: float = 0.5
    lambda_floor: float = 0.05
    lambda_mode: str = "fixed"      # "fixed" | "adaptive_ess"
    ess_target: float = 0.5         # for adaptive_ess: target ESS/num_samples
    lambda_cap: float = 1.0e4       # ceiling on the adaptive solve
    unroll: int = 1                 # hk.scan unroll; AF3 uses 4
    sigma_start: Optional[float] = None
    sigma_end: Optional[float] = None
    augment: bool = True
    verbose: bool = True


def lambda_ramp(sigma: jnp.ndarray, cfg: SMCConfig) -> jnp.ndarray:
    return cfg.lambda_max * jax.nn.sigmoid((cfg.sigma_on - sigma) / cfg.sigma_width)


def _ess_from_centred(lam: jnp.ndarray, v_centred: jnp.ndarray) -> jnp.ndarray:
    lw = -lam * v_centred
    lw = lw - jax.scipy.special.logsumexp(lw)
    return 1.0 / jnp.maximum(jnp.sum(jnp.exp(2.0 * lw)), 1e-12)


def adaptive_lambda(
    potential: jnp.ndarray, cfg: SMCConfig, n_particles: int
) -> jnp.ndarray:
    """Smallest lambda whose weights give ``ess_target * n_particles``.

    ESS is monotonically decreasing in lambda, so a fixed-iteration bisection is
    exact enough and jit-friendly.  Scale-free: it does not matter whether the
    potential is a normalised residual, an NLL, or an R-factor.

    Returns ``lambda_cap`` if even that cannot reach the target -- which is
    itself the useful signal, and is logged, because it means the ensemble's
    spread in V is too small to select on.
    """
    v = potential - jnp.mean(potential)
    target = cfg.ess_target * n_particles
    lo = jnp.asarray(0.0)
    hi = jnp.asarray(cfg.lambda_cap)

    def body(_, bounds):
        lo, hi = bounds
        mid = 0.5 * (lo + hi)
        too_flat = _ess_from_centred(mid, v) > target
        return (jnp.where(too_flat, mid, lo), jnp.where(too_flat, hi, mid))

    lo, hi = jax.lax.fori_loop(0, 40, body, (lo, hi))
    return 0.5 * (lo + hi)


def systematic_resample(key: jnp.ndarray, log_w: jnp.ndarray) -> jnp.ndarray:
    """Low-variance resampling indices from normalised log weights."""
    p = jax.nn.softmax(log_w)
    n = log_w.shape[0]
    cdf = jnp.cumsum(p)
    cdf = cdf / cdf[-1]
    u = jax.random.uniform(key) / n + jnp.arange(n) / n
    return jnp.searchsorted(cdf, u, side="left").astype(jnp.int32)


def effective_sample_size(log_w: jnp.ndarray) -> jnp.ndarray:
    p = jax.nn.softmax(log_w)
    return 1.0 / jnp.maximum(jnp.sum(p**2), 1e-12)


def _schedule_defaults():
    """AF3's own schedule constants, read from its function signature.

    Not hardcoded: ``smin``/``smax``/``p`` come from
    ``diffusion_head.noise_schedule``'s defaults and ``SIGMA_DATA`` from the
    module, so a retune upstream is picked up rather than silently ignored.
    ``tests/test_smc_sampler.py`` round-trips the inversion against AF3's own
    function to catch a signature change.
    """
    dh = _af3()
    params = inspect.signature(dh.noise_schedule).parameters
    try:
        smin = float(params["smin"].default)
        smax = float(params["smax"].default)
        p = float(params["p"].default)
    except (KeyError, TypeError) as exc:  # pragma: no cover
        raise RuntimeError(
            "alphafold3's noise_schedule no longer exposes smin/smax/p defaults; "
            "af3_neutron.smc.build_schedule needs updating"
        ) from exc
    return float(dh.SIGMA_DATA), smin, smax, p


def build_schedule(
    steps: int,
    sigma_start: Optional[float] = None,
    sigma_end: Optional[float] = None,
) -> jnp.ndarray:
    """AF3's noise schedule, optionally reparametrised to begin at ``sigma_start``.

    The inversion is done in plain Python/numpy, never with jnp ops.  Inside a
    ``jax.jit`` trace every jnp operation returns a tracer even when its inputs
    are Python floats -- constant folding happens in XLA, not at trace time -- so
    an earlier bisection that called ``float(noise_schedule(mid))`` raised
    ``ConcretizationTypeError``.  ``sigma_start`` is a static field of
    ``SMCConfig``, so resolving ``t0`` at trace time in numpy is both legal and
    free.

    Reparametrising over ``[t0, 1]`` rather than truncating the full grid
    matters: the EDM update propagates x_0 with weight
    ``a = step_scale * |dsigma| / sigma``, and ``a > 1`` over-relaxes and
    diverges.  Truncating a coarse schedule at 2 A leaves jumps of 8 -> 0.5 A,
    i.e. ``a = 1.4``.
    """
    if sigma_start is None and sigma_end is None:
        return noise_schedule(jnp.linspace(0, 1, steps + 1))
    sd, smin, smax, p = _schedule_defaults()
    a = smax ** (1.0 / p)
    b = smin ** (1.0 / p)

    def t_of(sigma):
        return float(
            np.clip(((float(sigma) / sd) ** (1.0 / p) - a) / (b - a), 0.0, 1.0 - 1e-9)
        )

    t0 = 0.0 if sigma_start is None else t_of(sigma_start)
    t1 = 1.0 if sigma_end is None else t_of(sigma_end)
    return noise_schedule(jnp.linspace(t0, t1, steps + 1))


def superposed_rmsd(a: jnp.ndarray, b: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
    """Kabsch-superposed, mask-weighted r.m.s.d. between two atom sets.

    Superposition is required, not optional: every particle receives its own
    random augmentation each step, so two particles live in different rigid
    frames and a raw coordinate r.m.s.d. between them measures the frames, not
    the structures.
    """
    n = jnp.maximum(jnp.sum(w), 1.0)
    ca = jnp.sum(a * w[:, None], axis=0) / n
    cb = jnp.sum(b * w[:, None], axis=0) / n
    p = (a - ca) * w[:, None]
    q = (b - cb) * w[:, None]
    h = p.T @ q
    u, _, vt = jnp.linalg.svd(h)
    d = jnp.sign(jnp.linalg.det(u) * jnp.linalg.det(vt))
    r = u @ jnp.diag(jnp.stack([jnp.ones(()), jnp.ones(()), d])) @ vt
    diff = ((a - ca) @ r) - (b - cb)
    return jnp.sqrt(jnp.sum(w * jnp.sum(diff**2, axis=-1)) / n)


def ensemble_diversity(positions: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    """Mean superposed pairwise r.m.s.d. across the particle ensemble.

    This is the other half of the V_spread diagnostic, and without it V_spread is
    uninterpretable.  A small V_spread has two opposite explanations:

    * diversity small too -> the ensemble has collapsed; the particles are the
      same structure, so there is nothing to select among and no lambda helps.
      Remedy: more diversity (higher sigma_start, repaint cycles, weaker
      gradient -- gradient guidance is a variance-reducing force and works
      against selection).
    * diversity large -> the particles genuinely differ but the data cannot tell
      them apart; the likelihood is flat along the directions the ensemble
      explores.  Remedy: none at the sampler level.  Get the model closer first.
    """
    p = positions.shape[0]
    flat = positions.reshape(p, -1, 3)
    w = mask.reshape(-1).astype(positions.dtype)
    i, j = np.triu_indices(p, k=1)
    pairs = jax.vmap(lambda ii, jj: superposed_rmsd(flat[ii], flat[jj], w))(
        jnp.asarray(i), jnp.asarray(j)
    )
    return jnp.mean(pairs)


def sample(
    denoising_step: Callable[[jnp.ndarray, jnp.ndarray], Any],
    batch: Any,
    key: jnp.ndarray,
    config: Any,
    smc_config: Optional[SMCConfig] = None,
    x_start: Optional[jnp.ndarray] = None,
) -> dict:
    """As ``diffusion_head.sample``, with optional likelihood-weighted resampling.

    ``denoising_step`` returns either ``x_0`` or ``(x_0, V)``.  Returns the same
    dict as AF3 (``atom_positions``, ``mask``) plus ``log_weights`` and
    ``potential``, so existing call sites that index ``['atom_positions']``
    are unaffected.

    Particles are equally weighted after any resampling step; if the last level
    did not resample, use ``log_weights`` (or just take
    ``argmin(potential)`` -- for structure determination you want the best
    particle, not the posterior mean, which would average distinct folds into
    something unphysical).
    """
    mask = batch.predicted_structure_info.atom_mask
    cfg = smc_config if smc_config is not None else SMCConfig()
    num_samples = config.num_samples
    use_smc = cfg.lambda_max > 0.0

    # ---- verbatim from alphafold3.model.network.diffusion_head.sample ----
    def apply_denoising_step(carry, noise_level):
        key, positions, noise_level_prev = carry
        key, key_noise, key_aug = jax.random.split(key, 3)

        if cfg.augment:
            positions = random_augmentation(
                rng_key=key_aug, positions=positions, mask=mask
            )

        gamma = config.gamma_0 * (noise_level > config.gamma_min)
        t_hat = noise_level_prev * (1 + gamma)

        noise_scale = config.noise_scale * jnp.sqrt(
            jnp.maximum(t_hat**2 - noise_level_prev**2, 0.0)
        )
        noise = noise_scale * jax.random.normal(key_noise, positions.shape)
        positions_noisy = positions + noise

        out = denoising_step(positions_noisy, t_hat)
        if isinstance(out, tuple) and len(out) == 3:
            # (x0, V_work, V_monitor).  V_monitor is never used for weighting --
            # it exists so the rank agreement between the two can be checked.
            # If the work-set and free-set losses rank the particles the same
            # way, V_spread is signal; if they are uncorrelated, it is noise.
            positions_denoised, potential, potential_mon = out
        elif isinstance(out, tuple):
            positions_denoised, potential = out
            potential_mon = jnp.zeros((), positions.dtype)
        else:
            positions_denoised = out
            potential = jnp.zeros((), positions.dtype)
            potential_mon = jnp.zeros((), positions.dtype)

        grad = (positions_noisy - positions_denoised) / t_hat
        d_t = noise_level - t_hat
        positions_out = positions_noisy + config.step_scale * d_t * grad
        # ---- end verbatim ----

        return (key, positions_out, noise_level), (potential, potential_mon, t_hat)

    vstep = hk.vmap(
        apply_denoising_step, in_axes=(0, None), split_rng=(not hk.running_init())
    )

    def scan_body(carry, noise_level):
        particles, log_w, lam_v_prev, gkey = carry
        particles, (potential, potential_mon, t_hat) = vstep(particles, noise_level)

        # Instrumentation must not depend on selection being enabled.  V_spread is
        # the number that decides whether selection can work at all, so it has to
        # be observable with lambda_max = 0 -- otherwise measuring the potential
        # and enabling selection are the same experiment, which is how a run with
        # the resolution window on (lambda=0) produced no V_spread at all.
        if not use_smc:
            if cfg.verbose:
                jax.debug.print(
                    "sigma {s:8.3f} | lambda      0.000 | V_mean {vm:9.4f} "
                    "V_spread {vs:9.2e} | CC {cc:6.3f} | div {dv:7.3f} A | "
                    "ESS {e:5.1f}/{n} | resample 0",
                    s=jnp.mean(t_hat),
                    vm=jnp.mean(potential),
                    vs=jnp.std(potential),
                    cc=jnp.sqrt(jnp.maximum(1.0 - jnp.mean(potential), 0.0)),
                    dv=ensemble_diversity(particles[1], mask),
                    e=jnp.asarray(float(num_samples)),
                    n=num_samples,
                )
            return (particles, log_w, lam_v_prev, gkey), None

        sigma = jnp.mean(t_hat)
        if cfg.lambda_mode not in ("fixed", "adaptive_ess"):
            raise ValueError(
                f"unknown lambda_mode {cfg.lambda_mode!r}; expected 'fixed' or "
                "'adaptive_ess'"
            )
        if cfg.lambda_mode == "adaptive_ess":
            # Solve for the lambda that puts ESS at ess_target * num_samples,
            # gated by the same sigma ramp so selection still switches on late.
            gate = jax.nn.sigmoid((cfg.sigma_on - sigma) / cfg.sigma_width) > 0.5
            lam = adaptive_lambda(potential, cfg, num_samples) * gate
        else:
            lam = lambda_ramp(sigma, cfg)

        # Feynman-Kac incremental weight: log w_t = -lam_t V_t + lam_{t-1} V_{t-1},
        # so the potential is not double counted as the ramp turns on.
        lam_v = lam * potential
        log_w = log_w - lam_v + lam_v_prev
        log_w = log_w - jax.scipy.special.logsumexp(log_w)

        ess = effective_sample_size(log_w)
        do_resample = jnp.logical_and(
            ess < cfg.ess_threshold * num_samples, lam > cfg.lambda_floor
        )

        gkey, k_rs = jax.random.split(gkey)
        idx = jnp.where(
            do_resample,
            systematic_resample(k_rs, log_w),
            jnp.arange(num_samples, dtype=jnp.int32),
        )

        keys, positions, nl_prev = particles
        # fold the particle index in so resampled duplicates diverge afterwards
        keys = jax.vmap(jax.random.fold_in)(keys[idx], jnp.arange(num_samples))
        particles = (keys, positions[idx], nl_prev[idx])
        lam_v = lam_v[idx]
        log_w = jnp.where(
            do_resample, jnp.full((num_samples,), -jnp.log(float(num_samples))), log_w
        )

        if cfg.verbose:
            jax.debug.print(
                "sigma {s:8.3f} | lambda {l:6.3f} | V_mean {vm:9.4f} "
                "V_spread {vs:9.2e} | CC {cc:6.3f} | div {dv:7.3f} A | "
                "ESS {e:5.1f}/{n} | resample {r}",
                s=sigma,
                l=lam,
                vm=jnp.mean(potential),
                vs=jnp.std(potential),
                cc=jnp.sqrt(jnp.maximum(1.0 - jnp.mean(potential), 0.0)),
                dv=ensemble_diversity(particles[1], mask),
                e=ess,
                n=num_samples,
                r=do_resample.astype(jnp.int32),
            )

        return (particles, log_w, lam_v, gkey), None

    noise_levels = build_schedule(config.steps, cfg.sigma_start, cfg.sigma_end)

    key, noise_key, global_key = jax.random.split(key, 3)
    positions = jax.random.normal(noise_key, (num_samples,) + mask.shape + (3,))
    positions *= noise_levels[0]
    if x_start is not None:
        # SDEdit warm start: noise a placed model instead of starting from noise
        positions = (
            jnp.broadcast_to(x_start, (num_samples,) + mask.shape + (3,)) + positions
        ) * mask[..., None]

    particles = (
        jax.random.split(key, num_samples),
        positions,
        jnp.tile(noise_levels[None, 0], (num_samples,)),
    )
    init = (
        particles,
        jnp.full((num_samples,), -jnp.log(float(num_samples))),
        jnp.zeros((num_samples,)),
        global_key,
    )

    (particles, log_w, lam_v, _), _ = hk.scan(
        scan_body, init, noise_levels[1:], unroll=cfg.unroll
    )
    _, positions_out, _ = particles

    if use_smc:
        # Order best-first so existing call sites that take `conformations[0]`
        # get the best particle rather than an arbitrary one.  lambda is a shared
        # scalar at any level, so argsort(lambda*V) == argsort(V).  Skipped when
        # SMC is off, which keeps the unguided path bit-identical to AF3.
        order = jnp.argsort(lam_v)
        positions_out = positions_out[order]
        log_w = log_w[order]
        lam_v = lam_v[order]

    return {
        "atom_positions": positions_out,
        "mask": jnp.tile(mask[None], (num_samples, 1, 1)),
        "log_weights": log_w,
        "potential": lam_v,
    }
