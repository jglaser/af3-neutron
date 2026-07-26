"""Regression tests for af3_neutron.smc.

The load-bearing test is :func:`test_disabled_matches_stock_af3`: the guided
sampler must reduce to AF3's own sampler exactly when SMC is off, so that
enabling SMC is the only thing that can ever change a result.  It is checked
against the real ``alphafold3.model.network.diffusion_head.sample`` -- there is
no reference copy in this suite, because comparing a copy against a copy would
prove nothing.  Where alphafold3 is unavailable those tests skip.

The pure SMC helpers need no AF3 and are tested unconditionally.
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from conftest import HAVE_AF3, requires_af3


# ==========================================================================
# pure SMC machinery -- no AF3 required
# ==========================================================================
def test_systematic_resample_respects_expected_counts(smc):
    w = jnp.asarray([0.7, 0.2, 0.05, 0.05])
    lw = jnp.log(w) - jax.scipy.special.logsumexp(jnp.log(w))
    idx = smc.systematic_resample(jax.random.PRNGKey(1), lw)
    assert idx.shape == (4,)
    assert 0 <= int(idx.min()) and int(idx.max()) < 4
    counts = np.bincount(np.asarray(idx), minlength=4)
    assert counts.sum() == 4
    # systematic resampling keeps each count within one of N*w
    for k in range(4):
        assert abs(counts[k] - 4 * float(w[k])) <= 1.0


def test_systematic_resample_is_unbiased(smc):
    w = jnp.asarray([0.5, 0.3, 0.15, 0.05])
    lw = jnp.log(w) - jax.scipy.special.logsumexp(jnp.log(w))
    keys = jax.random.split(jax.random.PRNGKey(0), 400)
    idx = jax.vmap(smc.systematic_resample, in_axes=(0, None))(keys, lw)
    freq = np.bincount(np.asarray(idx).ravel(), minlength=4) / idx.size
    np.testing.assert_allclose(freq, np.asarray(w), atol=0.02)


def test_effective_sample_size_bounds(smc):
    n = 8
    uniform = jnp.full((n,), -jnp.log(float(n)))
    assert float(smc.effective_sample_size(uniform)) == pytest.approx(n, rel=1e-5)
    degenerate = jnp.asarray([0.0] + [-60.0] * (n - 1))
    assert float(smc.effective_sample_size(degenerate)) == pytest.approx(1.0, abs=1e-4)
    mid = float(smc.effective_sample_size(jnp.asarray([0.0, -1.0, -2.0])))
    assert 1.0 <= mid <= 3.0


def test_lambda_ramp_is_monotone_and_bounded(smc):
    cfg = smc.SMCConfig(lambda_max=3.0, sigma_on=12.0, sigma_width=6.0)
    sig = jnp.asarray([100.0, 30.0, 18.0, 12.0, 6.0, 2.0, 0.1])
    lam = np.asarray(jax.vmap(lambda s: smc.lambda_ramp(s, cfg))(sig))
    assert np.all(np.diff(lam) > 0), "lambda must increase as sigma falls"
    assert np.all(lam >= 0.0) and np.all(lam <= 3.0)
    assert lam[3] == pytest.approx(1.5, rel=1e-5), "lambda_max/2 at sigma_on"


def test_lambda_zero_disables_selection(smc):
    cfg = smc.SMCConfig(lambda_max=0.0)
    for s in (0.1, 5.0, 50.0):
        assert float(smc.lambda_ramp(jnp.asarray(s), cfg)) == 0.0


@pytest.mark.skipif(HAVE_AF3, reason="only meaningful when alphafold3 is absent")
def test_sample_raises_clearly_without_af3(smc):
    """No silent fallback: without AF3 the sampler must refuse, not improvise."""
    with pytest.raises(ImportError, match="requires alphafold3"):
        smc.noise_schedule(jnp.asarray(0.5))


# ==========================================================================
# parity against the real AF3 sampler
# ==========================================================================
def _stock_positions(in_hk, af3_diffusion_head, denoising_step, batch, key, config):
    """AF3's own sampler, run inside an hk.transform apply context."""
    return in_hk(
        lambda: af3_diffusion_head.sample(
            denoising_step=denoising_step, batch=batch, key=key, config=config
        )
    )["atom_positions"]


def _guided(in_hk, smc, denoising_step, batch, key, config, **kw):
    """af3_neutron.smc.sample, run inside the same kind of context."""
    return in_hk(
        lambda: smc.sample(denoising_step, batch, key, config, **kw)
    )


@requires_af3
@pytest.mark.parametrize("kind", ["none", "lambda_zero"])
def test_disabled_matches_stock_af3(
    smc, in_hk, af3_diffusion_head, batch, sample_config, denoise_plain, kind
):
    key = jax.random.PRNGKey(7)
    cfg = None if kind == "none" else smc.SMCConfig(lambda_max=0.0, verbose=False)

    expected = _stock_positions(in_hk, af3_diffusion_head, denoise_plain, batch, key, sample_config)
    got = _guided(in_hk, smc, denoise_plain, batch, key, sample_config, smc_config=cfg)["atom_positions"]

    assert jnp.array_equal(expected, got), (
        "guided sampler diverged from stock AF3 with SMC disabled; max |diff| = "
        f"{float(jnp.max(jnp.abs(expected - got))):.3e}"
    )


@requires_af3
def test_tuple_returning_step_is_a_noop_when_disabled(
    smc, in_hk, af3_diffusion_head, batch, sample_config, denoise_plain, denoise_with_potential
):
    """Returning ``(x0, V)`` instead of ``x0`` must not perturb the trajectory."""
    key = jax.random.PRNGKey(11)
    expected = _stock_positions(in_hk, af3_diffusion_head, denoise_plain, batch, key, sample_config)
    got = _guided(
        in_hk, smc, denoise_with_potential, batch, key, sample_config,
        smc_config=smc.SMCConfig(lambda_max=0.0, verbose=False),
    )["atom_positions"]
    assert jnp.array_equal(expected, got)


@requires_af3
def test_best_first_ordering_is_not_applied_when_disabled(
    smc, in_hk, af3_diffusion_head, batch, sample_config, denoise_plain, denoise_with_potential
):
    """The best-first sort must not fire when SMC is off, or parity breaks."""
    key = jax.random.PRNGKey(13)
    expected = _stock_positions(in_hk, af3_diffusion_head, denoise_plain, batch, key, sample_config)
    got = _guided(
        in_hk, smc, denoise_with_potential, batch, key, sample_config,
        smc_config=smc.SMCConfig(lambda_max=0.0, verbose=False),
    )["atom_positions"]
    assert jnp.array_equal(expected, got)


@requires_af3
def test_output_contract_matches_af3_plus_extras(
    smc, in_hk, af3_diffusion_head, batch, sample_config, denoise_plain, denoise_with_potential
):
    """Keys and shapes AF3 returns must still be there, unchanged."""
    key = jax.random.PRNGKey(0)
    stock = in_hk(
        lambda: af3_diffusion_head.sample(
            denoising_step=denoise_plain, batch=batch, key=key, config=sample_config
        )
    )
    out = _guided(
        in_hk, smc, denoise_with_potential, batch, key, sample_config,
        smc_config=smc.SMCConfig(lambda_max=1.0, verbose=False),
    )
    assert set(stock).issubset(set(out))
    for k in stock:
        assert out[k].shape == stock[k].shape
    n = sample_config.num_samples
    assert out["log_weights"].shape == (n,)
    assert np.isclose(float(jax.scipy.special.logsumexp(out["log_weights"])), 0.0, atol=1e-5)
    assert np.all(np.isfinite(np.asarray(out["atom_positions"])))


# ==========================================================================
# schedule: the divergence guard
# ==========================================================================
@requires_af3
@pytest.mark.parametrize("steps,sigma_start", [(200, None), (200, 15.0), (60, 8.0), (30, 2.0)])
def test_warm_start_schedule_is_stable(
    smc, in_hk, make_sample_config, batch, atom_mask, denoise_plain, steps, sigma_start
):
    """The EDM propagation weight must stay below 1 or the trajectory diverges.

    ``a = step_scale * |dsigma| / sigma``; at ``a >= 1`` the update over-relaxes.
    Truncating a coarse schedule (rather than reparametrising it) produces jumps
    of 8 -> 0.5 A, i.e. ``a = 1.4``, which blows the sampler up.  This pins the
    reparametrisation, and checks sigma_start is honoured exactly.
    """
    cfg = make_sample_config(steps=steps, num_samples=2)
    sc = smc.SMCConfig(lambda_max=0.0, sigma_start=sigma_start, verbose=False)
    x_start = None if sigma_start is None else jnp.zeros(atom_mask.shape + (3,))
    out = _guided(in_hk, smc, denoise_plain, batch, jax.random.PRNGKey(0), cfg,
                  smc_config=sc, x_start=x_start)
    assert np.all(np.isfinite(np.asarray(out["atom_positions"]))), "trajectory diverged"

    sigmas = np.asarray(smc.build_schedule(cfg.steps, sigma_start))
    if sigma_start is not None:
        assert sigmas[0] == pytest.approx(sigma_start, rel=1e-4)
    a = np.max(1.5 * np.abs(np.diff(sigmas)) / np.maximum(sigmas[:-1], 1e-12))
    assert a < 1.0, f"EDM step weight {a:.3f} >= 1; schedule will over-relax"


@requires_af3
def test_warm_start_uses_the_supplied_model(
    smc, in_hk, make_sample_config, batch, atom_mask, denoise_plain, attractor
):
    """x_start must actually be used, not silently ignored.

    Compared after Kabsch superposition, because AF3's ``random_augmentation``
    applies a fresh random rotation *and* recentres every step.  A warm start
    therefore conveys internal geometry only -- never absolute position or
    orientation.  That is harmless here (the guidance operator re-derives the
    crystal frame each step, which is exactly why it must), but it means the
    property can only be asserted up to a rigid transform.
    """
    rng = np.random.default_rng(9)
    x_start = jnp.asarray(rng.normal(size=atom_mask.shape + (3,)) * 6.0, jnp.float32)
    sc = smc.SMCConfig(lambda_max=0.0, sigma_start=0.5, verbose=False)
    out = _guided(
        in_hk, smc, denoise_plain, batch, jax.random.PRNGKey(0),
        make_sample_config(steps=4, num_samples=4), smc_config=sc, x_start=x_start,
    )

    def superposed_rmsd(a, b):
        a = np.asarray(a, dtype=np.float64).reshape(-1, 3)
        b = np.asarray(b, dtype=np.float64).reshape(-1, 3)
        a, b = a - a.mean(0), b - b.mean(0)
        u, _, vt = np.linalg.svd(a.T @ b)
        d = np.sign(np.linalg.det(u) * np.linalg.det(vt))
        r = u @ np.diag([1.0, 1.0, d]) @ vt
        return float(np.sqrt((((a @ r) - b) ** 2).sum(-1).mean()))

    p0 = out["atom_positions"][0]
    to_start = superposed_rmsd(p0, x_start)
    to_attractor = superposed_rmsd(p0, attractor)
    assert to_start < to_attractor, (
        f"warm start ignored: rmsd to x_start {to_start:.2f} A not closer than "
        f"rmsd to the attractor {to_attractor:.2f} A"
    )


# ==========================================================================
# selection behaviour
# ==========================================================================
@requires_af3
def test_lambda_floor_blocks_resampling(
    smc, in_hk, make_sample_config, batch, denoise_with_potential
):
    """With lambda pinned under the floor, no resampling happens.

    Resampling on an uninformative potential locks the ensemble onto whichever
    particle was transiently lucky, and the collapse is permanent because churn
    noise is the only later source of divergence.  So the floor must hold.
    """
    cfg = make_sample_config(steps=30, num_samples=8)
    sc = smc.SMCConfig(lambda_max=1e-4, sigma_on=1e6, lambda_floor=0.05, verbose=False)
    out = _guided(in_hk, smc, denoise_with_potential, batch, jax.random.PRNGKey(2), cfg, smc_config=sc)
    pos = np.asarray(out["atom_positions"])
    pairwise = np.abs(pos[:, None] - pos[None, :]).reshape(8, 8, -1).max(-1)
    off_diag = pairwise[~np.eye(8, dtype=bool)]
    assert np.all(off_diag > 1e-6), "particles collapsed despite lambda below the floor"


@requires_af3
@pytest.mark.slow
def test_selection_does_not_raise_the_potential(
    smc, in_hk, make_sample_config, batch, denoise_with_potential, attractor
):
    """Enabling selection must not make the weighted ensemble worse.

    Asserted on the mean potential rather than a basin-occupancy fraction: the
    mean has far lower variance across seeds, so this is a usable CI gate.
    """
    cfg = make_sample_config(steps=50, num_samples=16)

    def mean_potential(sc):
        vals = []
        for seed in (0, 1, 2):
            out = _guided(in_hk, smc, denoise_with_potential, batch,
                          jax.random.PRNGKey(seed), cfg, smc_config=sc)
            v = jax.vmap(lambda p: jnp.mean(jnp.sum((p - attractor) ** 2, -1)))(
                out["atom_positions"])
            vals.append(float(jnp.mean(v)))
        return float(np.mean(vals))

    off = mean_potential(smc.SMCConfig(lambda_max=0.0, verbose=False))
    on = mean_potential(
        smc.SMCConfig(lambda_max=4.0, sigma_on=8.0, sigma_start=6.0, verbose=False))
    assert on <= off * 1.05, f"selection raised the mean potential: {on:.3f} vs {off:.3f}"


@requires_af3
@pytest.mark.slow
def test_best_particle_is_first(
    smc, in_hk, make_sample_config, batch, denoise_with_potential, attractor
):
    """Output is ordered best-first, so ``conformations[0]`` is the best particle.

    ``run_neutron_refine.py`` indexes ``conformations[0]``; without the ordering
    it would get an arbitrary member of the ensemble.
    """
    cfg = make_sample_config(steps=40, num_samples=12)
    sc = smc.SMCConfig(lambda_max=4.0, sigma_on=8.0, sigma_start=6.0, verbose=False)
    out = _guided(in_hk, smc, denoise_with_potential, batch, jax.random.PRNGKey(5), cfg, smc_config=sc)
    v = np.asarray(jax.vmap(lambda p: jnp.mean(jnp.sum((p - attractor) ** 2, -1)))(
        out["atom_positions"]))
    assert v[0] == pytest.approx(v.min(), rel=1e-5), (
        f"particle 0 is not the best: {v[0]:.4f} vs min {v.min():.4f}")
