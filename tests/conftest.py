"""Fixtures for the SMC sampler tests.

Nothing here mocks the AlphaFold 3 API.  The parity tests import the real
``alphafold3.model.network.diffusion_head`` and compare against the real
``diffusion_head.sample``; the batch is a real ``feat_batch.Batch`` holding a
real ``features.PredictedStructureInfo``.  If alphafold3 is not importable those
tests skip with a concrete reason rather than passing against a stand-in -- a
copy compared against a copy proves nothing about parity.

``haiku`` is the real dm-haiku.  ``af3_neutron.smc`` imports AF3 lazily, so the
pure SMC helpers are importable and fully tested even without alphafold3.
"""

import dataclasses
import importlib
import importlib.util
import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def _importable(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except Exception:
        return False


HAVE_AF3 = _importable("alphafold3.model.network.diffusion_head")

requires_af3 = pytest.mark.skipif(
    not HAVE_AF3,
    reason=(
        "alphafold3 is not importable; parity against the real "
        "diffusion_head.sample cannot be checked and is not faked"
    ),
)


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: statistical test, seconds not milliseconds")


@pytest.fixture(scope="session")
def smc():
    """The sampler under test.

    Imported via the installed package when available, otherwise loaded directly
    from ``src/`` so the pure helpers can be tested in a bare checkout.  No
    stubbing either way.
    """
    try:
        from af3_neutron import smc as module

        return module
    except ImportError:
        src = pathlib.Path(__file__).resolve().parents[1] / "src" / "af3_neutron" / "smc.py"
        spec = importlib.util.spec_from_file_location("_af3n_smc_under_test", src)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module


@pytest.fixture(scope="session")
def in_hk():
    """Run a thunk inside an ``hk.transform`` apply context.

    Both AF3's ``diffusion_head.sample`` and ``smc.sample`` call
    ``hk.running_init()``, ``hk.vmap`` and ``hk.scan``, all of which require a
    Haiku context -- calling either at module level raises
    ``ValueError: hk.running_init must be used as part of an hk.transform``.

    ``apply`` rather than ``init`` is used deliberately: ``hk.running_init()`` is
    True during init and False during apply, and AF3 passes
    ``split_rng=(not hk.running_init())``, so only the apply path exercises the
    real ``split_rng=True`` behaviour.  Params are ``{}`` because the sampler
    itself owns no ``hk.Module``; the denoiser is a plain callable argument.  The
    apply rng does not affect the result -- both samplers thread explicit keys
    through the scan carry and never draw ``hk.next_rng_key`` -- so parity is
    unaffected by it.
    """
    import haiku as hk

    def run(thunk, rng_seed=0):
        f = hk.transform(thunk)
        rng = jax.random.PRNGKey(rng_seed)
        try:
            return f.apply({}, rng)
        except Exception:
            return f.apply(f.init(rng), rng)

    return run


@pytest.fixture(scope="session")
def af3_diffusion_head():
    if not HAVE_AF3:
        pytest.skip("alphafold3 not importable")
    from alphafold3.model.network import diffusion_head

    return diffusion_head


@pytest.fixture
def make_sample_config(af3_diffusion_head):
    """Factory for AF3's own ``SampleConfig`` -- not a look-alike."""

    def make(steps=40, num_samples=8, **kw):
        return af3_diffusion_head.SampleConfig(
            steps=steps, num_samples=num_samples, **kw
        )

    return make


@pytest.fixture
def sample_config(make_sample_config):
    return make_sample_config()


@pytest.fixture
def n_tokens():
    return 24


@pytest.fixture
def n_atoms_per_token():
    return 3


@pytest.fixture
def atom_mask(n_tokens, n_atoms_per_token):
    return jnp.ones((n_tokens, n_atoms_per_token), dtype=bool)


@pytest.fixture
def batch(atom_mask, n_tokens):
    """A real ``feat_batch.Batch``.

    ``sample`` reads only ``predicted_structure_info.atom_mask``, so the other
    feature groups are left unpopulated -- but the container and the populated
    field are AF3's own dataclasses, not look-alikes.
    """
    if not HAVE_AF3:
        pytest.skip("alphafold3 not importable")
    from alphafold3.model import feat_batch, features

    info = features.PredictedStructureInfo(
        atom_mask=atom_mask,
        residue_center_index=jnp.zeros((n_tokens,), dtype=jnp.int32),
    )
    fields = [f.name for f in dataclasses.fields(feat_batch.Batch)]
    kwargs = {name: None for name in fields}
    kwargs["predicted_structure_info"] = info
    return feat_batch.Batch(**kwargs)


@pytest.fixture
def attractor(atom_mask):
    rng = np.random.default_rng(0)
    return jnp.asarray(rng.normal(size=atom_mask.shape + (3,)) * 6.0, jnp.float32)


@pytest.fixture
def denoise_plain(attractor):
    """Deterministic stand-in for the network: shrink toward a fixed target.

    This is a *test signal source*, not a mock of an AF3 interface -- AF3's
    `sample` takes `denoising_step` as a plain callable argument, and supplying a
    simple one is how the sampler is meant to be exercised.
    """

    def fn(x, sigma):
        k = 1.0 / (1.0 + (sigma / 6.0) ** 2)
        return attractor + k * (x - attractor)

    return fn


@pytest.fixture
def denoise_with_potential(attractor, denoise_plain):
    """Same denoiser, returning ``(x0, V)`` as the guided operator now does."""

    def fn(x, sigma):
        x0 = denoise_plain(x, sigma)
        return x0, jnp.mean(jnp.sum((x0 - attractor) ** 2, -1))

    return fn
