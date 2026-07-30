---
name: concise-docs
description: Cut over-long docstrings and comment walls to satisfy tools/lint_docs.py. Use when the docstring budget gate fails, when writing or reviewing docstrings in af3-neutron, or when a reviewer says the docs are too verbose.
---

# Cutting prose nobody reads

The gate is `tools/lint_docs.py`. Diagnose first:

```
uv run --no-project --python 3.12 python tools/lint_docs.py . --report
```

## What each code means and how to fix it

| Code | Offence | Fix |
|---|---|---|
| DOC101 | Docstring over budget | Keep the contract, move the reasoning to the commit message |
| DOC102 | `Notes`/`Why`/`Measured` section | Delete the section; the facts belong in the PR body |
| DOC103 | `*emphasis*` spam | Delete the asterisks; if a point needs emphasis it needs a shorter sentence |
| DOC201 | AI filler phrase | Delete the phrase, keep the claim |
| DOC301 | Comment wall | Extract a named function |
| DOC401 | File is more prose than code | Usually DOC101s in aggregate; cut those first |

## The cut

Keep, in this order: what it returns and in what units, what it mutates, what it
raises, non-obvious preconditions. That is usually one to four lines.

Move to the commit message: why this approach beat the alternative, benchmark
numbers, R-factor tables, the bug that motivated the code, PDB-entry-specific
measurements.

Delete outright: restatements of the signature, restatements of the function
name, tutorials on the domain, apologies, hedges.

## Worked example

Before — 47 lines, DOC101 + DOC102:

```python
def align_oracle_to_reference(...):
    """Aligns the unanchored AF3 oracle coordinates to an explicit absolute
    crystal lattice frame ...

    Notes
    -----
    If the reference was deposited in a *different* cell ... Measured on this
    system: 4BD0 (X-ray) is 72.500 72.500 97.670 while 4BD1 is 73.429 ...
        as-was (neither fix)   0.5228
        cell transfer only     0.5219
    ...
    """
```

After — 8 lines:

```python
def align_oracle_to_reference(...):
    """Move the oracle onto an absolute crystal frame taken from a reference.

    Parameters
    ----------
    mtz_path : str, optional
        Diffraction data; its cell header is authoritative for the whole
        pipeline. Prefer it over `target_cell`.
    target_cell : tuple of 6 floats, optional
        `(a, b, c, alpha, beta, gamma)`, angles in DEGREES. Overrides `mtz_path`.

    Run `resolve_reindexing` afterwards: a reference in a different setting is
    not fixed by the cell transfer alone.
    """
```

The R-factor table did not vanish — it moved to the commit that made the change,
where it is dated and attached to a diff.

## Rules

- Never satisfy the gate with `# lint-docs: allow`. It exists for generated files,
  pattern tables, and the gate's own test fixtures — code whose *subject* is the
  offending text. Reach for it anywhere else and you have missed the point.
- Never delete a *fact* that only exists in the docstring. Move it. Check `git log`
  or the PR body already records it; if not, put it in your commit message.
- Do not add a docstring just because a function lacks one. `D1xx` is deliberately
  not enabled — mandatory docstrings are what produced this filler.
