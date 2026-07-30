# af3-neutron conventions

Enforced by `.github/workflows/ci.yml`. Run the gate before pushing:

```
uv run --no-project --python 3.12 python tools/lint_docs.py .   # docstring budget
uvx --from 'ruff>=0.9,<1' ruff check . && uvx --from 'ruff>=0.9,<1' ruff format --check .
```

## Prose

A docstring says what a caller must know to call the thing: what it returns,
what units, what it mutates, what it raises. Nothing else.

- Max 12 lines per function/class docstring, 20 for a module. Blank lines are free.
- **Design rationale, measurements and R-factor tables go in the commit message
  or the PR description**, not in a docstring. They are dated and reviewable there;
  in a docstring they rot silently and nobody reads them.
- No `Notes`/`Why`/`Rationale`/`Background`/`Measured` sections. `Parameters`,
  `Returns` and `Raises` are fine.
- No AI filler: `leverage`, `delve into`, `in order to`, `it is important to
  note`, `moreover`, `furthermore`, `notably`. Full list in `tools/ai_tells.txt`.
- More than 8 consecutive comment lines is a design smell. Extract a named
  function instead: prefer `transfer_reference_through_fractional_space(...)` over
  a paragraph explaining that the next 15 lines transfer a reference through
  fractional space.

## Code

- Named constants, not magic numbers. Module-level `ALL_CAPS` with a short
  trailing comment for units or provenance:
  `SOLVENT_K_INIT = 0.35  # Fokkema 2018, table 2`
- Imports in three blocks: stdlib, third party, `af3_neutron`. `ruff` enforces it.
- One responsibility per module. Patterson code lives in `patterson.py`; a `run_*`
  script parses arguments and calls library code, it does not define it.
- Closures are idiomatic in JAX, but a closure over more than the arrays it
  differentiates should be a dataclass.
- Angles: every cell tuple in this codebase carries **degrees**. `biotite`'s
  `vectors_from_unitcell` takes radians and silently returns a plausible box if
  fed degrees. Go through `gemmi`.

## Tests

`tests/*` are exempt from the prose-ratio rule: a docstring naming the invariant
under test is that test's spec. The 12-line budget still applies.
