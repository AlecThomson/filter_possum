# CLAUDE.md

Context for Claude Code working in this repo.

## What this is

A small simulator of what a **single dish** and an **interferometer** recover of **diffuse polarised
emission** (a Faraday-thin RM screen), for the user to play with. The user wants it kept simple: a
Python API plus an example notebook, and results shown as plain 6-panel comparisons (true /
recovered / difference, for RM and PI) via `plot_comparison`. Don't add suites, scans, metrics
layers, derived summary figures or "how to read" notes unless asked.

`rm_scale.ipynb` is the original exploratory notebook. Leave it, `cosmicodes/` and `requirements.txt`
alone unless asked.

## Layout

- `src/filter_possum/simulator.py`: everything (`Sky`, `Band`/`BANDS`, `SingleDish`/`PARKES`,
  `Interferometer`/`ASKAP`/`MEERKAT`, `observe`, `rmsynth`, `plot_comparison`).
- `examples/simulator.ipynb`: walkthrough, committed with outputs.
- `tests/test_simulator.py`: analytic physics checks (~5 s). Must pass.

## Commands

```bash
uv sync
uv run pytest
uv run ruff check src tests && uv run ruff format --check src tests
```

## Model conventions (don't break these)

- SIN projection; pixels uniform in direction cosines, so the image FFT is the uv plane in wavelengths.
- Sky simulated on a `pad`× larger periodic box, central field analysed; the gradient folds back in
  the guard band so the box stays periodic.
- Responses evaluated at physical baseline b = |u| λ per channel. `uv_match=True` evaluates the dish
  at λ_max and the interferometer's inner edge at λ_min (exactly smoothing / uv-range matching).
- Q and U filtered together as one complex field with real, radially symmetric responses.
- Feather = `V_sd + (1 − T_sd) V_int`; joint = `(T_sd V_sd + T_int V_int)/(T_sd² + T_int² + eps)`.
- The `truth` cube is the unfiltered sky; its RM synthesis is a pipeline check (~1e-4 rad m⁻²).

## Verified analytic results (tests cover these)

- Weak screen: interferometer PI = 2λ0² |RM fluctuations on scales it sees|.
- Pure gradient: dish PI = band mean of T_sd(|∇RM| λ³ / π).
- Dish beam FWHM: 1.03 λ/D uniform, 1.157 λ/D for a 12 dB edge taper.
- A fixed cut in wavelengths at b_min/λ_min is exactly uv-range matching.

## History

Earlier versions of this repo had test suites (t1–t3), parameter scans and a metrics layer; they were
removed for simplicity and remain in git history (PR branches `uvfilter-simulation`, `t3-interferometers`).

## Working practices

- Python via `uv`; ruff rules in `pyproject.toml` (line length 120). Inline comments for non-obvious physics.
- No PII or secrets in code, commits, issues or results. The org uses the SKA `ska-privacy-filter`
  skill (https://gitlab.com/ska-telescope/ska-ai-skills, early development); run it over diffs and
  commit messages before pushing. Known false positives: hashes in `uv.lock`, `+@pytest…` in diffs.
- Only use Claude with Unrestricted Data.
