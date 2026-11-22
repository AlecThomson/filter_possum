# CLAUDE.md

Context for Claude Code working in this repo.

## What this is

A toy-model study of what a **single dish** (Murriyang/Parkes, 64 m) and an **interferometer**
(ASKAP-like, 30 m nominal inner uv edge) actually measure of **diffuse polarised emission**
across a POSSUM-like band (800–1088 MHz, 288 × 1 MHz), and how well their combination recovers
the truth. The physics question matters more than the code: every change should keep the
results interpretable and, where possible, checked against an analytic expectation.

`rm_scale.ipynb` is the original exploratory notebook (normalised units, constant uv cut).
Leave it, `cosmicodes/` and `requirements.txt` alone unless asked.

## Layout

- `src/filter_possum/uvfilter_sim.py` — everything: sky model, per-channel uv filtering,
  RM synthesis (rm-lite), metrics, plots, scans, test suites, CLI (`filter-possum-sim`).
- `tests/test_uvfilter_sim.py` — physics checks on small grids (~10 s). Must pass.
- `results/` — curated, committed snapshot (metrics/scan JSON, scans, transfer functions,
  headline figures). Regenerate deliberately; keep it ~≤20 MB (map PNGs are ~2 MB each,
  so commit only headline maps).
- `outputs/` — raw run outputs, gitignored.

## Commands

```bash
uv sync
uv run pytest
uv run ruff check src tests && uv run ruff format --check src tests
uv run filter-possum-sim --suite base|t1|t2            # whole suite
uv run filter-possum-sim --suite t2 --part runs        # parts: runs | scan_grad | scan_sigma
uv run filter-possum-sim --suite t2 --part scan_grad:joint
uv run filter-possum-sim --rm-grad 30 --sd-model aperture --int-inner soft --combine feather   # one run
```

Rough runtimes (1 CPU): example runs ~1–2 min per suite; each scan ~1.5–3.5 min
(2 cut modes × 6–8 values). Run long suites in parts.

## Model conventions (don't break these)

- Sky: p = 1, χ0 = 0, Faraday-thin per pixel; RM = mean + Gaussian random field
  (P(k) ∝ k^β, default β = −11/3, σ normalised over the padded box) + optional gradient.
- Geometry: SIN projection; pixels uniform in direction cosines, so the image FFT *is* the
  uv plane in wavelengths (w-term neglected; fine for 5°).
- The sky is simulated on a `pad`× larger periodic box and the central field analysed.
  The gradient is folded back in the guard band (`periodic_gradient`) to avoid a wrap
  discontinuity ringing through the uv filters.
- Responses are evaluated at **physical baseline** b = |u|·λ per channel (`cut_mode="baseline"`).
  `cut_mode="fixed"` evaluates SD at λ_max (common resolution) and the INT inner edge at
  λ_min (uv-range matching). Both responses are monotonic in b, so this equals smoothing /
  matching exactly.
- Q and U are filtered together as one complex field with real, radially symmetric
  transfer functions, so Q and U stay real.
- Combination (`combine`): feather = `V_sd + (1 − T_sd) V_int` (surface-brightness units, so no
  beam-area factor); joint = `(T_sd V_sd + T_int V_int)/(T_sd² + T_int² + eps)`, the noiseless
  Wiener / joint-deconvolution limit. `joint_eps` stands in for S/N until noise exists.
- The "Truth (no filtering)" row is a **pipeline check**: the unfiltered cube sampled at the
  channel frequencies (point-sampled, no bandwidth depolarisation) and RM-synthesised.
  Its residual is ~1e-4 rad m⁻² from sub-sample peak fitting.
- Defaults reproduce committed results exactly (fixed seed 42). New options must default to
  the existing behaviour; verify by comparing metrics JSON before/after.

## Output conventions

- RM-error maps: one **linear** colour scale shared by every row and run, ±RMSF FWHM/2
  (≈29.6 rad m⁻²) by default; `--err-lim` overrides. The user asked for this; don't revert it.
- Metrics JSON is rounded to 6 significant figures via `_dump` (readable diffs; also stops the
  privacy filter flagging 16-digit float mantissas as card numbers).
- New test suites get their own tag prefix (`t3_…`) so earlier results are never overwritten.
- Read files fully before rewriting them in place (an `open(f, "w")` before `json.load(open(f))`
  truncates the file).

## Verified analytic results (keep tests for these)

- Weak screen (2σλ² ≪ 1), uv-matched: PI_int ≈ 2λ0²|δRM_high-pass| (corr 0.999).
- Pure RM gradient: SD PI = band mean of T_sd(|∇RM| λ³ / π) (exact, 1e-4). Fringe baseline
  b = |∇RM| λ³ / π; the INT sees it once b ≳ 30 m. The Gaussian-beam formula is only
  approximate (~10% low once p < 0.5) — don't reintroduce it as "exact".
- Dish beam FWHM: 1.03 λ/D uniform, 1.157 λ/D for 12 dB edge taper.

## Headline findings so far

See README "Main findings". In short: the INT measures angle gradients, not PI; frequency-
dependent uv coverage creates spurious Faraday structure (uv matching cuts INT RM error 2–9×);
a 64 m dish beam-depolarises strongly for |∇RM| ≳ 30 rad m⁻² deg⁻¹; feathering loses 8–25% PI
(undeconvolved SD response, dip to 0.59 at b ≈ 25 m); joint combination recovers RM ≤0.07 rad m⁻²
and PI ≥0.98 (noiseless upper bound).

## Roadmap (agreed order; do one at a time, report results, then continue)

1. **Test 3 — Faraday-thick emission.** e.g. a Burn slab (uniform emission + rotation) and/or
   mixed screen + emitting layer, so frequency dependence is intrinsic. Compare SD / INT /
   feather / joint against the true Faraday spectrum (not just a peak RM); consider RM-CLEAN
   (rm-lite `tools_3d.rmclean`) and moments of the FDF.
2. **Test 4 — structure in intrinsic p and χ0** (so PI structure is real, not purely Faraday).
3. **Test 5 — noise and realisations.** Add thermal noise per instrument (sensitivity-weighted),
   re-derive `joint_eps` from S/N, and run multiple seeds for error bars on all scan metrics.
4. Optional: Stokes I and leakage; restoring beam with finer pixels (`--npix 256`); a real ASKAP
   uv density instead of the soft-edge model.

## Working practices

- Python via `uv`; ruff rule set pinned in `pyproject.toml` (line length 120).
- Inline comments for non-obvious physics/maths.
- No PII or secrets in code, commits, issues or results. The org uses the SKA
  `ska-privacy-filter` skill (from https://gitlab.com/ska-telescope/ska-ai-skills, early
  development); run it over diffs/commit messages before pushing. Known false positives:
  hashes/versions in `uv.lock`, `+@pytest…` in diffs.
- Only use Claude with Unrestricted Data.
