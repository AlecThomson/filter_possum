# POSSUM Filtering

A notebook to explore what is measured by an interferometer and a single dish given a model RM sky.

## Installation

Clone this repository recursively to get `cosmicodes` by [Andrea Bracco](https://github.com/abracco).

```bash
git clone --recurse-submodules https://github.com/AlecThomson/filter_possum.git
```

Then install Python requirements:

```bash
pip install -r requirements.txt
```

## uv-filtering simulation (`src/filter_possum/uvfilter_sim.py`)

A general version of the notebook experiment, in physical units. A constant-p (= 1),
Faraday-thin sky, RM = 30 rad m⁻² + Gaussian random field (P(k) ∝ k^−11/3) + optional
mean-RM gradient, is observed over 800–1088 MHz (288 × 1 MHz) on a 5 × 5° SIN grid.
Each channel's Q+iU is filtered in the uv plane by the instruments' responses at that
channel's wavelength (baseline b = |u| λ), then RM-synthesised with rm-lite.

| Instrument | Sharp model (`base`) | Realistic model (`t1`) |
|---|---|---|
| Single dish (Murriyang/Parkes, 64 m) | b ≤ 64 m | aperture autocorrelation, 12 dB edge taper (FWHM 1.157 λ/D) |
| Interferometer (ASKAP-like) | b ≥ 30 m | 30 m edge smeared over ±12 m (antenna aperture) |
| Combination (`t2`) | – | feather (CASA-style) or joint (least-squares/Wiener, noiseless limit of joint deconvolution) |

`--cut-mode fixed` evaluates the responses at a common wavelength for every channel
(single dish smoothed to common resolution; interferometer inner edge uv-range matched).

### Running

```bash
uv sync
uv run filter-possum-sim --suite base           # sharp cuts: examples + gradient/sigma scans
uv run filter-possum-sim --suite t1             # test 1: realistic responses
uv run filter-possum-sim --suite t2             # test 2: SD+INT combination
uv run filter-possum-sim --suite t2 --part scan_grad:joint   # one part (runs | scan_grad | scan_sigma)
uv run filter-possum-sim --suite t3             # test 3: interferometers (generic λ cut, ASKAP, MeerKAT)
uv run filter-possum-sim --array meerkat --band uhf --sd-model aperture   # one array/band preset
uv run filter-possum-sim --rm-grad 30 --sd-model aperture --int-inner soft --combine feather  # one run
uv run pytest                                    # physics checks (~10 s)
```

Raw outputs go to `outputs/` (gitignored). `results/` is a snapshot of all metrics
(JSON), scans, transfer functions and the headline figures. Difference maps share one
linear colour scale, ±RMSF FWHM/2 by default (`--err-lim` to change).

### Main findings (base case: σ(δRM) = 10 rad m⁻², realistic responses unless stated)

| | RM error, rms [rad m⁻²] | mean PI (true = 1) |
|---|---|---|
| Single dish | 1.5 | 0.88 |
| Interferometer | 8.1 (13.0 with sharp cuts) | 0.19 |
| Interferometer, uv-range matched | 3.3 | 0.16 |
| Feather | 0.77 | 0.92 |
| Joint | 0.04 | 0.99 |

* **An interferometer measures polarisation-angle gradients, not polarised intensity.** In
  the weak-screen limit its PI is 2λ₀²|δRM_high-pass| (correlation 0.999 with the simulation),
  yet its RM still follows the large-scale RM, carried by the small-scale structure.
* **Frequency-dependent uv coverage creates spurious Faraday structure.** Matching the uv
  range across the band cuts the interferometer's RM error by 2–9×.
* **An RM gradient puts Q/U fringes at baseline b = |∇RM| λ³ / π.** The single dish passes
  them with weight T_SD(b), which is exactly its PI for a pure gradient (tested to 10⁻³);
  the interferometer sees them only once b ≳ 30 m (31–79 rad m⁻² deg⁻¹ splits the band).
  With 64 m the two instruments cross near 40 rad m⁻² deg⁻¹, each recovering < 60%.
* **Feathering beats either instrument but loses 8–25% of PI**, because it does not
  deconvolve the single-dish response (combined transfer dips to 0.59 at b ≈ 25 m).
  **Joint combination recovers RM to ≤ 0.07 rad m⁻² and PI ≥ 0.98** at all gradients and σ
  tested, so the uv coverages overlap sufficiently; this is a noiseless upper bound.

### Test 3: what different interferometers tell us

Plain-language key (the figures use these terms):

* **Largest angular scale** an interferometer can see ≈ λ / (shortest baseline), in degrees. It is
  the inverse of the "inner uv cut" u_min used in the code (u_min = 80 λ ↔ 0.7°). Larger scales are
  invisible to it.
* **As observed vs matched**: with a fixed shortest baseline the largest scale shrinks as frequency
  rises, so each channel sees a different sky. "Matched" restricts every channel to the same largest
  scale (uv-range matching).
* **Polarised intensity recovered**: peak of the Faraday spectrum, as a fraction of the truth (which is
  1 everywhere in these simulations).
* **Stripes**: a uniform RM gradient rotates the polarisation angle linearly across the sky, so Q and U
  form stripes with spacing π / (|∇RM| λ²); the code calls this the gradient's "fringe".

A generic interferometer (sharp inner cut fixed in wavelengths, u_min = 10–320 λ) and parametric
ASKAP (shortest baseline 22 m, 12 m dishes) and MeerKAT (29 m, 13.5 m) arrays, on the common POSSUM band 1
and on native bands (ASKAP POSSUM band 1+2; MeerKAT UHF 544–1088 and L-band 856–1712 MHz).
Noiseless and ideally deconvolved, so only the coverage edges matter (uv density only matters once
there is noise). Same default sky as above. Figures (one question each), in `results/`:
`t3_fig1_what_int_measures` (maps), `t3_fig2_errors_at_pi_nulls`, `t3_fig3_inner_cut`, `t3_fig4_arrays`,
`t3_fig5_gradient`. Metrics and scan JSON are in `results/t3_*.json`; the per-run and scan plots that
`--suite t3` writes to `outputs/` are diagnostics.

| INT, default sky | u_min across band [λ] | RM error rms (bright half of PI) | uv-matched | mean PI |
|---|---|---|---|---|
| ASKAP, POSSUM band 1 | 59–80 | 7.3 (3.7) | 3.6 | 0.24 |
| MeerKAT, POSSUM band 1 | 77–105 | 7.7 (3.5) | 3.2 | 0.19 |
| ASKAP, POSSUM band 1+2 | 59–106 | 7.0 (3.7) | 2.8 | 0.19 |
| MeerKAT, UHF | 53–105 | 3.8 (2.9) | 2.5 | 0.36 |
| MeerKAT, L-band | 83–165 | 8.5 (3.9) | 2.4 | 0.09 |

* **An interferometer's RM map is the RM on scales larger than its inner cut; its PI map is the RM
  on smaller scales.** The large-scale RM rides on the phase of the small-scale structure, so the
  scale-resolved RM transfer function is ≈ 0.8–1 for u < u_min and falls to 0.5 at u ≈ u_min
  (0.6–1.4 u_min in a single realisation). Meanwhile PI = 2λ²|δRM(u > u_min)| (weak-screen law, now
  tested for any λ cut). Small-scale RM appears in PI, not in RM.
* **The interferometer's RM errors sit at its PI nulls** (where the high-passed RM changes sign).
  Keeping the brighter half of the PI map halves the rms error (e.g. 7.3 → 3.7 rad m⁻² for ASKAP).
  Any S/N mask will do this in practice.
* **A larger inner cut gives a *better* RM but less PI**: generic RM error 8.2 → 4.3 rad m⁻² (bright
  half 5.3 → 1.6) from 20 to 320 λ, while PI falls ∝ u_min^−0.95 (expected −(|β|−2)/2 = −0.83 for
  β = −11/3). Without noise this favours large cuts; with noise (test 5) the falling PI will win somewhere.
* **The band matters more than the array.** On the same band, ASKAP and MeerKAT agree to ~10% in RM
  error; MeerKAT's longer shortest baseline costs ~20% of PI. Lower frequencies give more PI
  (angle gradients ∝ λ²; UHF 0.36, L-band 0.09) and a much better RMSF (UHF FWHM 17 rad m⁻²).
  The wider the fractional band, the more a physical cut moves across it (2× at L-band) and the more
  spurious Faraday structure it creates. uv-range matching fixes this, cutting INT RM error 1.5–3.5×;
  L-band goes from worst (8.5) to best (2.4).
* **RM gradients:** the INT PI is the band mean of T_int at the fringe u = |∇RM| λ²/π. At 50 rad m⁻² deg⁻¹
  the fringe spans 69–128 λ over POSSUM band 1, so a cut ≲ 60 λ (ASKAP) keeps PI ≈ 1 where a
  64 m dish has depolarised to 0.35. Above ~110 λ the interferometer loses it.

Caveats: noiseless; one realisation (u₅₀ scatters by tens of per cent); no primary beam or mosaicking;
the 2.3′ grid reaches only u ≈ 730 λ, so outer edges and resolution are not modelled.

Not yet included: noise, Stokes I, Faraday-thick emission, structure in p, multiple realisations.
