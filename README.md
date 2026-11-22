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

Not yet included: noise, Stokes I, Faraday-thick emission, structure in p, multiple realisations.
