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

## Simulator (`src/filter_possum/simulator.py`)

A small simulator of what a single dish and an interferometer recover of a diffuse, Faraday-thin
polarised sky. The sky has polarised intensity 1 everywhere and an RM screen (mean + Kolmogorov
fluctuations + optional gradient) on a 5° × 5° field. Each frequency channel's Q + iU is filtered by
each instrument's uv response at that channel's wavelength, then RM synthesis (rm-lite) gives a
peak RM and peak PI per pixel.

```bash
uv sync
uv run pytest            # physics checks (~5 s)
```

```python
from filter_possum import simulator as fs

sky = fs.Sky(rm_mean=30, rm_sigma=10, rm_grad=0)          # rad m^-2; gradient in rad m^-2 per degree
obs = fs.observe(sky, fs.BANDS["possum1"], single_dish=fs.PARKES, interferometer=fs.ASKAP, combine="joint")
res = fs.rmsynth(obs.cubes["int"], obs.band)               # {"rm", "pi", "phi", "fwhm", ...}
fs.plot_comparison(obs, res, "ASKAP")                      # true / recovered / difference, for RM and PI
```

| Building block | Options |
|---|---|
| `Sky` | `rm_mean`, `rm_sigma`, `rm_slope`, `rm_grad`, `p0`, `npix`, `fov_deg`, `seed` |
| `Band`, `BANDS` | `possum1` (800–1088 MHz), `possum12` (+ 1296–1440 MHz), `uhf` (544–1088), `lband` (856–1712); or your own `(f_min, f_max, width)` segments |
| `SingleDish`, `PARKES` | `diameter_m`, `model` (`"aperture"` with `edge_taper_db`, or `"sharp"`) |
| `Interferometer`, `ASKAP`, `MEERKAT` | `shortest_baseline_m`, `dish_diameter_m`, `edge` (`"soft"`/`"sharp"`), `umin_lambda` (fixed cut in wavelengths), `restore_arcmin` |
| `observe` | `combine` (`"feather"`, `"joint"`), `uv_match` (same angular scales in every channel), `joint_eps` |

`examples/simulator.ipynb` walks through it. The model is noiseless and the interferometer is
perfectly deconvolved, so only its shortest baseline and dish size matter (not its uv density).
