"""What does a single dish vs. an interferometer measure of a diffuse Faraday screen?

Sky model (every pixel Faraday-thin):

    P(l, m, lambda^2) = p0 * exp(2i [chi0 + phi(l, m) lambda^2])
    phi(l, m) = phi_mean + G * l + delta_phi(l, m)

where ``delta_phi`` is a Gaussian random field with 2-D power spectrum
P(k) ~ k^beta and ``G`` is an optional mean-RM gradient (rad m^-2 deg^-1).

Image geometry: SIN projection, i.e. pixels are uniformly spaced in direction
cosines (l, m).  For a 5 deg field the w-term is negligible, so the 2-D DFT of
an (l, m) image is exactly the (u, v) plane in wavelengths.

Mock observation, channel by channel:
    * FFT P(l, m) -> V(u, v)   [u, v in wavelengths]
    * physical baseline length is b = |u| * lambda   [metres]
    * single dish     : keep b <= sd_bmax_m   (includes zero spacing)
    * interferometer  : keep b >= int_bmin_m  (no zero spacing)
  so the *angular* scales each instrument sees change across the band.
  ``cut_mode="fixed"`` instead applies the most conservative common uv range
  in wavelengths for every channel (what you get by uv-range matching /
  smoothing to a common resolution).

The sky is generated on a padded (``pad`` x) periodic box and the central
field is analysed, so the field edges are not artificial periodic boundaries.

Then RM synthesis (rm-lite) on truth / single dish / interferometer cubes and
comparison of peak RM, peak PI, and their angular power spectra.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import dask.array as da
import matplotlib
import matplotlib.lines
import matplotlib.ticker
import matplotlib.transforms

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from rm_lite.tools_3d.rmsynth import rmsynth_3d  # noqa: E402
from rm_lite.utils.synthesis import calc_faraday_peaks  # noqa: E402

C_M_S = 299_792_458.0
DEG = np.pi / 180.0
INSTRUMENTS = ("truth", "sd", "int")  # "comb" is appended when cfg.combine != "none"
LABELS = {"truth": "Truth (no filtering)", "sd": "Single dish", "int": "Interferometer", "comb": "Combined SD+INT"}
COLOURS = {"truth": "k", "sd": "tab:orange", "int": "tab:blue", "comb": "tab:green"}
OBS_ORDER = ("sd", "int", "comb")
ERR_LIM: float | None = None  # RM-error colour limit [rad m^-2] for all maps; None = RMSF FWHM / 2


def _obs(keys) -> list[str]:
    """Observed (non-truth) products present, in plotting order."""
    return [k for k in OBS_ORDER if k in keys]


@dataclass(frozen=True)
class SimConfig:
    """All knobs of one simulation run."""

    fov_deg: float = 5.0  # analysed field, SIN projection
    npix: int = 128  # pixels across the analysed field
    pad: int = 2  # sky simulated on pad*fov box, central fov analysed
    freq_min_hz: float = 800e6
    freq_max_hz: float = 1088e6
    chan_width_hz: float = 1e6  # POSSUM-like 288 x 1 MHz
    # optional list of (f_min, f_max, chan_width) [Hz] bands, e.g. POSSUM band 1 + band 2 with
    # the gap between them; overrides the three fields above when non-empty
    freq_segments: tuple[tuple[float, float, float], ...] = ()
    p0: float = 1.0  # constant polarised fraction (Stokes I = 1)
    chi0_rad: float = 0.0
    rm_mean: float = 30.0  # rad m^-2
    rm_sigma: float = 10.0  # rms of delta_phi over the padded box, rad m^-2
    rm_slope: float = -11.0 / 3.0  # 2-D power-spectrum slope of delta_phi
    rm_grad_deg: float = 0.0  # rad m^-2 per degree, along l
    grad_ramp_frac: float = 0.7  # fraction of padded box the gradient ramp spans
    sd_bmax_m: float = 64.0  # single-dish diameter (Murriyang/Parkes); runs before test 2 used 100 m
    int_bmin_m: float = 30.0
    cut_mode: str = "baseline"  # "baseline" (physical) or "fixed" (common uv range)
    # --- instrument response models (test 1: realistic responses)
    sd_model: str = "sharp"  # "sharp" (b <= sd_bmax_m) or "aperture" (tapered-dish autocorrelation)
    sd_edge_taper_db: float = 12.0  # illumination edge taper (power) for sd_model="aperture"
    int_inner: str = "sharp"  # "sharp" (b >= int_bmin_m) or "soft" (edge smeared by antenna aperture)
    ant_diam_m: float = 12.0  # antenna diameter setting the soft inner-edge width
    int_restore_arcmin: float = 0.0  # Gaussian restoring beam FWHM, fixed across band; 0 = none
    # generic interferometer (test 3): sharp inner cut at a fixed uv distance in wavelengths, the
    # same in every channel; > 0 overrides int_bmin_m / int_inner / cut_mode for the INT
    int_umin_lam: float = 0.0
    scale_metrics: bool = False  # add scale-resolved RM transfer functions to the metrics
    # --- SD + INT combination (test 2)
    combine: str = "none"  # "none", "feather" (CASA-style) or "joint" (Wiener-regularised)
    joint_eps: float = 1e-2  # Wiener regulariser for "joint" (stands in for finite S/N)
    phi_max: float = 400.0
    dphi: float = 2.0
    seed: int = 42

    @property
    def freqs(self) -> np.ndarray:
        # channel centres, e.g. 800.5 ... 1087.5 MHz
        segs = self.freq_segments or ((self.freq_min_hz, self.freq_max_hz, self.chan_width_hz),)
        out = []
        for f0, f1, df in segs:
            n = int(round((f1 - f0) / df))
            out.append(f0 + (np.arange(n) + 0.5) * df)
        return np.concatenate(out)

    @property
    def lams(self) -> np.ndarray:
        return C_M_S / self.freqs

    @property
    def dl(self) -> float:
        return self.fov_deg / self.npix * DEG  # pixel size in direction cosine

    @property
    def nbig(self) -> int:
        return self.npix * self.pad


# --------------------------------------------------------------------------- sky


def uv_grid(n: int, dl: float) -> np.ndarray:
    """|u| in wavelengths on the unshifted FFT grid of an n x n image."""
    u = np.fft.fftfreq(n, dl)
    return np.hypot(*np.meshgrid(u, u, indexing="xy"))


def gaussian_random_field(n: int, dl: float, slope: float, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Periodic zero-mean GRF with 2-D power spectrum ~ k^slope, std = sigma."""
    k = uv_grid(n, dl)
    amp = np.zeros_like(k)
    amp[k > 0] = k[k > 0] ** (slope / 2.0)  # amplitude = sqrt(power)
    noise = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    field = np.fft.ifft2(noise * amp).real
    field -= field.mean()
    return field * (sigma / field.std())


def periodic_gradient(n: int, dl: float, grad_per_rad: float, ramp_frac: float) -> np.ndarray:
    """Linear RM gradient along l that is periodic over the padded box.

    The ramp (slope G) covers |l| <= a; in the guard band it folds back so the
    field is continuous across the FFT wrap.  Keeps the analysed region a pure
    gradient without a wrap-around discontinuity ringing through sharp uv cuts.
    """
    x = (np.arange(n) - n / 2) * dl
    half = n * dl / 2
    a = ramp_frac * half
    prof = np.where(
        np.abs(x) <= a,
        grad_per_rad * x,
        np.sign(x) * grad_per_rad * a * (half - np.abs(x)) / (half - a),
    )
    return np.broadcast_to(prof[None, :], (n, n)).copy()


def make_rm_sky(cfg: SimConfig) -> np.ndarray:
    rng = np.random.default_rng(cfg.seed)
    n = cfg.nbig
    phi = np.full((n, n), cfg.rm_mean)
    if cfg.rm_sigma > 0:
        phi += gaussian_random_field(n, cfg.dl, cfg.rm_slope, cfg.rm_sigma, rng)
    if cfg.rm_grad_deg != 0:
        phi += periodic_gradient(n, cfg.dl, cfg.rm_grad_deg / DEG, cfg.grad_ramp_frac)
    return phi


def crop(arr: np.ndarray, cfg: SimConfig) -> np.ndarray:
    s = slice(cfg.nbig // 2 - cfg.npix // 2, cfg.nbig // 2 + cfg.npix // 2)
    return arr[..., s, s]


# ------------------------------------------------------------- mock observation


def aperture_autocorr(edge_taper_db: float, n: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """Radial autocorrelation of a circular dish illumination, vs rho = b / D in [0, 1].

    Illumination A(r) = 1 - (1 - tau) (2r/D)^2 (parabolic taper, tau = edge amplitude),
    the usual single-dish model. Its autocorrelation is the uv transfer function of the
    dish's power beam: 1 at zero spacing, falling to 0 at b = D. edge_taper_db = 0 gives a
    uniformly illuminated dish ("Chinese hat").
    """
    tau = 10 ** (-edge_taper_db / 20.0)
    x = (np.arange(n) - n / 2) / (n / 4)  # aperture radius = n/4 px, diameter n/2 -> lags fit without wrap
    r = np.hypot(*np.meshgrid(x, x))
    a = np.where(r <= 1.0, 1 - (1 - tau) * r**2, 0.0)  # r here is 2r/D
    ac = np.fft.fftshift(np.fft.ifft2(np.abs(np.fft.fft2(a)) ** 2).real)
    prof = ac[n // 2, n // 2 :] / ac[n // 2, n // 2]  # cut along one axis, lag in units of D/2
    rho = np.arange(prof.size) / (n / 4) / 2.0  # lag / D
    keep = rho <= 1.0
    return rho[keep], np.clip(prof[keep], 0, None)


def dish_fwhm_factor(edge_taper_db: float) -> float:
    """Power-beam FWHM in units of lambda/D, from the Hankel transform of the transfer function."""
    from scipy.special import j0

    rho, prof = _ac(edge_taper_db)
    t = np.linspace(0, 1.5, 3001)
    beam = np.array([np.trapezoid(prof * j0(2 * np.pi * rho * ti) * rho, rho) for ti in t])
    return float(2 * t[np.argmin(np.abs(beam / beam[0] - 0.5))])


def predicted_gradient_depol(cfg: SimConfig, grad_deg: float) -> float:
    """Exact band-mean single-dish PI for a pure linear RM gradient.

    A gradient G (rad m^-2 rad^-1) makes P a plane wave at u = G lambda^2 / pi, so the
    dish passes it with weight T_sd(u * lambda_beam): p(lambda) = T_sd(G lambda^2 lambda_beam / pi),
    with lambda_beam = lambda (physical) or lambda_max ("fixed"). T_sd is real and positive,
    so the phase (RM) is untouched and the FDF peak is the band mean of p(lambda).
    The Gaussian-beam approximation exp(-(2 lambda^2 G theta)^2 / (16 ln 2)) agrees to
    first order but underestimates p by ~10% once p < 0.5.
    """
    lam = cfg.lams
    lam_beam = np.full_like(lam, lam.max()) if cfg.cut_mode == "fixed" else lam
    b_fringe = (grad_deg / DEG) * lam**2 * lam_beam / np.pi
    return float(np.mean(sd_transfer(cfg, b_fringe)))


def predicted_gradient_int_pi(cfg: SimConfig, grad_deg: float) -> float:
    """Band-mean interferometer PI for a pure linear RM gradient (cf. predicted_gradient_depol).

    The gradient's Q/U fringe sits at u = G lambda^2 / pi; the INT passes it with weight T_int(u), real
    and positive, so the FDF peak is the band mean of T_int at the fringe. For a sharp cut in λ this is
    the fraction of channels whose fringe clears u_min. Exact for an infinite gradient; the simulated
    ramp is ~7 deg long, so each fringe is ~8 λ wide and a *sharp* cut is only matched to within that
    (the smooth single-dish response is not sensitive to it, hence its exact agreement).
    """
    u_fringe = (grad_deg / DEG) * cfg.lams**2 / np.pi
    return float(np.mean([uv_masks(cfg, np.array([u]), lam)[1][0] for u, lam in zip(u_fringe, cfg.lams, strict=True)]))


_AC_CACHE: dict[float, tuple[np.ndarray, np.ndarray]] = {}


def _ac(edge_taper_db: float) -> tuple[np.ndarray, np.ndarray]:
    if edge_taper_db not in _AC_CACHE:
        _AC_CACHE[edge_taper_db] = aperture_autocorr(edge_taper_db)
    return _AC_CACHE[edge_taper_db]


def sd_transfer(cfg: SimConfig, b_m: np.ndarray) -> np.ndarray:
    """Single-dish uv transfer function vs physical baseline b [m]."""
    if cfg.sd_model == "sharp":
        return (b_m <= cfg.sd_bmax_m).astype(float)
    if cfg.sd_model == "aperture":
        rho, prof = _ac(cfg.sd_edge_taper_db)
        return np.interp(b_m / cfg.sd_bmax_m, rho, prof, right=0.0)
    raise ValueError(cfg.sd_model)


def int_inner_transfer(cfg: SimConfig, b_m: np.ndarray) -> np.ndarray:
    """Interferometer inner-edge response vs physical baseline b [m].

    "soft": the nominal step at int_bmin_m is smeared over +/- ant_diam_m, because a
    baseline b made with dishes of diameter d samples |u| lambda in [b - d, b + d] with
    weight given by the antenna aperture autocorrelation (1-D cut, uniform illumination).
    """
    if cfg.int_inner == "sharp":
        return (b_m >= cfg.int_bmin_m).astype(float)
    if cfg.int_inner == "soft":
        rho, prof = _ac(0.0)
        x = np.concatenate([-rho[::-1], rho[1:]]) * cfg.ant_diam_m  # kernel support [-d, d]
        k = np.concatenate([prof[::-1], prof[1:]])
        cdf = np.cumsum(k) / k.sum()
        return np.interp(b_m - cfg.int_bmin_m, x, cdf, left=0.0, right=1.0)
    raise ValueError(cfg.int_inner)


def restore_transfer(cfg: SimConfig, uvdist_lam: np.ndarray) -> np.ndarray:
    """Gaussian restoring beam, same angular FWHM in every channel (common resolution)."""
    if cfg.int_restore_arcmin <= 0:
        return np.ones_like(uvdist_lam)
    theta = cfg.int_restore_arcmin / 60.0 * DEG
    return np.exp(-((np.pi * theta * uvdist_lam) ** 2) / (4 * np.log(2)))


def uv_masks(cfg: SimConfig, uvdist_lam: np.ndarray, lam: float) -> tuple[np.ndarray, np.ndarray]:
    """(single dish, interferometer) uv transfer functions for one channel.

    "baseline": responses evaluated at the channel's own wavelength (physical).
    "fixed": SD response evaluated at the longest wavelength (smoothing to a common
    resolution) and INT inner edge at the shortest (uv-range matching); because both
    responses are monotonic in b this is exactly what those operations achieve.
    Reduces to the original sharp masks for sd_model = int_inner = "sharp".
    """
    if cfg.cut_mode == "baseline":
        lam_sd = lam_int = lam
    elif cfg.cut_mode == "fixed":
        lam_sd, lam_int = cfg.lams.max(), cfg.lams.min()
    else:
        raise ValueError(cfg.cut_mode)
    t_sd = sd_transfer(cfg, uvdist_lam * lam_sd)
    if cfg.int_umin_lam > 0:
        inner = (uvdist_lam >= cfg.int_umin_lam).astype(float)  # generic: cut fixed in wavelengths
    else:
        inner = int_inner_transfer(cfg, uvdist_lam * lam_int)
    t_int = inner * restore_transfer(cfg, uvdist_lam)
    return t_sd, t_int


def combined_transfer(cfg: SimConfig, t_sd: np.ndarray, t_int: np.ndarray) -> np.ndarray:
    """Effective uv transfer function of the SD+INT combination for one channel.

    Each product is (its transfer function) x (true visibilities), so a linear
    combination of the two is another transfer function acting on the truth.

    feather: V = V_sd + (1 - T_sd) V_int    (CASA feather / immerge, in surface-brightness
             units so no beam-area rescaling).  T = T_sd + (1 - T_sd) T_int, i.e. the
             missing fraction is (1 - T_sd)(1 - T_int): the SD response is not deconvolved.
    joint:   V = (T_sd V_sd + T_int V_int) / (T_sd^2 + T_int^2 + eps)   (least-squares /
             Wiener estimate, the noiseless limit of joint deconvolution).
             T = S / (S + eps) with S = T_sd^2 + T_int^2.
    """
    if cfg.combine == "feather":
        return t_sd + (1.0 - t_sd) * t_int
    if cfg.combine == "joint":
        s2 = t_sd**2 + t_int**2
        return s2 / (s2 + cfg.joint_eps)
    raise ValueError(cfg.combine)


def observe(cfg: SimConfig, phi_big: np.ndarray) -> dict[str, np.ndarray]:
    """Return cropped complex P cubes (nchan, npix, npix) for truth / sd / int [/ comb]."""
    uvd = uv_grid(cfg.nbig, cfg.dl)
    nchan = cfg.freqs.size
    keys = INSTRUMENTS + (("comb",) if cfg.combine != "none" else ())
    out = {k: np.empty((nchan, cfg.npix, cfg.npix), np.complex128) for k in keys}
    for i, lam in enumerate(cfg.lams):
        p = cfg.p0 * np.exp(2j * (cfg.chi0_rad + phi_big * lam**2))
        vis = np.fft.fft2(p)
        m_sd, m_int = uv_masks(cfg, uvd, lam)
        out["truth"][i] = crop(p, cfg)
        # Q and U are filtered together as one complex field; transfer functions
        # are real and radially symmetric, so Q and U individually stay real.
        out["sd"][i] = crop(np.fft.ifft2(vis * m_sd), cfg)
        out["int"][i] = crop(np.fft.ifft2(vis * m_int), cfg)
        if "comb" in out:
            out["comb"][i] = crop(np.fft.ifft2(vis * combined_transfer(cfg, m_sd, m_int)), cfg)
    return out


# --------------------------------------------------------------- RM synthesis


def rm_synthesis(cube: np.ndarray, cfg: SimConfig, keep_pix: list[tuple[int, int]] | None = None) -> dict:
    """rm-lite 3-D RM synthesis + sub-sample peak finding on a complex P cube."""
    n = cube.shape[-1]
    q = da.from_array(cube.real, chunks=(-1, n, n))
    u = da.from_array(cube.imag, chunks=(-1, n, n))
    res = rmsynth_3d(
        q,
        u,
        cfg.freqs,
        weight_arr=np.ones(cfg.freqs.size),  # uniform weights, noiseless
        phi_max_radm2=cfg.phi_max,
        d_phi_radm2=cfg.dphi,
        log_level=40,
    )
    fdf = res.fdf_dirty_cube.compute()
    pk = calc_faraday_peaks(
        fdf,
        res.phi_arr_radm2,
        res.fwhm_rmsf_radm2,
        axis=0,
        lam_sq_0_m2=res.lam_sq_0_m2,
        lambda_sq_arr_m2=res.lambda_sq_arr_m2,
    )
    out = {
        "rm": np.asarray(pk.peak_rm_radm2),
        "pi": np.asarray(pk.peak_pi),
        "phi": np.asarray(res.phi_arr_radm2),
        "fwhm": float(res.fwhm_rmsf_radm2),
        "lam_sq_0": float(res.lam_sq_0_m2),
    }
    if keep_pix:
        out["fdf_pix"] = np.stack([fdf[:, y, x] for y, x in keep_pix])
    return out


# --------------------------------------------------------------- diagnostics


def radial_power(img: np.ndarray, dl: float, nbins: int = 28) -> tuple[np.ndarray, np.ndarray]:
    """Azimuthally averaged power spectrum of a (non-periodic) map.

    Mean removed and Hann-apodised to suppress edge leakage; normalised by the
    window's mean square so amplitudes are comparable between maps.
    """
    n = img.shape[0]
    w = np.outer(np.hanning(n), np.hanning(n))
    f = np.fft.fft2((img - img.mean()) * w)
    ps = np.abs(f) ** 2 * dl**2 / n**2 / np.mean(w**2)
    k = uv_grid(n, dl)
    edges = np.geomspace(1.0 / (n * dl), 0.5 / dl, nbins + 1)
    idx = np.digitize(k.ravel(), edges)
    centres = np.sqrt(edges[1:] * edges[:-1])
    filled = [i for i in range(1, nbins + 1) if np.any(idx == i)]
    pk = np.array([ps.ravel()[idx == i].mean() for i in filled])
    return centres[np.array(filled) - 1], pk


def rm_transfer(rec: np.ndarray, true: np.ndarray, dl: float, nbins: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Scale-resolved RM transfer function: Re<R T*> / <|T|^2> in radial uv bins.

    Projection of the recovered RM map onto the true one, scale by scale (both mean-removed,
    Hann-apodised like ``radial_power``). 1 = that scale is recovered, 0 = lost, < 0 = sign-flipped.
    Unlike a ratio of power spectra it is not inflated by spurious (uncorrelated) structure.
    """
    n = true.shape[0]
    w = np.outer(np.hanning(n), np.hanning(n))
    fr = np.fft.fft2((rec - np.nanmean(rec)) * w)
    ft = np.fft.fft2((true - true.mean()) * w)
    k = uv_grid(n, dl).ravel()
    edges = np.geomspace(1.0 / (n * dl), 0.5 / dl, nbins + 1)
    idx = np.digitize(k, edges)
    cross = (fr * ft.conj()).real.ravel()
    pt = (np.abs(ft) ** 2).ravel()
    filled = [i for i in range(1, nbins + 1) if np.any(idx == i)]
    t = np.array([cross[idx == i].sum() / pt[idx == i].sum() for i in filled])
    return np.sqrt(edges[1:] * edges[:-1])[np.array(filled) - 1], t


def transfer_u50(u: np.ndarray, t: np.ndarray) -> float:
    """uv distance where a transfer function, recovered (>= 0.5) at large scales, first falls below 0.5.

    Log-interpolated between bins; NaN if it never reaches 0.5, inf if it never falls below.
    """
    above = t >= 0.5
    if not above[0]:
        return float("nan")
    i = np.argmin(above)  # first False
    if above[i]:
        return float("inf")
    x0, x1 = np.log(u[i - 1]), np.log(u[i])
    return float(np.exp(x0 + (t[i - 1] - 0.5) / (t[i - 1] - t[i]) * (x1 - x0)))


def apply_transfer(img_big: np.ndarray, cfg: SimConfig, transfer: np.ndarray) -> np.ndarray:
    """Filter a padded-box map with a uv transfer function, cropped to the field."""
    return crop(np.fft.ifft2(np.fft.fft2(img_big) * transfer).real, cfg)


def metrics(rm_true: np.ndarray, rm_refs: dict, res: dict, cubes: dict) -> dict:
    out = {}
    tot = np.mean(np.abs(cubes["truth"]) ** 2)
    for k in res:
        d = res[k]["rm"] - rm_true
        pi = res[k]["pi"]
        out[k] = {
            "rm_mean_recovered": float(np.nanmean(res[k]["rm"])),
            "rm_bias": float(np.nanmean(d)),
            "rm_rms_err": float(np.sqrt(np.nanmean(d**2))),
            "rm_rms_err_pi_weighted": float(np.sqrt(np.nansum(pi * d**2) / np.nansum(pi))),
            "rm_corr_with_true": float(np.corrcoef(res[k]["rm"].ravel(), rm_true.ravel())[0, 1]),
            "frac_abs_err_gt_halfFWHM": float(np.mean(np.abs(d) > res[k]["fwhm"] / 2)),
            "pi_mean": float(np.nanmean(pi)),
            "pi_std": float(np.nanstd(pi)),
            "pi_p05_p95": [float(x) for x in np.nanpercentile(pi, [5, 95])],
            # band-averaged fraction of |P|^2 that survives the uv filter
            "qu_power_retained": float(np.mean(np.abs(cubes[k]) ** 2) / tot),
        }
    for k, ref in rm_refs.items():
        d = res[k]["rm"] - ref
        out[k]["rm_rms_err_vs_smoothed_truth"] = float(np.sqrt(np.nanmean(d**2)))
    return out


# --------------------------------------------------------------------- plots


def _imshow(ax, img, cfg, title, **kw):
    ext = [cfg.fov_deg / 2, -cfg.fov_deg / 2, -cfg.fov_deg / 2, cfg.fov_deg / 2]  # RA increases left
    im = ax.imshow(img, origin="lower", extent=ext, **kw)
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03).ax.tick_params(labelsize=7)


def plot_maps(cfg, rm_true, res, cubes, ichan, path, tag):
    lo, hi = np.percentile(rm_true, [1, 99])
    pad_ = 0.15 * (hi - lo)
    rm_kw = dict(cmap="RdBu_r", vmin=lo - pad_, vmax=hi + pad_)
    # one linear error scale for every row and every run (default +/- RMSF FWHM / 2)
    err_lim = ERR_LIM if ERR_LIM is not None else res["truth"]["fwhm"] / 2
    keys = list(cubes)
    fig, axs = plt.subplots(len(keys), 4, figsize=(15, 3.5 * len(keys)), constrained_layout=True)
    for r, k in enumerate(keys):
        _imshow(
            axs[r, 0],
            cubes[k][ichan].real,
            cfg,
            f"{LABELS[k]}: Q @ {cfg.freqs[ichan] / 1e6:.0f} MHz",
            cmap="RdBu_r",
            vmin=-1,
            vmax=1,
        )
        _imshow(axs[r, 1], res[k]["rm"], cfg, f"{LABELS[k]}: peak RM [rad m$^{{-2}}$]", **rm_kw)
        _imshow(
            axs[r, 2],
            res[k]["rm"] - rm_true,
            cfg,
            f"{LABELS[k]}: RM − true RM [rad m$^{{-2}}$]",
            cmap="PuOr_r",
            vmin=-err_lim,
            vmax=err_lim,
        )
        _imshow(axs[r, 3], res[k]["pi"], cfg, f"{LABELS[k]}: peak PI / p$_0$", cmap="viridis", vmin=0, vmax=1.05)
    for ax in axs[-1]:
        ax.set_xlabel("l offset [deg]", fontsize=8)
    fig.suptitle(f"{tag}   (true PI = 1 everywhere; truth row is RM-synthesis of the unfiltered cube)", fontsize=11)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _shade_cuts(ax, cfg):
    lmin, lmax = cfg.lams.min(), cfg.lams.max()
    if cfg.int_umin_lam > 0:
        ax.axvline(cfg.int_umin_lam, color=COLOURS["int"], ls=":", label=f"INT inner cut ({cfg.int_umin_lam:g} λ)")
        ax.axvspan(cfg.sd_bmax_m / lmax, cfg.sd_bmax_m / lmin, color=COLOURS["sd"], alpha=0.15, label="SD outer cut")
    elif cfg.cut_mode == "baseline":
        ax.axvspan(
            cfg.sd_bmax_m / lmax,
            cfg.sd_bmax_m / lmin,
            color=COLOURS["sd"],
            alpha=0.15,
            label=f"SD outer cut ({cfg.sd_bmax_m:.0f} m across band)",
        )
        ax.axvspan(
            cfg.int_bmin_m / lmax,
            cfg.int_bmin_m / lmin,
            color=COLOURS["int"],
            alpha=0.15,
            label=f"INT inner cut ({cfg.int_bmin_m:.0f} m across band)",
        )
    else:
        ax.axvline(cfg.sd_bmax_m / lmax, color=COLOURS["sd"], ls=":", label="SD common cut")
        ax.axvline(cfg.int_bmin_m / lmin, color=COLOURS["int"], ls=":", label="INT common cut")


def _scale_axis(ax):
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    sec = ax.secondary_xaxis(
        "top", functions=(lambda u: 1 / np.maximum(u, 1e-9) / DEG, lambda s: 1 / np.maximum(s, 1e-9) / DEG)
    )
    sec.set_xlabel("angular scale 1/u [deg]", fontsize=8)
    sec.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    sec.tick_params(labelsize=7)


def plot_spectra(cfg, rm_true, res, cubes, ichan, path, tag):
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    # (a) RM maps
    k, p = radial_power(rm_true, cfg.dl)
    axs[0].loglog(k, p, color="0.5", lw=4, alpha=0.5, label="input RM")
    for key in _obs(cubes):
        k, p = radial_power(res[key]["rm"], cfg.dl)
        axs[0].loglog(k, p, color=COLOURS[key], label=f"{LABELS[key]} RM")
        k, p = radial_power(res[key]["rm"] - rm_true, cfg.dl)
        axs[0].loglog(k, p, color=COLOURS[key], ls="--", label=f"{LABELS[key]} RM error")
    axs[0].set(title="Power spectrum of RM maps", ylabel="P(u) [(rad m$^{-2}$)$^2$ sr]")
    # (b) PI maps: truth is constant -> zero power at u > 0
    for key in _obs(cubes):
        k, p = radial_power(res[key]["pi"], cfg.dl)
        axs[1].loglog(k, p, color=COLOURS[key], label=f"{LABELS[key]} PI")
    k, p = radial_power(res["truth"]["pi"], cfg.dl)
    axs[1].loglog(k, np.maximum(p, 1e-30), color="0.5", lw=4, alpha=0.5, label="input PI (≈0, constant)")
    axs[1].set(title="Power spectrum of peak-PI maps", ylabel="P(u) [sr]", ylim=(1e-12, None))
    # (c) complex P at one channel
    for key in cubes:
        k, p = radial_power(cubes[key][ichan], cfg.dl)
        kw = dict(color="0.5", lw=4, alpha=0.5) if key == "truth" else dict(color=COLOURS[key])
        axs[2].loglog(k, p, label=LABELS[key], **kw)
    axs[2].set(
        title=f"Power spectrum of Q+iU @ {cfg.freqs[ichan] / 1e6:.0f} MHz", ylabel="P(u) [sr]", ylim=(1e-12, None)
    )
    for ax in axs:
        _shade_cuts(ax, cfg)
        _scale_axis(ax)
        ax.set_xlabel("uv distance u [wavelengths]")
        ax.legend(fontsize=7)
    fig.suptitle(tag, fontsize=11)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_diagnostics(cfg, rm_true, res, cubes, pix, path, tag):
    fig, axs = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    a = axs[0]
    tot = np.mean(np.abs(cubes["truth"]) ** 2, axis=(1, 2))
    for key in _obs(cubes):
        frac = np.mean(np.abs(cubes[key]) ** 2, axis=(1, 2)) / tot
        a[0].plot(cfg.freqs / 1e6, frac, color=COLOURS[key], label=LABELS[key])
    a[0].set(
        xlabel="frequency [MHz]",
        ylabel=r"$\langle|P_{\rm obs}|^2\rangle / \langle|P_{\rm true}|^2\rangle$",
        title="Polarised power retained per channel",
        ylim=(0, 1.05),
    )
    a[0].legend(fontsize=8)
    for key in _obs(cubes)[::-1]:  # SD drawn last so it stays visible
        a[1].plot(rm_true.ravel(), res[key]["rm"].ravel(), ",", color=COLOURS[key], alpha=0.5)
        a[1].plot([], [], "o", color=COLOURS[key], label=LABELS[key])
    lims = [np.nanmin(rm_true) - 10, np.nanmax(rm_true) + 10]
    a[1].plot(lims, lims, "k--", lw=1, label="1:1")
    a[1].set(
        xlabel="input RM [rad m$^{-2}$]",
        ylabel="recovered peak RM [rad m$^{-2}$]",
        title="Pixel-by-pixel RM",
        xlim=lims,
        ylim=(lims[0] - 30, lims[1] + 30),
    )
    a[1].legend(fontsize=8)
    fw = res["truth"]["fwhm"]
    bins = np.linspace(-fw, fw, 121)
    for key in _obs(cubes):
        a[2].hist(
            (res[key]["rm"] - rm_true).ravel(),
            bins=bins,
            histtype="step",
            color=COLOURS[key],
            label=LABELS[key],
            density=True,
            lw=1.5,
        )
    a[2].set(
        yscale="log", xlabel="RM − true RM [rad m$^{-2}$] (±RMSF FWHM)", ylabel="pdf", title="RM error distribution"
    )
    a[2].legend(fontsize=8)
    phi = res["truth"]["phi"]
    for j, (y, x) in enumerate(pix):
        ax = axs[1, j]
        for key in cubes:
            ax.plot(
                phi,
                np.abs(res[key]["fdf_pix"][j]),
                color=COLOURS[key],
                label=LABELS[key],
                lw=2.5 if key == "truth" else 1.3,
                alpha=0.5 if key == "truth" else 1,
            )
        ax.axvline(rm_true[y, x], color="k", ls=":", lw=1, label="input RM")
        ax.set(
            xlim=(rm_true[y, x] - 150, rm_true[y, x] + 150),
            xlabel=r"$\phi$ [rad m$^{-2}$]",
            ylabel=r"|F($\phi$)|",
            title=f"FDF, pixel (y={y}, x={x})",
        )
        ax.legend(fontsize=7)
    fig.suptitle(tag, fontsize=11)
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------- runs


def _rounded(obj, sig: int = 6):
    """Round floats for JSON output: readable diffs, no spurious 16-digit mantissas."""
    if isinstance(obj, float):
        return float(f"{obj:.{sig}g}")
    if isinstance(obj, dict):
        return {k: _rounded(v, sig) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_rounded(v, sig) for v in obj]
    return obj


def _dump(obj) -> str:
    return json.dumps(_rounded(obj), indent=2)


def run(cfg: SimConfig, outdir: Path | None = None, tag: str = "run") -> dict:
    phi_big = make_rm_sky(cfg)
    rm_true = crop(phi_big, cfg)
    cubes = observe(cfg, phi_big)
    n = cfg.npix
    pix = [(n // 2, n // 2), (n // 4, 3 * n // 4), (3 * n // 4, n // 3)] if outdir else None
    res = {k: rm_synthesis(cubes[k], cfg, keep_pix=pix) for k in cubes}
    # "best each instrument could hope for": input RM smoothed by its *resolution*
    # (SD response at lambda_0, or at lambda_max for "fixed"; INT restoring beam only,
    # since large-scale RM is not intrinsically lost by an interferometer)
    lam0 = np.sqrt(res["truth"]["lam_sq_0"])
    uvd = uv_grid(cfg.nbig, cfg.dl)
    lam_sd = cfg.lams.max() if cfg.cut_mode == "fixed" else lam0
    rm_refs = {
        "sd": apply_transfer(phi_big, cfg, sd_transfer(cfg, uvd * lam_sd)),
        "int": apply_transfer(phi_big, cfg, restore_transfer(cfg, uvd)),
    }
    if "comb" in cubes:
        rm_refs["comb"] = rm_refs["int"]
    m = metrics(rm_true, rm_refs, res, cubes)
    m["input"] = {
        "rm_mean": float(rm_true.mean()),
        "rm_std": float(rm_true.std()),
        "fwhm_rmsf": res["truth"]["fwhm"],
        "lam_sq_0": res["truth"]["lam_sq_0"],
    }
    if cfg.scale_metrics:
        for k in res:
            rm_k = np.nan_to_num(res[k]["rm"], nan=np.nanmean(res[k]["rm"]))
            u, t = rm_transfer(rm_k, rm_true, cfg.dl)
            m[k]["rm_transfer"] = t.tolist()
            m[k]["rm_transfer_u50"] = transfer_u50(u, t)
            # an observer masks on PI (S/N): RM error over the brighter half of this product's PI map
            bright = res[k]["pi"] >= np.nanmedian(res[k]["pi"])
            m[k]["rm_rms_err_bright_half"] = float(np.sqrt(np.nanmean((res[k]["rm"] - rm_true)[bright] ** 2)))
        m["input"]["rm_transfer_u"] = u.tolist()
        m["input"]["freq_range_mhz"] = [cfg.freqs.min() / 1e6, cfg.freqs.max() / 1e6]
    if outdir is not None:
        outdir.mkdir(parents=True, exist_ok=True)
        ichan = int(np.argmin(np.abs(cfg.freqs - np.sqrt(cfg.freqs.min() * cfg.freqs.max()))))
        label = (
            f"{tag}: <RM>={cfg.rm_mean:g}, σ(δRM)={cfg.rm_sigma:g} rad m⁻², β={cfg.rm_slope:.2f}, "
            f"∇RM={cfg.rm_grad_deg:g} rad m⁻² deg⁻¹, cuts={cfg.cut_mode}\n"
            f"SD={cfg.sd_model} {cfg.sd_bmax_m:g} m, "
            + (
                f"INT u ≥ {cfg.int_umin_lam:g} λ, "
                if cfg.int_umin_lam > 0
                else f"INT b ≥ {cfg.int_bmin_m:g} m ({cfg.int_inner}, D={cfg.ant_diam_m:g} m), "
            )
            + f"{cfg.freqs.min() / 1e6:.0f}–{cfg.freqs.max() / 1e6:.0f} MHz, "
            f"restore={cfg.int_restore_arcmin:g}′, combine={cfg.combine}"
        )
        plot_maps(cfg, rm_true, res, cubes, ichan, outdir / f"{tag}_maps.png", label)
        plot_spectra(cfg, rm_true, res, cubes, ichan, outdir / f"{tag}_power_spectra.png", label)
        plot_diagnostics(cfg, rm_true, res, cubes, pix, outdir / f"{tag}_diagnostics.png", label)
        (outdir / f"{tag}_metrics.json").write_text(_dump({"config": asdict(cfg), "metrics": m}))
    return m


def fringe_gradient(b_m: float, lam: float) -> float:
    """|grad RM| [rad m^-2 deg^-1] whose Faraday fringes sit at baseline b_m.

    A local RM gradient g (rad m^-2 rad^-1) winds the angle as 2 g lambda^2 l, i.e.
    a fringe at u = g lambda^2 / pi wavelengths, i.e. a baseline b = g lambda^3 / pi.
    """
    return np.pi * b_m / lam**3 * DEG


def plot_scan(out: dict, param: str, cfg: SimConfig, outdir: Path, xlabel: str, prefix: str = "") -> None:
    values = out["values"]
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    for mode, ls in (("baseline", "-"), ("fixed", ":")):
        for key in _obs(out[mode][0]):
            lab = f"{LABELS[key]} ({mode} cuts)"
            for ax, met in zip(axs, ("rm_rms_err", "rm_bias", "pi_mean"), strict=True):
                ax.plot(values, [r[key][met] for r in out[mode]], ls, marker="o", color=COLOURS[key], label=lab)
    if param == "rm_sigma":
        axs[0].plot(values, [r["input"]["rm_std"] for r in out["baseline"]], "k--", lw=1, label="input RM std")
    if param == "rm_grad_deg":
        # analytic fringe thresholds across the band (see fringe_gradient)
        lmin, lmax = cfg.lams.min(), cfg.lams.max()
        for ax in axs:
            ax.axvspan(
                fringe_gradient(cfg.int_bmin_m, lmax),
                fringe_gradient(cfg.int_bmin_m, lmin),
                color=COLOURS["int"],
                alpha=0.12,
                label="fringe crosses INT inner cut within band",
            )
            if cfg.sd_model == "sharp":
                ax.axvline(
                    fringe_gradient(cfg.sd_bmax_m, lmax),
                    color=COLOURS["sd"],
                    lw=1,
                    ls="--",
                    label="fringe reaches SD outer cut (band bottom)",
                )
        if cfg.sd_model == "aperture":
            # analytic check: (PI at zero gradient) x (dish transfer at the gradient's Faraday fringe)
            for mode, ls in (("baseline", "-"), ("fixed", ":")):
                c = replace(cfg, cut_mode=mode)
                p0 = out[mode][0]["sd"]["pi_mean"]
                g = np.linspace(min(values), max(values), 101)
                axs[2].plot(
                    g,
                    [p0 * predicted_gradient_depol(c, gi) for gi in g],
                    ls,
                    color="k",
                    lw=0.8,
                    label=f"SD fringe-transfer prediction ({mode})",
                )
            axs[2].legend(fontsize=7)
    axs[0].axhline(out["baseline"][0]["input"]["fwhm_rmsf"] / 2, color="0.6", lw=0.8, label="RMSF FWHM / 2")
    axs[0].set(ylabel="rms(RM − true) [rad m$^{-2}$]", title="RM error")
    axs[1].axhline(0, color="0.6", lw=0.8)
    axs[1].set(ylabel="mean(RM − true) [rad m$^{-2}$]", title="RM bias")
    axs[2].set(ylabel="mean peak PI / p$_0$", title="Recovered PI (true = 1)", ylim=(0, 1.05))
    for ax in axs:
        ax.set_xlabel(xlabel)
    axs[0].legend(fontsize=7)
    fig.suptitle(
        f"{prefix or 'sharp cuts'}: SD={cfg.sd_model} {cfg.sd_bmax_m:g} m, INT inner={cfg.int_inner}, "
        f"restore={cfg.int_restore_arcmin:g}′, combine={cfg.combine}",
        fontsize=10,
    )
    fig.savefig(outdir / f"{prefix}scan_{param}.png", dpi=110)
    plt.close(fig)


def scan(base: SimConfig, param: str, values: list[float], outdir: Path, xlabel: str, prefix: str = "") -> dict:
    """Scan one parameter for both cut modes and plot RM error, RM bias and mean PI.

    A generic-interferometer scan (int_umin_lam) runs physical cuts only: its INT cut is fixed in
    wavelengths already, so "fixed" would only change the single dish.
    """
    out = {"values": values}
    modes = ("baseline",) if param == "int_umin_lam" else ("baseline", "fixed")
    for mode in modes:
        out[mode] = [run(replace(base, **{param: v}, cut_mode=mode)) for v in values]
    (outdir / f"{prefix}scan_{param}.json").write_text(_dump(out))
    if param == "int_umin_lam":
        plot_ucut_scan(out, base, outdir, prefix)
    else:
        plot_scan(out, param, base, outdir, xlabel, prefix)
    return out


def _array_ucut_ranges() -> list[tuple[str, float, float, str]]:
    """(label, u_min at top of band, u_min at bottom of band, colour): each array's shortest baseline in λ."""
    out = []
    for i, (arr, band) in enumerate(ARRAY_RUNS):
        c = array_config(SimConfig(), arr, band)
        out.append((f"{arr} {band}", c.int_bmin_m / c.lams.max(), c.int_bmin_m / c.lams.min(), f"C{i + 3}"))
    return out


def plot_ucut_scan(out: dict, cfg: SimConfig, outdir: Path, prefix: str) -> None:
    """Generic interferometer: what an inner uv cut u_min [λ] does to RM and PI, scale by scale."""
    values = np.array(out["values"], float)
    runs = out["baseline"]
    fig, axs = plt.subplots(1, 4, figsize=(20, 4.4), constrained_layout=True)
    a = axs[0]
    a.plot(values, [r["int"]["rm_rms_err"] for r in runs], "o-", color=COLOURS["int"], label="INT rms error")
    a.plot(values, [r["int"]["rm_rms_err_bright_half"] for r in runs], "s--", color=COLOURS["int"],
           label="INT rms error, brighter half of PI")  # fmt: skip
    a.axhline(runs[0]["sd"]["rm_rms_err"], color=COLOURS["sd"], lw=1, label="single dish rms error")
    a.axhline(runs[0]["input"]["rm_std"], color="k", ls="--", lw=1, label="input RM std")
    a.axhline(runs[0]["input"]["fwhm_rmsf"] / 2, color="0.6", lw=0.8, label="RMSF FWHM / 2")
    a.set(ylabel="[rad m$^{-2}$]", title="Peak-RM error vs inner cut", yscale="log")
    axs[1].plot(values, [r["int"]["rm_corr_with_true"] for r in runs], "o-", color=COLOURS["int"], label="RM corr")
    axs[1].plot(values, [r["int"]["pi_mean"] for r in runs], "s-", color="tab:purple", label="mean PI (true 1)")
    axs[1].plot(values, [r["int"]["qu_power_retained"] for r in runs], "^:", color="tab:gray", label="Q/U power kept")
    if cfg.rm_grad_deg != 0:
        g = np.geomspace(values.min(), values.max(), 200)
        axs[1].plot(g, [predicted_gradient_int_pi(replace(cfg, rm_sigma=0.0, int_umin_lam=v), cfg.rm_grad_deg)
                        for v in g], "k-", lw=0.8, label="pure-gradient fringe prediction")  # fmt: skip
    axs[1].set(title="INT: RM correlation, PI and Q/U power", ylim=(0, 1.05))
    for ax in axs[:2]:
        for lab, lo, hi, col in _array_ucut_ranges():
            ax.axvspan(lo, hi, color=col, alpha=0.12, label=f"{lab} shortest baseline")
        ax.set(xlabel="INT inner cut u$_{min}$ [λ]", xscale="log")
        ax.legend(fontsize=6)
    # scale-resolved RM transfer for each cut
    u = np.array(runs[0]["input"]["rm_transfer_u"])
    cmap = plt.get_cmap("viridis")
    for i, (v, r) in enumerate(zip(values, runs, strict=True)):
        col = cmap(i / max(len(values) - 1, 1))
        axs[2].semilogx(u, r["int"]["rm_transfer"], color=col, label=f"u$_{{min}}$={v:g} λ")
        axs[2].axvline(v, color=col, ls=":", lw=0.8)
    axs[2].semilogx(u, runs[0]["sd"]["rm_transfer"], color=COLOURS["sd"], lw=2.5, alpha=0.6, label="single dish")
    axs[2].axhline(1, color="0.6", lw=0.8)
    axs[2].axhline(0, color="0.6", lw=0.8)
    axs[2].set(xlabel="uv distance u [λ]", ylabel="Re⟨RM̂ RM*⟩ / ⟨|RM|²⟩", title="INT RM transfer function",
               ylim=(-0.3, 1.2))  # fmt: skip
    axs[2].legend(fontsize=6, ncol=2)
    _scale_axis(axs[2])
    axs[3].plot(values, [r["int"]["rm_transfer_u50"] for r in runs], "o-", color=COLOURS["int"])
    axs[3].plot(values, values, "k--", lw=1, label="u$_{50}$ = u$_{min}$")
    axs[3].set(xlabel="INT inner cut u$_{min}$ [λ]", ylabel="u$_{50}$ [λ]", xscale="log", yscale="log",
               title="RM recovered down to scale 1/u$_{50}$ (transfer falls to 0.5)")  # fmt: skip
    axs[3].legend(fontsize=7)
    fig.suptitle(
        f"{prefix.rstrip('_')}: generic interferometer (sharp cut in λ, same every channel), "
        f"{cfg.freqs.min() / 1e6:.0f}–{cfg.freqs.max() / 1e6:.0f} MHz, σ(δRM)={cfg.rm_sigma:g}, "
        f"∇RM={cfg.rm_grad_deg:g}, SD={cfg.sd_model} {cfg.sd_bmax_m:g} m",
        fontsize=10,
    )
    fig.savefig(outdir / f"{prefix}scan_int_umin_lam.png", dpi=110)
    plt.close(fig)


def plot_array_comparison(summary: dict, cfgs: dict[str, SimConfig], path: Path) -> None:
    """ASKAP / MeerKAT across bands: RM transfer functions and headline metrics side by side."""
    tags = [t for t in summary if not t.endswith("_fixedcuts")]
    fig, axs = plt.subplots(1, 3, figsize=(19, 4.8), constrained_layout=True)
    for i, tag in enumerate(tags):
        m, c = summary[tag], cfgs[tag]
        col = f"C{i}"
        u = np.array(m["input"]["rm_transfer_u"])
        axs[0].semilogx(u, m["int"]["rm_transfer"], color=col, label=tag.removeprefix("t3_"))
        fx = summary.get(f"{tag}_fixedcuts")
        if fx:
            axs[0].semilogx(u, fx["int"]["rm_transfer"], ":", color=col)
        axs[0].axvspan(c.int_bmin_m / c.lams.max(), c.int_bmin_m / c.lams.min(), color=col, alpha=0.08)
    axs[0].semilogx(u, summary[tags[0]]["sd"]["rm_transfer"], color=COLOURS["sd"], lw=2.5, alpha=0.5,
                    label=f"single dish ({tags[0].removeprefix('t3_')})")  # fmt: skip
    axs[0].axhline(1, color="0.6", lw=0.8)
    axs[0].axhline(0, color="0.6", lw=0.8)
    axs[0].set(xlabel="uv distance u [λ]", ylabel="RM transfer", ylim=(-0.3, 1.2),
               title="INT RM transfer (solid: physical cuts, dotted: uv-matched)")  # fmt: skip
    axs[0].legend(fontsize=7)
    _scale_axis(axs[0])
    x = np.arange(len(tags))
    w = 0.38
    for j, (suffix, hatch) in enumerate((("", None), ("_fixedcuts", "//"))):
        vals = [summary.get(t + suffix, {}).get("int", {}) for t in tags]
        axs[1].bar(x + (j - 0.5) * w, [v.get("rm_rms_err", np.nan) for v in vals], w, hatch=hatch,
                   color=[f"C{i}" for i in range(len(tags))], edgecolor="k",
                   label="physical cuts" if j == 0 else "uv-matched")  # fmt: skip
        axs[2].bar(x + (j - 0.5) * w, [v.get("pi_mean", np.nan) for v in vals], w, hatch=hatch,
                   color=[f"C{i}" for i in range(len(tags))], edgecolor="k")  # fmt: skip
    axs[1].set(ylabel="rms(RM − true) [rad m$^{-2}$]", title="INT peak-RM error")
    axs[2].set(ylabel="mean peak PI (true 1)", title="INT recovered PI")
    for ax in axs[1:]:
        ax.set_xticks(x, [t.removeprefix("t3_").replace("_", "\n", 1) for t in tags], fontsize=8)
    axs[1].legend(fontsize=7)
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ------------------------------------------------- test 3 headline figures (one question each)

# validated 3-slot categorical palette (dataviz reference palette, light mode, all-pairs CVD-safe);
# aqua is < 3:1 on white, so wherever it appears it is direct-labelled
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
DIVERGING = matplotlib.colors.LinearSegmentedColormap.from_list(
    "bgo", ["#184f95", "#3987e5", "#e9e8e4", "#ee7d4f", "#a33a10"]
)
SEQ_BLUE = matplotlib.colors.LinearSegmentedColormap.from_list(
    "blue", ["#f7f9fc", "#b7d3f6", "#5598e7", "#1c5cab", "#0d366b"]
)


def _style(ax, grid_axis="y"):
    """Recessive chrome: no top/right spines, hairline solid grid, muted ticks."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(colors=INK2, labelsize=9, width=0.6)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)


def _title(fig_or_ax, headline: str, sub: str = "", fig=True):
    """Headline states the takeaway in plain words; subtitle states the setup."""
    if fig:
        fig_or_ax.text(0.01, 0.985, headline, fontsize=13, weight="bold", color=INK, va="top", ha="left")
        if sub:
            fig_or_ax.text(0.01, 0.935, sub, fontsize=9.5, color=INK2, va="top", ha="left")
    else:
        fig_or_ax.set_title(headline, fontsize=10.5, color=INK, loc="left")


def _howto(fig, text: str):
    """'How to read' footnote along the bottom of the figure."""
    fig.text(0.01, 0.015, "How to read: " + text, fontsize=9, color=INK, va="bottom", ha="left", wrap=True,
             bbox=dict(fc="#f3f2ee", ec="none", boxstyle="round,pad=0.5"))  # fmt: skip


def _las_deg(u_lam):
    """Largest angular scale an interferometer sees for shortest baseline u [wavelengths]: ~1/u radians."""
    return 1.0 / np.asarray(u_lam, float) / DEG


def _las_axis(ax):
    ax.set_xscale("log")
    ax.set_xticks([0.2, 0.3, 0.5, 1, 2, 3, 5], ["0.2", "0.3", "0.5", "1", "2", "3", "5"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("Largest angular scale the interferometer can see [deg]  (larger → shorter baselines)")


def _int_maps(cfg: SimConfig) -> dict:
    """INT RM/PI maps plus the analytic references: RM on scales above the cut, 2 lambda0^2 |RM below it|."""
    phi = make_rm_sky(cfg)
    res = rm_synthesis(observe(cfg, phi)["int"], cfg)
    uvd = uv_grid(cfg.nbig, cfg.dl)
    if cfg.int_umin_lam > 0:
        t_in = (uvd >= cfg.int_umin_lam).astype(float)
    else:
        lam_ref = cfg.lams.min() if cfg.cut_mode == "fixed" else np.sqrt(res["lam_sq_0"])
        t_in = int_inner_transfer(cfg, uvd * lam_ref)
    return {
        "rm": res["rm"],
        "pi": res["pi"],
        "rm_true": crop(phi, cfg),
        "rm_large": apply_transfer(phi, cfg, 1.0 - t_in),  # what the INT's RM should follow
        # weak-screen law: PI_int = 2 lambda0^2 |high-passed RM|
        "pi_pred": 2 * res["lam_sq_0"] * np.abs(apply_transfer(phi - cfg.rm_mean, cfg, t_in)),
    }


NATIVE = "as observed (largest scale shrinks with frequency)"
MATCHED = "matched (same largest scale at every frequency)"


def fig_what_int_measures(path: Path) -> None:
    """Fig 1: an interferometer's RM is the large-scale RM; its PI is the small-scale RM's magnitude."""
    cfg = replace(array_config(SimConfig(**REALISTIC), "askap", "possum1"), cut_mode="fixed", rm_sigma=2.0)
    las = cfg.lams.min() / cfg.int_bmin_m / DEG
    m = _int_maps(cfg)
    ext = [cfg.fov_deg / 2, -cfg.fov_deg / 2, -cfg.fov_deg / 2, cfg.fov_deg / 2]
    fig, axs = plt.subplots(2, 2, figsize=(10, 10.6))
    fig.subplots_adjust(left=0.04, right=0.9, top=0.86, bottom=0.12, wspace=0.08, hspace=0.18)
    lim = np.percentile(np.abs(m["rm_large"] - cfg.rm_mean), 99)
    rm_kw = dict(cmap=DIVERGING, vmin=cfg.rm_mean - lim, vmax=cfg.rm_mean + lim)
    pi_kw = dict(cmap=SEQ_BLUE, vmin=0, vmax=np.percentile(m["pi_pred"], 99))
    panels = [
        (m["rm_large"], f"True RM, smoothed: structure smaller than {las:.1f}° removed", rm_kw),
        (m["rm"], "RM measured by the interferometer", rm_kw),
        (m["pi_pred"], f"Predicted: 2λ² × size of the RM structure smaller than {las:.1f}°", pi_kw),
        (m["pi"], "Polarised intensity measured by the interferometer", pi_kw),
    ]
    for ax, (img, title, kw) in zip(axs.flat, panels, strict=True):
        ax.imshow(img, origin="lower", extent=ext, **kw)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    for row, lab in enumerate(("RM [rad m⁻²]", "polarised intensity\n(true value = 1 everywhere)")):
        pos = axs[row, 1].get_position()
        cb = fig.colorbar(axs[row, 1].images[0], cax=fig.add_axes([0.915, pos.y0, 0.015, pos.height]))
        cb.set_label(lab, color=INK, fontsize=9)
        cb.ax.tick_params(labelsize=8, colors=INK2)
        cb.outline.set_visible(False)
    for row, (a, b) in enumerate(((m["rm"], m["rm_large"]), (m["pi"], m["pi_pred"]))):
        c = np.corrcoef(a.ravel(), b.ravel())[0, 1]
        axs[row, 1].text(0.02, 0.02, f"matches the left panel: correlation {c:.2f}", transform=axs[row, 1].transAxes,
                         fontsize=9, color=INK, bbox=dict(fc="white", ec="none", alpha=0.85))  # fmt: skip
    headline = (
        "The interferometer's RM map shows the smooth RM;\nits polarised-intensity map shows the fine RM structure"
    )
    _title(fig, headline,
           f"Simulated 5° × 5° field: polarised intensity is 1 everywhere, only the RM varies (30 ± 2 rad m⁻²).\n"
           f"ASKAP-like interferometer, 800–1088 MHz, seeing angular scales up to {las:.1f}° at every frequency."
           )  # fmt: skip
    _howto(
        fig,
        "compare each row left to right. Top: the interferometer's RM agrees with the smoothed true RM, apart from "
        "scattered speckles.\nBottom: the interferometer does not see the true polarised intensity (1 everywhere). Its "
        "polarised intensity is instead set by how\nmuch the RM varies on scales it can see, and matches that "
        "prediction exactly. The speckles (top right) sit where this is near zero.",
    )
    fig.savefig(path, dpi=130, facecolor="white")
    plt.close(fig)


def fig_errors_at_pi_nulls(path: Path) -> None:
    """Fig 2: where the INT RM errors are: binned by the INT's own PI."""
    fig, ax = plt.subplots(figsize=(9.5, 6))
    fig.subplots_adjust(left=0.09, right=0.66, top=0.8, bottom=0.27)
    base = array_config(SimConfig(**REALISTIC), "askap", "possum1")
    nb = 10
    centres = np.arange(nb) + 1
    for cfg, col, lab in ((base, C2, NATIVE), (replace(base, cut_mode="fixed"), C1, MATCHED)):
        m = _int_maps(cfg)
        err = np.abs(m["rm"] - m["rm_true"]).ravel()
        rank = np.argsort(np.argsort(m["pi"].ravel())) / err.size  # PI percentile of each pixel
        med = [np.median(err[(rank >= i / nb) & (rank < (i + 1) / nb)]) for i in range(nb)]
        ax.plot(centres, med, "-o", color=col, lw=2, ms=6, mec="white", mew=1.5)
        ax.annotate(lab.replace(" (", "\n("), (centres[-1], med[-1]), xytext=(10, 0), textcoords="offset points",
                    va="center", color=INK, fontsize=9.5, annotation_clip=False)  # fmt: skip
    ax.set_xticks(centres, ["faintest\n10%"] + [""] * (nb - 2) + ["brightest\n10%"])
    ax.set(xlabel="Pixels sorted into tenths by the polarised intensity the interferometer measures",
           ylabel="Typical RM error [rad m⁻²]", ylim=(0, None))  # fmt: skip
    _style(ax)
    _title(
        fig,
        "The interferometer's RMs are only badly wrong where its polarised intensity is faint",
        "Simulated 5° × 5° field, RM 30 ± 10 rad m⁻², ASKAP-like interferometer, 800–1088 MHz, no noise.",
    )
    _howto(fig, "each dot is the median RM error of one tenth of the pixels. The faintest tenth (left) has large RM "
                "errors; beyond\nthe faintest ~20% the error is flat. So a normal signal-to-noise cut on polarised "
                "intensity removes the worst RMs.\nKeeping the same largest angular scale at every frequency (blue) "
                "lowers the error everywhere.")  # fmt: skip
    fig.savefig(path, dpi=130, facecolor="white")
    plt.close(fig)


def _arrays_on_axis(ax, top: float = 0.97, step: float = 0.065):
    """Where each real array sits: its largest angular scale (lambda / shortest baseline) across its band.

    One row each in axes-fraction y (works on linear and log axes), direct-labelled.
    """
    trans = matplotlib.transforms.blended_transform_factory(ax.transData, ax.transAxes)
    names = {"possum1": "800–1088 MHz", "uhf": "UHF", "lband": "L-band"}
    for i, (arr, band) in enumerate((("askap", "possum1"), ("meerkat", "possum1"), ("meerkat", "uhf"),
                                     ("meerkat", "lband"))):  # fmt: skip
        c = array_config(SimConfig(), arr, band)
        lo, hi = c.lams.min() / c.int_bmin_m / DEG, c.lams.max() / c.int_bmin_m / DEG
        y = top - i * step
        ax.plot([lo, hi], [y, y], color=INK2, lw=2, solid_capstyle="butt", transform=trans, clip_on=False)
        ax.annotate(f"{'ASKAP' if arr == 'askap' else 'MeerKAT'} {names[band]}", (hi, y), xycoords=trans,
                    xytext=(4, 0), textcoords="offset points", va="center", fontsize=7.5, color=INK2)  # fmt: skip


def fig_inner_cut(outdir: Path, path: Path) -> None:
    """Fig 3: generic interferometer: RM error and PI vs the largest angular scale it sees."""
    d = json.loads((outdir / "t3_generic_scan_int_umin_lam.json").read_text())
    x = _las_deg(d["values"])
    runs = d["baseline"]
    fig, axs = plt.subplots(1, 2, figsize=(13, 6.2))
    fig.subplots_adjust(left=0.06, right=0.97, top=0.78, bottom=0.27, wspace=0.22)
    b = axs[0]
    y = np.array([r["int"]["pi_mean"] for r in runs])
    b.plot(x, y, "-o", color=C1, lw=2, ms=6, mec="white", mew=1.5)
    slope = (11 / 3 - 2) / 2  # rms of RM structure below scale θ grows as θ^((|β| - 2) / 2) for P(k) ~ k^β
    ref = y[5] * (x / x[5]) ** slope
    b.plot(x, ref, color=INK2, lw=1)
    b.annotate("expected from the RM power spectrum", (x[7], ref[7]), xytext=(8, -4), textcoords="offset points",
               ha="left", va="top", fontsize=8.5, color=INK2)  # fmt: skip
    b.set(yscale="log", ylabel="Polarised intensity recovered (true = 1)", ylim=(0.03, 6))
    b.set_yticks([0.05, 0.1, 0.2, 0.5, 1], ["0.05", "0.1", "0.2", "0.5", "1"])
    b.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    _title(b, "Polarised intensity recovered: more with larger scales", fig=False)
    a = axs[1]
    for key, col, lab in (("rm_rms_err", C2, "all pixels"), ("rm_rms_err_bright_half", C1, "brighter half of pixels")):
        yy = [r["int"][key] for r in runs]
        a.plot(x, yy, "-o", color=col, lw=2, ms=6, mec="white", mew=1.5)
        a.annotate(lab, (x[-1], yy[-1]), xytext=(-8, 0 if key == "rm_rms_err" else 12), textcoords="offset points",
                   ha="right", va="center",
                   color=INK, fontsize=9.5)  # fmt: skip
    sd = runs[0]["sd"]["rm_rms_err"]
    a.axhline(sd, color=INK2, lw=1)
    a.annotate(f"64 m single dish ({sd:.1f})", (x[0], sd), xytext=(0, -12), textcoords="offset points", ha="right",
               fontsize=8.5, color=INK2)  # fmt: skip
    a.set(ylabel="RM error, rms [rad m⁻²]", ylim=(0, 12))
    _title(a, "RM error: also larger with larger scales", fig=False)
    for ax in axs:
        _las_axis(ax)
        ax.set_xlim(0.04, 8)
        _arrays_on_axis(ax)
        _style(ax)
    _title(fig, "Seeing larger angular scales recovers more polarised intensity, but gives worse RMs",
           "Idealised interferometer that sees every angular scale up to a given size, the same at every frequency.\n"
           "Simulated 5° × 5° field, RM 30 ± 10 rad m⁻², polarised intensity 1 everywhere, 800–1088 MHz, no noise."
           )  # fmt: skip
    _howto(
        fig,
        "moving right = the interferometer sees larger scales (it has shorter baselines). Left: it recovers more of "
        "the true\npolarised intensity. Right: but its RMs get worse, because the larger-scale RM variations it now "
        "sees rotate the\npolarisation angle by more than a radian across the band (until, far right, it sees almost "
        "the whole field). Grey bars:\nwhere ASKAP and MeerKAT sit; their largest scale changes across their band. "
        "Without noise, small scales look best for\nRM; with noise, the faint polarised intensity there will push "
        "the other way.",
    )
    fig.savefig(path, dpi=130, facecolor="white")
    plt.close(fig)


def fig_arrays(outdir: Path, path: Path) -> None:
    """Fig 4: ASKAP vs MeerKAT by band: dumbbells, as observed -> matched largest scale."""
    s = json.loads((outdir / "t3_summary.json").read_text())
    rows = [("askap", "possum1", "ASKAP, 800–1088 MHz"), ("meerkat", "possum1", "MeerKAT, 800–1088 MHz"),
            ("askap", "possum12", "ASKAP, 800–1088 + 1296–1440 MHz"), ("meerkat", "uhf", "MeerKAT UHF, 544–1088 MHz"),
            ("meerkat", "lband", "MeerKAT L-band, 856–1712 MHz")]  # fmt: skip
    fig, axs = plt.subplots(1, 2, figsize=(13, 6.2), sharey=True)
    fig.subplots_adjust(left=0.2, right=0.97, top=0.74, bottom=0.25, wspace=0.12)
    y = np.arange(len(rows))[::-1]
    for ax, key, xlab, title in ((axs[0], "rm_rms_err", "RM error, rms [rad m⁻²]", "RM error (lower is better)"),
                                 (axs[1], "pi_mean", "Polarised intensity recovered (true = 1)",
                                  "Polarised intensity recovered (higher is better)")):  # fmt: skip
        for yi, (arr, band, _) in zip(y, rows, strict=True):
            a = s[f"t3_{arr}_{band}"]["int"][key]
            b = s[f"t3_{arr}_{band}_fixedcuts"]["int"][key]
            ax.plot([a, b], [yi, yi], color=GRID, lw=3, solid_capstyle="round", zorder=1)
            ax.scatter([a], [yi], s=70, color=C2, edgecolor="white", lw=1.5, zorder=3)
            ax.scatter([b], [yi], s=70, color=C1, edgecolor="white", lw=1.5, zorder=3)
        ax.set(xlabel=xlab, xlim=(0, None))
        _title(ax, title, fig=False)
        _style(ax, grid_axis="x")
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axs[0].set_yticks(y, [r[2] for r in rows], fontsize=10, color=INK)
    axs[0].set_ylim(-0.6, len(rows) - 0.3)
    key = [
        matplotlib.lines.Line2D([], [], ls="", marker="o", ms=8, mfc=c, mec="white", label=lab)
        for c, lab in ((C2, NATIVE), (C1, MATCHED))
    ]
    fig.legend(handles=key, loc="upper left", bbox_to_anchor=(0.2, 0.86), frameon=False, fontsize=9.5, ncol=2)
    _title(fig, "The frequency band matters more than ASKAP vs MeerKAT",
           "Interferometer only. Simulated 5° × 5° field, RM 30 ± 10 rad m⁻², polarised intensity 1 everywhere, "
           "no noise.\nASKAP: 36 × 12 m dishes, shortest baseline 22 m.  MeerKAT: 64 × 13.5 m dishes, shortest 29 m."
           )  # fmt: skip
    _howto(fig, "each row is one telescope and band. Orange: as observed. Blue: every frequency restricted to the "
                "same largest angular scale.\nThe top two rows (same band, different array) are close; changing the "
                "band moves the results much more. Lower frequencies\n(UHF) recover more polarised intensity, because "
                "Faraday rotation grows as λ². Matching the scales costs a little polarised\nintensity but cuts the RM "
                "error 1.5–3.5×, most for the widest band (L-band, 8.5 → 2.4 rad m⁻²).")  # fmt: skip
    fig.savefig(path, dpi=130, facecolor="white")
    plt.close(fig)


def fig_gradient(outdir: Path, path: Path) -> None:
    """Fig 5: a strong RM gradient survives an interferometer only if it sees scales larger than its stripes."""
    d = json.loads((outdir / "t3_generic_grad50_scan_int_umin_lam.json").read_text())
    x = _las_deg(d["values"])
    runs = d["baseline"]
    cfg = replace(SimConfig(**REALISTIC), rm_sigma=0.0, rm_grad_deg=50.0)
    # the gradient winds the angle into Q/U stripes with period pi / (|grad RM| lambda^2) radians
    period = np.pi / (50.0 / DEG * cfg.lams**2) / DEG
    fig, ax = plt.subplots(figsize=(10, 6.4))
    fig.subplots_adjust(left=0.08, right=0.96, top=0.77, bottom=0.26)
    ax.axvspan(period.min(), period.max(), color="#f3f2ee", zorder=0)
    ax.annotate(f"stripe spacing\n{period.min():.2f}–{period.max():.2f}°", (np.sqrt(period.min() * period.max()), 0.04),
                ha="center", va="bottom", fontsize=8.5, color=INK2)  # fmt: skip
    g = np.geomspace(1 / x.max() / DEG, 1 / x.min() / DEG, 300)  # u grid
    pred = np.array([predicted_gradient_int_pi(replace(cfg, int_umin_lam=v), 50.0) for v in g])
    ax.plot(_las_deg(g), pred, color=INK2, lw=1)
    i = np.argmin(np.abs(pred - 0.5))
    ax.annotate("prediction: fraction of frequency channels\nwhose stripe spacing fits within the largest scale",
                (_las_deg(g[i]), pred[i]), xytext=(0.02, 0.55), textcoords="axes fraction", fontsize=8.5, color=INK2,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))  # fmt: skip
    yy = [r["int"]["pi_mean"] for r in runs]
    ax.plot(x, yy, "-o", color=C1, lw=2, ms=6, mec="white", mew=1.5)
    ax.annotate("interferometer", (x[0], yy[0]), xytext=(-6, 8), textcoords="offset points", ha="right", fontsize=9.5,
                color=INK)  # fmt: skip
    sd = runs[0]["sd"]["pi_mean"]
    ax.axhline(sd, color=C2, lw=2)
    ax.annotate(f"64 m single dish ({sd:.2f})", (x[0], sd), xytext=(0, 6), textcoords="offset points", ha="right",
                fontsize=9.5,
                color=INK)  # fmt: skip
    ax.set(ylabel="Polarised intensity recovered (true = 1)", ylim=(0, 1.55))
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    _las_axis(ax)
    ax.set_xlim(0.06, 8)
    _arrays_on_axis(ax)
    _style(ax)
    _title(fig, "A strong RM gradient survives if the interferometer sees scales larger than its stripes",
           "RM gradient of 50 rad m⁻² per degree across the simulated field (plus RM 30 ± 10 rad m⁻²), 800–1088 MHz.\n"
           "The gradient twists the polarisation angle into stripes across the sky; their spacing shrinks with "
           "frequency.")  # fmt: skip
    _howto(
        fig,
        "moving right = the interferometer sees larger scales. Once its largest scale is bigger than the stripe "
        "spacing (shaded),\nit keeps all the polarised intensity; when it is smaller, the stripes are filtered out. "
        "ASKAP and MeerKAT sit right in that\ntransition. The 64 m dish (orange) loses most of it because its ~0.3° "
        "beam averages over neighbouring stripes, which cancel.",
    )
    fig.savefig(path, dpi=130, facecolor="white")
    plt.close(fig)


def t3_figures(outdir: Path) -> None:
    """The five test-3 headline figures (needs the runs and scan_ucut JSON in outdir)."""
    fig_what_int_measures(outdir / "t3_fig1_what_int_measures.png")
    fig_errors_at_pi_nulls(outdir / "t3_fig2_errors_at_pi_nulls.png")
    fig_inner_cut(outdir, outdir / "t3_fig3_inner_cut.png")
    fig_arrays(outdir, outdir / "t3_fig4_arrays.png")
    fig_gradient(outdir, outdir / "t3_fig5_gradient.png")


def plot_transfer(cfgs: dict[str, SimConfig], path: Path) -> None:
    """uv transfer functions of each response model at the band edges, both cut modes."""
    u = np.linspace(0, 700, 1401)
    fig, axs = plt.subplots(1, len(cfgs), figsize=(6.5 * len(cfgs), 4.3), constrained_layout=True, squeeze=False)
    for ax, (name, cfg) in zip(axs[0], cfgs.items(), strict=True):
        for f_hz, ls in ((cfg.freqs.min(), "-"), (cfg.freqs.max(), "--")):
            t_sd, t_int = uv_masks(cfg, u, C_M_S / f_hz)
            ax.plot(u, t_sd, ls, color=COLOURS["sd"], label=f"SD @ {f_hz / 1e6:.0f} MHz")
            ax.plot(u, t_int, ls, color=COLOURS["int"], label=f"INT @ {f_hz / 1e6:.0f} MHz")
            if cfg.combine != "none":
                ax.plot(
                    u,
                    combined_transfer(cfg, t_sd, t_int),
                    ls,
                    color=COLOURS["comb"],
                    label=f"{cfg.combine} @ {f_hz / 1e6:.0f} MHz",
                )
        ax.set(
            title=f"{name} (cuts={cfg.cut_mode}, D={cfg.sd_bmax_m:g} m)",
            xlabel="uv distance u [wavelengths]",
            ylabel="transfer function",
            ylim=(-0.03, 1.05),
        )
        ax.legend(fontsize=7)
    fig.savefig(path, dpi=110)
    plt.close(fig)


REALISTIC = dict(sd_model="aperture", int_inner="soft")
GRAD_VALUES = [0, 5, 10, 20, 35, 50, 75, 100]
SIGMA_VALUES = [1, 3, 5, 10, 20, 40]
GRAD_LABEL = "mean-RM gradient [rad m$^{-2}$ deg$^{-1}$]"
SIGMA_LABEL = r"$\sigma(\delta RM)$ [rad m$^{-2}$]"


# Test 3: interferometers. Parametric: in this noiseless, ideally deconvolved limit only the coverage
# edges matter (uv density / weighting only matters with noise), and the 2.3' pixel grid only reaches
# u ~ 730 λ (b ~ 200-300 m), far below either array's longest baselines, so only the shortest baseline
# and the dish diameter (which smears the inner edge, see int_inner_transfer) enter.
ARRAYS = {
    "askap": dict(int_bmin_m=22.0, ant_diam_m=12.0, int_inner="soft"),  # 36 x 12 m, shortest baseline ~22 m
    "meerkat": dict(int_bmin_m=29.0, ant_diam_m=13.5, int_inner="soft"),  # 64 x 13.5 m, shortest ~29 m
}
BANDS = {  # (f_min, f_max, channel width) segments [Hz]
    "possum1": ((800e6, 1088e6, 1e6),),  # POSSUM band 1: the common band
    "possum12": ((800e6, 1088e6, 1e6), (1296e6, 1440e6, 1e6)),  # ASKAP POSSUM band 1 + band 2
    "uhf": ((544e6, 1088e6, 2e6),),  # MeerKAT UHF
    "lband": ((856e6, 1712e6, 2e6),),  # MeerKAT L-band
}
ARRAY_RUNS = [("askap", "possum1"), ("meerkat", "possum1"), ("askap", "possum12"), ("meerkat", "uhf"),
              ("meerkat", "lband")]  # fmt: skip
UCUT_VALUES = [10, 20, 30, 45, 60, 80, 110, 160, 240, 320]
UCUT_LABEL = "INT inner cut u$_{min}$ [λ]"


def array_config(cfg: SimConfig, array: str, band: str) -> SimConfig:
    """Apply an array preset (shortest baseline, dish diameter) and a band preset."""
    return replace(cfg, **ARRAYS[array], freq_segments=BANDS[band], int_umin_lam=0.0)


def _examples(cfg: SimConfig, prefix: str, fixed: bool = True) -> dict[str, SimConfig]:
    """Standard example cases: base, gradient 50, weak screen, and uv-matched versions."""
    out = {
        f"{prefix}": cfg,
        f"{prefix}_grad50": replace(cfg, rm_grad_deg=50.0),
        f"{prefix}_weak_screen": replace(cfg, rm_sigma=2.0),
    }
    if fixed:
        out[f"{prefix}_fixedcuts"] = replace(cfg, cut_mode="fixed")
        out[f"{prefix}_weak_screen_fixedcuts"] = replace(cfg, rm_sigma=2.0, cut_mode="fixed")
    return out


def suite(name: str, base: SimConfig, outdir: Path, part: str = "all") -> None:
    """Run one test suite, or one part of it.

    name: "base"  - sharp cuts (original tests)
          "t1"    - realistic responses (aperture-tapered dish, soft INT inner edge)
          "t2"    - realistic responses + SD/INT combination (feather and joint)
          "t3"    - interferometers: generic cut in λ, ASKAP and MeerKAT on common and native bands
    part: "all", "runs", "scan_grad", "scan_sigma" (t2 scans take a ":feather"/":joint" suffix);
          t3 has "runs", "scan_ucut" (":weak" / ":grad" suffixes pick one sky) and "figures"
          (the five headline figures, from the runs/scan JSON; the per-run and scan plots are diagnostics)
    """
    real = replace(base, **REALISTIC)
    global ERR_LIM
    if name == "t3":
        if ERR_LIM is None:  # one RM-error colour scale for every band: the POSSUM band-1 RMSF FWHM / 2
            ERR_LIM = rm_synthesis(np.ones((base.freqs.size, 2, 2), complex), base)["fwhm"] / 2
        t3 = replace(real, scale_metrics=True)
        cfgs = {}
        for arr, band in ARRAY_RUNS:
            c = array_config(t3, arr, band)
            cfgs[f"t3_{arr}_{band}"] = c
            cfgs[f"t3_{arr}_{band}_fixedcuts"] = replace(c, cut_mode="fixed")
        part, _, which = part.partition(":")
        if part in ("all", "runs"):
            plot_transfer({k: v for k, v in cfgs.items() if not k.endswith("_fixedcuts")},
                          outdir / "t3_transfer_functions.png")  # fmt: skip
            summary = {tag: run(c, outdir, tag=tag) for tag, c in cfgs.items()}
            (outdir / "t3_summary.json").write_text(_dump(summary))
            plot_array_comparison(summary, cfgs, outdir / "t3_array_comparison.png")
        skies = {"t3_generic_": t3, "t3_generic_weak_screen_": replace(t3, rm_sigma=2.0),
                 "t3_generic_grad50_": replace(t3, rm_grad_deg=50.0)}  # fmt: skip
        if part in ("all", "scan_ucut"):
            for prefix, c in skies.items():
                if which and which not in prefix.removeprefix("t3_generic"):
                    continue
                scan(c, "int_umin_lam", UCUT_VALUES, outdir, UCUT_LABEL, prefix=prefix)
        if part in ("all", "figures"):
            t3_figures(outdir)
        return
    if name == "base":
        runs = _examples(base, "base")
        scans = {"": base}
        transfer = None
    elif name == "t1":
        runs = {
            "t1a_sd_aperture": replace(base, sd_model="aperture"),
            "t1b_int_soft": replace(base, int_inner="soft"),
            **_examples(real, "t1c_realistic"),
        }
        scans = {"t1_realistic_": real}
        transfer = {"sharp": base, "realistic": real, "realistic, fixed": replace(real, cut_mode="fixed")}
    elif name == "t2":
        runs = {
            **_examples(replace(real, combine="feather"), "t2_feather"),
            **_examples(replace(real, combine="joint"), "t2_joint", fixed=False),
        }
        scans = {"t2_feather_": replace(real, combine="feather"), "t2_joint_": replace(real, combine="joint")}
        transfer = {
            "feather": replace(real, combine="feather"),
            "feather, fixed": replace(real, combine="feather", cut_mode="fixed"),
            "joint": replace(real, combine="joint"),
        }
    else:
        raise ValueError(name)
    part, _, which = part.partition(":")
    if part in ("all", "runs"):
        if transfer:
            plot_transfer(transfer, outdir / f"{name}_transfer_functions.png")
        summary = {tag: run(cfg, outdir, tag=tag) for tag, cfg in runs.items()}
        (outdir / f"{name}_summary.json").write_text(_dump(summary))
    for prefix, cfg in scans.items():
        if which and which not in prefix:
            continue
        if part in ("all", "scan_grad"):
            scan(cfg, "rm_grad_deg", GRAD_VALUES, outdir, GRAD_LABEL, prefix=prefix)
        if part in ("all", "scan_sigma"):
            scan(cfg, "rm_sigma", SIGMA_VALUES, outdir, SIGMA_LABEL, prefix=prefix)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", type=Path, default=Path("outputs"))  # gitignored; see results/ for a snapshot
    ap.add_argument("--rm-mean", type=float, default=30.0)
    ap.add_argument("--rm-sigma", type=float, default=10.0)
    ap.add_argument("--rm-slope", type=float, default=-11 / 3)
    ap.add_argument("--rm-grad", type=float, default=0.0, help="rad m^-2 deg^-1")
    ap.add_argument("--cut-mode", choices=("baseline", "fixed"), default="baseline")
    ap.add_argument("--npix", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sd-model", choices=("sharp", "aperture"), default="sharp")
    ap.add_argument("--sd-taper-db", type=float, default=12.0)
    ap.add_argument("--int-inner", choices=("sharp", "soft"), default="sharp")
    ap.add_argument("--ant-diam", type=float, default=12.0)
    ap.add_argument("--int-restore-arcmin", type=float, default=0.0)
    ap.add_argument(
        "--suite",
        choices=("base", "t1", "t2", "t3"),
        default=None,
        help="test suite: base = sharp cuts, t1 = realistic responses, t2 = SD+INT combination, "
        "t3 = interferometers (generic λ cut, ASKAP, MeerKAT)",
    )
    ap.add_argument(
        "--part", default="all", help="all | runs | scan_grad | scan_sigma (t2 scans: e.g. scan_grad:feather)"
    )
    ap.add_argument("--sd-diam", type=float, default=64.0, help="single-dish diameter [m]")
    ap.add_argument("--combine", choices=("none", "feather", "joint"), default="none")
    ap.add_argument("--array", choices=tuple(ARRAYS), default=None, help="INT preset (sets inner edge, dish)")
    ap.add_argument("--band", choices=tuple(BANDS), default=None, help="frequency preset")
    ap.add_argument("--int-umin-lam", type=float, default=0.0, help="generic INT: sharp inner cut [λ]")
    ap.add_argument(
        "--err-lim",
        type=float,
        default=None,
        help="linear colour limit for all RM-error maps [rad m^-2]; default RMSF FWHM / 2",
    )
    args = ap.parse_args()
    global ERR_LIM
    ERR_LIM = args.err_lim

    cfg = SimConfig(
        rm_mean=args.rm_mean,
        rm_sigma=args.rm_sigma,
        rm_slope=args.rm_slope,
        rm_grad_deg=args.rm_grad,
        cut_mode=args.cut_mode,
        npix=args.npix,
        seed=args.seed,
        sd_model=args.sd_model,
        sd_edge_taper_db=args.sd_taper_db,
        int_inner=args.int_inner,
        ant_diam_m=args.ant_diam,
        int_restore_arcmin=args.int_restore_arcmin,
        sd_bmax_m=args.sd_diam,
        combine=args.combine,
        int_umin_lam=args.int_umin_lam,
    )
    if args.array:
        cfg = replace(cfg, **ARRAYS[args.array])
    if args.band:
        cfg = replace(cfg, freq_segments=BANDS[args.band])
    args.outdir.mkdir(parents=True, exist_ok=True)
    if args.suite:
        suite(args.suite, cfg, args.outdir, args.part)
    else:
        print(json.dumps(run(cfg, args.outdir, tag="single"), indent=2))


if __name__ == "__main__":
    main()
