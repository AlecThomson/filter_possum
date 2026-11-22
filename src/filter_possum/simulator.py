"""Simulate what a single dish and an interferometer recover of a diffuse Faraday screen.

Sky: polarised intensity p0 everywhere, Faraday-thin, with

    P(l, m, lambda^2) = p0 exp(2i [chi0 + RM(l, m) lambda^2])
    RM = rm_mean + Gaussian random field (P(k) ~ k^rm_slope, rms rm_sigma) + optional gradient

on a 5 x 5 deg SIN-projected field. Pixels are uniform in direction cosines, so the 2-D FFT of
each channel *is* its uv plane in wavelengths (w-term neglected). Each channel's Q + iU is
multiplied in the uv plane by an instrument's response at that channel's wavelength (physical
baseline b = |u| lambda), then RM synthesis (rm-lite) gives peak RM and peak PI per pixel.

Typical use::

    from filter_possum import simulator as fs

    sky = fs.Sky(rm_sigma=10, rm_grad=20)
    obs = fs.observe(sky, fs.BANDS["possum1"], fs.SingleDish(), fs.ASKAP, combine="joint")
    res = {k: fs.rmsynth(cube, obs.band) for k, cube in obs.cubes.items()}
    fs.plot_comparison(obs, res["int"], "Interferometer")
"""

from __future__ import annotations

from dataclasses import dataclass, field

import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
from rm_lite.tools_3d.rmsynth import rmsynth_3d
from rm_lite.utils.synthesis import calc_faraday_peaks

C_M_S = 299_792_458.0
DEG = np.pi / 180.0


# ----------------------------------------------------------------------------- sky


def uv_grid(n: int, dl: float) -> np.ndarray:
    """|u| in wavelengths on the unshifted FFT grid of an n x n image with pixel size dl [rad]."""
    u = np.fft.fftfreq(n, dl)
    return np.hypot(*np.meshgrid(u, u, indexing="xy"))


def gaussian_random_field(n: int, dl: float, slope: float, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Periodic zero-mean Gaussian random field with 2-D power spectrum ~ k^slope and std sigma."""
    k = uv_grid(n, dl)
    amp = np.zeros_like(k)
    amp[k > 0] = k[k > 0] ** (slope / 2.0)  # amplitude = sqrt(power)
    noise = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    f = np.fft.ifft2(noise * amp).real
    f -= f.mean()
    return f * (sigma / f.std())


def periodic_gradient(n: int, dl: float, grad_per_rad: float, ramp_frac: float) -> np.ndarray:
    """Linear gradient along l that folds back in the guard band, so the padded box stays periodic.

    Without the fold-back, the FFT wrap would put a step in the RM that rings through every uv filter.
    """
    x = (np.arange(n) - n / 2) * dl
    half = n * dl / 2
    a = ramp_frac * half
    prof = np.where(np.abs(x) <= a, grad_per_rad * x, np.sign(x) * grad_per_rad * a * (half - np.abs(x)) / (half - a))
    return np.broadcast_to(prof[None, :], (n, n)).copy()


@dataclass(frozen=True)
class Sky:
    """Diffuse, Faraday-thin polarised sky with a turbulent RM screen and optional RM gradient.

    The sky is generated on a `pad`-times larger periodic box; only the central `fov_deg` field is
    analysed, so the field edges are not artificial periodic boundaries.
    """

    rm_mean: float = 30.0  # rad m^-2
    rm_sigma: float = 10.0  # rms of the RM fluctuations over the padded box, rad m^-2
    rm_slope: float = -11.0 / 3.0  # 2-D power-spectrum slope of the fluctuations (Kolmogorov)
    rm_grad: float = 0.0  # RM gradient along l, rad m^-2 per degree
    p0: float = 1.0  # polarised intensity (constant)
    chi0_rad: float = 0.0  # intrinsic polarisation angle (constant)
    fov_deg: float = 5.0
    npix: int = 128
    pad: int = 2
    grad_ramp_frac: float = 0.7  # fraction of the padded box the gradient ramp spans
    seed: int = 42

    @property
    def dl(self) -> float:
        """Pixel size in direction cosine (radians)."""
        return self.fov_deg / self.npix * DEG

    @property
    def nbig(self) -> int:
        return self.npix * self.pad

    def rm_padded(self) -> np.ndarray:
        """RM over the whole padded box."""
        rng = np.random.default_rng(self.seed)
        rm = np.full((self.nbig, self.nbig), self.rm_mean, dtype=float)  # ints allowed in the call
        if self.rm_sigma > 0:
            rm += gaussian_random_field(self.nbig, self.dl, self.rm_slope, self.rm_sigma, rng)
        if self.rm_grad != 0:
            rm += periodic_gradient(self.nbig, self.dl, self.rm_grad / DEG, self.grad_ramp_frac)
        return rm

    def crop(self, arr: np.ndarray) -> np.ndarray:
        """Central analysed field of a padded-box array (works on cubes too)."""
        s = slice(self.nbig // 2 - self.npix // 2, self.nbig // 2 + self.npix // 2)
        return arr[..., s, s]

    def rm(self) -> np.ndarray:
        """True RM over the analysed field."""
        return self.crop(self.rm_padded())

    @property
    def extent(self) -> list[float]:
        """imshow extent in degrees (RA increasing to the left)."""
        h = self.fov_deg / 2
        return [h, -h, -h, h]


# ---------------------------------------------------------------------------- band


@dataclass(frozen=True)
class Band:
    """Frequency coverage as one or more (f_min, f_max, channel width) segments [Hz]."""

    segments: tuple[tuple[float, float, float], ...] = ((800e6, 1088e6, 1e6),)

    @property
    def freqs(self) -> np.ndarray:
        """Channel centres [Hz], ascending."""
        out = [f0 + (np.arange(round((f1 - f0) / df)) + 0.5) * df for f0, f1, df in self.segments]
        return np.concatenate(out)

    @property
    def lams(self) -> np.ndarray:
        return C_M_S / self.freqs


BANDS = {
    "possum1": Band(((800e6, 1088e6, 1e6),)),  # ASKAP POSSUM band 1, 288 x 1 MHz
    "possum12": Band(((800e6, 1088e6, 1e6), (1296e6, 1440e6, 1e6))),  # POSSUM bands 1 + 2
    "uhf": Band(((544e6, 1088e6, 2e6),)),  # MeerKAT UHF
    "lband": Band(((856e6, 1712e6, 2e6),)),  # MeerKAT L-band
}


# --------------------------------------------------------------------- instruments

_AC_CACHE: dict[float, tuple[np.ndarray, np.ndarray]] = {}


def aperture_autocorr(edge_taper_db: float, n: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """Radial autocorrelation of a circular dish illumination, vs rho = b / D in [0, 1].

    Illumination A(r) = 1 - (1 - tau)(2r/D)^2 with tau the edge amplitude. Its autocorrelation is the
    uv transfer function of the dish's power beam: 1 at zero spacing, 0 at b = D. 0 dB = uniform dish.
    """
    if edge_taper_db not in _AC_CACHE:
        tau = 10 ** (-edge_taper_db / 20.0)
        x = (np.arange(n) - n / 2) / (n / 4)  # aperture radius n/4 px, so lags up to D fit without wrapping
        r = np.hypot(*np.meshgrid(x, x))
        a = np.where(r <= 1.0, 1 - (1 - tau) * r**2, 0.0)
        ac = np.fft.fftshift(np.fft.ifft2(np.abs(np.fft.fft2(a)) ** 2).real)
        prof = ac[n // 2, n // 2 :] / ac[n // 2, n // 2]
        rho = np.arange(prof.size) / (n / 4) / 2.0
        keep = rho <= 1.0
        _AC_CACHE[edge_taper_db] = (rho[keep], np.clip(prof[keep], 0, None))
    return _AC_CACHE[edge_taper_db]


def dish_beam_fwhm(edge_taper_db: float) -> float:
    """Power-beam FWHM in units of lambda / D (Hankel transform of the aperture transfer function)."""
    from scipy.special import j0

    rho, prof = aperture_autocorr(edge_taper_db)
    t = np.linspace(0, 1.5, 3001)
    beam = np.array([np.trapezoid(prof * j0(2 * np.pi * rho * ti) * rho, rho) for ti in t])
    return float(2 * t[np.argmin(np.abs(beam / beam[0] - 0.5))])


@dataclass(frozen=True)
class SingleDish:
    """Total-power single dish (default: Murriyang/Parkes, 64 m, 12 dB edge taper).

    model "aperture": tapered-dish autocorrelation (realistic beam); "sharp": keep b <= diameter.
    """

    diameter_m: float = 64.0
    model: str = "aperture"
    edge_taper_db: float = 12.0

    def transfer(self, b_m: np.ndarray) -> np.ndarray:
        """uv response vs physical baseline b [m]."""
        if self.model == "sharp":
            return (b_m <= self.diameter_m).astype(float)
        if self.model == "aperture":
            rho, prof = aperture_autocorr(self.edge_taper_db)
            return np.interp(b_m / self.diameter_m, rho, prof, right=0.0)
        raise ValueError(self.model)


@dataclass(frozen=True)
class Interferometer:
    """Interferometer with no zero spacing, perfectly deconvolved (noiseless).

    edge "soft": the response rises over shortest_baseline_m +/- dish_diameter_m, because a baseline b
    made with dishes of diameter d samples |u| lambda in [b - d, b + d]; "sharp": keep b >= shortest.
    umin_lambda > 0: instead a sharp cut fixed in wavelengths, the same in every channel.
    restore_arcmin > 0: Gaussian restoring beam of that FWHM in every channel.
    """

    shortest_baseline_m: float = 30.0
    dish_diameter_m: float = 12.0
    edge: str = "soft"
    umin_lambda: float = 0.0
    restore_arcmin: float = 0.0

    def transfer(self, u_lam: np.ndarray, lam: float) -> np.ndarray:
        """uv response at uv distance u [wavelengths], evaluating the inner edge at wavelength lam."""
        if self.umin_lambda > 0:
            t = (u_lam >= self.umin_lambda).astype(float)
        elif self.edge == "sharp":
            t = (u_lam * lam >= self.shortest_baseline_m).astype(float)
        elif self.edge == "soft":
            rho, prof = aperture_autocorr(0.0)  # uniformly illuminated antenna
            x = np.concatenate([-rho[::-1], rho[1:]]) * self.dish_diameter_m
            k = np.concatenate([prof[::-1], prof[1:]])
            cdf = np.cumsum(k) / k.sum()
            t = np.interp(u_lam * lam - self.shortest_baseline_m, x, cdf, left=0.0, right=1.0)
        else:
            raise ValueError(self.edge)
        if self.restore_arcmin > 0:
            theta = self.restore_arcmin / 60.0 * DEG
            t = t * np.exp(-((np.pi * theta * u_lam) ** 2) / (4 * np.log(2)))
        return t


ASKAP = Interferometer(shortest_baseline_m=22.0, dish_diameter_m=12.0)  # 36 x 12 m
MEERKAT = Interferometer(shortest_baseline_m=29.0, dish_diameter_m=13.5)  # 64 x 13.5 m
PARKES = SingleDish(diameter_m=64.0)


def responses(
    u_lam: np.ndarray,
    lam: float,
    band: Band,
    single_dish: SingleDish | None,
    interferometer: Interferometer | None,
    uv_match: bool = False,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """(single dish, interferometer) uv responses for one channel.

    uv_match=True gives every channel the same angular scales: the dish is evaluated at the longest
    wavelength (smoothing to a common resolution) and the interferometer's inner edge at the shortest
    (uv-range matching). Both responses are monotonic in b, so this is exactly what those operations do.
    """
    lam_sd = band.lams.max() if uv_match else lam
    lam_int = band.lams.min() if uv_match else lam
    t_sd = None if single_dish is None else single_dish.transfer(u_lam * lam_sd)
    t_int = None if interferometer is None else interferometer.transfer(u_lam, lam_int)
    return t_sd, t_int


def combined_response(t_sd: np.ndarray, t_int: np.ndarray, combine: str, joint_eps: float = 1e-2) -> np.ndarray:
    """uv response of the single dish + interferometer combination.

    feather: V = V_sd + (1 - T_sd) V_int (surface-brightness units), so T = T_sd + (1 - T_sd) T_int;
             the dish response is not deconvolved.
    joint:   V = (T_sd V_sd + T_int V_int) / (T_sd^2 + T_int^2 + eps), the noiseless limit of joint
             deconvolution; eps stands in for finite signal-to-noise.
    """
    if combine == "feather":
        return t_sd + (1.0 - t_sd) * t_int
    if combine == "joint":
        s2 = t_sd**2 + t_int**2
        return s2 / (s2 + joint_eps)
    raise ValueError(combine)


# ------------------------------------------------------------------------- observe


@dataclass
class Observation:
    """Complex Q + iU cubes (nchan, npix, npix) over the analysed field.

    cubes["truth"] is the unfiltered sky; "sd", "int" and "comb" are present when observed.
    """

    sky: Sky
    band: Band
    cubes: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def rm_true(self) -> np.ndarray:
        return self.sky.rm()

    @property
    def pi_true(self) -> np.ndarray:
        return np.full((self.sky.npix, self.sky.npix), self.sky.p0)


def observe(
    sky: Sky,
    band: Band = BANDS["possum1"],
    single_dish: SingleDish | None = PARKES,
    interferometer: Interferometer | None = ASKAP,
    combine: str | None = None,
    uv_match: bool = False,
    joint_eps: float = 1e-2,
) -> Observation:
    """Observe the sky channel by channel with each instrument (and optionally their combination)."""
    if combine and (single_dish is None or interferometer is None):
        raise ValueError("combine needs both a single dish and an interferometer")
    rm = sky.rm_padded()
    uvd = uv_grid(sky.nbig, sky.dl)
    keys = ["truth"] + ["sd"] * (single_dish is not None) + ["int"] * (interferometer is not None)
    keys += ["comb"] * bool(combine)
    nchan = band.freqs.size
    cubes = {k: np.empty((nchan, sky.npix, sky.npix), np.complex128) for k in keys}
    for i, lam in enumerate(band.lams):
        p = sky.p0 * np.exp(2j * (sky.chi0_rad + rm * lam**2))
        vis = np.fft.fft2(p)
        t_sd, t_int = responses(uvd, lam, band, single_dish, interferometer, uv_match)
        cubes["truth"][i] = sky.crop(p)
        # Q and U are filtered together as one complex field; the responses are real and radially
        # symmetric, so Q and U individually stay real.
        if t_sd is not None:
            cubes["sd"][i] = sky.crop(np.fft.ifft2(vis * t_sd))
        if t_int is not None:
            cubes["int"][i] = sky.crop(np.fft.ifft2(vis * t_int))
        if combine:
            cubes["comb"][i] = sky.crop(np.fft.ifft2(vis * combined_response(t_sd, t_int, combine, joint_eps)))
    return Observation(sky, band, cubes)


def uv_filter(img_padded: np.ndarray, sky: Sky, response: np.ndarray) -> np.ndarray:
    """Apply a uv response to a padded-box map (e.g. the true RM) and crop to the field."""
    return sky.crop(np.fft.ifft2(np.fft.fft2(img_padded) * response).real)


# -------------------------------------------------------------------- RM synthesis


def rmsynth(cube: np.ndarray, band: Band, phi_max: float = 400.0, dphi: float = 2.0, keep_fdf: bool = False) -> dict:
    """RM synthesis (rm-lite, uniform weights) and sub-sample peak finding on a complex P cube.

    Returns rm (peak Faraday depth), pi (peak polarised intensity), phi, fwhm (RMSF) and, with
    keep_fdf, the dirty Faraday dispersion function cube (nphi, npix, npix).
    """
    n = cube.shape[-1]
    res = rmsynth_3d(
        da.from_array(cube.real, chunks=(-1, n, n)),
        da.from_array(cube.imag, chunks=(-1, n, n)),
        band.freqs,
        weight_arr=np.ones(band.freqs.size),
        phi_max_radm2=phi_max,
        d_phi_radm2=dphi,
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
    if keep_fdf:
        out["fdf"] = fdf
    return out


# ---------------------------------------------------------------------------- plot


def plot_comparison(obs: Observation, res: dict, title: str = "", rm_err_lim: float | None = None):
    """Six panels: true / recovered / recovered - true, for RM (top row) and PI (bottom row).

    True and recovered share a colour scale. rm_err_lim defaults to half the RMSF FWHM.
    """
    sky = obs.sky
    rm_t, pi_t = obs.rm_true, obs.pi_true
    lo, hi = np.percentile(rm_t, [1, 99])
    pad = 0.15 * (hi - lo) + 1e-9
    elim = rm_err_lim if rm_err_lim is not None else res["fwhm"] / 2
    rows = [
        ("RM [rad m⁻²]", rm_t, res["rm"], dict(cmap="RdBu_r", vmin=lo - pad, vmax=hi + pad), elim),
        ("PI", pi_t, res["pi"], dict(cmap="viridis", vmin=0, vmax=1.05 * sky.p0), sky.p0),
    ]
    fig, axs = plt.subplots(2, 3, figsize=(14, 8.5), constrained_layout=True)
    for r, (name, true, rec, kw, lim) in enumerate(rows):
        for c, (img, lab, k) in enumerate(
            ((true, "true", kw), (rec, "recovered", kw), (rec - true, "recovered − true", None))
        ):
            k = k or dict(cmap="PuOr_r", vmin=-lim, vmax=lim)
            im = axs[r, c].imshow(img, origin="lower", extent=sky.extent, **k)
            axs[r, c].set_title(f"{name.split(' ')[0]}: {lab}")
            fig.colorbar(im, ax=axs[r, c], fraction=0.046, pad=0.03, label=name)
    for ax in axs[1]:
        ax.set_xlabel("l [deg]")
    for ax in axs[:, 0]:
        ax.set_ylabel("m [deg]")
    if title:
        fig.suptitle(title)
    return fig
