"""Physics checks for the uv-filtering simulation.

Small grids (32-64 px) keep the suite fast; each test checks a claim the
results rely on rather than implementation details.
"""

from dataclasses import replace

import numpy as np
import pytest

from filter_possum import uvfilter_sim as sim

SMALL = sim.SimConfig(npix=32)
REAL = replace(SMALL, **sim.REALISTIC)


def test_truth_cube_recovers_input_rm_and_pi():
    """RM synthesis of the unfiltered cube is a pipeline check: errors << RMSF FWHM."""
    m = sim.run(SMALL)
    assert m["truth"]["rm_rms_err"] < 1e-2
    assert m["truth"]["pi_mean"] == pytest.approx(1.0, abs=1e-4)


@pytest.mark.parametrize(("taper_db", "fwhm"), [(0.0, 1.03), (12.0, 1.157)])
def test_dish_beam_fwhm_matches_textbook(taper_db, fwhm):
    """Uniform dish ~1.03 lambda/D; 12 dB edge taper ~1.16 lambda/D."""
    assert sim.dish_fwhm_factor(taper_db) == pytest.approx(fwhm, abs=0.01)


def test_sd_aperture_transfer_is_normalised_and_bounded():
    b = np.linspace(0, 1.5 * REAL.sd_bmax_m, 500)
    t = sim.sd_transfer(REAL, b)
    assert t[0] == pytest.approx(1.0)
    assert np.all(np.diff(t) <= 1e-12)  # monotonically non-increasing
    assert np.all(t[b >= REAL.sd_bmax_m] == 0)


def test_soft_inner_edge_shape():
    """Smeared over +/- one antenna diameter, half-power at the nominal edge."""
    c, d = REAL.int_bmin_m, REAL.ant_diam_m
    w = sim.int_inner_transfer(REAL, np.array([c - d - 1, c, c + d + 1]))
    assert w == pytest.approx([0.0, 0.5, 1.0], abs=0.01)


def test_fixed_mode_is_frequency_independent():
    cfg = replace(REAL, cut_mode="fixed")
    u = np.linspace(0, 600, 301)
    lo = sim.uv_masks(cfg, u, cfg.lams.max())
    hi = sim.uv_masks(cfg, u, cfg.lams.min())
    np.testing.assert_allclose(lo, hi)


def test_combined_transfer_functions():
    """Feather keeps the SD response undeconvolved (dip); joint is ~1 where either covers."""
    u = np.linspace(0, 600, 601)
    t_sd, t_int = sim.uv_masks(REAL, u, REAL.lams.mean())
    feather = sim.combined_transfer(replace(REAL, combine="feather"), t_sd, t_int)
    joint = sim.combined_transfer(replace(REAL, combine="joint", joint_eps=1e-4), t_sd, t_int)
    assert np.all(feather <= 1 + 1e-12) and np.all(feather >= np.maximum(t_sd, t_int) - 1e-12)
    assert feather.min() < 0.7  # the feather dip near b ~ 25 m
    assert joint.min() > 0.99


def test_weak_screen_interferometer_pi_traces_high_passed_rm():
    """For 2 sigma lambda^2 << 1: PI_int ~= 2 lambda_0^2 |delta_phi_HP| (uv-matched cuts)."""
    cfg = replace(sim.SimConfig(npix=64), rm_sigma=1.0, cut_mode="fixed")
    phi = sim.make_rm_sky(cfg)
    res = sim.rm_synthesis(sim.observe(cfg, phi)["int"], cfg)
    u_int = cfg.int_bmin_m / cfg.lams.min()
    hp_mask = sim.uv_grid(cfg.nbig, cfg.dl) >= u_int
    hp = sim.apply_transfer(phi - cfg.rm_mean, cfg, hp_mask)
    pred = 2 * res["lam_sq_0"] * np.abs(hp)
    assert np.corrcoef(res["pi"].ravel(), pred.ravel())[0, 1] > 0.99
    assert res["pi"].mean() / pred.mean() == pytest.approx(1.0, abs=0.05)


@pytest.mark.parametrize("grad", [20.0, 50.0, 75.0])
def test_single_dish_gradient_depolarisation_is_fringe_transfer(grad):
    """Pure RM gradient: SD PI = band mean of T_sd(|grad RM| lambda^3 / pi), exactly."""
    cfg = replace(REAL, npix=64, rm_sigma=0.0, rm_grad_deg=grad)
    m = sim.run(cfg)
    assert m["sd"]["pi_mean"] == pytest.approx(sim.predicted_gradient_depol(cfg, grad), abs=2e-3)


def test_joint_combination_recovers_truth():
    m = sim.run(replace(REAL, combine="joint", joint_eps=1e-4))
    assert m["comb"]["rm_rms_err"] < 0.05
    assert m["comb"]["pi_mean"] > 0.99
    assert m["int"]["rm_rms_err"] > 10 * m["comb"]["rm_rms_err"]  # INT alone is much worse


# --------------------------------------------------------------- test 3: interferometers


def test_generic_lambda_cut_equals_uv_matched_sharp_cut():
    """A sharp cut fixed in wavelengths at b_min / lambda_min is exactly uv-range matching."""
    cfg = replace(SMALL, cut_mode="fixed")
    gen = replace(SMALL, int_umin_lam=SMALL.int_bmin_m / SMALL.lams.min())
    phi = sim.make_rm_sky(cfg)
    np.testing.assert_array_equal(sim.observe(cfg, phi)["int"], sim.observe(gen, phi)["int"])


@pytest.mark.parametrize("umin", [40.0, 120.0])
def test_generic_cut_weak_screen_pi_traces_high_passed_rm(umin):
    """Weak-screen result holds for any fixed-λ cut: PI_int ~= 2 lambda_0^2 |delta_phi_HP(u >= u_min)|."""
    cfg = replace(sim.SimConfig(npix=64), rm_sigma=1.0, int_umin_lam=umin)
    phi = sim.make_rm_sky(cfg)
    res = sim.rm_synthesis(sim.observe(cfg, phi)["int"], cfg)
    hp = sim.apply_transfer(phi - cfg.rm_mean, cfg, sim.uv_grid(cfg.nbig, cfg.dl) >= umin)
    pred = 2 * res["lam_sq_0"] * np.abs(hp)
    assert np.corrcoef(res["pi"].ravel(), pred.ravel())[0, 1] > 0.99
    assert res["pi"].mean() / pred.mean() == pytest.approx(1.0, abs=0.05)


def test_rm_transfer_is_one_for_truth_and_tracks_a_known_filter():
    m = sim.run(replace(SMALL, scale_metrics=True))
    np.testing.assert_allclose(m["truth"]["rm_transfer"], 1.0, atol=1e-3)
    # a pure linear filter of the RM map gives back (approximately, given apodisation) its transfer function
    cfg = sim.SimConfig(npix=64)
    phi = sim.make_rm_sky(cfg)
    uvd = sim.uv_grid(cfg.nbig, cfg.dl)
    filt = sim.apply_transfer(phi, cfg, uvd >= 50.0)
    u, t = sim.rm_transfer(filt, sim.crop(phi, cfg), cfg.dl)
    assert np.all(t[u < 25] < 0.15) and np.all(t[u > 100] > 0.85)


def test_band_presets():
    """Default band is POSSUM band 1; band 1 + 2 keeps the gap; array preset sets the INT edge."""
    np.testing.assert_array_equal(sim.SimConfig().freqs, replace(SMALL, freq_segments=sim.BANDS["possum1"]).freqs)
    f = replace(SMALL, freq_segments=sim.BANDS["possum12"]).freqs
    assert f.size == 288 + 144 and not np.any((f > 1088e6) & (f < 1296e6))
    c = sim.array_config(REAL, "askap", "possum1")
    assert sim.int_inner_transfer(c, np.array([22.0]))[0] == pytest.approx(0.5, abs=0.01)


@pytest.mark.parametrize("umin", [60.0, 80.0, 100.0, 120.0])  # in-band fringes span 69-128 λ at 50 rad m^-2 deg^-1
def test_generic_cut_gradient_pi_is_fraction_of_channels_passing(umin):
    """Pure gradient: INT PI ~= fraction of channels whose fringe u = |grad RM| lambda^2 / pi clears u_min.

    The ramp is finite (grad_ramp_frac x padded box ~ 7 deg), so each fringe is spread over
    w ~ 1/ramp ~ 8 λ; a sharp cut resolves that, so the PI is bracketed by the prediction at u_min -/+ w.
    """
    cfg = replace(sim.SimConfig(npix=64), rm_sigma=0.0, rm_grad_deg=50.0, int_umin_lam=umin)
    w = 1.0 / (cfg.grad_ramp_frac * cfg.nbig * cfg.dl)

    def frac(u):
        return np.mean(50.0 / sim.DEG * cfg.lams**2 / np.pi >= u)

    assert sim.predicted_gradient_int_pi(cfg, 50.0) == pytest.approx(frac(umin))
    pi = sim.rm_synthesis(sim.observe(cfg, sim.make_rm_sky(cfg))["int"], cfg)["pi"].mean()
    assert frac(umin + w) - 5e-3 <= pi <= frac(umin - w) + 5e-3


def test_interferometer_rm_errors_sit_at_pi_nulls():
    """Weak screen: INT RM is well defined where its PI (~ 2 lambda^2 |delta_phi_HP|) is bright;
    the error comes from the nulls, so the brighter half has < half the overall rms error. RM
    structure is recovered (via the phase) at scales larger than the cut: transfer falls to 0.5 at u_min."""
    m = sim.run(replace(sim.SimConfig(npix=64), rm_sigma=2.0, int_umin_lam=80.0, scale_metrics=True))["int"]
    assert m["rm_rms_err_bright_half"] < 0.5 * m["rm_rms_err"]
    assert m["rm_transfer_u50"] == pytest.approx(80.0, rel=0.25)
    assert np.mean(m["rm_transfer"][:4]) > 0.6
