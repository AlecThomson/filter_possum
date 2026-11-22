"""Physics checks on the simulator, on small grids (~10 s). Each test checks a result the simulator
should reproduce analytically, not an implementation detail."""

import numpy as np
import pytest

from filter_possum import simulator as fs

SKY = fs.Sky(npix=32)
BAND = fs.BANDS["possum1"]


def test_unfiltered_sky_is_recovered_exactly():
    """RM synthesis of the unfiltered cube is a pipeline check: RM to << RMSF, PI = p0."""
    obs = fs.observe(SKY, BAND, None, None)
    res = fs.rmsynth(obs.cubes["truth"], BAND)
    assert np.sqrt(np.mean((res["rm"] - obs.rm_true) ** 2)) < 1e-2
    assert res["pi"].mean() == pytest.approx(1.0, abs=1e-4)


@pytest.mark.parametrize(("taper_db", "fwhm"), [(0.0, 1.03), (12.0, 1.157)])
def test_dish_beam_fwhm_matches_textbook(taper_db, fwhm):
    assert fs.dish_beam_fwhm(taper_db) == pytest.approx(fwhm, abs=0.01)


def test_single_dish_response_is_normalised_and_bounded():
    sd = fs.SingleDish()
    b = np.linspace(0, 1.5 * sd.diameter_m, 500)
    t = sd.transfer(b)
    assert t[0] == pytest.approx(1.0)
    assert np.all(np.diff(t) <= 1e-12)
    assert np.all(t[b >= sd.diameter_m] == 0)


def test_soft_inner_edge_is_half_power_at_shortest_baseline():
    i = fs.ASKAP
    c, d = i.shortest_baseline_m, i.dish_diameter_m
    w = i.transfer(np.array([c - d - 1, c, c + d + 1]), 1.0)  # lam = 1 m, so u [λ] = b [m]
    assert w == pytest.approx([0.0, 0.5, 1.0], abs=0.01)


def test_uv_match_is_frequency_independent_and_equals_a_lambda_cut():
    u = np.linspace(0, 600, 301)
    lo = fs.responses(u, BAND.lams.max(), BAND, fs.PARKES, fs.ASKAP, uv_match=True)
    hi = fs.responses(u, BAND.lams.min(), BAND, fs.PARKES, fs.ASKAP, uv_match=True)
    np.testing.assert_allclose(lo, hi)
    sharp = fs.Interferometer(edge="sharp")
    a = fs.observe(SKY, BAND, None, sharp, uv_match=True).cubes["int"]
    b = fs.observe(SKY, BAND, None, fs.Interferometer(umin_lambda=sharp.shortest_baseline_m / BAND.lams.min())).cubes[
        "int"
    ]
    np.testing.assert_array_equal(a, b)


def test_combination_responses():
    """Feather keeps the dish response undeconvolved (a dip); joint is ~1 wherever either instrument covers."""
    u = np.linspace(0, 600, 601)
    t_sd, t_int = fs.responses(u, BAND.lams.mean(), BAND, fs.PARKES, fs.Interferometer())
    feather = fs.combined_response(t_sd, t_int, "feather")
    joint = fs.combined_response(t_sd, t_int, "joint", joint_eps=1e-4)
    assert np.all(feather <= 1 + 1e-12) and np.all(feather >= np.maximum(t_sd, t_int) - 1e-12)
    assert feather.min() < 0.7
    assert joint.min() > 0.99


@pytest.mark.parametrize("umin", [40.0, 120.0])
def test_weak_screen_interferometer_pi_traces_small_scale_rm(umin):
    """For 2 sigma lambda^2 << 1: PI_int = 2 lambda_0^2 |RM fluctuations on scales the interferometer sees|."""
    sky = fs.Sky(npix=64, rm_sigma=1.0)
    obs = fs.observe(sky, BAND, None, fs.Interferometer(umin_lambda=umin))
    res = fs.rmsynth(obs.cubes["int"], BAND)
    hp = fs.uv_filter(sky.rm_padded() - sky.rm_mean, sky, fs.uv_grid(sky.nbig, sky.dl) >= umin)
    pred = 2 * res["lam_sq_0"] * np.abs(hp)
    assert np.corrcoef(res["pi"].ravel(), pred.ravel())[0, 1] > 0.99
    assert res["pi"].mean() / pred.mean() == pytest.approx(1.0, abs=0.05)


@pytest.mark.parametrize("grad", [20.0, 50.0, 75.0])
def test_single_dish_gradient_depolarisation(grad):
    """Pure gradient: Q/U stripes at u = |grad RM| lambda^2 / pi; the dish passes them with weight
    T_sd(u lambda), so its PI is the band mean of that, exactly."""
    sky = fs.Sky(npix=64, rm_sigma=0.0, rm_grad=grad)
    res = fs.rmsynth(fs.observe(sky, BAND, fs.PARKES, None).cubes["sd"], BAND)
    pred = np.mean(fs.PARKES.transfer((grad / fs.DEG) * BAND.lams**3 / np.pi))
    assert res["pi"].mean() == pytest.approx(pred, abs=2e-3)


def test_joint_combination_recovers_truth():
    obs = fs.observe(SKY, BAND, fs.PARKES, fs.Interferometer(), combine="joint", joint_eps=1e-4)
    res = fs.rmsynth(obs.cubes["comb"], BAND)
    assert np.sqrt(np.mean((res["rm"] - obs.rm_true) ** 2)) < 0.05
    assert res["pi"].mean() > 0.99


def test_band_presets():
    assert BAND.freqs.size == 288
    f = fs.BANDS["possum12"].freqs
    assert f.size == 288 + 144 and not np.any((f > 1088e6) & (f < 1296e6))


def test_plot_comparison_has_six_panels():
    obs = fs.observe(SKY, BAND, fs.PARKES, None)
    fig = fs.plot_comparison(obs, fs.rmsynth(obs.cubes["sd"], BAND), "single dish")
    assert len([a for a in fig.axes if a.get_title()]) == 6


def test_integer_parameters_are_accepted():
    sky = fs.Sky(npix=32, rm_mean=30, rm_sigma=10, rm_grad=5)
    np.testing.assert_allclose(sky.rm(), fs.Sky(npix=32, rm_mean=30.0, rm_sigma=10.0, rm_grad=5.0).rm())
