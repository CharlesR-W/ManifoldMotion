"""Matched-regime fixture (9-dimensional blob, as at the real b7) and the flow / thick rule sets.

The full fixture (tests/matched.py: groups smooth, weakstretch, weakcomp, rank8, nearsing, vertex, fold,
curved, controls; seeds 0 and 1) runs on Modal; its numbers are in the run report.  Frozen per-field
tolerances (|estimate - analytic truth|, set on seed 0, verified on seed 1):
    ic_log_gain_max 0.1;  ic_E_trace 0.2 x median |E| (0.81 here);  ic_strain_sq, ic_spin_sq 20% of the
    cloud median;  released-point false-release rate <= 10% wherever held-out (intercept) >= 0.8.
The fast tests below check the rule sets' masking, allowlists and totals; the slow one reruns the smooth
group at k = 256 and checks the frozen tolerances.
"""
import numpy as np
import pytest

from mm import fixtures as fx, step, totals
from mm.frame import Rules, analyse_frame, rerule
from mm import graph


@pytest.fixture(scope="module")
def blob_step():
    rng = np.random.default_rng(5)
    b = fx.MatchedBlob(rng, 0.55)
    X = b.sample(2500, rng)
    Yc = b.f(X)
    from mm import sphere
    Y = sphere.exp_map(Yc, sphere.tangent_project(Yc, 0.02 * rng.standard_normal(Yc.shape)))
    return X, Y


def test_flow_mode_masks_and_allowlist(blob_step):
    X, Y = blob_step
    o = step.measure_step(X, Y, 1.0, rules=Rules.flow(), operator=dict(k=96), mask=False)
    om = step.apply_release(o)
    assert np.all(o["dim"] == 9)                                  # nine directions or withheld
    allowed = step.MODE_FIELDS["flow"]
    for key in step.FIELD_RULE:
        if key in om and key not in allowed:
            assert np.isnan(np.asarray(om[key], float)).all(), key
    # normal split, attribution, mass balance, log J never released in flow mode
    for key in ("ke_norm", "ke_tan", "tan_fraction", "E_flow", "mass_balance", "logJ", "log_stretches", "hencky"):
        assert np.isnan(np.asarray(om[key], float)).all(), key
    # released flow fields: finite somewhere, NaN exactly where their rule fails
    for key in ("ic_log_gain_max", "ic_E_trace", "ic_spin_sq", "relmotion_slope", "heldout_ic"):
        m = step.release_mask(o, key)
        assert m.any(), key
        assert np.isnan(om[key][~m]).all()
    # totals from raw and masked output agree and honour the allowlist
    t_raw, t_m = totals.step_totals(o), totals.step_totals(om)
    for name in totals.TOTAL_FIELDS:
        a, b = t_raw[name], t_m[name]
        assert a["mass_fraction"] == b["mass_fraction"], name
        np.testing.assert_allclose(a["value"], b["value"], equal_nan=True)
        if name not in allowed:
            assert a["mass_fraction"] == 0.0, name
    sp = t_m["split:relmotion_shift_slope_res"]
    np.testing.assert_allclose(sp["relmotion"]["value"], sp["relmotion_shift"]["value"] + sp["relmotion_slope"]["value"]
                               + sp["relmotion_icres"]["value"], rtol=1e-9)
    # budget exact per point
    r = o["relmotion"] - o["relmotion_shift"] - o["relmotion_slope"] - o["relmotion_icres"]
    assert np.nanmax(np.abs(r) / o["relmotion"]) < 1e-9


def test_thick_mode_is_a_rerule_of_strict(blob_step):
    X, _ = blob_step
    op = graph.build_operator(X, k=96, kind="adaptive", eps=0.25, k_bw=16)
    fs = analyse_frame(op, Rules.strict())
    ft1 = analyse_frame(op, Rules.thick())
    ft2 = rerule(fs, Rules.thick())
    assert np.array_equal(ft1.ok_tangent, ft2.ok_tangent)
    assert not (ft1.ok_tangent & ~ft1.rank_stable).any()
    with pytest.raises(ValueError):
        rerule(fs, Rules.flow())


def test_thick_allowlist(blob_step):
    X, Y = blob_step
    o = step.measure_step(X, Y, 1.0, rules=Rules.thick(), operator=dict(k=96), mask=True)
    for key in ("E", "logJ", "ic_E_trace", "E_flow", "mass_balance", "relmotion"):
        assert np.isnan(np.asarray(o[key], float)).all(), key


@pytest.mark.slow
def test_matched_smooth_group(tmp_path):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    import matched
    matched.run_group("smooth", 1, 256, 0.25, tmp_path)
    ref = np.load(tmp_path / "smooth_s1_k256_e0.25_clean.npz")
    Escale = np.nanmedian(np.sqrt(ref["true_rate_total"]))
    # frozen: <= 10% false releases at held-out (intercept) >= 0.9; <= 15% at ~0.8 (fixture: 7-10%)
    for var, fmax in (("clean", 0.10), ("endpoint0.3", 0.10), ("endpoint0.6", 0.10), ("endpoint1.0", 0.15)):
        d = np.load(tmp_path / f"smooth_s1_k256_e0.25_{var}.npz")
        for key, tk, tol, rule in (("ic_E_trace", "true_E_trace", 0.2 * Escale, "rel_strain_ic"),
                                   ("ic_strain_sq", "true_strain_sq", 0.2 * np.nanmedian(ref["true_strain_sq"]), "rel_strain_ic"),
                                   ("ic_spin_sq", "true_spin_sq", 0.2 * np.nanmedian(ref["true_spin_sq"]), "rel_strain_ic")):
            sel = (d[rule] > 0) & np.isfinite(d[key])
            assert sel.mean() > 0.5, (var, key)
            assert np.mean(np.abs(d[key][sel] - d[tk][sel]) > tol) <= fmax, (var, key)
    sh = np.load(tmp_path / "smooth_s1_k256_e0.25_shuffled.npz")
    assert (sh["rel_strain_ic"] > 0).mean() < 0.01 and (sh["rel_strain_finite_ic"] > 0).mean() < 0.01
    ev = np.load(tmp_path / "smooth_s1_k256_e0.25_edge0.6.npz")
    assert np.nanmedian(ev["anch_endpoint"]) > 3 * np.nanmedian(ev["ic_endpoint"])      # anchor bias


def test_flow_fit_quantities_need_nine_directions():
    """A 9-dimensional blob next to a thin 2-dimensional patch: fit-free relative motion is released on
    both, every flow fit quantity (free-intercept budget, fit scores, strain) only where all nine
    directions are resolved; the additive split's common support likewise."""
    rng = np.random.default_rng(6)
    b = fx.MatchedBlob(rng, 0.3)
    Xb = b.sample(1500, rng)
    fr = fx.random_frame(3, rng)
    from mm import sphere
    Xt = fx.great_sphere(1500, 2, rng, frame=fr)[0]
    X = np.concatenate([Xb, Xt])
    Y = sphere.exp_map(X, sphere.tangent_project(X, 0.05 * rng.standard_normal(X.shape)))
    o = step.measure_step(X, Y, 1.0, rules=Rules.flow(), operator=dict(k=64), mask=False)
    om = step.apply_release(o)
    thin = np.arange(len(X)) >= 1500
    assert (o["dim"][thin] < 9).mean() > 0.9 and (o["dim"][~thin] == 9).mean() > 0.9
    assert np.isfinite(om["relmotion"][thin]).mean() > 0.9                # fit-free: rank-free
    for key in ("relmotion_shift", "relmotion_slope", "relmotion_icres", "heldout_ic", "heldout_v_ic",
                "intercept_rel", "cond_centred", "ic_log_gain_max", "ic_E_trace", "ic_spin_sq"):
        fin = np.isfinite(np.asarray(om[key], float))
        assert not (fin & (o["dim"] < 9)).any(), key
    assert np.isfinite(om["relmotion_slope"][~thin]).mean() > 0.5
    tot = totals.step_totals(om)
    sp = tot["split:relmotion_shift_slope_res"]
    assert sp["relmotion"]["mass_fraction"] <= np.mean(o["dim"] == 9) + 1e-12
    for name in step.ANCHORED_DIAG:                                # anchored fits never public in flow mode
        assert not step.release_mask(o, name).any(), name
        if name in tot:
            assert tot[name]["mass_fraction"] == 0.0, name


def test_thick_tilt_needs_stable_planes_not_a_good_fit():
    """Thin great 2-sphere rotated out of its plane, images shuffled: endpoint planes are stable and equal
    rank, the offset prediction is useless.  Tilt is released (plane geometry), the strain is not."""
    from scipy.linalg import expm
    rng = np.random.default_rng(7)
    fr = fx.random_frame(4, rng)
    X = fx.great_sphere(3000, 2, rng, frame=fr[:, :3])[0]
    A = np.outer(fr[:, 3], fr[:, 0]) - np.outer(fr[:, 0], fr[:, 3])
    Y = (X @ expm(0.3 * A).T)[rng.permutation(len(X))]
    o = step.measure_step(X, Y, 1.0, rules=Rules.thick(), operator=dict(k=64), mask=False)
    rel = o["release"]
    planes = rel["tangent"] & o["ok_tangent1"] & (o["tgt_dim"] == o["dim"])
    assert planes.mean() > 0.3
    assert rel["strain_finite"].mean() < 0.05
    om = step.apply_release(o)
    assert np.array_equal(np.isfinite(om["tilt_max"]), planes & (o["dim"] < 9) & np.isfinite(o["tilt_max"]))
    # strict mode keeps its prediction gate: the same step releases no strict tilt
    os_ = step.measure_step(X, Y, 1.0, rules=Rules.strict(), operator=dict(k=64), mask=True)
    assert np.isfinite(os_["tilt_max"]).mean() < 0.05
    assert not (np.isfinite(os_["tilt_max"]) & ~os_["release"]["strain_finite"]).any()
