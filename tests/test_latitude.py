"""Fixture family 4: a latitude subsphere moved from polar angle theta to theta' at fixed longitude, and
shear from an explicit map with a known induced metric.

Latitude: x = 2 (cos theta e0 + sin theta omega), omega on the unit 2-sphere of span(e1, e2, e3).
Moving to theta' at fixed omega scales every chord by s = sin theta' / sin theta, keeps the tangent plane
(the meridian transport fixes T_x M), and the displacement v = 2 (theta' - theta) n is purely normal with
grad_u v = (theta' - theta) cot(theta) u.  Hence: every stretch s, log J = m log s, strain rate
E = thetadot cot(theta) T at any step size, all of it from the normal-motion term, zero tilt, and mass
balance (uniform density on both spheres) exact.

Shear: the projective map x -> 2 A x / |A x| of a great 2-sphere onto itself, A = exp(h S) with S
symmetric trace-free in the span: finite stretches are the singular values of the exact differential;
the rate is E = (T S T - (x.S x / 4) T) h / dt + O(h^2).
"""
import numpy as np
import pytest
from scipy.linalg import expm

from mm import sphere, step, totals
from mm import fixtures as fx
from helpers import fib_great_sphere

R = sphere.R
M = 2


def latitude_cloud(N, theta, fr):
    om = fx.fibonacci_sphere(N, 0.0)
    return fx.latitude_points(om, theta, fr), om


def run_latitude(N, theta, theta1, tau, k, dt=1.0, seed=40):
    fr = fx.random_frame(4, np.random.default_rng(seed))
    X, om = latitude_cloud(N, theta, fr)
    Y = fx.latitude_points(om, theta1, fr)
    o = step.measure_step(X, Y, dt, operator=dict(kind="fixed", tau=tau, k=k), mask=False)
    return X, Y, o, fr


def latitude_errors(o, theta, theta1, dt):
    ok = o["dim"] == M
    s = np.sin(theta1) / np.sin(theta)
    rate = (theta1 - theta) / dt / np.tan(theta)          # thetadot cot(theta)
    st = o["stretches"][ok, :M]
    return dict(
        frac_dim=ok.mean(),
        stretch=np.max(np.abs(np.median(st, 0) / s - 1)),
        stretch_rms=np.sqrt(np.mean((st / s - 1) ** 2)),
        logJ=np.median(np.abs(o["logJ"][ok] - M * np.log(s))),
        trE=np.median(np.abs(o["E_trace"][ok] / (M * rate) - 1)),
        E_iso=np.median(o["E_shear"][ok]) / abs(rate),
        flow=np.median(np.abs(o["E_flow_trace"][ok])) / abs(M * rate),
        normal=np.median(np.abs(o["E_normal_trace"][ok] / (M * rate) - 1)),
        tilt=np.median(o["tilt_max"][ok]),
        tan=np.median(o["tan_fraction"][ok]),
        mass=np.median(o["mass_balance"][ok]),
        mass_rms=np.sqrt(np.nanmean(o["mass_balance"][ok] ** 2)),
    )


@pytest.mark.parametrize("theta,theta1", [(0.9, 1.0), (1.2, 0.8), (0.5, 0.55)])
def test_latitude_move(theta, theta1):
    dt = 0.5
    X, Y, o, fr = run_latitude(8000, theta, theta1, tau=0.004, k=256, dt=dt)
    e = latitude_errors(o, theta, theta1, dt)
    print("latitude", theta, theta1, {k: round(float(v), 5) for k, v in e.items()})
    assert e["frac_dim"] > 0.99
    assert e["stretch"] < 2e-3 and e["stretch_rms"] < 5e-3
    assert e["logJ"] < 4e-3
    assert e["trE"] < 0.01 and e["E_iso"] < 0.01
    assert e["flow"] < 0.01 and e["normal"] < 0.01         # all of it from the normal-motion term
    assert e["tilt"] < 0.01
    assert e["tan"] < 1e-3
    assert abs(e["mass"]) < 0.01 and e["mass_rms"] < 0.05
    # sign: expansion is positive when sin grows
    s = np.sin(theta1) / np.sin(theta)
    assert np.sign(np.median(o["E_trace"])) == np.sign(np.log(s))
    # tangential motion is absent, all kinetic energy is normal
    ok = o["dim"] == M
    np.testing.assert_allclose(np.median(o["ke_norm"][ok] / o["ke"][ok]), 1.0, atol=1e-3)
    np.testing.assert_allclose(o["travel"], 2 * abs(theta1 - theta), rtol=1e-12)
    # every release rule holds on this clean fixture (attribution: stable projector field)
    rel = {k: float(v[ok].mean()) for k, v in o["release"].items()}
    print("release", rel, "proj_rms med", np.median(o["proj_rms_angle"][ok]), "proj_agree med",
          np.median(o["proj_agree"][ok]), "attribution_noise med", np.median(o["attribution_noise"][ok]),
          "split med", np.median(o["split_angle"][ok]), "|E| med", np.median(np.sqrt(o["strain_sq"][ok])))
    for k in ("tangent", "strain", "strain_finite", "logJ", "tilt", "attribution", "mass_balance"):
        assert rel[k] > 0.95, (k, rel)


@pytest.mark.slow
def test_latitude_convergence_sweep():
    """Bandwidth and sample-size sweep: errors decrease and reach the stated tolerance."""
    theta, theta1, dt = 0.9, 1.0, 1.0
    rows = []
    for N, tau, k in ((4000, 0.008, 128), (16000, 0.002, 128), (64000, 0.0005, 128)):
        X, Y, o, fr = run_latitude(N, theta, theta1, tau=tau, k=k, dt=dt)
        rows.append((N, tau, latitude_errors(o, theta, theta1, dt)))
        print(rows[-1])
    for key, tol in (("stretch", 5e-4), ("trE", 3e-3), ("flow", 3e-3), ("tilt", 3e-3)):
        vals = [r[2][key] for r in rows]
        assert vals[-1] < vals[0], (key, vals)
        assert vals[-1] < tol, (key, vals)


def test_latitude_random_sampling_and_totals():
    """Random (not low-discrepancy) sampling: same answers within sampling noise; totals and mass fractions."""
    rng = np.random.default_rng(41)
    fr = fx.random_frame(4, rng)
    theta, theta1, dt = 0.9, 1.0, 1.0
    X, om = fx.latitude_sphere(12000, M, theta, rng, frame=fr)
    Y = fx.latitude_points(om, theta1, fr)
    o = step.measure_step(X, Y, dt, operator=dict(kind="fixed", tau=0.006, k=256), mask=False)
    e = latitude_errors(o, theta, theta1, dt)
    print("release", {k: round(float(v.mean()), 3) for k, v in o["release"].items()}, "trunc", np.median(o["trunc_tail"]))
    print("latitude random", {k: round(float(v), 5) for k, v in e.items()})
    assert e["stretch"] < 0.01 and e["trE"] < 0.03 and e["normal"] < 0.03 and e["flow"] < 0.03
    assert abs(e["mass"]) < 0.03
    om_ = step.apply_release(o)
    tot = totals.step_totals(om_)
    # masked totals: mass fractions in [0, 1], additive splits add up on their common sets
    for name, val in tot.items():
        if name.startswith("split:"):
            parts = list(val.values())
            whole = parts[0]["value"]
            if np.isfinite(whole):
                np.testing.assert_allclose(whole, sum(p["value"] for p in parts[1:]), rtol=1e-9)
        elif isinstance(val, dict):
            assert 0.0 <= val["mass_fraction"] <= 1.0
    # totals from raw and from masked output are identical (every total selects on its release rule)
    tot_raw = totals.step_totals(o)
    for name, val in tot.items():
        if isinstance(val, dict) and "value" in val:
            np.testing.assert_allclose(tot_raw[name]["value"], val["value"], rtol=1e-12, equal_nan=True)
            assert tot_raw[name]["mass_fraction"] == val["mass_fraction"], name
        elif name.startswith("split:"):
            for f, part in val.items():
                np.testing.assert_allclose(tot_raw[name][f]["value"], part["value"], rtol=1e-12, equal_nan=True)
    s = np.sin(theta1) / np.sin(theta)
    assert tot["logJ"]["mass_fraction"] > 0.5
    np.testing.assert_allclose(tot["logJ"]["value"], M * np.log(s), rtol=0.02)
    np.testing.assert_allclose(tot["E_trace"]["value"], M * (theta1 - theta) / np.tan(theta), rtol=0.03)


# ------------------------------------------------------------------------------------------------------
# shear from an explicit map

def shear_case(N, h, tau, k, dt=1.0, seed=42):
    fr = fx.random_frame(3, np.random.default_rng(seed))
    X = fib_great_sphere(N, 0.5, fr)
    S3 = np.diag([1.0, -0.3, -0.7])
    S = fr @ S3 @ fr.T
    A = expm(h * S)
    Y = fx.projective_map(X, A)
    o = step.measure_step(X, Y, dt, operator=dict(kind="fixed", tau=tau, k=k), mask=False)
    T = (fr @ fr.T)[None] - X[:, :, None] * X[:, None, :] / R**2
    D = fx.projective_differential(X, A)
    s_ex = np.linalg.svd(D @ T, compute_uv=False)[:, :M]
    E_ex = (T @ S @ T - (np.einsum("ni,ij,nj->n", X, S, X) / R**2)[:, None, None] * T) * h / dt
    return X, Y, o, s_ex, E_ex


@pytest.mark.parametrize("h", [0.05, 0.4])
def test_shear_map(h):
    X, Y, o, s_ex, E_ex = shear_case(16000, h, tau=0.002, k=128)
    ok = o["dim"] == M
    st = o["stretches"][ok, :M]
    err_s = np.median(np.abs(st / s_ex[ok] - 1), 0)
    ls = np.log(s_ex[ok])
    shear_ex = np.sum(ls**2, 1) - np.sum(ls, 1) ** 2 / M
    err_shear = np.median(np.abs(o["fin_shear"][ok] - shear_ex)) / np.median(shear_ex)
    print("shear h", h, "stretch err", err_s, "fin_shear rel err", err_shear)
    assert err_s.max() < 3e-3
    assert err_shear < 0.02
    if h <= 0.05:   # rate form is first order in the step
        E = o["E"][ok]
        relE = np.median(np.linalg.norm(E - E_ex[ok], axis=(1, 2)) / np.linalg.norm(E_ex[ok], axis=(1, 2)))
        print("shear rate rel err", relE)
        assert relE < 0.05
