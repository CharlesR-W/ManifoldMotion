"""Fixture family 2: identity map on a non-uniformly sampled great 2-subsphere, and the operator tests of
SPEC Section 2 for both the fixed-bandwidth and the adaptive operator.

Exact (any bandwidth): identity map gives F = T (unit stretches), zero motion, strain, tilt; undefined
ratios are NaN.  Continuum (the cloud: a great 2-sphere with density prop. to exp(kappa (x.e) / 2),
kappa = 2):
  Gamma(f, f) -> |T c|^2 for f = c . x           (diffusion scale; cannot see a wrong drift)
  L f -> -m f / 4 + grad log rho . T c            (needs the right drift)
  L X -> -m x / 4 + grad log rho                  (tangential L X = grad log rho != 0 where rho varies)
  G -> T (top m eigenvalues 1)
A deliberately wrong bandwidth convention (exp(-d^2 / 2 tau)) and wrong degree corrections (alpha = 0, 1;
adaptive without the 1 / (sigma_i sigma_j) factor) must fail the same checks.
"""
import numpy as np
import pytest

from mm import graph, step, sphere
from mm import fixtures as fx
from helpers import fib_great_sphere, operator_errors, sigma_field

KAPPA = 2.0
M = 2
# tolerances of the continuum checks at the bandwidths used below (relative errors).  The fixed operator
# needs k large enough that the kNN rows contain the kernel at the densest point (density ratio e^4 between
# the poles here): with k = 280 the dense pole is truncated and Gamma comes out 3% low.
TOL = dict(gamma=0.03, L=0.05, drift=0.08, G=0.05)


def passes(e, tol=TOL):
    return all(e[k] < v for k, v in tol.items())


@pytest.fixture(scope="module")
def fr():
    return fx.random_frame(3, np.random.default_rng(1))


@pytest.fixture(scope="module")
def fib16(fr):
    return fib_great_sphere(16000, KAPPA, fr)


@pytest.mark.parametrize("kind", ["fixed", "adaptive"])
def test_identity_map_exact(fr, kind):
    rng = np.random.default_rng(2)
    X = fx.great_sphere(4000, 2, rng, kappa=KAPPA, frame=fr)[0]
    opk = dict(kind="fixed", tau=0.005, k=128) if kind == "fixed" else dict(kind="adaptive", eps=0.25, k_bw=16, k=128)
    o = step.measure_step(X, X.copy(), 1.0, operator=opk, mask=False)
    ok = o["dim"] == M
    assert ok.mean() > 0.95
    np.testing.assert_allclose(o["stretches"][ok, :M], 1.0, atol=1e-12)
    np.testing.assert_allclose(o["logJ"][ok], 0.0, atol=1e-12)
    # F = G G_T^+ = the retained projector, which is the great sphere's T (offsets lie in its plane)
    T = (fr @ fr.T)[None] - X[ok][:, :, None] * X[ok][:, None, :] / sphere.R**2
    np.testing.assert_allclose(o["F"][ok], T, atol=1e-11)
    for key in ("travel", "relmotion", "grad_sq", "strain_sq", "spin_sq", "tilt_sq", "E_trace"):
        assert np.nanmax(np.abs(o[key][ok])) == 0.0, key
    np.testing.assert_allclose(o["tilt_max"][ok], 0.0, atol=1e-7)
    np.testing.assert_allclose(o["explained"][ok], 1.0, atol=1e-12)
    np.testing.assert_allclose(o["heldout"][ok], 1.0, atol=1e-10)
    mb = o["mass_balance"][ok]
    assert np.isfinite(mb).mean() > 0.9           # NaN only where the density estimator withholds a value
    np.testing.assert_allclose(mb[np.isfinite(mb)], 0.0, atol=1e-10)
    # zero displacement: ratios with |v| in the denominator are undefined, not 0
    assert np.isnan(o["tan_fraction"]).all() and np.isnan(o["heldout_v"]).all()
    assert np.isnan(o["rigid"]["rigid_fraction"])
    assert o["no_relative_motion"][ok].all()


def test_fixed_operator_continuum(fr, fib16):
    op = graph.build_operator(fib16, k=900, tau=0.005)
    e = operator_errors(op, fr, KAPPA, M)
    print("fixed", e)
    assert passes(e), e
    assert e["dim_ok"] > 0.99
    np.testing.assert_allclose(e["drift_coef"], 1.0, atol=0.05)


def test_wrong_bandwidth_convention_is_caught(fr, fib16):
    """exp(-d^2 / 2 tau) halves the kernel variance: Gamma and G come out at 1/2 and the checks fail."""
    op = graph.build_operator(fib16, k=900, tau=0.005, kernel_denominator=2.0)
    e = operator_errors(op, fr, KAPPA, M)
    print("wrong convention", e)
    assert not passes(e)
    assert e["gamma"] > 0.4 and e["G"] > 0.4


@pytest.mark.parametrize("alpha,coef", [(0.0, 2.0), (1.0, 0.0)])
def test_wrong_drift_caught_only_by_L(fr, fib16, alpha, coef):
    """alpha = 0 doubles the drift, alpha = 1 removes it: Gamma on linear functions cannot see this (it passes),
    L on a coordinate along the density gradient does."""
    op = graph.build_operator(fib16, k=900, tau=0.005, alpha=alpha)
    e = operator_errors(op, fr, KAPPA, M)
    print("alpha", alpha, e)
    assert e["gamma"] < TOL["gamma"]
    assert e["L"] > 3 * TOL["L"] and e["drift"] > 3 * TOL["drift"]
    np.testing.assert_allclose(e["drift_coef"], coef, atol=0.1)


def test_adaptive_prescribed_sigma(fr, fib16):
    """The exact construction with a smooth bandwidth not tied to the density: same limit.  A prescribed
    sigma does not follow the density, so where the cloud is sparse (density ratio e^4 between the poles)
    the kernel holds few points; the check is made where n_eff >= 30 (the bulk of the mass)."""
    sig = sigma_field(fib16, fr, 0.14)
    op = graph.build_operator(fib16, k=1000, kind="adaptive", eps=0.25, sigma=sig)
    neff = 1.0 / np.asarray(op.P.multiply(op.P).sum(1)).ravel()
    sel = neff >= 30
    assert sel.mean() > 0.7
    e = operator_errors(op, fr, KAPPA, M, sel=sel)
    print("adaptive prescribed", e, "selected", sel.mean())
    assert passes(e), e
    np.testing.assert_allclose(e["drift_coef"], 1.0, atol=0.05)
    # without the 1 / (sigma_i sigma_j) weight the drift is grad log(rho sigma^2): caught
    op2 = graph.build_operator(fib16, k=1000, kind="adaptive", eps=0.25, sigma=sig, sigma_correction=False)
    e2 = operator_errors(op2, fr, KAPPA, M, sel=sel)
    print("adaptive without sigma correction", e2)
    assert not passes(e2) and e2["drift"] > 3 * TOL["drift"]


def test_adaptive_knn_pilot(fr, fib16):
    op = graph.build_operator(fib16, k=400, kind="adaptive", eps=0.25, k_bw=64)
    e = operator_errors(op, fr, KAPPA, M)
    print("adaptive kNN pilot", e)
    assert passes(e), e
    np.testing.assert_allclose(e["drift_coef"], 1.0, atol=0.05)


@pytest.mark.slow
@pytest.mark.parametrize("kind", ["fixed", "adaptive"])
def test_operator_convergence_sweep(fr, kind):
    """Random (vMF, kappa = 1) sampling, joint sample-size / bandwidth sweep N = 2000 -> 32000.

    Gamma and G are checked pointwise (sampling noise ~ sqrt(2 / n_eff) shrinks as n_eff grows).  L and the
    drift carry pointwise noise ~ |grad f| sqrt(2 / (tau n_eff)), which does not shrink along any sweep that
    fits in memory here (n tau^2 must grow), so for them the decisive check is the bias: the error averaged
    in 20 bins along the density gradient, and the global drift coefficient."""
    kappa = 1.0
    rows = []
    for N in (2000, 8000, 32000):
        X = fx.great_sphere(N, 2, np.random.default_rng(3), kappa=kappa, frame=fr)[0]
        if kind == "fixed":
            tau = 0.02 * (2000 / N) ** 0.5
            k = int(min(1.25 * (N * 0.184 / 4) * np.pi * 36 * tau, 1200))      # contains the kernel at the pole
            op = graph.build_operator(X, k=k, tau=tau)
        else:
            kbw = int(16 * (N / 2000) ** 0.5)
            op = graph.build_operator(X, k=6 * kbw, kind="adaptive", eps=0.25, k_bw=kbw)
        e = operator_errors(op, fr, kappa, M)
        rows.append((N, e))
        print(kind, N, {k: round(float(v), 4) for k, v in e.items()})
    for key in ("gamma", "G", "L_binned", "drift_binned"):
        vals = [r[1][key] for r in rows]
        assert vals[-1] < vals[0], (key, vals)
    last = rows[-1][1]
    # what random sampling reaches at N = 32000 (characterisation; the tight continuum checks are the
    # low-discrepancy fixtures above): Gamma and G ~ 10% pointwise (ratio-estimator bias ~ 1 / n_eff
    # included), the drift coefficient within 3%, binned (bias) errors of L and the drift ~10% and ~30-40%.
    assert last["gamma"] < 0.1 and last["G"] < 0.1, last
    assert abs(last["drift_coef"] - 1.0) < 0.03, last
    assert last["L_binned"] < 0.2 and last["drift_binned"] < 0.5, last
    # finding: pointwise L and drift are not estimable at these sample sizes on random samples (noise
    # ~ sqrt(2 / (tau n_eff)) |grad f| exceeds the signal); recorded so that a change is noticed
    assert last["L"] > 1.0 and last["drift"] > 1.0, last
