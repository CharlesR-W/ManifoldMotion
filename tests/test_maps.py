"""Fixture family 6: a smooth nonlinear one-to-one map (stretches, log J and mass balance against the exact
differential and density), two sheets that merge (densities add; flagged, not an error), and a shuffled
or noisy target (held-out prediction against a permutation distribution).

Map: x -> 2 A x / |A x| on a great 2-sphere (A acts inside its span), differential
(2 / |A x|)(I - yhat yhat^T) A.  Source density rho0 prop. to exp(kappa (x.e)/2) (Fisher volume); the
target density is rho1(Y) = rho0(x) / J(x).
"""
import numpy as np
import pytest

from mm import sphere, step
from mm import fixtures as fx
from helpers import fib_great_sphere

R = sphere.R
M = 2


def nonlinear_case(N, tau, k, kappa=1.0, seed=60, random=False):
    rng = np.random.default_rng(seed)
    fr = fx.random_frame(3, rng)
    if random:
        X = fx.great_sphere(N, 2, rng, kappa=kappa, frame=fr)[0]
    else:
        X = fib_great_sphere(N, kappa, fr)
    A3 = np.array([[1.3, 0.4, 0.0], [-0.2, 0.9, 0.3], [0.1, 0.0, 0.7]])
    A = fr @ A3 @ fr.T + (np.eye(10) - fr @ fr.T)
    Y = fx.projective_map(X, A)
    o = step.measure_step(X, Y, 1.0, operator=dict(kind="fixed", tau=tau, k=k), mask=False, density_k=64)
    T = (fr @ fr.T)[None] - X[:, :, None] * X[:, None, :] / R**2
    D = fx.projective_differential(X, A) @ T
    s_ex = np.linalg.svd(D, compute_uv=False)[:, :M]
    # exact density: vMF on the unit 2-sphere with parameter kappa, w.r.t. Fisher area (radius 2)
    logZ = np.log(4 * np.pi * np.sinh(kappa) / kappa) + 2 * np.log(R)
    logrho0 = kappa * (X @ fr[:, 0]) / R - logZ
    return X, Y, o, s_ex, D, logrho0


def test_nonlinear_map_stretches_and_density():
    X, Y, o, s_ex, D, logrho0 = nonlinear_case(16000, tau=0.002, k=400)
    ok = o["dim"] == M
    st = o["stretches"][ok, :M]
    err = np.median(np.abs(st / s_ex[ok] - 1), 0)
    logJ_ex = np.sum(np.log(s_ex), 1)
    eJ = np.median(np.abs(o["logJ"][ok] - logJ_ex[ok]))
    Ferr = np.median(np.linalg.norm(o["F"][ok] - D[ok], axis=(1, 2)) / np.linalg.norm(D[ok], axis=(1, 2)))
    print("nonlinear map: stretch err", err, "logJ err", eJ, "F rel err", Ferr)
    assert err.max() < 3e-3 and eJ < 5e-3 and Ferr < 5e-3
    # the operator-independent density against the exact one (fixed-k kNN: bias small, noise ~ 1/sqrt(k))
    d0 = o["logrho0"][ok] - logrho0[ok]
    print("density (k = 64): mean err", np.nanmean(d0), "rms", np.sqrt(np.nanmean(d0**2)))
    assert abs(np.nanmean(d0)) < 0.03 and np.sqrt(np.nanmean(d0**2)) < 0.1


def test_nonlinear_map_mass_balance():
    """rho1(Y) J = rho0(x).  The single-sheet check needs the target-density neighbours' sources inside the
    validated fit patch (3 kernel radii), so the bandwidth here is larger (tau = 0.008) and density_k = 32;
    the map compresses areas by up to ~3x, so with density_k = 64 or tau = 0.002 the check withholds."""
    rng = np.random.default_rng(60)
    fr = fx.random_frame(3, rng)
    X = fib_great_sphere(12000, 1.0, fr)
    A3 = np.array([[1.3, 0.4, 0.0], [-0.2, 0.9, 0.3], [0.1, 0.0, 0.7]])
    A = fr @ A3 @ fr.T + (np.eye(10) - fr @ fr.T)
    Y = fx.projective_map(X, A)
    o = step.measure_step(X, Y, 1.0, operator=dict(kind="fixed", tau=0.008, k=400), mask=False, density_k=32)
    ok = o["dim"] == M
    mb = o["mass_balance"][ok]
    print("mass balance: median", np.nanmedian(mb), "rms", np.sqrt(np.nanmean(mb**2)), "single sheet",
          np.mean(o["single_sheet"][ok]), "unassessed mean", np.nanmean(o["unassessed_frac"][ok]))
    print("release", {k: round(float(v.mean()), 3) for k, v in o["release"].items()})
    assert np.mean(o["single_sheet"][ok]) > 0.9
    assert abs(np.nanmedian(mb)) < 0.02 and np.sqrt(np.nanmean(mb**2)) < 0.15
    assert o["release"]["mass_balance"].mean() > 0.5


@pytest.mark.slow
def test_nonlinear_map_sweep_random_sampling():
    """Random sampling, sample-size and bandwidth sweep: stretch and log J errors decrease."""
    rows = []
    for N, tau, k in ((4000, 0.008, 160), (16000, 0.004, 200), (64000, 0.002, 300)):
        X, Y, o, s_ex, D, logrho0 = nonlinear_case(N, tau, k, random=True)
        ok = o["dim"] == M
        st = o["stretches"][ok, :M]
        e = dict(stretch=np.sqrt(np.mean((st / s_ex[ok] - 1) ** 2)),
                 logJ=np.sqrt(np.mean((o["logJ"][ok] - np.sum(np.log(s_ex[ok]), 1)) ** 2)),
                 mass=np.nanmedian(o["mass_balance"][ok]),
                 heldout=np.nanmedian(o["heldout"][ok]), released=o["release"]["logJ"].mean())
        rows.append((N, tau, e))
        print(rows[-1])
    for key, tol in (("stretch", 0.03), ("logJ", 0.04)):
        vals = [r[2][key] for r in rows]
        assert vals[-1] < vals[0] and vals[-1] < tol, (key, vals)


def test_two_sheets_merge():
    """Sheet B (a far cap) is mapped on top of sheet A: target densities add (mass balance log 2) and
    the foreign-source flag catches it; an unmerged control is single-sheet."""
    rng = np.random.default_rng(61)
    Q = fx.random_frame(6, rng)
    fa, fb = Q[:, :3], Q[:, 3:]
    N = 6000
    om_a = fx.unit_sphere(N, 2, rng)
    om_b = fx.unit_sphere(N, 2, rng)
    om_a, om_b = om_a[om_a[:, 0] > 0.0], om_b[om_b[:, 0] > 0.0]          # hemispheres
    XA, XB = R * om_a @ fa.T, R * om_b @ fb.T
    swap = fa @ fb.T + fb @ fa.T + (np.eye(10) - fa @ fa.T - fb @ fb.T)      # B -> A's place
    X0 = np.concatenate([XA, XB])
    X1 = np.concatenate([XA, XB @ swap.T])
    # tau = 0.008 (n_eff ~ 24): the 32 density neighbours lie within the validated source patch
    o = step.measure_step(X0, X1, 1.0, operator=dict(kind="fixed", tau=0.008, k=128), mask=False, density_k=32)
    inner = np.concatenate([om_a[:, 0], om_b[:, 0]]) > 0.4          # away from the hemisphere rims
    ok = inner & (o["dim"] == M)
    print("merge: single_sheet frac", o["single_sheet"][ok].mean(), "foreign median", np.nanmedian(o["foreign_frac"][ok]),
          "mass balance median", np.nanmedian(o["mass_balance"][ok]))
    assert o["single_sheet"][ok].mean() < 0.05
    np.testing.assert_allclose(np.nanmedian(o["mass_balance"][ok]), np.log(2), atol=0.08)
    assert not o["release"]["mass_balance"][ok].any()
    # control: no merge
    o2 = step.measure_step(XA, XA @ fx.rotation(10, (0, 1), 0.3, fa).T, 1.0,
                           operator=dict(kind="fixed", tau=0.008, k=128), mask=False, density_k=32)
    ok2 = (om_a[:, 0] > 0.4) & (o2["dim"] == M)
    print("control: single_sheet frac", o2["single_sheet"][ok2].mean(), "mass median", np.nanmedian(o2["mass_balance"][ok2]))
    assert o2["single_sheet"][ok2].mean() > 0.95
    assert abs(np.nanmedian(o2["mass_balance"][ok2])) < 0.05


def test_parallel_sheets_merged_under_strong_compression():
    """Two parallel latitude patches (normal separation 2 delta = 0.5, about 5 source radii) mapped onto the same target with a 10x
    compression along the sheet: the tangential ambiguity r_Y / sigma_min exceeds the separation, so only
    the normal-separation bound can see the merge; mass balance must not be released there."""
    rng = np.random.default_rng(63)
    fr = fx.random_frame(4, rng)
    N = 6000
    om_a, om_b = fx.unit_sphere(N, 2, rng), fx.unit_sphere(N, 2, rng)
    om_a, om_b = om_a[om_a[:, 0] > 0.5], om_b[om_b[:, 0] > 0.5]           # caps around e1 of the omega-sphere
    theta, delta = 0.9, 0.25
    XA = fx.latitude_points(om_a, theta, fr)
    XB = fx.latitude_points(om_b, theta + delta, fr)
    C = np.diag([1.0, 1.0, 0.1])                                              # compress one tangent direction

    def squash(om):
        z = om @ C.T
        return z / np.linalg.norm(z, axis=1, keepdims=True)

    X0 = np.concatenate([XA, XB])
    X1 = np.concatenate([fx.latitude_points(squash(om_a), theta, fr), fx.latitude_points(squash(om_b), theta, fr)])
    o = step.measure_step(X0, X1, 1.0, operator=dict(kind="fixed", tau=0.003, k=128), mask=False, density_k=32)
    inner = np.concatenate([om_a[:, 0], om_b[:, 0]]) > 0.7
    ok = inner & (o["dim"] == M) & np.isfinite(o["sigma_min"])
    amb = 3 * np.nanmedian(o["radius"][ok]) / np.nanmedian(o["sigma_min"][ok])
    print("parallel sheets: separation", 2 * delta, "median tangential ambiguity ~", amb, "source radius",
          np.median(o["radius"][ok]), "single-sheet frac", np.mean(o["single_sheet"][ok]),
          "mass balance median", np.nanmedian(o["mass_balance"][ok]))
    assert np.median(o["sigma_min"][ok]) < 0.2
    assert np.mean(o["single_sheet"][ok]) < 0.05
    assert not o["release"]["mass_balance"][ok].any()


def test_in_plane_merge_under_anisotropic_compression():
    """Two disjoint patches of one great 2-sphere, side by side along s, both carried onto the same target
    patch (B translated by 0.6 in s) while t is compressed 100x: every source offset is tangent (the normal
    bound sees nothing) and r_Y / sigma_min exceeds the separation (the tangential bound accepts the other
    patch); only the comparison of target offsets with F u sees the merge."""
    rng = np.random.default_rng(64)
    fr = fx.random_frame(3, rng)
    Np = 2000
    sA = rng.uniform(-0.25, 0.25, Np)
    sB = rng.uniform(0.35, 0.85, Np)
    tA, tB = rng.uniform(-0.25, 0.25, Np), rng.uniform(-0.25, 0.25, Np)

    def pts(s, t):
        z = np.column_stack([np.ones_like(s), s, t])
        return R * (z / np.linalg.norm(z, axis=1, keepdims=True)) @ fr.T

    X0 = np.concatenate([pts(sA, tA), pts(sB, tB)])
    X1 = np.concatenate([pts(sA, 0.01 * tA), pts(sB - 0.6, 0.01 * tB)])
    o = step.measure_step(X0, X1, 1.0, operator=dict(kind="fixed", tau=0.0008, k=200), mask=False, density_k=32)
    s_all, t_all = np.concatenate([sA, sB - 0.6]), np.concatenate([tA, tB])
    inner = (np.abs(s_all) < 0.18) & (np.abs(t_all) < 0.18)
    ok = inner & (o["dim"] == M) & np.isfinite(o["sigma_min"])
    amb = 3 * np.nanmedian(o["tgt_radius"][ok]) / np.nanmedian(o["sigma_min"][ok])
    print("in-plane merge: sigma_min median", np.median(o["sigma_min"][ok]), "tangential ambiguity ~", amb,
          "single-sheet frac", np.mean(o["single_sheet"][ok]), "mass balance median", np.nanmedian(o["mass_balance"][ok]))
    assert np.median(o["sigma_min"][ok]) < 0.02
    assert np.mean(o["single_sheet"][ok]) < 0.05
    assert not o["release"]["mass_balance"][ok].any()
    # control: patch A alone under the same map stays single-sheet
    o2 = step.measure_step(pts(sA, tA), pts(sA, 0.01 * tA), 1.0, operator=dict(kind="fixed", tau=0.0008, k=200),
                           mask=False, density_k=32)
    ok2 = (np.abs(sA) < 0.18) & (np.abs(tA) < 0.18) & (o2["dim"] == M) & np.isfinite(o2["sigma_min"])
    # under 100x compression the 32 target-density neighbours come from far beyond the validated source
    # patch: the control is *unassessed* (withheld), not merged: no neighbour inside the patch is inconsistent
    print("in-plane control: single-sheet frac", np.mean(o2["single_sheet"][ok2]), "unassessed median",
          np.nanmedian(o2["unassessed_frac"][ok2]), "inconsistent max", np.nanmax(o2["inconsistent_frac"][ok2]))
    assert np.nanmedian(o2["unassessed_frac"][ok2]) > 0.5
    assert np.nanmean(o2["inconsistent_frac"][ok2]) < 0.01
    assert np.nanmean(o["inconsistent_frac"][ok]) > 0.1 or np.nanmean(o["unassessed_frac"][ok]) > 0.5


def test_merge_along_compressed_direction():
    """As above but the patches are separated along the compressed direction t: after the 100x compression
    the wrong-sheet neighbours sit within r_Y of the anchor's image and are mispredicted only by
    sigma_min D ~ 1.7 r_Y, so neither the density radius nor the tangential ambiguity can be the yardstick;
    the fit's own error is."""
    rng = np.random.default_rng(65)
    fr = fx.random_frame(3, rng)
    Np = 2000
    sA, sB = rng.uniform(-0.25, 0.25, Np), rng.uniform(-0.25, 0.25, Np)
    tA, tB = rng.uniform(-0.25, 0.25, Np), rng.uniform(0.35, 0.85, Np)

    def pts(s, t):
        z = np.column_stack([np.ones_like(s), s, t])
        return R * (z / np.linalg.norm(z, axis=1, keepdims=True)) @ fr.T

    X0 = np.concatenate([pts(sA, tA), pts(sB, tB)])
    X1 = np.concatenate([pts(sA, 0.01 * tA), pts(sB, 0.01 * (tB - 0.6))])
    o = step.measure_step(X0, X1, 1.0, operator=dict(kind="fixed", tau=0.0008, k=200), mask=False, density_k=32)
    s_all, t_all = np.concatenate([sA, sB]), np.concatenate([tA, tB - 0.6])
    inner = (np.abs(s_all) < 0.18) & (np.abs(t_all) < 0.18)
    ok = inner & (o["dim"] == M) & np.isfinite(o["sigma_min"])
    print("compressed-direction merge: sigma_min * D / r_Y ~",
          np.median(o["sigma_min"][ok]) * 1.2 / np.nanmedian(o["tgt_radius"][ok]),
          "single-sheet frac", np.mean(o["single_sheet"][ok]))
    assert np.mean(o["single_sheet"][ok]) < 0.05
    assert not o["release"]["mass_balance"][ok].any()
    o2 = step.measure_step(pts(sA, tA), pts(sA, 0.01 * tA), 1.0, operator=dict(kind="fixed", tau=0.0008, k=200),
                           mask=False, density_k=32)
    ok2 = (np.abs(sA) < 0.18) & (np.abs(tA) < 0.18) & (o2["dim"] == M) & np.isfinite(o2["sigma_min"])
    print("compressed control: single-sheet frac", np.mean(o2["single_sheet"][ok2]), "unassessed median",
          np.nanmedian(o2["unassessed_frac"][ok2]), "inconsistent mean", np.nanmean(o2["inconsistent_frac"][ok2]))
    assert np.nanmean(o2["inconsistent_frac"][ok2]) < 0.01
    assert not o2["release"]["mass_balance"][ok2].any() or np.nanmedian(o2["unassessed_frac"][ok2]) < 0.05


def test_shuffled_and_noisy_target():
    X, Y, o, s_ex, D, logrho0 = nonlinear_case(6000, tau=0.005, k=128)
    ok = o["dim"] == M
    true = o["heldout"]
    rng = np.random.default_rng(62)
    perms = []
    for _ in range(3):
        Yp = Y[rng.permutation(len(Y))]
        op = step.measure_step(X, Yp, 1.0, operator=dict(kind="fixed", tau=0.005, k=128), mask=False)
        perms.append(op)
    ph = np.stack([p["heldout"] for p in perms])
    pe = np.stack([p["explained"] for p in perms])
    print("held-out: true median", np.nanmedian(true[ok]), "perm median", np.nanmedian(ph[:, ok]),
          "perm 99%", np.nanpercentile(ph[:, ok], 99), "perm in-sample explained median", np.nanmedian(pe[:, ok]))
    assert np.nanmedian(true[ok]) > 0.99
    assert np.nanmedian(ph[:, ok]) < 0.05
    # pointwise: the true score beats every permutation at almost every point
    assert np.mean(true[ok] > np.nanmax(ph[:, ok], axis=0)) > 0.99
    # in-sample fraction is positive even for a shuffled target: why a held-out score is needed
    assert np.nanmedian(pe[:, ok]) > 0.02
    assert all(p["release"]["strain_finite"].mean() < 0.01 for p in perms)
    # noise of increasing size relative to the neighbourhood radius: held-out score decreases
    r = np.median(o["radius"])
    med = []
    for lev in (0.0, 1.0, 3.0):
        Yn = sphere.exp_map(Y, sphere.tangent_project(Y, lev * r / 3.0 * rng.standard_normal(Y.shape)))
        on = step.measure_step(X, Yn, 1.0, operator=dict(kind="fixed", tau=0.005, k=128), mask=False)
        med.append(np.nanmedian(on["heldout"][ok]))
    print("noise levels / radius", (0.0, 1.0, 3.0), "held-out medians", med)
    assert all(a > b for a, b in zip(med[:-1], med[1:]))
    assert med[-1] < 0.2
