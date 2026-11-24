"""Fixture family 3: rigid motions.

* a Killing field V = Omega x supplied directly: zero strain, K -> Q Omega T, rigid fraction exactly 1;
* finite rotations at several angles: unit stretches exactly (y_j = R u_j), log J = 0, co-rotating motion
  zero; the log-displacement strain is O(theta^2) per step with a known leading term
  -(theta^2 / 2)(T P T - (x.P x / 4) T), P the projector on the rotation plane, so it tends to zero only
  as a rate;
* a full-rank (9-dimensional) patch where every normal quantity vanishes exactly;
* one endpoint pair with two different time intervals: finite fields identical, rates scale 1 / dt,
  squared rates 1 / dt^2, release flags identical.
"""
import numpy as np
import pytest
from scipy.linalg import expm

from mm import sphere, step
from mm import fixtures as fx
from helpers import fib_great_sphere

R = sphere.R
OP = dict(kind="fixed", tau=0.005, k=200)


@pytest.fixture(scope="module")
def cloud():
    fr = fx.random_frame(3, np.random.default_rng(30))
    return fib_great_sphere(12000, 1.0, fr), fr


@pytest.fixture(scope="module")
def small_cloud():
    """For the exact-algebra checks (any cloud will do)."""
    fr = fx.random_frame(3, np.random.default_rng(38))
    return fib_great_sphere(4000, 1.0, fr), fr


OP_SMALL = dict(kind="fixed", tau=0.015, k=128)


def test_killing_field_supplied_directly(cloud):
    X, fr = cloud
    rng = np.random.default_rng(31)
    Om = fx.random_skew(10, rng, 0.3)
    dt = 0.5
    V = X @ Om.T
    o = step.measure_step(X, None, dt, v=V * dt, operator=OP, mask=False)
    ok = o["dim"] == 2
    assert ok.mean() > 0.99
    # rigid fit: exact
    assert abs(o["rigid"]["rigid_fraction"] - 1.0) < 1e-10
    np.testing.assert_allclose(o["ke_rigid"], o["ke"], rtol=1e-9)
    # continuum: K -> Q Omega T, strain -> 0
    T = (fr @ fr.T)[None] - X[:, :, None] * X[:, None, :] / R**2
    Q = sphere.tangent_projector(X)
    Kex = Q @ Om @ T
    K = o["K_d"] / dt
    rel = np.linalg.norm((K - Kex)[ok], axis=(1, 2)) / np.linalg.norm(Kex[ok], axis=(1, 2))
    strain_rel = np.sqrt(o["strain_sq"][ok] / o["grad_sq"][ok])
    print("Killing: median rel |K - Q Om T|", np.median(rel), "median |E|/|KT|", np.median(strain_rel))
    assert np.median(rel) < 0.02
    assert np.median(strain_rel) < 0.02
    # spin and tilt carry the motion
    spin_ex = np.sum((T @ Om @ T) ** 2, axis=(1, 2))
    np.testing.assert_allclose(np.median(o["spin_sq"][ok] / spin_ex[ok]), 1.0, atol=0.03)


@pytest.mark.parametrize("theta", [0.02, 0.4, 1.0])
def test_finite_rotation_unit_stretches(small_cloud, theta):
    X, fr = small_cloud
    rng = np.random.default_rng(32)
    A = fx.random_skew(10, rng)
    A /= np.linalg.norm(A, 2)
    Rm = expm(theta * A)
    Y = X @ Rm.T
    o = step.measure_step(X, Y, 1.0, operator=OP_SMALL, mask=False)
    ok = o["dim"] == 2
    # exact algebra: y_j = log_{R x_i} R x_j = R u_j, so F = R T and every stretch is 1
    np.testing.assert_allclose(o["stretches"][ok, :2], 1.0, atol=1e-9)
    np.testing.assert_allclose(o["logJ"][ok], 0.0, atol=1e-9)
    np.testing.assert_allclose(o["hencky"][ok], 0.0, atol=1e-9)
    T = (fr @ fr.T)[None] - X[:, :, None] * X[:, None, :] / R**2
    np.testing.assert_allclose(o["F"][ok], (Rm[None] @ T)[ok], atol=1e-9)
    # the finite rigid fit finds the motion of the cloud: nothing moves in the co-rotating frame
    np.testing.assert_allclose(o["cr_travel"], 0.0, atol=1e-10)
    assert o["rigid"]["procrustes_fraction"] > 1 - 1e-12
    np.testing.assert_allclose(o["cr_relmotion"][ok], 0.0, atol=1e-18)
    np.testing.assert_allclose(o["cr_tilt_max"][ok], 0.0, atol=1e-6)


def test_finite_tilt_against_exact_planes(small_cloud):
    """Rotation moving the great sphere out of its span: target tangent planes are R T_x M exactly; the
    finite tilt is the set of principal angles between PT_{x -> Rx} T_x M and R T_x M.  The estimated
    tangent bases are exact here (offsets lie in the plane), so the angles agree to rounding."""
    X, fr = small_cloud
    a = fr[:, 0]
    b = fx.random_frame(4, np.random.default_rng(37))[:, 3]
    b -= fr @ (fr.T @ b)
    b /= np.linalg.norm(b)
    A = np.outer(b, a) - np.outer(a, b)
    Rm = expm(0.5 * A)
    Y = X @ Rm.T
    o = step.measure_step(X, Y, 1.0, operator=OP_SMALL, mask=False)
    ok = (o["dim"] == 2) & (o["tgt_dim"] == 2)
    # exact tangent bases of the great sphere at x, and their images
    E = sphere.tangent_basis(X)
    T = (fr @ fr.T)[None] - X[:, :, None] * X[:, None, :] / R**2
    U = np.linalg.svd(T @ E, full_matrices=False)[0][:, :, :2]
    ang = step.principal_angles(sphere.parallel_transport_matrix(X, Y, U), Rm[None] @ U)
    np.testing.assert_allclose(o["tilt_angles"][ok, :2], ang[ok], atol=1e-6)
    assert np.median(o["tilt_max"][ok]) > 0.01                       # the fixture does tilt
    np.testing.assert_allclose(o["cr_tilt_max"][ok], 0.0, atol=1e-6)


def test_log_displacement_strain_is_second_order():
    """v = log_x(R x) for R = exp(theta A), A a rotation in one coordinate plane (projector P):
    v = theta A x - (theta^2 / 2) Q P x + O(theta^3), so E dt = -(theta^2 / 2)(T P T - (x.P x / 4) T)
    + O(theta^3) (the Killing part contributes no strain)."""
    fr = fx.random_frame(3, np.random.default_rng(33))
    X = fib_great_sphere(8000, 0.0, fr)
    a = fr[:, 0] + fr[:, 1]
    a /= np.linalg.norm(a)
    b = fx.random_frame(4, np.random.default_rng(34))[:, 3]
    b -= a * (a @ b)
    b /= np.linalg.norm(b)
    A = np.outer(b, a) - np.outer(a, b)
    P = np.outer(a, a) + np.outer(b, b)
    T = (fr @ fr.T)[None] - X[:, :, None] * X[:, None, :] / R**2
    pred_unit = -0.5 * (T @ P @ T - (np.einsum("ni,ij,nj->n", X, P, X) / R**2)[:, None, None] * T)
    rows = []
    for theta in (0.4, 0.2, 0.1):
        Y = X @ expm(theta * A).T
        o = step.measure_step(X, Y, 1.0, operator=dict(kind="fixed", tau=0.0075, k=200), mask=False)
        ok = o["dim"] == 2
        E = o["E"][ok]
        pred = theta**2 * pred_unit[ok]
        err = np.linalg.norm(E - pred, axis=(1, 2)) / np.linalg.norm(pred, axis=(1, 2)).mean()
        rows.append((theta, np.sqrt(np.mean(np.sum(E**2, (1, 2)))) / theta, np.median(err)))
    print("theta, |E dt| / theta, rel err vs theta^2 prediction:", rows)
    ratios = [r[1] for r in rows]
    # |E dt| / theta -> 0 linearly in theta: halves when theta halves
    for r1, r2 in zip(ratios[:-1], ratios[1:]):
        assert 0.4 < r2 / r1 < 0.6
    assert rows[-1][2] < 0.05


def test_full_rank_patch_normal_quantities_vanish():
    rng = np.random.default_rng(35)
    z = rng.standard_normal((5000, 10))
    X = R * z / np.linalg.norm(z, axis=1, keepdims=True)
    A = np.eye(10) + 0.2 * rng.standard_normal((10, 10))
    Y = fx.projective_map(X, A)
    o = step.measure_step(X, Y, 1.0, operator=dict(kind="fixed", tau=0.5, k=200), mask=False)
    full = (o["dim"] == 9) & (o["tgt_dim"] == 9)
    assert full.mean() > 0.5, np.bincount(o["dim"])
    assert o["full_rank"][full].all()
    scale = np.median(o["ke"])
    assert np.abs(o["ke_norm"][full]).max() < 1e-20 * scale + 1e-30
    np.testing.assert_allclose(o["ke_tan"][full], o["ke"][full], rtol=1e-10)
    assert np.abs(o["tilt_sq"][full]).max() < 1e-20 * np.median(o["grad_sq"][full])
    np.testing.assert_allclose(o["tilt_max"][full], 0.0, atol=1e-6)
    # released as full-rank: the normal split and tilt are undefined (not "no extrinsic motion")
    assert not o["release"]["normal"][full].any()
    assert not o["release"]["tilt"][full].any()
    om = step.apply_release(o)
    assert np.isnan(om["ke_norm"][full]).all() and np.isnan(om["tilt_max"][full]).all()


def test_two_time_intervals(small_cloud):
    X, fr = small_cloud
    rng = np.random.default_rng(36)
    A = np.eye(10) + 0.3 * fr @ rng.standard_normal((3, 3)) @ fr.T
    Y = fx.projective_map(X, A)
    o1 = step.measure_step(X, Y, 1.0, operator=OP_SMALL, mask=False)
    dt = 3.7
    o2 = step.measure_step(X, Y, dt, operator=OP_SMALL, mask=False)
    finite = ["travel", "step_angle", "stretches", "log_stretches", "logJ", "hencky", "fin_vol", "fin_shear",
              "tilt_angles", "tilt_max", "explained", "heldout", "heldout_v", "trB", "F", "K_d",
              "rate_finite_gap", "dim", "radius", "mass_balance"]
    rate1 = ["speed", "E", "E_trace", "E_shear", "E_top", "E_flow", "E_normal", "E_normal_skew"]
    rate2 = ["ke", "ke_tan", "ke_norm", "ke_rigid", "relmotion", "relmotion_fit", "relmotion_res", "grad_sq",
             "strain_sq", "spin_sq", "tilt_sq", "rate_vol", "rate_shear", "rate_total"]

    def cmp(a, b):
        np.testing.assert_allclose(np.asarray(a, float), np.asarray(b, float), rtol=1e-10, atol=1e-14,
                                   equal_nan=True)

    for k in finite:
        cmp(o2[k], o1[k])
    for k in rate1:
        cmp(o2[k] * dt, o1[k])
    for k in rate2:
        cmp(o2[k] * dt**2, o1[k])
    for r in o1["release"]:
        assert np.array_equal(o1["release"][r], o2["release"][r]), r
    assert abs(o1["rigid"]["rigid_fraction"] - o2["rigid"]["rigid_fraction"]) < 1e-12
