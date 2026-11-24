"""Sphere geometry: stable distance, log / exp maps, exact parallel transport, conversions from log
probabilities (including float32 input near simplex vertices).  geomstats is an independent cross-check."""
import numpy as np
import pytest

from mm import sphere

R = sphere.R


def rand_sphere(rng, n, d=10):
    z = rng.standard_normal((n, d))
    return R * z / np.linalg.norm(z, axis=1, keepdims=True)


def rand_tangent(rng, x, scale=1.0):
    return sphere.tangent_project(x, scale * rng.standard_normal(x.shape))


def test_distance_stable_for_close_and_far_points():
    rng = np.random.default_rng(0)
    x = rand_sphere(rng, 50)
    for eps in (1e-3, 1e-7, 1e-11):
        xi = rand_tangent(rng, x)
        xi *= eps / np.linalg.norm(xi, axis=1, keepdims=True)
        y = sphere.exp_map(x, xi)
        d = sphere.fisher_distance(x, y)
        # y itself is rounded to ~4e-16 per coordinate: that is the attainable absolute accuracy
        np.testing.assert_allclose(d, eps, rtol=1e-9, atol=2e-15)
    # nearly antipodal (outside the orthant, but the formula must hold on the whole sphere)
    xi = rand_tangent(rng, x)
    xi *= (np.pi * R - 1e-6) / np.linalg.norm(xi, axis=1, keepdims=True)
    y = sphere.exp_map(x, xi)
    np.testing.assert_allclose(sphere.fisher_distance(x, y), np.pi * R - 1e-6, rtol=1e-9)
    # logp form 4 atan2(|s - s'|, |s + s'|)
    p = rng.dirichlet(np.ones(10), 20)
    q = rng.dirichlet(np.ones(10), 20)
    s, t = np.sqrt(p), np.sqrt(q)
    ref = 4 * np.arctan2(np.linalg.norm(s - t, axis=1), np.linalg.norm(s + t, axis=1))
    np.testing.assert_allclose(sphere.fisher_distance_logp(np.log(p), np.log(q)), ref, rtol=1e-12)
    np.testing.assert_allclose(ref, 2 * np.arccos(np.clip(np.sum(s * t, 1), -1, 1)), rtol=1e-8)


def test_log_exp_inverse_and_norm():
    rng = np.random.default_rng(1)
    x = rand_sphere(rng, 200)
    for scale in (1e-9, 1e-3, 0.5, 2.5, 5.5):
        xi = rand_tangent(rng, x)
        xi *= scale / np.linalg.norm(xi, axis=1, keepdims=True)
        y = sphere.exp_map(x, xi)
        np.testing.assert_allclose(np.linalg.norm(y, axis=1), R, rtol=1e-14)
        lg = sphere.log_map(x, y)
        np.testing.assert_allclose(lg, xi, atol=1e-12 * max(scale, 1), rtol=1e-9)
        np.testing.assert_allclose(np.linalg.norm(lg, axis=1), sphere.fisher_distance(x, y), rtol=1e-12)
        np.testing.assert_allclose(np.sum(lg * x, 1), 0.0, atol=1e-14 * max(scale, 1))


@pytest.fixture(scope="module")
def geomstats_sphere():
    if not hasattr(np, "trapz"):           # geomstats 2.8 imports numpy.trapz (removed in numpy 2.4)
        np.trapz = np.trapezoid
    hs = pytest.importorskip("geomstats.geometry.hypersphere")
    return hs.Hypersphere(dim=9)


def test_log_map_against_geomstats(geomstats_sphere):
    rng = np.random.default_rng(2)
    x = rand_sphere(rng, 40)
    y = rand_sphere(rng, 40)
    # radius-2 sphere = 2 x unit sphere: log scales by 2
    for i in range(40):
        ref = 2.0 * np.asarray(geomstats_sphere.metric.log(y[i] / R, x[i] / R))
        np.testing.assert_allclose(sphere.log_map(x[i], y[i]), ref, atol=1e-10)


def test_parallel_transport_against_geomstats(geomstats_sphere):
    rng = np.random.default_rng(3)
    x = rand_sphere(rng, 40)
    y = rand_sphere(rng, 40)
    xi = rand_tangent(rng, x)
    for i in range(40):
        ref = np.asarray(geomstats_sphere.metric.parallel_transport(xi[i], x[i] / R, end_point=y[i] / R))
        np.testing.assert_allclose(sphere.parallel_transport(x[i], y[i], xi[i]), ref, atol=1e-10)


def test_parallel_transport_against_ode():
    """Integrate the Levi-Civita transport ODE d xi / dt = -(xi . g') g / R^2 along the geodesic g."""
    rng = np.random.default_rng(4)
    for _ in range(5):
        x = rand_sphere(rng, 1)[0]
        y = rand_sphere(rng, 1)[0]
        xi = rand_tangent(rng, x)
        v = sphere.log_map(x, y)
        nst = 4000
        h = 1.0 / nst

        def geo(t):
            return sphere.exp_map(x, t * v)

        def dgeo(t):
            a = np.linalg.norm(v) / R
            e = v / np.linalg.norm(v)
            return -np.sin(a * t) * a * x + np.cos(a * t) * a * R * e

        def f(t, z):
            return -np.dot(z, dgeo(t)) * geo(t) / R**2

        z = xi.copy()
        for k in range(nst):
            t = k * h
            k1 = f(t, z)
            k2 = f(t + h / 2, z + h / 2 * k1)
            k3 = f(t + h / 2, z + h / 2 * k2)
            k4 = f(t + h, z + h * k3)
            z = z + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        np.testing.assert_allclose(sphere.parallel_transport(x, y, xi), z, atol=1e-9)


def test_parallel_transport_properties():
    rng = np.random.default_rng(5)
    x = rand_sphere(rng, 100)
    y = sphere.exp_map(x, rand_tangent(rng, x, 0.7))
    xi = rand_tangent(rng, x)
    zt = sphere.parallel_transport(x, y, xi)
    np.testing.assert_allclose(np.sum(zt * y, 1), 0, atol=1e-13)                       # tangent at y
    np.testing.assert_allclose(np.linalg.norm(zt, axis=1), np.linalg.norm(xi, axis=1), rtol=1e-13)
    np.testing.assert_allclose(sphere.parallel_transport(y, x, zt), xi, atol=1e-13)      # inverse
    # the geodesic's own velocity: PT(log_x y) = -log_y x
    np.testing.assert_allclose(sphere.parallel_transport(x, y, sphere.log_map(x, y)),
                               -sphere.log_map(y, x), atol=1e-12)
    A = np.stack([rand_tangent(rng, x) for _ in range(3)], axis=-1)
    PA = sphere.parallel_transport_matrix(x, y, A)
    for k in range(3):
        np.testing.assert_allclose(PA[..., k], sphere.parallel_transport(x, y, A[..., k]), atol=1e-14)


def test_tangent_basis_orthonormal():
    rng = np.random.default_rng(6)
    x = rand_sphere(rng, 50)
    x[0] = R * np.eye(10)[9]
    x[1] = -R * np.eye(10)[9]
    x[2] = R * np.eye(10)[0]
    E = sphere.tangent_basis(x)
    np.testing.assert_allclose(np.swapaxes(E, 1, 2) @ E, np.broadcast_to(np.eye(9), (50, 9, 9)), atol=1e-14)
    np.testing.assert_allclose(np.einsum("ni,nik->nk", x, E), 0, atol=1e-14)


def test_face_distance_is_distance_to_vertex_and_edge():
    rng = np.random.default_rng(7)
    p = rng.dirichlet(np.full(10, 0.3), 100)
    lp = np.log(p)
    top = np.argmax(p, 1)
    V = R * np.eye(10)[top]
    np.testing.assert_allclose(sphere.face_distance(lp, 1), sphere.fisher_distance(sphere.sphere_from_logp(lp), V),
                               rtol=1e-10)
    # nearest point of the edge spanned by the two largest classes: renormalised restriction
    o = np.argsort(-p, 1)[:, :2]
    q = np.zeros_like(p)
    np.put_along_axis(q, o, np.take_along_axis(p, o, 1), 1)
    q /= q.sum(1, keepdims=True)
    np.testing.assert_allclose(sphere.face_distance(lp, 2),
                               sphere.fisher_distance(sphere.sphere_from_logp(lp), R * np.sqrt(q)), rtol=1e-10)


def _logp32_from_logits(z):
    """float32 log-softmax as a network would produce it (z - max - log sum exp(z - max), in float32)."""
    z = z.astype(np.float32)
    zm = z - z.max(1, keepdims=True)
    return zm - np.log(np.sum(np.exp(zm), 1, keepdims=True, dtype=np.float32))


def test_float32_input_near_vertex_keeps_relative_precision():
    """Points within 1e-9 .. 1e-2 of a vertex, given as float32 log-probabilities: neighbour distances
    computed from the float32 input agree with the float64 truth to a small relative error, and the
    per-point input resolution bounds the coordinate error."""
    rng = np.random.default_rng(8)
    n = 400
    depth = rng.uniform(4, 40, n)                       # z_top - z_other: 1 - p_top from ~1e-17 to ~1e-1
    z = rng.normal(0, 1.0, (n, 10)) - depth[:, None]
    z[:, 3] = 25.0
    z = z.astype(np.float32).astype(np.float64)         # the network's float32 logits are the truth
    zm = z - z.max(1, keepdims=True)
    p = np.exp(zm) / np.exp(zm).sum(1, keepdims=True)
    X_true = R * np.sqrt(p)
    lp32 = _logp32_from_logits(z)
    X = sphere.sphere_from_logp(lp32)
    res = sphere.input_resolution(lp32)
    err = np.linalg.norm(X - X_true, axis=1)
    assert np.all(err <= res), (err / res).max()
    # pairs of points: relative distance error stays at the input's relative precision (~1e-6), also for
    # pairs at Fisher distance ~1e-9 from each other
    i = np.arange(n - 1)
    d_true = sphere.fisher_distance(X_true[i], X_true[i + 1])
    d = sphere.fisher_distance(X[i], X[i + 1])
    rel = np.abs(d - d_true) / d_true
    print("float32 input: max relative distance error", rel.max(), "min distance", d_true.min())
    assert np.max(rel) < 1e-5, np.max(rel)
    # nearly coincident near-vertex pairs: perturb the small logits slightly
    z2 = z.copy()
    z2[:, :3] += 1e-3
    z2 = z2.astype(np.float32).astype(np.float64)
    p2 = np.exp(z2 - z2.max(1, keepdims=True))
    X2_true = R * np.sqrt(p2 / p2.sum(1, keepdims=True))
    X2 = sphere.sphere_from_logp(_logp32_from_logits(z2))
    deep = depth > 25
    dd_true = sphere.fisher_distance(X_true[deep], X2_true[deep])
    dd = sphere.fisher_distance(X[deep], X2[deep])
    print("near-vertex pair distances", dd_true.min(), dd_true.max(), "max rel err", np.max(np.abs(dd / dd_true - 1)))
    assert dd_true.max() < 1e-6
    assert np.max(np.abs(dd / dd_true - 1)) < 1e-2
    # near-vertex points get a near-vertex resolution, generic ones a float32-sized one
    vd = sphere.face_distance(lp32, 1)
    assert np.all(res[vd < 1e-6] < 1e-10)


def test_logp_roundtrip_and_simplex():
    rng = np.random.default_rng(9)
    p = rng.dirichlet(np.ones(10), 50)
    X = sphere.sphere_from_logp(np.log(p))
    np.testing.assert_allclose(np.linalg.norm(X, axis=1), R, rtol=1e-15)
    np.testing.assert_allclose(np.exp(sphere.logp_from_sphere(X)), p, rtol=1e-12)
    lp = np.log(p)
    lp[0, 2] = -np.inf
    X = sphere.sphere_from_logp(lp)
    assert X[0, 2] == 0.0 and np.isfinite(X).all()
