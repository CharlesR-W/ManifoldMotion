"""Fixture family 7: asymmetric patch against orthant faces, controlled thin dimensions, and a cluster
shrinking towards a simplex vertex (drift and rank bias, coverage, rank loss reported as such).
"""
import numpy as np
import pytest

from mm import sphere, step, frame, graph
from mm import fixtures as fx

R = sphere.R
M = 2


def octant_patch(N, rng):
    """Uniform sample of the great 2-sphere spanned by b1 = (e0..e4)/sqrt5, b2 = (e5..e8)/2, e9, restricted to
    the positive orthant {x.b1 >= 0, x.b2 >= 0, x.e9 >= 0}: a patch bounded by orthant faces.  Returns X and
    the Fisher distance of each point to the patch boundary (R arcsin(min face coordinate / R))."""
    b1 = np.r_[np.ones(5), np.zeros(5)] / np.sqrt(5)
    b2 = np.r_[np.zeros(5), np.ones(4), 0.0] / 2.0
    e9 = np.eye(10)[9]
    fr = np.column_stack([b1, b2, e9])
    om = np.abs(fx.unit_sphere(N, 2, rng))
    X = R * om @ fr.T
    dist = R * np.arcsin(np.clip(om.min(1), 0, 1))
    return X, dist, fr


def test_orthant_face_one_sided_and_coverage():
    rng = np.random.default_rng(70)
    X, dist, fr = octant_patch(10000, rng)
    assert (X >= 0).all()
    tau = 0.001
    w = np.sqrt(2 * tau)                      # kernel width (per-coordinate sd)
    op = graph.build_operator(X, k=160, tau=tau)
    fa = frame.analyse_frame(op)
    interior, edge = dist > 5 * w, dist < 0.5 * w
    print("orthant: interior one_sided median", np.median(fa.one_sided[interior]),
          "edge median", np.median(fa.one_sided[edge]),
          "released interior / edge", fa.ok_tangent[interior].mean(), fa.ok_tangent[edge].mean(),
          "dim==2 frac", np.mean(fa.dim == 2))
    assert np.median(fa.one_sided[edge]) > 0.6           # ~0.8 for a flat boundary
    assert np.mean(fa.one_sided[interior] < 0.5) > 0.98
    assert fa.ok_tangent[edge].mean() < 0.1 and fa.ok_tangent[interior].mean() > 0.9
    assert np.mean(fa.dim == M) > 0.95                   # no rank bias from the face
    # drift (uniform density, great sphere: exact answer 0): biased at the face by ~ sqrt(2 / pi) w / tau,
    # small inside (sampling noise only)
    T = (fr @ fr.T)[None] - X[:, :, None] * X[:, None, :] / R**2
    Td = np.einsum("nij,nj->ni", T, fa.drift)
    dn = np.linalg.norm(Td, axis=1)
    print("drift |T L X| interior median", np.median(dn[interior]), "edge median", np.median(dn[edge]),
          "predicted edge", np.sqrt(2 / np.pi) * w / tau)
    assert np.median(dn[edge]) > 5 * np.median(dn[interior])
    np.testing.assert_allclose(np.median(dn[edge]), np.sqrt(2 / np.pi) * w / tau, rtol=0.35)
    # a step near the face: released points are accurate, coverage is reported (< 1)
    Y = fx.projective_map(X, fr @ np.diag([1.2, 1.0, 0.8]) @ fr.T + np.eye(10) - fr @ fr.T)
    o = step.measure_step(X, Y, 1.0, op0=op, operator=dict(kind="fixed", tau=tau, k=160))
    cov = np.mean(np.isfinite(o["logJ"]))
    print("orthant step: logJ released mass fraction", cov)
    assert 0.3 < cov < 0.98


def test_density_boundary_correction():
    """Item 9 density on the octant patch (uniform, Fisher area 2 pi): exact log rho = -log(2 pi).  At the
    faces a kNN ball is cut by the boundary; the half-space correction removes the log 2 deficit there.
    Corners (two faces within the ball) are withheld (NaN) rather than mis-corrected."""
    from mm import density
    rng = np.random.default_rng(74)
    X, dist, fr = octant_patch(20000, rng)
    E = sphere.tangent_basis(X)
    T = (fr @ fr.T)[None] - X[:, :, None] * X[:, None, :] / R**2
    U = np.zeros((len(X), 10, 9))
    U[:, :, :2] = np.linalg.svd(T @ E, full_matrices=False)[0][:, :, :2]
    dim = np.full(len(X), 2)
    folds = density.fold_split(len(X), 0)
    k = 32
    on = density.knn_log_density(X, dim, U, folds, k=k, boundary=True)
    off = density.knn_log_density(X, dim, U, folds, k=k, boundary=False)
    exact = -np.log(2 * np.pi)
    rk = np.nanmedian(on["r_k"])
    face = dist < 0.2 * rk
    inner = dist > 2 * rk
    e_on, e_off = on["logrho"] - exact, off["logrho"] - exact
    print("density: inner bias", np.nanmean(e_on[inner]), "face bias without / with correction",
          np.nanmean(e_off[face]), np.nanmean(e_on[face]), "face withheld", np.mean(np.isnan(e_on[face])))
    assert abs(np.nanmean(e_on[inner])) < 0.03
    assert np.nanmean(e_off[face]) < -0.4                       # ~ -log 2 without the correction
    assert abs(np.nanmean(e_on[face])) < 0.15


@pytest.mark.parametrize("ratio,expect_dim", [(0.1, 2), (3.0, 3)])
def test_thin_dimension(ratio, expect_dim):
    """A great 2-sphere thickened along a normal direction f3 by a uniform Fisher offset z with
    sd(z) = ratio * sqrt(2 tau).  Thin (z << kernel): lambda_3 = (Var z + z_i^2) / (2 tau) and the retained
    dimension is 2; thick: the third direction is a tangent direction (lambda_3 -> 1, dim 3)."""
    rng = np.random.default_rng(71)
    fr = fx.random_frame(4, rng)
    tau = 0.003
    sd = ratio * np.sqrt(2 * tau)
    b = np.sqrt(3) * sd                                  # z uniform on [-b, b]
    N = 20000 if expect_dim == 2 else 60000
    om = fx.unit_sphere(N, 2, rng)
    z = rng.uniform(-b, b, N)
    beta = z / R
    X = R * (np.cos(beta)[:, None] * (om @ fr[:, :3].T) + np.sin(beta)[:, None] * fr[:, 3])
    k = 128 if expect_dim == 2 else 256
    op = graph.build_operator(X, k=k, tau=tau)
    fa = frame.analyse_frame(op)
    print("thin ratio", ratio, "dims", np.bincount(fa.dim, minlength=4), "median eig", np.median(fa.eig[:, :4], 0),
          "trunc", np.nanmedian(fa.trunc_tail))
    assert np.mean(fa.dim == expect_dim) > 0.9
    if expect_dim == 2:
        pred = (b**2 / 3 + z**2) / (2 * tau)
        np.testing.assert_allclose(np.median(fa.eig[:, 2] / pred), 1.0, atol=0.15)
        assert np.median(fa.eigengap) > 10


def vertex_cluster(N, rng, scale):
    """2-d cluster at Fisher distance 2r from the vertex e0: x = 2(cos r e0 + sin r zeta(phi)), r in
    [0.01, 0.02] * scale, zeta(phi) = cos(phi) g0 + sin(phi) g1 in the positive orthant."""
    g0 = np.r_[0.0, 1, 1, 1, 0, 0, 0, 0, 0, 0] / np.sqrt(3)
    g1 = np.r_[0.0, 0, 0, 0, 1, 1, 0, 0, 0, 0] / np.sqrt(2)
    r0, r1 = 0.01 * scale, 0.02 * scale
    r = np.sqrt(rng.uniform(r0**2, r1**2, N))           # area element ~ r dr dphi
    phi = rng.uniform(0.3, 1.2, N)
    return r, phi, (r0, r1), (g0, g1)


def vertex_points(r, phi, g):
    zeta = np.cos(phi)[:, None] * g[0] + np.sin(phi)[:, None] * g[1]
    return R * (np.cos(r)[:, None] * np.eye(10)[0] + np.sin(r)[:, None] * zeta)


@pytest.mark.parametrize("scale", [1.0, 1e-5])
@pytest.mark.parametrize("lam", [0.5, 1e-3, 1e-6])
def test_cluster_shrinking_to_vertex(scale, lam):
    """r -> lam r at fixed phi: stretches lam (radial) and sin(lam r) / sin r (angular), log J their log-sum.
    scale = 1e-5 puts the source cloud within ~4e-7 of the vertex (real late-site clouds sit at 1e-9 .. 1e-3)."""
    rng = np.random.default_rng(72)
    N = 8000
    r, phi, (r0, r1), g = vertex_cluster(N, rng, scale)
    X = vertex_points(r, phi, g)
    Y = vertex_points(lam * r, phi, g)
    spacing = np.sqrt(0.9 * 4 * (r1**2 - r0**2) / 2 / N)
    tau = (2.5 * spacing) ** 2 / 2
    o = step.measure_step(X, Y, 1.0, operator=dict(kind="fixed", tau=tau, k=256), mask=False)
    w = np.sqrt(2 * tau)
    inner = ((2 * (r - r0) > 5 * w) & (2 * (r1 - r) > 5 * w)
             & (2 * r * np.sin(phi - 0.3) > 5 * w) & (2 * r * np.sin(1.2 - phi) > 5 * w))
    ok = inner & (o["dim"] == M)
    assert ok.sum() > 0.3 * N
    # target neighbourhoods below the float64 resolution floor of the representation (1e-12) are reported
    # as rank loss (log J withheld), not as a number
    unres = ok & (lam * o["radius"] <= o["res1"])
    assert o["rank_loss"][unres].all() and np.isnan(o["logJ"][unres]).all()
    ok = ok & ~unres
    if ok.sum() < 0.3 * N:
        return
    s_ex = np.sort(np.column_stack([np.full(N, lam), np.sin(lam * r) / np.sin(r)]), 1)[:, ::-1]
    err = np.median(np.abs(o["stretches"][ok, :M] / s_ex[ok] - 1), 0)
    eJ = np.median(np.abs(o["logJ"][ok] - np.log(s_ex[ok]).sum(1)))
    print("vertex scale", scale, "lam", lam, "stretch err", err, "logJ err", eJ, "res0", np.median(o["res0"]))
    assert err.max() < 0.01 and eJ < 0.02
    assert not o["rank_loss"][ok].any()


def test_cluster_collapse_reported_as_rank_loss():
    rng = np.random.default_rng(73)
    N = 6000
    r, phi, (r0, r1), g = vertex_cluster(N, rng, 1.0)
    X = vertex_points(r, phi, g)
    spacing = np.sqrt(0.9 * 4 * (r1**2 - r0**2) / 2 / N)
    tau = (2.5 * spacing) ** 2 / 2
    OP = dict(kind="fixed", tau=tau, k=256)
    # total collapse onto the vertex
    Y = np.broadcast_to(R * np.eye(10)[0], X.shape).copy()
    o = step.measure_step(X, Y, 1.0, operator=OP, mask=False)
    ok = o["dim"] == M
    assert o["rank_loss"][ok].all() and o["collapsed"][ok].all()
    assert np.isnan(o["logJ"][ok]).all()
    assert np.isnan(o["log_stretches"][ok]).all()
    # below-resolution target spread: flagged (collapsed, rank loss), nothing finite is released
    assert not o["release"]["strain_finite"].any() and not o["release"]["logJ"].any()
    # collapse of one direction (phi -> constant): one stretch ~1, the other lost
    Y2 = vertex_points(r, np.full(N, 0.7), g)
    o2 = step.measure_step(X, Y2, 1.0, operator=OP, mask=False)
    ok2 = o2["dim"] == M
    inner = ok2 & (r > r0 + 0.002) & (r < r1 - 0.002) & (phi > 0.4) & (phi < 1.1)
    print("partial collapse: rank loss frac", o2["rank_loss"][inner].mean(), "top stretch median",
          np.median(o2["stretches"][inner, 0]))
    assert o2["rank_loss"][inner].mean() > 0.99
    np.testing.assert_allclose(np.median(o2["stretches"][inner, 0]), 1.0, atol=0.02)
    assert np.isnan(o2["logJ"][inner]).all()
