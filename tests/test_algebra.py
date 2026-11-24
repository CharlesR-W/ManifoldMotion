"""Fixture family 1 (algebra only) and the exact identities of the operator and of measure_step.

Everything here holds to rounding for any data: transpose convention, truncated fit, residual closure,
Gram-weighted versus Frobenius energy; the carre du champ identity, reversibility, x.LX = -tr Gamma(X, X);
and the additive splits of measure_step."""
import numpy as np
import pytest

from mm import sphere, graph, frame, step
from mm import fixtures as fx

R = sphere.R


def _design(rng, x, spreads, k=60, symmetric=False):
    """Offsets u_j in the tangent at x along orthonormal tangent directions with the given spreads."""
    E = sphere.tangent_basis(x[None])[0]
    Q, _ = np.linalg.qr(E @ rng.standard_normal((9, 9)))           # random orthonormal tangent frame
    d = len(spreads)
    c = rng.standard_normal((k, d)) * np.asarray(spreads)
    if symmetric:
        c = np.concatenate([c, -c])
    u = c @ Q[:, :d].T
    a = rng.uniform(0.5, 1.5, u.shape[0])
    if symmetric:
        a = np.concatenate([a[: k], a[: k]])
    return a, u, Q[:, :d]


def _fit(a, u, resp, x, rel_threshold=0.1):
    G = frame._weighted_moment(a[None], u[None])
    lam, B = frame._eig_in_tangent(G, sphere.tangent_basis(x[None]))
    dim = frame.retained_dim(lam, rel_threshold, np.array([True]), np.array([0.0]))
    Gi = frame.truncated_pinv(B, lam, dim)
    out = frame.regress(a[None], u[None], resp[None], Gi)
    return {k: v[0] for k, v in out.items()}, int(dim[0]), G[0], B[0], lam[0]


def test_transpose_convention_anisotropic_design():
    rng = np.random.default_rng(0)
    x = R * np.eye(10)[0]
    a, u, U = _design(rng, x, [1.0, 0.7, 0.5])
    A = rng.standard_normal((10, 10))
    y = u @ A.T                                    # y_j = A u_j
    f, m, G, *_ = _fit(a, u, y, x)
    assert m == 3
    T = U @ U.T
    np.testing.assert_allclose(f["A"], A @ T, atol=1e-11)            # rows output, columns input
    assert np.abs(f["A"] - A.T @ T).max() > 0.1                       # the transpose is not a solution
    np.testing.assert_allclose(f["A"] @ u.T, y.T, atol=1e-11)
    assert f["res"] < 1e-20 * f["total"] + 1e-25


def test_truncated_fit_dropped_direction_and_closure():
    rng = np.random.default_rng(1)
    x = sphere.project_to_sphere(rng.standard_normal(10))
    a, u, U = _design(rng, x, [1.0, 0.8, 0.05])          # third direction: eigenvalue ratio 0.0025 < 0.1
    A = rng.standard_normal((10, 10))
    y = u @ A.T
    f, m, G, B, lam = _fit(a, u, y, x)
    assert m == 2
    T2 = B[:, :2] @ B[:, :2].T
    # only the source predictor is truncated: F = M G_T^+ = A G G_T^+, and F = F T2
    np.testing.assert_allclose(f["A"], f["A"] @ T2, atol=1e-12)
    np.testing.assert_allclose(f["A"], A @ G @ frame.truncated_pinv(B[None], lam[None], np.array([2]))[0], atol=1e-11)
    # dropped direction ends up in the residual, and the budget closes exactly
    assert f["res"] > 1e-4 * f["total"]
    np.testing.assert_allclose(f["total"], f["fit"] + f["res"], rtol=1e-12)
    Bm = frame._weighted_moment(a[None], y[None])[0]
    np.testing.assert_allclose(Bm, f["A"] @ G @ f["A"].T + f["Rm"], atol=1e-12 * np.abs(Bm).max())
    np.testing.assert_allclose(np.trace(f["Rm"]), f["res"], rtol=1e-12)
    assert np.linalg.eigvalsh(f["Rm"]).min() > -1e-12 * np.abs(Bm).max()          # R >= 0


def test_known_nonlinear_residual():
    """y = A u + c |u|^2 e on a symmetric design: the quadratic part is orthogonal to the linear predictor
    (odd moments vanish), so F = A T exactly and R = sum a c^2 |u|^4 e e^T exactly."""
    rng = np.random.default_rng(2)
    x = sphere.project_to_sphere(rng.standard_normal(10))
    a, u, U = _design(rng, x, [1.0, 0.6], symmetric=True)
    A = rng.standard_normal((10, 10))
    e = rng.standard_normal(10)
    c = 0.7
    y = u @ A.T + c * np.sum(u**2, 1)[:, None] * e
    f, m, G, *_ = _fit(a, u, y, x)
    T = U @ U.T
    np.testing.assert_allclose(f["A"], A @ T, atol=1e-11)
    Rexp = np.sum(a * c**2 * np.sum(u**2, 1) ** 2) * np.outer(e, e)
    np.testing.assert_allclose(f["Rm"], Rexp, atol=1e-10 * np.abs(Rexp).max())


def test_gram_weighted_not_frobenius_energy():
    rng = np.random.default_rng(3)
    x = sphere.project_to_sphere(rng.standard_normal(10))
    a, u, U = _design(rng, x, [1.0, 0.4])
    K = rng.standard_normal((10, 10))
    w = u @ K.T + 0.01 * rng.standard_normal(u.shape)
    f, m, G, *_ = _fit(a, u, w, x)
    fro = np.sum(f["A"] ** 2)
    np.testing.assert_allclose(f["total"], f["fit"] + f["res"], rtol=1e-12)
    np.testing.assert_allclose(f["fit"], np.trace(f["A"] @ G @ f["A"].T), rtol=1e-12)
    # anisotropic design: Frobenius norm of the gradient is not the fitted energy
    assert abs(f["total"] - (fro + f["res"])) > 0.1 * f["total"]
    # calibrated design G = T (rescale the design to unit eigenvalues): the two agree
    lam = np.linalg.eigvalsh(G)[::-1][:2]
    coords = u @ U
    Gc = (a[:, None] * coords).T @ coords
    W, Vv = np.linalg.eigh(Gc)
    white = coords @ Vv / np.sqrt(W) @ Vv.T
    u2 = white @ U.T
    f2, *_ = _fit(a, u2, u2 @ K.T, x)
    np.testing.assert_allclose(f2["fit"], np.sum((f2["A"]) ** 2), rtol=1e-10)


# ------------------------------------------------------------------------------------------------------
# operator identities (exact for any row-stochastic P)

@pytest.fixture(scope="module", params=["fixed", "adaptive"])
def op_small(request):
    rng = np.random.default_rng(10)
    X = fx.great_sphere(1500, 3, rng, kappa=2.0, frame=fx.random_frame(4, rng))[0]
    X = sphere.project_to_sphere(X + 0.02 * rng.standard_normal(X.shape))   # a thickened, noisy cloud
    if request.param == "fixed":
        return graph.build_operator(X, k=64, tau=0.01)
    return graph.build_operator(X, k=64, kind="adaptive", eps=0.1, k_bw=8)


def test_operator_rows_stochastic_and_reversible(op_small):
    op = op_small
    P = op.P
    np.testing.assert_allclose(np.asarray(P.sum(1)).ravel(), 1.0, rtol=1e-13)
    assert (P.data >= 0).all()
    # pi_i L_ij symmetric with pi = d_i tau_i (L = diag(1/tau)(P - I))
    S = (P.multiply(op.pi[:, None] / op.tau[:, None])).tocsr()
    diff = abs(S - S.T)
    assert diff.max() <= 1e-12 * abs(S).max()
    # distances stored on the same pattern
    assert op.D.nnz == P.nnz


def test_carre_du_champ_identity(op_small):
    op = op_small
    rng = np.random.default_rng(11)
    f = rng.standard_normal(op.N)
    h = op.X @ rng.standard_normal(10)
    g = op.gamma(f, h)[:, 0, 0]
    ref = 0.5 * (op.L(f * h) - f * op.L(h) - h * op.L(f))
    np.testing.assert_allclose(g, ref, atol=1e-9 * np.abs(ref).max())


def test_x_dot_LX_is_minus_trace_gamma(op_small):
    """x_i . (L X)_i = -tr Gamma(X, X)_i for the chord coordinates, at every bandwidth (|x| = 2)."""
    op = op_small
    LX = op.L(op.X)
    lhs = np.sum(op.X * LX, 1)
    G = op.gamma(op.X)
    rhs = -np.trace(G, axis1=1, axis2=2)
    np.testing.assert_allclose(lhs, rhs, rtol=1e-9, atol=1e-10 * np.abs(rhs).max())


# ------------------------------------------------------------------------------------------------------
# measure_step: exact additive identities on an arbitrary (noisy, nonlinear) step

@pytest.fixture(scope="module")
def noisy_step():
    rng = np.random.default_rng(20)
    fr = fx.random_frame(4, rng)
    X0 = fx.great_sphere(2000, 3, rng, kappa=1.0, frame=fr)[0]
    X0 = sphere.project_to_sphere(X0 + 0.01 * rng.standard_normal(X0.shape))
    A = np.eye(10) + 0.3 * rng.standard_normal((10, 10))
    X1 = fx.projective_map(X0, A)
    X1 = sphere.exp_map(X1, sphere.tangent_project(X1, 0.01 * rng.standard_normal(X1.shape)))
    o = step.measure_step(X0, X1, 0.7, operator=dict(kind="fixed", tau=0.01, k=64), mask=False)
    return X0, X1, o


def test_step_budgets_close_exactly(noisy_step):
    X0, X1, o = noisy_step
    ok = o["dim"] > 0
    assert ok.mean() > 0.9

    def close(a, b, rtol=1e-10):
        np.testing.assert_allclose(a[ok], b[ok], rtol=rtol, atol=1e-12 * np.nanmax(np.abs(b[ok])))

    close(o["trB"], o["trFGF"] + o["trR"])
    close(o["relmotion"], o["relmotion_fit"] + o["relmotion_res"])
    close(o["cr_relmotion"], o["cr_relmotion_fit"] + o["cr_relmotion_res"])
    close(o["grad_sq"], o["strain_sq"] + o["spin_sq"] + o["tilt_sq"])
    close(o["cr_grad_sq"], o["cr_strain_sq"] + o["cr_spin_sq"] + o["cr_tilt_sq"])
    close(o["rate_total"], o["rate_vol"] + o["rate_shear"])
    close(o["strain_sq"], o["rate_total"])
    fin = ok & ~o["rank_loss"]
    np.testing.assert_allclose(o["fin_total"][fin], (o["fin_vol"] + o["fin_shear"])[fin], rtol=1e-10)
    np.testing.assert_allclose(o["logJ"][fin], np.nansum(o["log_stretches"], 1)[fin], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(o["logJ"][fin], np.trace(o["hencky"], axis1=1, axis2=2)[fin], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(o["E_trace"][ok], np.trace(o["E"], axis1=1, axis2=2)[ok], rtol=1e-9, atol=1e-12)
    # item 8: the two terms add up to the strain exactly (w^T + w^N = w)
    np.testing.assert_allclose((o["E_flow"] + o["E_normal"])[ok], o["E"][ok], atol=1e-10 * np.nanmax(np.abs(o["E"])))
    # kinetic energy splits
    np.testing.assert_allclose((o["ke_tan"] + o["ke_norm"])[ok], o["ke"][ok], rtol=1e-12)
    np.testing.assert_allclose(np.mean(o["ke_rigid"] + o["ke_nonrigid"]), np.mean(o["ke"]), rtol=1e-10)
    np.testing.assert_allclose(o["ke_rigid"] + o["ke_nonrigid"] + o["ke_cross"], o["ke"], rtol=1e-10,
                               atol=1e-14 * o["ke"].max())
    # finite stretches are singular values of F on the retained tangent
    i = np.where(ok)[0][0]
    m = o["dim"][i]
    s = np.linalg.svd(o["F"][i], compute_uv=False)[:m]
    np.testing.assert_allclose(o["stretches"][i, :m], s, rtol=1e-10)


def test_step_undefined_is_nan_not_zero():
    """Unresolved neighbourhoods (duplicates), zero displacement and zero relative motion give NaN."""
    rng = np.random.default_rng(21)
    X0 = fx.great_sphere(400, 2, rng)[0]
    X0[:40] = X0[0]                                  # 40 exact duplicates: an atom
    X1 = X0.copy()
    o = step.measure_step(X0, X1, 1.0, operator=dict(kind="fixed", tau=1e-6, k=32), mask=False)
    atom = np.arange(40)
    assert np.all(o["dim"][atom] == 0)
    for key in ("F", "K_d", "E", "grad_sq", "strain_sq", "ke_tan", "ke_norm", "relmotion", "relmotion_fit",
                "stretches", "logJ", "explained", "heldout", "tan_fraction", "tilt_max"):
        assert np.all(np.isnan(o[key][atom])), key
    assert np.isnan(o["tan_fraction"]).all()                 # |v| = 0 everywhere
    assert np.isnan(o["rigid"]["rigid_fraction"])
    assert np.all(o["no_relative_motion"][o["dim"] > 0])
    om = step.apply_release(o)
    for key, rule in step.FIELD_RULE.items():
        bad = ~o["release"][rule]
        assert np.all(np.isnan(np.asarray(om[key], dtype=float)[bad])), key


def test_generator_at_tiny_kernel_time():
    """A cloud of Fisher size ~1e-8 next to a vertex (kernel times ~1e-17): L of a constant is exactly 0 and
    L is invariant to adding a constant (computed from differences, not (P f - f) / tau)."""
    rng = np.random.default_rng(12)
    N = 800
    zeta = np.abs(rng.standard_normal((N, 3)))
    zeta /= np.linalg.norm(zeta, axis=1, keepdims=True)
    r = 1e-8 * rng.uniform(1, 2, N)
    X = R * np.column_stack([np.cos(r), np.sin(r)[:, None] * zeta, np.zeros((N, 6))])
    for opk in (dict(kind="fixed", tau=float(np.median(graph.knn(X, 9)[1][:, 8]) ** 2)),
                dict(kind="adaptive", eps=0.25, k_bw=8)):
        op = graph.build_operator(X, k=48, **opk)
        assert np.median(op.tau) < 1e-15
        assert np.abs(op.L(np.ones(N))).max() == 0.0
        # (P f - f) / tau would turn the rounding of the row sums into O(1) garbage here.  Identities that
        # need second-order information in the chord coordinates (x.LX = -tr Gamma(X, X)) are not
        # representable at this scale: the top coordinate 2 cos r rounds to 2 for r < 1e-8.


def test_neighbour_resolution_gates_release():
    """An anchor with a precise target but neighbours whose target resolution exceeds their offsets: the
    target neighbourhood is unresolved for that anchor (collapsed, held-out undefined, stretches lost),
    even though its own resolution is tiny."""
    rng = np.random.default_rng(13)
    X = fx.great_sphere(1500, 2, rng)[0]
    op = dict(kind="fixed", tau=0.01, k=64)
    r1 = np.full(len(X), 10.0)            # every target offset is below resolution ...
    r1[:20] = 1e-12                       # ... except that these anchors are themselves precise
    o = step.measure_step(X, X.copy(), 1.0, operator=op, mask=False, res1=r1)
    a = np.arange(20)
    assert o["collapsed"][a].all() and np.isnan(o["heldout"][a]).all() and o["rank_loss"][a].all()
    assert not o["release"]["strain_finite"][a].any() and not o["release"]["relmotion"][a].any()
    # the same cloud with resolved targets: released (control)
    o2 = step.measure_step(X, X.copy(), 1.0, operator=op, mask=False)
    assert not o2["collapsed"][a].any() and np.isfinite(o2["heldout"][a]).all()


def test_cached_frame_resolution_must_match():
    """A frame analysed at one resolution cannot be reused with another (rank selection depends on it),
    including below 1e-8 where an absolute tolerance would hide the difference."""
    from mm import frame as frame_mod
    rng = np.random.default_rng(14)
    X = fx.great_sphere(600, 2, rng)[0]
    op = graph.build_operator(X, k=32, tau=0.01)
    f0 = frame_mod.analyse_frame(op, resolution=1e-12)
    with pytest.raises(ValueError):
        step.measure_step(X, X.copy(), 1.0, frame0=f0, res0=5e-9, operator=dict(kind="fixed", tau=0.01, k=32))
    o = step.measure_step(X, X.copy(), 1.0, frame0=f0, operator=dict(kind="fixed", tau=0.01, k=32), mask=False)
    np.testing.assert_array_equal(o["res0"], f0.res)
