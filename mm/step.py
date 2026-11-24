"""Per-point kinematic fields of one step (SPEC Section 3, items 1-11) and their release flags (Section 4).

Two frames X (time l) and Y (time l+1) of the same N material points, and the physical interval dt.
Displacement v_i = log_{x_i} Y_i (or a supplied tangent field); rates divide by dt, squared rates by dt^2.

Local moments at anchor i (frame-l operator, a_j = P_ij / 2 tau_i):
    u_j = log_{x_i} x_j,  y_j = log_{Y_i} Y_j,  w_j = PT_{j->i} v_j - v_i
    G = sum a u u^T,  M = sum a y u^T,  B = sum a y y^T,  C_v = sum a w u^T
    F = M G_T^+,  K_d = C_v G_T^+
G_T^+ is the pseudo-inverse truncated to the retained eigenspace T of G, so only the source predictor is
truncated (F = F T); responses keep all their sphere-tangent components (y_j is tangent at Y_i, w_j at x_i).
All matrices are 10 x 10 in ambient coordinates (rows: output, columns: input): regressing on the
truncated coordinates z = U^T u and lifting back, A_z U^T = M U (U^T G U)^-1 U^T = M G_T^+, so the
"lift to ambient coordinates" of the spec is built in.  Exact budgets (tests check them to rounding):
tr B = tr(F G F^T) + tr R and sum a |w|^2 = tr(K_d G K_d^T) + tr R_v, with R, R_v the weighted
residual second moments computed from the residual vectors themselves.

Rate gradient K = K_d / dt; with N = Q_x - T:
    E = sym(T K T) (strain rate), W = skew(T K T) (in-plane spin), N K T (tilt);
    |K T|^2 = |E|^2 + |W|^2 + |N K T|^2 exactly (K = K T because G_T^+ = T G_T^+ T).
Finite: singular values sigma_k of F on T (principal stretches), right singular vectors (source
directions), Hencky (1/2) log(F^T F) on T, lambda_k = log sigma_k, log J = sum lambda_k (withheld if a
singular value vanishes: rank loss).
Item 8: K_T and K_N are the same regression applied to w^T_j = PT_{j->i}(T_j v_j) - T_i v_i and
w^N_j = PT_{j->i}(N_j v_j) - N_i v_i (each neighbour's own projector); w^T + w^N = w, so
E = sym(T K_T T)/dt + sym(T K_N T)/dt exactly, the terms estimating sym grad_M V_T and -<II_S(., .), V_N>.
Released where the projector field is stable: equal-rank neighbours' transported projectors close to T_i
(proj_rms_angle), >= 90% of the neighbour mass with a released tangent, and a split-sample error
(E_flow recomputed with the even- and odd-half projector fields; half the difference) below 20% of the
larger term.
Item 10: principal angles between PT_{x->Y}(T_x M_l) and T_Y M_{l+1} (target tangent from the frame-(l+1)
operator), and the map-consistency angles between F(T_x M_l) and T_Y M_{l+1}.
Item 11: Killing fit V ~ Omega x (Omega C + C Omega = D - D^T, solved in the eigenbasis of C with the
minimum-norm convention for unidentified rotations), and the SO(10) Procrustes rotation R of the paired
positions (chordal loss); co-rotating fields use Y' = R^T Y.

Resolution and masking.  res0, res1 are per-point Fisher-distance resolutions of the two frames (from
sphere.input_resolution for log-probability input, else rules.resolution).  A quantity whose defining
norm or denominator is below resolution is NaN (never 0) even in the raw output: points with an
unresolved neighbourhood (dim 0) get NaN gradients, splits and budgets; tan_fraction needs |v| above
res0 + res1; a ratio such as `explained` needs its denominator resolved.  With mask=True every field is
also NaN where its release rule fails (the point keeps its mass; totals report the mass fraction they
cover).  Raw values are available with mask=False; the boolean rules are in out['release'].  Motion
below resolution is flagged (collapsed: the next-frame neighbourhood; no_relative_motion: the
displacement differences; rank_loss: a stretch, judged in its own source direction) and never released
as a finite strain: the estimate there is not a measurement of zero.  Co-rotating strain and tilt have
their own held-out gate (heldout_cr), since removing a large rigid motion can leave an unpredictable
remainder.
"""
from __future__ import annotations

import numpy as np

from . import sphere
from .density import fold_split, knn_log_density
from .frame import TDIM, Frame, Rules, _weighted_moment, analyse_frame, regress
from .graph import Operator, build_operator, run_batches

# Default operator: see DEFAULTS in the package docstring (mm/__init__.py) for the reasons.
DEFAULT_OPERATOR = dict(kind="adaptive", k=128, eps=0.25, k_bw=16)


# ----------------------------------------------------------------------------------------------------------
# rigid fits (item 11)

def killing_fit(X: np.ndarray, V: np.ndarray, rel_tol: float = 1e-10) -> dict:
    """Least-squares Killing field V ~ Omega x on the sphere (Omega antisymmetric), equal image weights.

    Normal equation Omega C + C Omega = D - D^T with C = sum x x^T, D = sum V x^T.  In the eigenbasis of C
    (eigenvalues c_a) it decouples: Omega_ab = Q_ab / (c_a + c_b).  Pairs with c_a + c_b <= rel_tol * max
    are rotations that do not move the cloud; they are set to 0 (minimum-norm solution) and counted.
    Returns Omega, the fitted field, conditioning max/min (c_a + c_b) over identified pairs, and the number
    of unidentified pairs.
    """
    C = X.T @ X
    D = V.T @ X
    c, Uc = np.linalg.eigh(C)
    Q = Uc.T @ (D - D.T) @ Uc
    S = c[:, None] + c[None, :]
    iu = np.triu_indices(len(c), 1)
    ok = S > rel_tol * S[iu].max()
    with np.errstate(divide="ignore", invalid="ignore"):
        Om = np.where(ok, Q / np.where(ok, S, 1.0), 0.0)
    np.fill_diagonal(Om, 0.0)
    Om = Uc @ Om @ Uc.T
    Om = 0.5 * (Om - Om.T)
    pairs = S[iu]
    okp = ok[iu]
    cond = pairs[okp].max() / pairs[okp].min() if okp.any() else np.inf
    fit = X @ Om.T
    return dict(Omega=Om, fit=fit, cond=float(cond), unidentified=int((~okp).sum()))


def procrustes_rotation(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """R in SO(n) minimising sum_i |Y_i - R X_i|^2 (Kabsch with determinant correction)."""
    H = Y.T @ X
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(U @ Vt))
    Dg = np.ones(H.shape[0])
    Dg[-1] = d if d != 0 else 1.0
    return (U * Dg) @ Vt


# ----------------------------------------------------------------------------------------------------------
# held-out score of a source-linear fit

def _heldout(a, u, resp, B, msk, lam_max, n_folds, fits=None):
    """Cross-validated score pieces for the fit resp_j ~ A u_j within the retained subspace.

    Folds: neighbour rank (distance order) mod n_folds.  Each fold refits the training design
    sum_train a z z^T (z = U^T u, U the full-data retained basis: the score is conditional on that basis).
    Returns (sum SSE_test, sum_test a |resp|^2) per row; if `fits` (b, n_folds, p, n) is given the training
    fits are stored in it (for the jackknife of the stretches).
    """
    k = u.shape[1]
    pos = np.arange(k)[None, :]
    sse = np.zeros(u.shape[0])
    den = np.zeros(u.shape[0])
    Bm = B * msk[:, None, :]
    pad = (~msk) * np.maximum(lam_max, 1e-300)[:, None]
    for f in range(n_folds):
        test = (pos % n_folds) == f
        atr = a * ~test
        ate = a * test
        Gtr = _weighted_moment(atr, u)
        Mtr = _weighted_moment(atr, u, resp)
        Gr = np.swapaxes(Bm, 1, 2) @ Gtr @ Bm
        Gr = Gr + pad[:, :, None] * np.eye(TDIM)[None]
        Mr = Mtr @ Bm
        A = Mr @ np.linalg.pinv(Gr, rcond=1e-12, hermitian=True) @ np.swapaxes(Bm, 1, 2)
        if fits is not None:
            fits[:, f] = A
        r = resp - u @ np.swapaxes(A, 1, 2)
        sse += np.einsum("bk,bki,bki->b", ate, r, r)
        den += np.einsum("bk,bki,bki->b", ate, resp, resp)
    return sse, den


def _centred_fit(an, u, resp, Bm, pad):
    """Free-intercept weighted least squares resp_j ~ A u_j + c on the weights `an` (non-self neighbours),
    within the retained basis Bm (masked columns zero, padded by `pad`).  Returns A (b, p, n), c (b, p), the
    centred design eigenvalues (b, 9) in the retained coordinates (padded ones excluded by the caller)."""
    s = an.sum(1)
    s = np.where(s > 0, s, 1.0)
    ub = np.einsum("bk,bki->bi", an, u) / s[:, None]
    yb = np.einsum("bk,bki->bi", an, resp) / s[:, None]
    z = (u - ub[:, None, :]) @ Bm
    Gz = np.swapaxes(an[..., None] * z, 1, 2) @ z + pad[:, :, None] * np.eye(TDIM)[None]
    Mz = np.swapaxes(an[..., None] * (resp - yb[:, None, :]), 1, 2) @ z
    Az = Mz @ np.linalg.pinv(Gz, rcond=1e-12, hermitian=True)
    A = Az @ np.swapaxes(Bm, 1, 2)
    c = yb - np.einsum("bpi,bi->bp", A, ub)
    return A, c, np.linalg.eigvalsh(Gz - pad[:, :, None] * np.eye(TDIM)[None])   # unpadded, ascending


def _heldout_family(a, u, resp, B, msk, lam_max, n_folds, nonself, fits=None):
    """Held-out SSE (same folds and test neighbours as _heldout, self excluded: it contributes 0 to the
    anchored fit) for the intercept fit and the constant-offset baseline (training-weighted mean of the
    responses).  Returns sse_intercept, sse_const, den, and the per-fold intercept fits if `fits` given."""
    k = u.shape[1]
    pos = np.arange(k)[None, :]
    Bm = B * msk[:, None, :]
    pad = (~msk) * np.maximum(lam_max, 1e-300)[:, None]
    sse_i = np.zeros(u.shape[0])
    sse_c = np.zeros(u.shape[0])
    den = np.zeros(u.shape[0])
    for f in range(n_folds):
        test = ((pos % n_folds) == f) & nonself
        atr = a * (~test & nonself)
        ate = a * test
        A, c, _ = _centred_fit(atr, u, resp, Bm, pad)
        if fits is not None:
            fits[:, f] = A
        r = resp - u @ np.swapaxes(A, 1, 2) - c[:, None, :]
        sse_i += np.einsum("bk,bki,bki->b", ate, r, r)
        st = atr.sum(1)
        cm = np.einsum("bk,bki->bi", atr, resp) / np.where(st > 0, st, 1.0)[:, None]
        rc = resp - cm[:, None, :]
        sse_c += np.einsum("bk,bki,bki->b", ate, rc, rc)
        den += np.einsum("bk,bki,bki->b", ate, resp, resp)
    return sse_i, sse_c, den


# ----------------------------------------------------------------------------------------------------------

def principal_angles(A: np.ndarray, Bm: np.ndarray) -> np.ndarray:
    """Principal angles (ascending) between column spaces of orthonormal A (n, d, p) and B (n, d, q);
    returns (n, min(p, q)).  Small angles from the sines, large from the cosines (atan2 of both)."""
    p, q = A.shape[2], Bm.shape[2]
    if p > q:
        A, Bm = Bm, A
        p, q = q, p
    cos = np.linalg.svd(np.swapaxes(Bm, 1, 2) @ A, compute_uv=False)[:, :p]       # descending
    perp = A - Bm @ (np.swapaxes(Bm, 1, 2) @ A)
    sin = np.sort(np.linalg.svd(perp, compute_uv=False), axis=1)[:, :p]           # ascending
    return np.arctan2(np.clip(sin, 0, 1), np.clip(cos, 0, 1))


def _orthonormal_columns(A: np.ndarray) -> np.ndarray:
    Q, _ = np.linalg.qr(A)
    return Q


def _fro2(A: np.ndarray) -> np.ndarray:
    return np.einsum("nij,nij->n", A, A)


def measure_step(
    X0: np.ndarray,
    X1: np.ndarray | None = None,
    dt: float = 1.0,
    v: np.ndarray | None = None,
    *,
    operator: dict | None = None,
    op0: Operator | None = None,
    op1: Operator | None = None,
    frame0: Frame | None = None,
    frame1: Frame | None = None,
    rules: Rules | None = None,
    labels: np.ndarray | None = None,
    res0: np.ndarray | float | None = None,
    res1: np.ndarray | float | None = None,
    density_k: int = 32,
    density_seed: int = 0,
    density_boundary: bool = True,
    mask: bool = True,
    budget: int = 32768,
    n_jobs: int = 4,
) -> dict:
    """All per-point fields of SPEC Section 3 items 1-11 for the step X0 -> X1 over the interval dt.

    X0, X1: (N, 10) points of the radius-2 sphere (use measure_step_logp for log-probabilities).
    v: optional displacement field (tangent at X0) to use instead of log_{X0} X1 (e.g. a Killing field
       supplied directly); if X1 is None it is exp_{X0}(v).
    operator: kwargs of graph.build_operator (kind, k, tau | eps, k_bw, ...), merged over
       DEFAULT_OPERATOR and used for both frames unless op0/op1 (or frames) are given.  kind="fixed"
       needs tau.
    labels: optional cluster labels for the competing-cluster part of the tangent rule (unassessed if None).
    res0, res1: per-point Fisher resolutions of the two frames (default rules.resolution).
    Returns a flat dict of arrays (keys documented in FIELD_DOC) plus 'release' (dict of boolean rules),
    'rigid' (global fits) and 'meta'.
    """
    rules = rules or Rules()
    operator = dict(DEFAULT_OPERATOR, **(operator or {}))
    if operator.get("kind") == "fixed":
        operator.pop("eps", None), operator.pop("k_bw", None)
    X0 = np.asarray(X0, dtype=np.float64)
    if X1 is None:
        if v is None:
            raise ValueError("need X1 or v")
        v = sphere.tangent_project(X0, np.asarray(v, dtype=np.float64))
        X1 = sphere.exp_map(X0, v)
    X1 = np.asarray(X1, dtype=np.float64)
    if v is None:
        v = sphere.log_map(X0, X1)
    else:
        v = sphere.tangent_project(X0, np.asarray(v, dtype=np.float64))
    N, n = X0.shape
    dt = float(dt)
    if not dt > 0:
        raise ValueError("dt must be positive")
    r0 = np.broadcast_to(np.asarray(rules.resolution if res0 is None else res0, float), (N,)).copy()
    r1 = np.broadcast_to(np.asarray(rules.resolution if res1 is None else res1, float), (N,)).copy()
    r0 = np.maximum(r0, rules.resolution)
    r1 = np.maximum(r1, rules.resolution)
    # a supplied frame was analysed at its own resolution: use it, and refuse a different one (the rank
    # selection depends on it)
    for fr_, rr, name in ((frame0, res0, "res0"), (frame1, res1, "res1")):
        if fr_ is not None and rr is not None and not np.allclose(
                np.maximum(np.broadcast_to(np.asarray(rr, float), (N,)), rules.resolution), fr_.res,
                rtol=1e-9, atol=0.0):
            raise ValueError(f"{name} differs from the resolution the supplied frame was analysed at")
    if frame0 is not None:
        r0 = np.asarray(frame0.res, float).copy()
    if frame1 is not None:
        r1 = np.asarray(frame1.res, float).copy()

    if frame0 is None:
        op0 = op0 or build_operator(X0, n_jobs=n_jobs, **operator)
        frame0 = analyse_frame(op0, rules, labels, resolution=r0, n_jobs=n_jobs)
    if frame1 is None:
        op1 = op1 or build_operator(X1, n_jobs=n_jobs, **operator)
        frame1 = analyse_frame(op1, rules, labels, resolution=r1, n_jobs=n_jobs)
    op0, op1 = frame0.op, frame1.op
    f0, f1 = frame0, frame1
    dim = f0.dim
    has_nb = dim > 0                               # resolved neighbourhood with a retained subspace

    # ---- global rigid fits (item 11)
    V = v / dt
    kf = killing_fit(X0, V)
    Rrot = procrustes_rotation(X0, X1)
    Yr = X1 @ Rrot              # rows: R^T Y_i
    vr = sphere.log_map(X0, Yr)

    # ---- tangential / normal parts with each point's own projector
    T0 = f0.T()
    vT = np.einsum("nij,nj->ni", T0, v)
    vN = v - vT
    vrT = np.einsum("nij,nj->ni", T0, vr)

    names = ("F", "K_d", "K_T", "K_N", "K_cr", "K_Te", "K_To")
    out_mats = {k: np.zeros((N, n, n)) for k in names}
    sc = {k: np.full(N, np.nan) for k in (
        "trB", "trFGF", "trR", "q", "q_fit", "q_res", "cr_q", "cr_q_fit", "cr_q_res",
        "ho_sse", "ho_den", "hov_sse", "hov_den", "hocr_sse", "hocr_den", "proj_rms", "proj_agree", "rmsY",
        "rmsW", "rmsWcr", "rn0", "rn1", "rnv", "attr_rej",
        # anchor / baseline diagnostics (y: next-frame offsets, v: displacement increments)
        "id_sse", "hoic_sse", "hoc_sse", "hons_den", "ho3_sse", "ho6_sse",
        "hovic_sse", "hovc_sse", "hovns_den", "c_y", "c_v", "cond_c", "rms_y_ns", "rms_v_ns",
        "q_shift", "q_slope", "q_icres")}
    F_folds = np.zeros((N, rules.n_folds, n, n))
    K_folds = np.zeros((N, rules.n_folds, n, n))
    Kic_folds = np.zeros((N, rules.n_folds, n, n))
    F_ic = np.full((N, n, n), np.nan)
    K_ic = np.full((N, n, n), np.nan)
    T0f = T0.reshape(N, n * n)
    # half-sample tangential parts (even / odd-ranked neighbours' projectors) for the split-sample error of
    # the attribution: E_flow recomputed with each half's projector field
    Ue, Uo = f0.split_U
    vTe = np.einsum("nik,nk->ni", Ue, np.einsum("nik,ni->nk", Ue, v))
    vTo = np.einsum("nik,nk->ni", Uo, np.einsum("nik,ni->nk", Uo, v))
    fields = np.stack([v, vT, vN, vr, vTe, vTo], axis=1)                 # (N, 6, n)
    speed0 = np.linalg.norm(v, axis=1)

    def batch(rows, idx, w):
        xi, xj = X0[rows], X0[idx]
        Yi, Yj = X1[rows], X1[idx]
        b, k = idx.shape
        a = w / (2.0 * op0.tau[rows])[:, None]
        u = sphere.log_map(xi[:, None, :], xj)
        y = sphere.log_map(Yi[:, None, :], Yj)
        xib = np.broadcast_to(xi[:, None, :], xj.shape)
        # displacement differences, transported from x_j to the anchor x_i: [v, T v, N v, corotated v]
        fj = fields[idx]                                                 # (b, k, 4, n)
        c = sphere.R2 + np.einsum("bki,bi->bk", xj, xi)
        coef = np.einsum("bi,bkfi->bkf", xi, fj) / c[:, :, None]
        wall = fj - coef[..., None] * (xj + xib)[:, :, None, :] - fields[rows][:, None, :, :]
        Gi = f0.ginv(rows)
        resps = (y, wall[:, :, 0], wall[:, :, 1], wall[:, :, 2], wall[:, :, 3], wall[:, :, 4], wall[:, :, 5])
        fits = [regress(a, u, rsp, Gi) for rsp in resps]
        for name, ft in zip(names, fits):
            out_mats[name][rows] = ft["A"]
        for pre, ft in (("", fits[0]), ("q", fits[1]), ("cr_q", fits[4])):
            if pre == "":
                sc["trB"][rows], sc["trFGF"][rows], sc["trR"][rows] = ft["total"], ft["fit"], ft["res"]
            else:
                sc[pre][rows], sc[pre + "_fit"][rows], sc[pre + "_res"][rows] = ft["total"], ft["fit"], ft["res"]
        wv = wall[:, :, 0]
        sc["rmsY"][rows] = np.sqrt((w * np.square(y).sum(-1)).sum(-1))
        sc["rmsW"][rows] = np.sqrt((w * np.square(wv).sum(-1)).sum(-1))
        sc["rmsWcr"][rows] = np.sqrt((w * np.square(wall[:, :, 3]).sum(-1)).sum(-1))
        # resolution of the neighbourhood's pairwise responses (anchor and neighbour both count):
        # P-weighted rms of r_i + r_j for source offsets, target offsets and displacement differences
        p0 = r0[rows][:, None] + r0[idx]
        p1 = r1[rows][:, None] + r1[idx]
        sc["rn0"][rows] = np.sqrt((w * p0**2).sum(1))
        sc["rn1"][rows] = np.sqrt((w * p1**2).sum(1))
        sc["rnv"][rows] = np.sqrt((w * (p0 + p1) ** 2).sum(1))

        Bb, mk = f0.basis[rows], f0.mask[rows]
        lmax = f0.eig[rows, 0]
        ff = np.zeros((b, rules.n_folds, n, n))
        sc["ho_sse"][rows], sc["ho_den"][rows] = _heldout(a, u, y, Bb, mk, lmax, rules.n_folds, fits=ff)
        F_folds[rows] = ff
        sc["hov_sse"][rows], sc["hov_den"][rows] = _heldout(a, u, wv, Bb, mk, lmax, rules.n_folds)
        sc["hocr_sse"][rows], sc["hocr_den"][rows] = _heldout(a, u, wall[:, :, 3], Bb, mk, lmax, rules.n_folds)
        kfo = np.zeros((b, rules.n_folds, n, n))
        _heldout(a, u, wv, Bb, mk, lmax, rules.n_folds, fits=kfo)
        K_folds[rows] = kfo

        # ---- anchor bias and baselines (SPEC review, 2026-11-24): the anchored fit absorbs any response
        # offset c shared by all neighbours (e.g. the anchor's own endpoint scatter) as A + c mu^T S^-1.
        nonself = idx != rows[:, None]
        an = a * nonself
        pad = (~mk) * np.maximum(lmax, 1e-300)[:, None]
        Bmk = Bb * mk[:, None, :]
        Ay, cy, ev = _centred_fit(an, u, y, Bmk, pad)
        Av, cv, _ = _centred_fit(an, u, wv, Bmk, pad)
        F_ic[rows], K_ic[rows] = Ay, Av
        # three-way budget of the relative motion on non-self neighbours (OLS with intercept: residual
        # orthogonal to the fitted values, so shift + slope + residual = total exactly):
        #   shift = sum a |c + A u_bar|^2 (= sum a |w_bar|^2), slope = sum a |A (u - u_bar)|^2
        sa = an.sum(1)
        ubar = np.einsum("bk,bki->bi", an, u) / np.where(sa > 0, sa, 1.0)[:, None]
        wbar = np.einsum("bk,bki->bi", an, wv) / np.where(sa > 0, sa, 1.0)[:, None]
        sl = (u - ubar[:, None, :]) @ np.swapaxes(Av, 1, 2)
        rr = wv - wbar[:, None, :] - sl
        sc["q_shift"][rows] = sa * (wbar**2).sum(1)
        sc["q_slope"][rows] = np.einsum("bk,bki,bki->b", an, sl, sl)
        sc["q_icres"][rows] = np.einsum("bk,bki,bki->b", an, rr, rr)
        # centred design condition number on the retained coordinates (masked ones give eigenvalue 0)
        mcount = mk.sum(1)
        evd = ev[:, ::-1]
        lam_low = evd[np.arange(b), np.maximum(mcount - 1, 0)]
        with np.errstate(invalid="ignore", divide="ignore"):
            sc["cond_c"][rows] = np.where((mcount > 0) & (lam_low > 0), evd[:, 0] / lam_low, np.inf)
        sw_ = an.sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            sc["rms_y_ns"][rows] = np.sqrt((an * (y**2).sum(-1)).sum(1) / sw_)
            sc["rms_v_ns"][rows] = np.sqrt((an * (wv**2).sum(-1)).sum(1) / sw_)
        sc["c_y"][rows] = np.linalg.norm(cy, axis=1)
        sc["c_v"][rows] = np.linalg.norm(cv, axis=1)
        kic = np.zeros((b, rules.n_folds, n, n))
        sc["hoic_sse"][rows], sc["hoc_sse"][rows], sc["hons_den"][rows] = _heldout_family(
            a, u, y, Bb, mk, lmax, rules.n_folds, nonself)
        sc["hovic_sse"][rows], sc["hovc_sse"][rows], sc["hovns_den"][rows] = _heldout_family(
            a, u, wv, Bb, mk, lmax, rules.n_folds, nonself, fits=kic)
        Kic_folds[rows] = kic
        # transported-identity baseline: y_j ~ PT_{x_i -> Y_i} u_j (no fitting, so in- and out-of-sample)
        PTu = sphere.parallel_transport(xib, np.broadcast_to(Yi[:, None, :], xj.shape), u)
        r_id = y - PTu
        sc["id_sse"][rows] = np.einsum("bk,bki,bki->b", an, r_id, r_id)
        # anchored fits on the strongest 3 and 6 source directions only (held-out gain of the weak ones)
        ar9 = np.arange(TDIM)[None, :]
        sc["ho3_sse"][rows], _ = _heldout(a, u, y, Bb, mk & (ar9 < 3), lmax, rules.n_folds)
        sc["ho6_sse"][rows], _ = _heldout(a, u, y, Bb, mk & (ar9 < 6), lmax, rules.n_folds)

        # projector-field stability: tr(T_i PT_{j->i} T_j PT^T) with PT xi = xi - (x_i.xi)(x_i + x_j)/c
        # equals <T_i, T_j>_F - 2 (T_i x_j).(T_j x_i) / c + (x_j^T T_i x_j)(x_i^T T_j x_i) / c^2
        Ti = T0[rows]
        Tjf = T0f[idx]                                                   # (b, k, n*n)
        tt = (Tjf @ T0f[rows][:, :, None])[..., 0]
        Tixj = xj @ Ti                                                   # (b, k, n) (T_i symmetric)
        Tjxi = np.einsum("bkij,bj->bki", Tjf.reshape(b, k, n, n), xi)
        tr = (tt - 2.0 * np.sum(Tixj * Tjxi, -1) / c
              + np.sum(Tixj * xj, -1) * np.sum(Tjxi * xib, -1) / c**2)
        mi, mj = dim[rows][:, None], dim[idx]
        # equal rank m: m - tr = sum_a sin^2(theta_a); rms sine over angles and equal-rank neighbours
        same = mj == mi
        sin2 = np.clip(mi - tr, 0.0, None) / np.maximum(mi, 1)
        wn = w * (idx != rows[:, None])
        sw = wn.sum(1)
        ws = wn * same
        with np.errstate(invalid="ignore", divide="ignore"):
            sc["proj_rms"][rows] = np.sqrt((ws * sin2).sum(1) / ws.sum(1))
            agree = same & f0.ok_tangent[idx]
            sc["proj_agree"][rows] = (wn * agree).sum(1) / sw
        # worst case of the neighbours whose projector is not trusted (other rank or tangent rule failed):
        # replacing T_j by any projector changes w^T_j by at most |v_j|, so K_T moves by at most
        # sum_rejected a_j |v_j| |G_T^+ u_j| (Frobenius), and E_flow by that / dt
        rej = (idx != rows[:, None]) & ~agree
        gu = np.linalg.norm(u @ np.swapaxes(Gi, 1, 2), axis=2)
        sc["attr_rej"][rows] = (a * rej * speed0[idx] * gu).sum(1)

    run_batches(op0, batch, budget=budget, n_jobs=n_jobs)

    # undefined where there is no resolved neighbourhood: NaN, never 0
    for k in names:
        out_mats[k][~has_nb] = np.nan
    rn0, rn1, rnv = sc["rn0"], sc["rn1"], sc["rnv"]
    if f0.res_nb is not None:
        rn0 = np.maximum(rn0, f0.res_nb)
    nb_resolved = f0.radius > rn0
    for k in ("q", "cr_q", "rmsW", "rmsWcr"):
        sc[k][~nb_resolved] = np.nan
    for k in ("trB", "trFGF", "trR", "q_fit", "q_res", "cr_q_fit", "cr_q_res", "ho_sse", "ho_den",
              "hov_sse", "hov_den", "hocr_sse", "hocr_den", "proj_rms", "proj_agree"):
        sc[k][~has_nb] = np.nan

    F, Kd, KT, KN, Kcr, KTe, KTo = (out_mats[k] for k in names)
    K = Kd / dt
    Q0 = sphere.tangent_projector(X0)
    Nproj = Q0 - T0

    # ---- speeds and kinetic energy (items 1, 2, 11)
    travel = np.linalg.norm(v, axis=1)
    vres = r0 + r1                                   # resolution of a displacement
    o = {}
    o["v"] = v
    o["travel"] = travel
    o["speed"] = travel / dt
    o["step_angle"] = travel / sphere.R
    o["ke"] = 0.5 * travel**2 / dt**2
    nbn = np.where(has_nb, 1.0, np.nan)
    o["ke_tan"] = 0.5 * np.sum(vT**2, 1) / dt**2 * nbn
    o["ke_norm"] = 0.5 * np.sum(vN**2, 1) / dt**2 * nbn
    with np.errstate(invalid="ignore", divide="ignore"):
        o["tan_fraction"] = np.where((travel > vres) & has_nb, np.sum(vT**2, 1) / travel**2, np.nan)
    o["full_rank"] = dim == TDIM
    o["ke_rigid"] = 0.5 * np.sum(kf["fit"] ** 2, 1)
    o["ke_nonrigid"] = 0.5 * np.sum((V - kf["fit"]) ** 2, 1)
    # ke = ke_rigid + ke_nonrigid + ke_cross per point; the cross term averages to 0 over all images only
    o["ke_cross"] = np.sum(kf["fit"] * (V - kf["fit"]), 1)
    crt = np.linalg.norm(vr, axis=1)
    o["cr_travel"] = crt
    o["cr_ke_tan"] = 0.5 * np.sum(vrT**2, 1) / dt**2 * nbn
    o["cr_ke_norm"] = 0.5 * np.sum((vr - vrT) ** 2, 1) / dt**2 * nbn

    # ---- relative motion budgets (items 3, 4); squared rates per dt^2
    o["relmotion"] = sc["q"] / dt**2
    o["relmotion_fit"] = sc["q_fit"] / dt**2
    o["relmotion_res"] = sc["q_res"] / dt**2
    o["cr_relmotion"] = sc["cr_q"] / dt**2
    o["cr_relmotion_fit"] = sc["cr_q_fit"] / dt**2
    o["cr_relmotion_res"] = sc["cr_q_res"] / dt**2
    o["F"], o["K_d"] = F, Kd

    def split(Km):
        TKT = T0 @ Km @ T0
        Em = 0.5 * (TKT + np.swapaxes(TKT, 1, 2))
        Wm = 0.5 * (TKT - np.swapaxes(TKT, 1, 2))
        NKT = Nproj @ Km @ T0
        return Em, Wm, NKT

    E, W, NKT = split(K)
    o["E"] = E
    o["grad_sq"] = _fro2(K @ T0)
    o["strain_sq"] = _fro2(E)
    o["spin_sq"] = _fro2(W)
    o["tilt_sq"] = _fro2(NKT)
    Ecr, Wcr, NKTcr = split(Kcr / dt)
    o["cr_grad_sq"] = _fro2((Kcr / dt) @ T0)
    o["cr_strain_sq"] = _fro2(Ecr)
    o["cr_spin_sq"] = _fro2(Wcr)
    o["cr_tilt_sq"] = _fro2(NKTcr)

    # ---- item 8: two-term attribution
    E_tan, _, _ = split(KT / dt)
    TKNT = T0 @ (KN / dt) @ T0
    E_nor = 0.5 * (TKNT + np.swapaxes(TKNT, 1, 2))
    o["E_flow"] = E_tan
    o["E_normal"] = E_nor
    o["E_flow_trace"] = np.trace(E_tan, axis1=1, axis2=2)
    o["E_normal_trace"] = np.trace(E_nor, axis1=1, axis2=2)
    o["E_normal_skew"] = np.sqrt(_fro2(0.5 * (TKNT - np.swapaxes(TKNT, 1, 2))))
    o["proj_rms_angle"] = sc["proj_rms"]
    o["proj_agree"] = sc["proj_agree"]
    # split-sample error of the attribution: E_flow with the even- and the odd-half projector fields (same
    # output projector T); half their difference estimates the full-sample projector-noise error of E_flow
    # (and, as the two terms add to E exactly, of E_normal)
    Ee, _, _ = split(KTe / dt)
    Eo, _, _ = split(KTo / dt)
    o["attribution_noise"] = 0.5 * np.sqrt(_fro2(Ee - Eo))
    o["attribution_rejected"] = sc["attr_rej"] / dt

    # ---- explained fractions (item 4 / Section 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        o["explained"] = np.where(sc["rmsY"] > rn1, sc["trFGF"] / sc["trB"], np.nan)
        hd = np.sqrt(np.maximum(sc["ho_den"] * 2.0 * op0.tau, 0.0))        # rms held-out |y| (P weights)
        o["heldout"] = np.where(hd > rn1, 1.0 - sc["ho_sse"] / sc["ho_den"], np.nan)
        hvd = np.sqrt(np.maximum(sc["hov_den"] * 2.0 * op0.tau, 0.0))
        o["heldout_v"] = np.where(hvd > rnv, 1.0 - sc["hov_sse"] / sc["hov_den"], np.nan)
        hcd = np.sqrt(np.maximum(sc["hocr_den"] * 2.0 * op0.tau, 0.0))
        o["heldout_cr"] = np.where(hcd > rnv, 1.0 - sc["hocr_sse"] / sc["hocr_den"], np.nan)
    o["trB"], o["trFGF"], o["trR"] = sc["trB"], sc["trFGF"], sc["trR"]
    o["step_K"] = np.sqrt(_fro2(Kd))

    # ---- baselines and anchor diagnostics.  All scores: 1 - SSE / sum a |response|^2 over the same
    # non-self test neighbours, i.e. improvement over predicting a zero response (not a centred R^2).
    with np.errstate(invalid="ignore", divide="ignore"):
        okd = (hd > rn1) & has_nb
        okv = (hvd > rnv) & has_nb
        o["heldout_identity"] = np.where(okd, 1.0 - sc["id_sse"] / sc["hons_den"], np.nan)
        o["heldout_const"] = np.where(okd, 1.0 - sc["hoc_sse"] / sc["hons_den"], np.nan)
        o["heldout_ic"] = np.where(okd, 1.0 - sc["hoic_sse"] / sc["hons_den"], np.nan)
        o["heldout_top3"] = np.where(okd, 1.0 - sc["ho3_sse"] / sc["ho_den"], np.nan)
        o["heldout_top6"] = np.where(okd, 1.0 - sc["ho6_sse"] / sc["ho_den"], np.nan)
        o["heldout_v_const"] = np.where(okv, 1.0 - sc["hovc_sse"] / sc["hovns_den"], np.nan)
        o["heldout_v_ic"] = np.where(okv, 1.0 - sc["hovic_sse"] / sc["hovns_den"], np.nan)
        o["intercept_rel"] = np.where(okd, sc["c_y"] / sc["rms_y_ns"], np.nan)
        o["intercept_v_rel"] = np.where(okv, sc["c_v"] / sc["rms_v_ns"], np.nan)
    o["cond_centred"] = np.where(has_nb, sc["cond_c"], np.nan)
    nbq = np.where(nb_resolved, 1.0, np.nan)
    o["relmotion_shift"] = sc["q_shift"] / dt**2 * nbq
    o["relmotion_slope"] = sc["q_slope"] / dt**2 * nbq
    o["relmotion_icres"] = sc["q_icres"] / dt**2 * nbq
    o["F_ic"], o["K_ic"] = F_ic, K_ic

    def jk_var(folds):
        """Delete-a-group jackknife variance (summed over entries) of sym and skew parts of T K T / dt."""
        Tf = T0[:, None]
        Kf = Tf @ (folds / dt) @ Tf
        Ef = 0.5 * (Kf + np.swapaxes(Kf, 2, 3))
        Wf = 0.5 * (Kf - np.swapaxes(Kf, 2, 3))
        g = folds.shape[1]
        ve = (g - 1) / g * np.sum((Ef - Ef.mean(1, keepdims=True)) ** 2, axis=(1, 2, 3))
        vw = (g - 1) / g * np.sum((Wf - Wf.mean(1, keepdims=True)) ** 2, axis=(1, 2, 3))
        return ve, vw

    # squared norms carry a positive noise term even at zero true strain / spin: subtract the jackknife
    # variance (debiased; can be negative where the signal is below its noise)
    ve, vw = jk_var(K_folds)
    o["strain_sq_var"], o["spin_sq_var"] = ve, vw
    o["strain_sq_db"] = o["strain_sq"] - ve
    o["spin_sq_db"] = o["spin_sq"] - vw
    Kic = K_ic / dt
    Eic = 0.5 * (T0 @ Kic @ T0 + np.swapaxes(T0 @ Kic @ T0, 1, 2))
    Wic = 0.5 * (T0 @ Kic @ T0 - np.swapaxes(T0 @ Kic @ T0, 1, 2))
    vei, vwi = jk_var(Kic_folds)
    o["ic_E_trace"] = np.trace(Eic, axis1=1, axis2=2)
    o["ic_strain_sq"] = _fro2(Eic)
    o["ic_spin_sq"] = _fro2(Wic)
    o["ic_strain_sq_db"] = o["ic_strain_sq"] - vei
    o["ic_spin_sq_db"] = o["ic_spin_sq"] - vwi
    o["ic_log_stretches"] = np.full((N, TDIM), np.nan)
    o["ic_log_gain_max"] = np.full(N, np.nan)
    o["ic_logJ"] = np.full(N, np.nan)
    for m in range(1, TDIM + 1):
        sel = np.where(dim == m)[0]
        if sel.size == 0:
            continue
        sv = np.linalg.svd(F_ic[sel] @ f0.basis[sel][:, :, :m], compute_uv=False)
        with np.errstate(divide="ignore", invalid="ignore"):
            lsv = np.log(sv)
        o["ic_log_stretches"][sel, :m] = lsv
        o["ic_log_gain_max"][sel] = lsv[:, 0]
        o["ic_logJ"][sel] = lsv.sum(1)
    for key in ("ic_E_trace", "ic_strain_sq", "ic_spin_sq", "ic_strain_sq_db", "ic_spin_sq_db", "ic_logJ"):
        o[key] = np.where(has_nb, o[key], np.nan)

    # ---- per-dimension post-processing: SVD of F, spectra of E, tilt
    o["stretches"] = np.full((N, TDIM), np.nan)
    o["log_stretches"] = np.full((N, TDIM), np.nan)
    o["stretch_dirs"] = np.full((N, TDIM, n), np.nan)
    o["hencky"] = np.full((N, n, n), np.nan)
    o["logJ"] = np.full(N, np.nan)
    o["rank_loss"] = np.zeros(N, dtype=bool)
    for key in ("E_trace", "E_shear", "E_top", "rate_vol", "rate_shear", "rate_total", "fin_vol",
                "fin_shear", "fin_total", "rate_finite_gap", "tilt_max", "tilt_rms", "map_angle_max",
                "cr_tilt_max", "cr_tilt_rms", "sigma_min", "sigma_max", "log_sigma_min_se", "logJ_se"):
        o[key] = np.full(N, np.nan)
    o["E_top_dir"] = np.full((N, n), np.nan)
    o["tilt_dim_mismatch"] = np.zeros(N, dtype=bool)
    o["tilt_angles"] = np.full((N, TDIM), np.nan)
    o["cr_tilt_angles"] = np.full((N, TDIM), np.nan)
    R_UY = np.einsum("ab,nbk->nak", Rrot.T, f1.basis)

    for m in range(1, TDIM + 1):
        sel = np.where(dim == m)[0]
        if sel.size == 0:
            continue
        U = f0.basis[sel][:, :, :m]
        FU = F[sel] @ U
        Ul, s, Vt = np.linalg.svd(FU, full_matrices=False)
        dirs = U @ np.swapaxes(Vt, 1, 2)                            # (b, n, m) source directions
        smax = s[:, 0]
        # a stretch is lost if it is negligible against the largest, or if it maps the source spread in its
        # own direction d_k (sqrt(2 tau d_k^T G d_k), the P-weighted rms offset along d_k) below the target
        # resolution
        spread = np.sqrt(2.0 * op0.tau[sel][:, None]
                         * np.einsum("bl,bkl->bk", f0.eig[sel][:, :m], Vt**2))
        vanish = ((s <= rules.rank_tol * smax[:, None])
                  | (s * spread <= rn1[sel][:, None]))
        rl = vanish.any(1)
        with np.errstate(divide="ignore"):
            ls = np.where(vanish, np.nan, np.log(np.where(vanish, 1.0, s)))
        o["stretches"][sel, :m] = s
        o["log_stretches"][sel, :m] = ls
        o["sigma_min"][sel], o["sigma_max"][sel] = s[:, -1], smax
        # delete-a-group jackknife over the held-out folds (each fold fit leaves one fold out)
        sf = np.linalg.svd(F_folds[sel] @ U[:, None], compute_uv=False)        # (b, folds, m)
        with np.errstate(divide="ignore", invalid="ignore"):
            lsf = np.log(sf)
        nf = sf.shape[1]
        for key, th in (("log_sigma_min_se", lsf[:, :, -1]), ("logJ_se", lsf.sum(2))):
            with np.errstate(invalid="ignore"):
                o[key][sel] = np.sqrt((nf - 1) / nf * np.sum((th - th.mean(1, keepdims=True)) ** 2, 1))
        # direction of a stretch is undefined where it is (numerically) degenerate with a neighbour
        gap_tol = 1e-6 * np.maximum(smax, 1e-300)[:, None]
        degen = np.zeros_like(s, dtype=bool)
        if m > 1:
            dd = np.abs(np.diff(s, axis=1)) <= gap_tol
            degen[:, :-1] |= dd
            degen[:, 1:] |= dd
        o["stretch_dirs"][sel, :m, :] = np.where(degen[:, :, None], np.nan, np.swapaxes(dirs, 1, 2))
        H = (dirs * ls[:, None, :]) @ np.swapaxes(dirs, 1, 2)      # NaN rows where a stretch vanished
        o["hencky"][sel] = H
        o["rank_loss"][sel] = rl
        sum_l = np.where(rl, np.nan, np.nansum(ls, 1))
        sum_l2 = np.where(rl, np.nan, np.nansum(ls**2, 1))
        o["logJ"][sel] = sum_l
        o["fin_vol"][sel] = sum_l**2 / m
        o["fin_shear"][sel] = sum_l2 - sum_l**2 / m
        o["fin_total"][sel] = sum_l2

        Em = np.swapaxes(U, 1, 2) @ E[sel] @ U
        ev, evec = np.linalg.eigh(Em)
        tr = ev.sum(1)
        e2 = np.einsum("bij,bij->b", Em, Em)
        o["E_trace"][sel] = tr
        o["E_shear"][sel] = np.sqrt(np.maximum(e2 - tr**2 / m, 0.0))
        o["E_top"][sel] = ev[:, -1]
        o["rate_vol"][sel] = tr**2 / m
        o["rate_shear"][sel] = e2 - tr**2 / m
        o["rate_total"][sel] = e2
        top = (U @ evec[:, :, -1:])[:, :, 0]
        rad = np.maximum(f0.radius[sel], r0[sel])
        res_E = np.maximum(1e-8 * np.sqrt(e2), vres[sel] / (rad * dt))
        if m > 1:
            und = (ev[:, -1] - ev[:, -2]) <= res_E
        else:
            und = np.sqrt(e2) <= res_E
        o["E_top_dir"][sel] = np.where(und[:, None], np.nan, top)
        Hn = np.sqrt(np.einsum("bij,bij->b", H, H))
        Edt = np.sqrt(e2) * dt
        diff = np.sqrt(np.einsum("bij,bij->b", H - E[sel] * dt, H - E[sel] * dt))
        scale = np.maximum(Hn, Edt)
        with np.errstate(invalid="ignore", divide="ignore"):
            o["rate_finite_gap"][sel] = np.where(scale > vres[sel] / rad, diff / scale, np.nan)

        # tilt: transported source tangent vs target tangent, grouped by target dimension
        PTU = sphere.parallel_transport_matrix(X0[sel], X1[sel], U)
        PTUr = sphere.parallel_transport_matrix(X0[sel], Yr[sel], U)
        Fq = _orthonormal_columns(Ul)
        for m1 in range(1, TDIM + 1):
            s1 = np.where(f1.dim[sel] == m1)[0]
            if s1.size == 0:
                continue
            g = sel[s1]
            UY = f1.basis[g][:, :, :m1]
            ang = principal_angles(PTU[s1], UY)
            p = ang.shape[1]
            o["tilt_angles"][g, :p] = ang
            o["tilt_max"][g] = ang.max(1)
            o["tilt_dim_mismatch"][g] = m1 != m
            o["tilt_rms"][g] = np.sqrt(np.mean(ang**2, 1))
            o["map_angle_max"][g] = np.where(rl[s1], np.nan, principal_angles(Fq[s1], UY).max(1))
            angr = principal_angles(PTUr[s1], R_UY[g][:, :, :m1])
            o["cr_tilt_angles"][g, :p] = angr
            o["cr_tilt_max"][g] = angr.max(1)
            o["cr_tilt_rms"][g] = np.sqrt(np.mean(angr**2, 1))

    # ---- item 9: densities and mass balance
    folds = fold_split(N, density_seed)
    d0 = knn_log_density(X0, dim, f0.basis, folds, k=density_k, boundary=density_boundary, n_jobs=n_jobs,
                         res=r0)
    d1 = knn_log_density(X1, dim, f1.basis, folds, k=density_k, boundary=density_boundary, n_jobs=n_jobs,
                         res=r1)
    o["logrho0"], o["logrho1"] = d0["logrho"], d1["logrho"]
    o["boundary0"], o["boundary1"] = d0["boundary"], d1["boundary"]
    o["mass_balance"] = d1["logrho"] + o["logJ"] - d0["logrho"]
    # single-sheet check on the neighbours that set the target density (other-fold kNN of Y_i): under a
    # locally one-to-one map from one sheet their sources lie in x_i's own source patch, i.e. within the
    # tangential ambiguity far_factor r_k(Y) / sigma_min of x_i along T_x M and within far_factor times
    # the source radius normally to it.  A neighbour outside either bound comes from another sheet (or the
    # map is not locally one-to-one): the fraction of such neighbours is foreign_frac.  The normal bound
    # does not grow with compression, so parallel sheets mapped together are caught however strong the
    # compression along the sheet; a third test compares each neighbour's target offset with F u, within
    # the fit's own (extrapolated) error, not within the density radius: two sheets separated along a
    # compressed direction land within r_Y of each other but are mispredicted by sigma_min D.
    tj = d1["nbr"]
    rY = d1["r_k"]
    foreign = np.full(N, np.nan)
    unassessed = np.full(N, np.nan)
    inconsistent = np.full(N, np.nan)
    good = (tj >= 0).all(1) & np.isfinite(o["sigma_min"]) & (o["sigma_min"] > 0) & np.isfinite(rY) & has_nb
    gi = np.where(good)[0]
    for s0 in range(0, gi.size, 4096):
        g = gi[s0 : s0 + 4096]
        us = sphere.log_map(X0[g][:, None, :], X0[tj[g]])                      # (b, k, n)
        ut = np.einsum("bij,bkj->bki", T0[g], us)
        tan_d = np.linalg.norm(ut, axis=2)
        nor_d = np.linalg.norm(us - ut, axis=2)
        tb = rules.far_factor * rY[g] / o["sigma_min"][g]
        nb = rules.far_factor * np.maximum(f0.radius[g], r0[g])
        # consistency with the local fit: the target offset of a same-sheet neighbour is F u up to the
        # fit's own residual scale (catches in-plane merges that the two bounds above accept when the
        # map compresses one direction strongly)
        yt = sphere.log_map(X1[g][:, None, :], X1[tj[g]])
        miss = np.linalg.norm(yt - np.einsum("bij,bkj->bki", F[g], us), axis=2)
        # allowed prediction error: endpoint resolution plus the fit's own rms error (the larger of the
        # in-sample and held-out residuals, P-weighted, Fisher units) extrapolated quadratically from the
        # kernel radius to the neighbour's source offset (second-order Taylor term of a smooth map)
        res_rms = np.sqrt(np.maximum(np.nan_to_num(np.maximum(sc["trR"][g], sc["ho_sse"][g])), 0.0)
                          * 2.0 * op0.tau[g])
        rad = np.maximum(f0.radius[g], rn0[g])
        ud = np.linalg.norm(us, axis=2) / rad[:, None]
        # the fit is validated on the kernel neighbourhood; a density neighbour whose source lies beyond
        # far_factor radii of it is outside the validated patch: unassessed (counted against the sheet),
        # never accepted on an extrapolated tolerance
        outside = ud > rules.far_factor
        mb = rules.far_factor * (rn1[g][:, None] + res_rms[:, None] * np.maximum(1.0, ud**2))
        bad_fit = (miss > mb) & ~outside
        unassessed[g] = np.mean(outside, axis=1)
        inconsistent[g] = np.mean(bad_fit, axis=1)
        foreign[g] = np.mean((tan_d > tb[:, None]) | (nor_d > nb[:, None]) | bad_fit | outside, axis=1)
    o["foreign_frac"] = foreign
    o["unassessed_frac"] = unassessed
    o["inconsistent_frac"] = inconsistent
    o["single_sheet"] = foreign <= rules.foreign_max

    # ---- diagnostics
    for key in ("dim", "eig", "radius", "n_eff", "cond", "eigengap", "split_angle", "one_sided", "drift",
                "trunc_tail", "competing"):
        o[key] = getattr(f0, key)
        o["tgt_" + key] = getattr(f1, key)
    o["tau"] = op0.tau
    o["res0"], o["res1"] = r0, r1
    o["ok_tangent0"] = f0.ok_tangent
    o["ok_tangent1"] = f1.ok_tangent

    # ---- release rules
    resolved = f0.radius > rn0
    rel = {}
    rel["always"] = np.ones(N, dtype=bool)
    rel["neighbourhood"] = (resolved & (f0.n_eff >= rules.n_eff_min)
                            & (np.nan_to_num(f0.trunc_tail, nan=1.0) <= rules.trunc_tail_max))
    rel["tangent"] = f0.ok_tangent & resolved
    rel["normal"] = rel["tangent"] & (dim < TDIM)
    # below-resolution motion is flagged and not released (the estimate there is not a measurement of 0)
    o["collapsed"] = sc["rmsY"] <= rn1         # next-frame neighbourhood below resolution (observed collapse)
    o["no_relative_motion"] = sc["rmsW"] <= rnv
    o["no_relative_motion_cr"] = sc["rmsWcr"] <= rnv
    rel["relmotion"] = rel["neighbourhood"] & ~o["no_relative_motion"]
    rel["cr_relmotion"] = rel["neighbourhood"] & ~o["no_relative_motion_cr"]
    with np.errstate(invalid="ignore"):
        design = (rel["tangent"] & (f0.n_eff >= rules.n_eff_per_dim_strain * (dim + 1))
                  & (f0.cond <= rules.cond_max_strain))
        rel["strain_finite"] = design & (np.nan_to_num(o["heldout"], nan=-np.inf) >= rules.heldout_min)
        rel["strain"] = design & (np.nan_to_num(o["heldout_v"], nan=-np.inf) >= rules.heldout_v_min)
        rel["cr_strain"] = design & (np.nan_to_num(o["heldout_cr"], nan=-np.inf) >= rules.heldout_v_min)
        # anchor agreement (flow mode): anchored and free-intercept fits must agree (see Rules.anchor_tol)
        agree_y = np.abs(np.nan_to_num(o["heldout"], nan=-np.inf) - np.nan_to_num(o["heldout_ic"], nan=np.inf)) \
            <= rules.anchor_score_tol
        agree_v = np.abs(np.nan_to_num(o["heldout_v"], nan=-np.inf) - np.nan_to_num(o["heldout_v_ic"], nan=np.inf)) \
            <= rules.anchor_score_tol
        rel["strain_finite"] &= agree_y
        rel["strain"] &= agree_v
        # free-intercept fits: score, gain of the slope over a constant-offset predictor, centred conditioning
        cc_ok = np.nan_to_num(o["cond_centred"], nan=np.inf) <= rules.cond_centred_max
        hy, hcy = np.nan_to_num(o["heldout_ic"], nan=-np.inf), np.nan_to_num(o["heldout_const"], nan=np.inf)
        hv, hcv = np.nan_to_num(o["heldout_v_ic"], nan=-np.inf), np.nan_to_num(o["heldout_v_const"], nan=np.inf)
        rel["strain_finite_ic"] = design & cc_ok & (hy >= rules.heldout_min) & (hy - hcy >= rules.slope_gain_min)
        rel["strain_ic"] = design & cc_ok & (hv >= rules.heldout_v_min) & (hv - hcv >= rules.slope_gain_min)
        # largest gain: noise inflates the top singular value when the true deformation is weak, so the fit
        # must also beat 'no deformation' (transported identity) clearly, unless that baseline is near exact
        hid = np.nan_to_num(o["heldout_identity"], nan=-np.inf)
        rel["gain_ic"] = rel["strain_finite_ic"] & ((hy - hid >= rules.identity_gain_min)
                                                    | (hid >= rules.identity_exact_min))
        # stretches: the weakest stretch must be stable across held-out folds; log J also its own jackknife
        rel["stretches"] = (rel["strain_finite"]
                            & (np.nan_to_num(o["log_sigma_min_se"], nan=np.inf) <= rules.stretch_se_max))
        rel["logJ"] = (rel["stretches"] & ~o["rank_loss"]
                       & (np.nan_to_num(o["logJ_se"], nan=np.inf) <= rules.stretch_se_max))
        rel["finite_rate"] = rel["strain"] & rel["logJ"]
        rel["grad_full"] = rel["strain"] & (dim == TDIM)
    rel["relmotion_split"] = rel["tangent"] & rel["relmotion"]
    # fit-based relative-motion quantities (free-intercept budget, fit scores): the fit must use all the
    # directions the mode promises (flow: the tangent rule requires nine resolved directions)
    rel["relmotion_fitsplit"] = rel["tangent"] & rel["relmotion"]
    rel["cr_relmotion_split"] = rel["tangent"] & rel["cr_relmotion"]
    rel["tilt_rate"] = rel["strain"] & (dim < TDIM)
    rel["cr_tilt_rate"] = rel["cr_strain"] & (dim < TDIM)
    rel["tilt"] = rel["strain_finite"] & (dim < TDIM) & f1.ok_tangent & (f1.dim == dim)
    # angles between endpoint subspaces: a property of the two planes, not of any regression -- released
    # where the leading subspace passes the tangent rule at both endpoints with the same rank
    rel["tilt_planes"] = rel["tangent"] & (dim < TDIM) & f1.ok_tangent & (f1.dim == dim)
    # rule of the tilt angles: plane geometry alone in thick mode; strict mode keeps its prediction gate
    rel["tilt_angles"] = rel["tilt_planes"] if rules.mode == "thick" else rel["tilt"]
    big = np.maximum(np.sqrt(_fro2(E_tan)), np.sqrt(_fro2(E_nor)))
    with np.errstate(invalid="ignore"):
        noise_ok = ((np.nan_to_num(o["attribution_noise"], nan=np.inf)
                     <= rules.attribution_noise_max * np.nan_to_num(big))
                    & (np.nan_to_num(o["attribution_rejected"], nan=np.inf)
                       <= rules.attribution_noise_max * np.nan_to_num(big)))
        rel["attribution"] = (rel["strain"] & (np.nan_to_num(sc["proj_rms"], nan=np.inf) <= rules.proj_angle_max)
                              & (np.nan_to_num(sc["proj_agree"], nan=0.0) >= rules.proj_agree_min) & noise_ok)
    rel["mass_balance"] = (rel["logJ"] & f1.ok_tangent & (f1.dim == dim) & (o["single_sheet"] == True)  # noqa: E712
                           & np.isfinite(o["mass_balance"]))
    o["release"] = rel

    vtot = np.sum(V**2)
    o["rigid"] = dict(
        Omega=kf["Omega"], omega_cond=kf["cond"], omega_unidentified=kf["unidentified"],
        rigid_fraction=(float(1.0 - np.sum((V - kf["fit"]) ** 2) / vtot)
                        if np.sqrt(np.mean(np.sum(v**2, 1))) > np.mean(vres) else float("nan")),
        R=Rrot,
        procrustes_rss=float(np.sum((X1 - X0 @ Rrot.T) ** 2)),
        procrustes_fraction=(float(1.0 - np.sum((X1 - X0 @ Rrot.T) ** 2) / np.sum((X1 - X0) ** 2))
                             if np.sqrt(np.mean(np.sum((X1 - X0) ** 2, 1))) > np.mean(vres) else float("nan")),
    )
    o["meta"] = dict(N=N, dt=dt, operator=op0.params, rules=rules.as_dict(), density_k=density_k)
    if mask:
        o = apply_release(o)
    return o


# field -> release rule
FIELD_RULE = {
    "v": "always", "travel": "always", "speed": "always", "step_angle": "always", "ke": "always",
    "ke_rigid": "always", "ke_nonrigid": "always", "cr_travel": "always",
    "ke_tan": "normal", "ke_norm": "normal", "tan_fraction": "normal",
    "cr_ke_tan": "normal", "cr_ke_norm": "normal",
    "relmotion": "relmotion", "cr_relmotion": "cr_relmotion",
    "relmotion_fit": "relmotion_split", "relmotion_res": "relmotion_split", "cr_relmotion_fit": "cr_relmotion_split",
    "cr_relmotion_res": "cr_relmotion_split", "grad_sq": "strain", "spin_sq": "strain", "cr_grad_sq": "cr_strain",
    "cr_spin_sq": "cr_strain", "explained": "relmotion_split", "F": "strain_finite", "K_d": "strain",
    "strain_sq_db": "strain", "spin_sq_db": "strain",
    "heldout": "relmotion_fitsplit", "heldout_v": "relmotion_fitsplit", "heldout_top3": "relmotion_fitsplit",
    "heldout_top6": "relmotion_fitsplit",
    # baselines that fit nothing direction-dependent (transported identity, constant offset): rank-free
    "heldout_identity": "relmotion", "heldout_const": "relmotion", "heldout_v_const": "relmotion",
    "heldout_ic": "relmotion_fitsplit", "heldout_v_ic": "relmotion_fitsplit", "intercept_rel": "relmotion_fitsplit",
    "intercept_v_rel": "relmotion_fitsplit", "cond_centred": "relmotion_fitsplit",
    "F_ic": "strain_finite_ic", "K_ic": "strain_ic", "ic_log_stretches": "strain_finite_ic",
    "ic_logJ": "strain_finite_ic", "ic_E_trace": "strain_ic", "ic_strain_sq": "strain_ic",
    "ic_spin_sq": "strain_ic", "ic_strain_sq_db": "strain_ic", "ic_spin_sq_db": "strain_ic",
    "ic_log_gain_max": "gain_ic",
    "relmotion_shift": "relmotion_fitsplit", "relmotion_slope": "relmotion_fitsplit",
    "relmotion_icres": "relmotion_fitsplit",
    "ke_cross": "always",
    "E": "strain", "strain_sq": "strain", "cr_strain_sq": "cr_strain", "E_trace": "strain", "E_shear": "strain",
    "E_top": "strain", "E_top_dir": "strain", "rate_vol": "strain", "rate_shear": "strain",
    "rate_total": "strain",
    "stretches": "stretches", "log_stretches": "stretches", "stretch_dirs": "stretches",
    "hencky": "logJ", "fin_vol": "logJ", "fin_shear": "logJ", "fin_total": "logJ",
    "rate_finite_gap": "finite_rate",
    "tilt_sq": "tilt_rate", "cr_tilt_sq": "cr_tilt_rate",
    "logJ": "logJ",
    "tilt_angles": "tilt_angles", "tilt_max": "tilt_angles", "tilt_rms": "tilt_angles", "map_angle_max": "tilt",
    "cr_tilt_angles": "tilt_angles", "cr_tilt_max": "tilt_angles", "cr_tilt_rms": "tilt_angles",
    "E_flow": "attribution", "E_normal": "attribution", "E_flow_trace": "attribution",
    "E_normal_trace": "attribution", "E_normal_skew": "attribution",
    "logrho0": "mass_balance", "logrho1": "mass_balance", "mass_balance": "mass_balance",
}

FIELD_DOC = {
    # item 1-2
    "v": "displacement log_x Y (N, 10), tangent at x; Fisher units per step",
    "travel": "Fisher travel distance |v| per step",
    "speed": "|v| / dt",
    "step_angle": "displacement angle |v| / 2 (step-size diagnostic)",
    "ke": "kinetic energy |V|^2 / 2, V = v / dt",
    "ke_tan": "kinetic energy of the tangential part T v (own retained projector)",
    "ke_norm": "kinetic energy of the normal part (Q - T) v",
    "tan_fraction": "|T v|^2 / |v|^2 (NaN when |v| is below resolution)",
    "full_rank": "retained dimension 9: no normal space, split trivially tangential (flag)",
    # item 11 (rate)
    "ke_rigid": "kinetic energy of the least-squares Killing field Omega x (rate fit, all images)",
    "ke_nonrigid": "kinetic energy of V - Omega x",
    "ke_cross": "<Omega x, V - Omega x>: ke = ke_rigid + ke_nonrigid + ke_cross; mean over all images is 0",
    "cr_travel": "|log_x(R^T Y)|: travel in the co-rotating frame of the finite Procrustes fit",
    "cr_ke_tan": "co-rotating tangential kinetic energy", "cr_ke_norm": "co-rotating normal kinetic energy",
    # items 3-4
    "relmotion": "q = sum a |w|^2 / dt^2: Dirichlet energy density of the displacement field",
    "relmotion_fit": "tr(K G K^T): Gram-weighted fitted part of q",
    "relmotion_res": "tr R_v / dt^2: residual part of q (relmotion = fit + res exactly)",
    "cr_relmotion": "q in the co-rotating frame", "cr_relmotion_fit": "its fitted part",
    "cr_relmotion_res": "its residual part",
    "F": "deformation gradient M G_T^+ (N, 10, 10), maps T_x to T_Y",
    "K_d": "displacement gradient C_v G_T^+ (N, 10, 10); rate K = K_d / dt",
    "grad_sq": "|K T|^2 = strain_sq + spin_sq + tilt_sq",
    "strain_sq": "|sym(T K T)|^2", "spin_sq": "|skew(T K T)|^2", "tilt_sq": "|N K T|^2",
    "cr_grad_sq": "co-rotating |K T|^2", "cr_strain_sq": "co-rotating strain", "cr_spin_sq": "co-rotating spin",
    "cr_tilt_sq": "co-rotating tilt",
    "explained": "tr(F G F^T) / tr B: in-sample fraction of next-frame separation explained",
    "heldout": "held-out score 1 - SSE_test / sum_test a |y|^2 of y ~ F u (can be negative)",
    "heldout_v": "held-out score of w ~ K_d u (gates the rate strain)",
    "heldout_cr": "held-out score of the co-rotating differences (gates co-rotating strain and tilt)",
    "trB": "tr B", "trFGF": "tr F G F^T", "trR": "tr R (trB = trFGF + trR exactly)",
    "step_K": "|K_d|: dimensionless step size of the gradient",
    # items 5-7
    "E": "strain rate sym(T K T) (N, 10, 10)",
    "E_trace": "volume expansion rate tr E", "E_shear": "|trace-free part of E|",
    "E_top": "largest eigenvalue of E (maximal stretching rate)", "E_top_dir": "its eigenvector (NaN if degenerate)",
    "rate_vol": "(tr E)^2 / m", "rate_shear": "|E|^2 - (tr E)^2 / m", "rate_total": "|E|^2",
    "stretches": "singular values of F on T (principal stretches), descending, (N, 9) padded NaN",
    "log_stretches": "lambda_k = log sigma_k (NaN where a stretch vanishes)",
    "stretch_dirs": "source directions of the stretches (N, 9, 10)",
    "hencky": "Hencky strain (1/2) log(F^T F) on T (N, 10, 10)",
    "logJ": "log J = sum lambda_k (NaN on rank loss)", "rank_loss": "a stretch vanished (flag)",
    "fin_vol": "(sum lambda)^2 / m", "fin_shear": "sum lambda^2 - (sum lambda)^2 / m", "fin_total": "sum lambda^2",
    "sigma_min": "smallest stretch", "sigma_max": "largest stretch",
    "log_sigma_min_se": "jackknife s.e. of log sigma_min over held-out folds", "logJ_se": "jackknife s.e. of log J",
    "rate_finite_gap": "|H - E dt| / max(|H|, |E| dt): disagreement of finite and rate strain",
    # item 8
    "E_flow": "sym(T K_T T): strain from flow within the cloud (sym grad_M V_T)",
    "E_normal": "sym(T K_N T): strain from normal motion of a curved cloud (-<II, V_N>)",
    "E_flow_trace": "tr E_flow", "E_normal_trace": "tr E_normal",
    "E_normal_skew": "|skew(T K_N T)| (zero in the continuum: a consistency check)",
    "proj_rms_angle": "P-weighted rms principal-angle sine between T_i and transported neighbour T_j",
    "proj_agree": "P-mass of neighbours with the same dimension and a released tangent",
    "attribution_noise": "split-sample error of E_flow / E_normal: |E_flow(even) - E_flow(odd)| / 2",
    # baselines and free-intercept fits (all scores: 1 - SSE / sum a |response|^2 on the same non-self
    # held-out neighbours: improvement over predicting zero, not a centred R^2)
    "heldout_identity": "score of the transported-identity predictor y_j = PT_{x_i -> Y_i} u_j (no deformation)",
    "heldout_const": "held-out score of a constant-offset predictor (training-weighted mean of the y_j)",
    "heldout_ic": "held-out score of the free-intercept fit y_j ~ A u_j + c (non-self neighbours, same folds)",
    "heldout_top3": "held-out score of the anchored fit on the 3 strongest source directions only",
    "heldout_top6": "... on the 6 strongest (heldout - heldout_top6 = held-out gain of the 3 weakest)",
    "heldout_v_const": "constant-offset score for the displacement differences w_j",
    "heldout_v_ic": "free-intercept held-out score for w_j ~ K u_j + c",
    "intercept_rel": "|c| of the free-intercept y-fit / rms |y_j| (non-self)",
    "intercept_v_rel": "|c| of the free-intercept w-fit / rms |w_j|",
    "cond_centred": "condition number of the centred design on the retained directions",
    "relmotion_shift": "shared-shift part sum a |w_bar|^2 / dt^2 of the relative motion (free-intercept fit)",
    "relmotion_slope": "linear part sum a |K (u - u_bar)|^2 / dt^2 (free-intercept fit)",
    "relmotion_icres": "unexplained part; relmotion = shift + slope + icres exactly",
    "F_ic": "free-intercept fit of y on u (N, 10, 10)", "K_ic": "free-intercept fit of w on u (N, 10, 10)",
    "ic_log_stretches": "log singular values of F_ic on the retained directions",
    "ic_log_gain_max": "largest log singular value of F_ic: top gain of the best graph-weighted linear predictor",
    "ic_logJ": "sum of ic_log_stretches (not released in flow mode)",
    "ic_E_trace": "trace of sym(T K_ic T) / dt", "ic_strain_sq": "|sym(T K_ic T)|^2 / dt^2",
    "ic_spin_sq": "|skew(T K_ic T)|^2 / dt^2",
    "ic_strain_sq_db": "ic_strain_sq minus its fold jackknife variance", "ic_spin_sq_db": "same for spin",
    "strain_sq_db": "strain_sq minus its fold jackknife variance", "spin_sq_db": "same for spin",
    "attribution_rejected": "worst-case shift of E_flow from untrusted neighbour projectors: sum a_j |V_j| |G^+ u_j|",
    # item 9
    "logrho0": "cross-fitted kNN log density at x (common dimension)", "logrho1": "same at Y",
    "mass_balance": "log rho1(Y) + log J - log rho0(x) (0 for single-sheet mass conservation)",
    "boundary0": "density boundary correction applied at x", "boundary1": "at Y",
    "foreign_frac": "fraction of target-density neighbours not shown to come from x's own sheet (any test)",
    "unassessed_frac": "... of them whose source lies beyond far_factor kernel radii (outside the validated fit)",
    "inconsistent_frac": "... of them inside the validated patch whose target offset contradicts F u",
    "single_sheet": "foreign_frac <= foreign_max (flag)",
    # item 10
    "tilt_angles": "principal angles between PT_{x->Y} T_x M_l and T_Y M_{l+1} (N, 9)",
    "tilt_max": "largest tilt angle", "tilt_rms": "rms tilt angle",
    "tilt_dim_mismatch": "source and target retained dimensions differ (flag)",
    "map_angle_max": "largest angle between F(T_x M_l) and T_Y M_{l+1} (map consistency)",
    "cr_tilt_angles": "tilt angles in the co-rotating frame", "cr_tilt_max": "", "cr_tilt_rms": "",
    # diagnostics (tgt_* are the same at frame l+1)
    "dim": "retained dimension m of G", "eig": "eigenvalues of G (N, 9), descending",
    "radius": "physical radius sqrt(sum_j P_ij d_ij^2)", "n_eff": "1 / sum_j P_ij^2",
    "cond": "lambda_1 / lambda_m", "eigengap": "lambda_m / lambda_{m+1}",
    "split_angle": "split-sample subspace angle (rad)", "one_sided": "tangential mean offset / spread",
    "drift": "sum_j P_ij u_j / tau (-> grad log rho + H_S)", "trunc_tail": "kernel second moment lost to kNN truncation",
    "competing": "radius / distance to nearest other-label point (NaN without labels)",
    "tau": "row times tau_i", "res0": "per-point resolution, frame l", "res1": "frame l+1",
    "ok_tangent0": "tangent rule at frame l", "ok_tangent1": "at frame l+1",
    "collapsed": "next-frame neighbourhood below resolution (flag)",
    "no_relative_motion": "neighbour displacement differences below resolution (flag)",
    "no_relative_motion_cr": "co-rotating displacement differences below resolution (flag)",
}


# Explicit per-mode allowlists: a field not listed is never released in that mode, whatever its rule says,
# so relaxing the tangent rule cannot release, e.g., the strain attribution or the mass balance.
_MOTION = {"v", "travel", "speed", "step_angle", "ke", "ke_rigid", "ke_nonrigid", "ke_cross", "cr_travel"}
_RELMOTION = {"relmotion", "cr_relmotion", "heldout_identity", "heldout_const", "heldout_ic", "heldout_v_const",
              "heldout_v_ic", "intercept_rel", "intercept_v_rel", "cond_centred"}
# anchored-fit quantities: withheld in flow mode (SPEC 4a); exported only as diagnostics, under names that
# cannot be confused with public ones (fields_modal: 'diag:anchored:...')
ANCHORED_DIAG = ["relmotion_fit", "relmotion_res", "cr_relmotion_fit", "cr_relmotion_res", "explained", "heldout",
                 "heldout_v", "heldout_top3", "heldout_top6"]
MODE_FIELDS = {
    "strict": None,                                   # every field under its own rule
    # flow: set from the matched fixture (tests/test_matched.py); see Rules.flow
    # flow: free-intercept fits only (the anchored fit absorbs shared offsets; fixture: 6x larger F error
    # under endpoint scatter); smallest stretches, Hencky and log J are not released (fixture: false-release
    # 30-100% at held-out 0.9 and below, and noise floors hide rank loss)
    "flow": _MOTION | _RELMOTION | {"relmotion_shift", "relmotion_slope", "relmotion_icres", "ic_log_gain_max",
                                    "ic_E_trace", "ic_strain_sq", "ic_spin_sq"},
    "thick": _MOTION | {"ke_tan", "ke_norm", "tan_fraction", "tilt_angles", "tilt_max", "tilt_rms"},
}


def release_mask(o: dict, key: str) -> np.ndarray:
    """Boolean per point: is field `key` released (its rule holds and the mode allows it)?"""
    rel = o["release"]
    allowed = MODE_FIELDS.get(o["meta"]["rules"].get("mode", "strict"))
    rule = FIELD_RULE.get(key, "always")
    if allowed is not None and key not in allowed:
        return np.zeros_like(rel["always"])
    return rel[rule]


def apply_release(o: dict) -> dict:
    """Copy of a measure_step result with every field NaN where its release rule fails or its mode does
    not allow it."""
    out = dict(o)
    for key in FIELD_RULE:
        if key not in o:
            continue
        arr = np.array(o[key], dtype=np.float64, copy=True)
        arr[~release_mask(o, key)] = np.nan
        out[key] = arr
    out["masked"] = True
    return out


def measure_step_logp(logp0: np.ndarray, logp1: np.ndarray, dt: float = 1.0, *,
                      resolution_factor: float = 4.0, **kw) -> dict:
    """measure_step on log-probability arrays (N, 10) of the same N points at two times.

    Points are X = 2 sqrt(p) in float64 (sphere.sphere_from_logp) and the per-point resolutions come from
    the input precision (sphere.input_resolution, so float32 input is not over-interpreted).
    """
    r0 = sphere.input_resolution(logp0, factor=resolution_factor)
    r1 = sphere.input_resolution(logp1, factor=resolution_factor)
    kw.setdefault("res0", r0)
    kw.setdefault("res1", r1)
    return measure_step(sphere.sphere_from_logp(logp0), sphere.sphere_from_logp(logp1), dt, **kw)


def path_length(frames: list[np.ndarray], logp: bool = True) -> np.ndarray:
    """Accumulated Fisher path length per point over consecutive frames (item 1)."""
    Xs = [sphere.sphere_from_logp(f) if logp else np.asarray(f, dtype=np.float64) for f in frames]
    return np.sum([sphere.fisher_distance(a, b) for a, b in zip(Xs[:-1], Xs[1:])], axis=0)
