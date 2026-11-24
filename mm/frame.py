"""Per-frame local analysis: the local second moment G of SPEC Section 2 and the tangent-space diagnostics
used by the release rules of SPEC Section 4.

At anchor i with weights a_j = P_ij / (2 tau_i) and sphere-log offsets u_j = log_{x_i} x_j,

    G = sum_j a_j u_j u_j^T.

G is eigendecomposed in an orthonormal basis of the sphere's tangent space at x_i (9-dimensional; the radial
direction is excluded exactly, so the retained dimension is at most 9).  The retained tangent space T is the
span of the eigenvectors with eigenvalue >= rel_threshold * lambda_max.  In the calibrated limit G -> T (all
retained eigenvalues -> 1); at finite bandwidth it is not a projector and is only used as a regression
design.

Diagnostics (per point):
  eig         the 9 eigenvalues of G, descending
  dim         retained dimension m (0 if the neighbourhood is unresolved)
  radius      physical radius sqrt(sum_j P_ij d_ij^2) (Fisher units)
  n_eff       effective neighbour count 1 / sum_j P_ij^2
  cond        lambda_1 / lambda_m on the retained space
  eigengap    lambda_m / lambda_{m+1} (inf when m = 9)
  split_angle largest principal angle (rad) between the top-m eigenspaces of G built from the even- and
              odd-ranked neighbours (rank = distance order); the two half-sample bases are kept
              (split_U) for the split-sample error of the strain attribution
  one_sided   sqrt(mu_T^T S_T^-1 mu_T) with mu = sum_j P_ij u_j, S = sum_j P_ij u_j u_j^T, both restricted
              to the retained tangent (mu_T = U^T mu, S_T = U^T S U = 2 tau diag(lambda)): the neighbourhood
              mean in units of its own spread, inside the cloud.  O(sqrt(tau) |grad log rho|) in the
              interior; about 0.8 against a flat boundary (orthant face, edge of a patch) or for a
              neighbourhood straddling a gap.  Only the tangential mean is used: on a curved cloud the
              normal mean is O(radius^2) and of order the spread itself (sqrt(m / (m + 2))), so it would
              wrongly flag every point of a uniform latitude sphere.
  drift       mu / tau_i (sphere-log version of L X without the radial part): T drift -> grad log rho and
              N drift -> H_S (mean curvature vector in Fisher space) in the interior
  trunc_tail  omitted second moment Pr(chi2_{m+2} > r^2 / 2 tau_i), r = min(kNN radius, cutoff radius), of
              the kernel for a locally uniform m-dimensional cloud (conservative: rows are union-kNN sets)
  ok_tangent  the tangent release rule (see Rules)
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from scipy.stats import chi2

from . import sphere
from .graph import Operator, gather_distances, run_batches

TDIM = sphere.AMBIENT - 1  # 9: dimension of the sphere's tangent space


# Flow-mode gates, set from the matched-regime fixture (tests/test_matched.py; numbers in its docstring).
FLOW_GATES = dict(n_eff_per_dim_strain=2.0, cond_max_strain=np.inf, heldout_min=0.5, heldout_v_min=0.5,
                  slope_gain_min=0.5, cond_centred_max=50.0, identity_gain_min=0.3)


@dataclass
class Rules:
    """Thresholds of the release rules (SPEC Section 4).  Defaults are heuristics meant to be swept."""

    rel_threshold: float = 0.1        # retained eigenvalues >= rel_threshold * lambda_max
    resolution: float = 1e-12         # Fisher-distance resolution floor (float64); per-point input
                                      # resolutions (sphere.input_resolution) are used where larger
    n_eff_min: float = 10.0           # tangent rule: n_eff >= max(n_eff_min, 2 (m + 1))
    eigengap_min: float = 2.0         # tangent rule: lambda_m / lambda_{m+1}
    split_angle_max: float = 0.35     # tangent rule: split-sample subspace angle (rad)
    one_sided_max: float = 0.5        # tangent rule: one-sidedness
    trunc_tail_max: float = 0.01      # tangent rule: kernel mass lost to kNN truncation
    competing_ratio: float = 0.5      # tangent rule, only if labels are given (they must identify competing
                                      # components; otherwise the criterion is reported as unassessed):
                                      # radius <= ratio * distance to the nearest point of another label
    n_eff_per_dim_strain: float = 5.0 # strain rule: n_eff >= n_eff_per_dim_strain * (m + 1)
    cond_max_strain: float = 5.0      # strain rule: lambda_1 / lambda_m
    heldout_min: float = 0.5          # finite-strain rule: held-out score of the fit y ~ F u
    heldout_v_min: float = 0.5        # rate-strain rule: held-out score of the fit w ~ K_d u (or w = 0)
    stretch_se_max: float = 0.1       # log J / Hencky rule: jackknife s.e. of log sigma_min over folds
    rank_tol: float = 1e-6            # singular value <= rank_tol * sigma_max (or image below resolution)
                                      # is reported as rank loss
    proj_angle_max: float = 0.35      # attribution rule: P-weighted rms sine of principal angles between
                                      # T_i and the transported neighbour projectors (equal-rank neighbours)
    proj_agree_min: float = 0.99      # attribution rule: P-mass of neighbours with same m and tangent ok
    attribution_noise_max: float = 0.2  # attribution rule: split-sample error of the attribution
                                      # (step.FIELD_DOC['attribution_noise']) <= this times the larger term
    foreign_max: float = 0.05         # mass balance: largest fraction of target-density neighbours not shown
                                      # to come from the anchor's own sheet (step.py, single-sheet check)
    far_factor: float = 3.0           # single-sheet check: tangential bound far_factor r_k(Y) / sigma_min,
                                      # normal bound and validated patch far_factor source radii, allowed
                                      # misprediction far_factor (resolution + extrapolated fit error)
    n_folds: int = 5                  # held-out folds (by distance rank mod n_folds)
    anchor_score_tol: float = np.inf  # strain rules: |held-out(anchored) - held-out(free intercept)| <= this
    slope_gain_min: float = 0.0       # intercept-fit rules: held-out(intercept fit) - held-out(constant) >= this
    cond_centred_max: float = np.inf  # intercept-fit rules: condition number of the centred design
    identity_gain_min: float = 0.0    # largest-gain rule: held-out(intercept fit) - held-out(identity) >= this ...
    identity_exact_min: float = 0.95  # ... or held-out(identity) >= this (no deformation, near noise-free)
    mode: str = "strict"              # name of the rule set (see the constructors below)

    def as_dict(self) -> dict:
        return asdict(self)

    # ---- named rule sets -------------------------------------------------------------------------------
    # The defaults ("strict") ask the neighbourhood to look like a small patch of a thin submanifold: an
    # eigengap below the retained directions, a stable split-sample tangent, a neighbourhood centred on its
    # anchor and a kernel fully inside the kNN rows.  The lens clouds at sites stem..b8 fail that test at
    # every scale 10,000 points resolve: the local second moment has no eigengap (median spectrum relative
    # to the largest about 1, .79, .65, .50, .34, .23, .14, .08, .05 at b7), every point sits at the edge of
    # its own neighbourhood (one_sided ~ 0.85, as for any point of a 9-dimensional blob) and the kernel is
    # truncated.  These are properties of a 9-dimensional anisotropic blob, not estimator failures, so two
    # further rule sets are defined, each released only to the accuracy its matched fixture supports
    # (tests/test_matched.py).

    @classmethod
    def strict(cls, **kw) -> "Rules":
        """The default rules: thin-submanifold reading, all tangent criteria."""
        return cls(mode="strict", **kw)

    @classmethod
    def flow(cls, **kw) -> "Rules":
        """Full-dimensional reading: the cloud is treated as a 9-dimensional blob in S^9.

        The tangent space is the whole sphere tangent (every direction above the resolution floor is
        retained, rel_threshold = 0), so normal quantities are identically zero and are reported as not
        applicable (the 'normal' release requires m < 9).  The deformation gradient is the weighted
        least-squares fit in all 9 directions.  No eigengap, split-angle, one-sidedness or truncation
        requirement: they test for a thin submanifold, which this reading does not assume.  What remains:
        a resolved neighbourhood, the effective neighbour count, the conditioning of the 9-dimensional
        design and the held-out score of the fit, with the thresholds set from the matched 9-dimensional
        fixture (tests/test_matched.py).  Every quantity is a quantity *at the scale of the neighbourhood*
        (radius reported alongside), not a pointwise derivative.

        What the fixture decided (tests/matched.py, seeds 0 and 1, k = 256):
          * The anchored fit (through the anchor) absorbs any offset shared by all neighbour responses
            (e.g. the anchor's own endpoint scatter) as A + c mu^T S^-1; with one-sidedness ~0.85 this is
            large: under endpoint scatter at held-out 0.9 its prediction-weighted error of F is 6% against
            1% for the free-intercept fit (and 1.2% / 1.4% under independent edge noise of the same size).
            Flow mode therefore releases free-intercept fits only (F_ic, K_ic and their derived scalars).
          * Gates (FLOW_GATES): held-out score of the intercept fit >= 0.5, its gain over the constant-offset
            predictor >= 0.5 (shuffled targets and a piecewise vertex collapse otherwise pass), centred design
            condition number <= 50, n_eff >= 2 (m + 1); the largest gain additionally must beat the
            transported-identity predictor by 0.3 (else noise inflates it by ~0.2 on a non-deforming cloud),
            EXCEPT where the no-deformation score itself is >= 0.95 (identity_exact_min): there the cloud is
            essentially undeformed and near noise-free (a rigid rotation scores 0.97), the 0.3 margin cannot be
            met, and the gain is released without it.  No released value of the lens grid used this exception.
          * Released, with released-point false-release rates <= 10% at |error| tolerances: largest log
            gain 0.1; expansion rate (trace of sym K) 0.2 x median |E|; |sym K|^2 and |skew K|^2 20% of
            their cloud medians.  Exceptions: a strongly curved map (variation length 0.1) gives 15-29%
            (secant at scale r versus derivative), scatter at held-out 0.7 gives up to 21% on 23% coverage.
          * Not released: smallest log-stretches, Hencky measures, log J (false-release 30-100% at held-out
            0.9 and below; scatter hides an exact rank-8 map as log-stretch ~ -3 and moves a -5 stretch to
            -3), and the anchored fitted / residual split, whose fitted fraction is inflated by up to +0.14.
            The free-intercept budget (shared shift + slope + residual) is exact and is the headline split.
        """
        base = dict(mode="flow", rel_threshold=0.0, eigengap_min=0.0, split_angle_max=np.inf,
                    one_sided_max=np.inf, trunc_tail_max=np.inf, **FLOW_GATES)
        base.update(kw)
        return cls(**base)

    @classmethod
    def thick(cls, **kw) -> "Rules":
        """Thick/thin reading: retained subspace by the relative eigenvalue threshold (0.1), with the
        eigengap, split-angle and neighbour-count requirements kept but one-sidedness and truncation
        dropped (both fail for any point of a full-dimensional blob).  Used only for the tangential /
        normal split of the displacement and for tilt, read as motion along the cloud's thick directions
        versus across its thin ones; strain-type fields are not read in this mode."""
        base = dict(mode="thick", one_sided_max=np.inf, trunc_tail_max=np.inf)
        base.update(kw)
        return cls(**base)


@dataclass
class Frame:
    """Result of analyse_frame; arrays are per point."""

    op: Operator
    eig: np.ndarray        # (N, 9)
    basis: np.ndarray      # (N, 10, 9) ambient eigenvectors of G, columns sorted by eigenvalue (desc)
    dim: np.ndarray        # (N,)
    radius: np.ndarray
    n_eff: np.ndarray
    cond: np.ndarray
    eigengap: np.ndarray
    split_angle: np.ndarray
    one_sided: np.ndarray
    drift: np.ndarray      # (N, 10)
    trunc_tail: np.ndarray
    competing: np.ndarray  # radius / distance to nearest point with another label (nan if no labels)
    ok_tangent: np.ndarray
    rules: Rules
    res: np.ndarray        # (N,) Fisher-distance resolution used for this frame
    split_U: tuple | None = None   # (U_even, U_odd): top-dim eigenvectors of the two half-sample G's
    res_nb: np.ndarray | None = None   # (N,) neighbourhood offset resolution sqrt(sum_j P_ij (r_i + r_j)^2)
    eig_c: np.ndarray | None = None    # (N, 9) eigenvalues of the centred second moment
    rank_stable: np.ndarray | None = None  # (N,) same rank at cutoff/2, 2 cutoff and for the centred moment

    @property
    def X(self) -> np.ndarray:
        return self.op.X

    @property
    def mask(self) -> np.ndarray:
        """(N, 9) boolean: which basis columns are retained."""
        return np.arange(TDIM)[None, :] < self.dim[:, None]

    def U(self) -> np.ndarray:
        """Retained basis with non-retained columns zeroed, (N, 10, 9)."""
        return self.basis * self.mask[:, None, :]

    def T(self) -> np.ndarray:
        """Tangent projectors (N, 10, 10)."""
        U = self.U()
        return U @ np.swapaxes(U, 1, 2)

    def ginv(self, rows: np.ndarray | None = None) -> np.ndarray:
        """Truncated pseudo-inverse G_T^+ = U diag(1/lambda) U^T over the retained eigenspace."""
        sl = slice(None) if rows is None else rows
        return truncated_pinv(self.basis[sl], self.eig[sl], self.dim[sl])


def truncated_pinv(basis: np.ndarray, eig: np.ndarray, dim: np.ndarray) -> np.ndarray:
    """G_T^+ = sum_{k < dim} lambda_k^-1 b_k b_k^T for eigenpairs sorted by decreasing eigenvalue.

    basis (b, n, r), eig (b, r), dim (b,) -> (b, n, n).  Rows with dim 0 give the zero matrix.
    """
    msk = np.arange(eig.shape[1])[None, :] < np.asarray(dim)[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = np.where(msk, 1.0 / np.where(msk, eig, 1.0), 0.0)
    return (basis * inv[:, None, :]) @ np.swapaxes(basis, 1, 2)


def regress(a: np.ndarray, u: np.ndarray, resp: np.ndarray, Ginv: np.ndarray) -> dict:
    """Weighted least squares of responses on source offsets, truncated to the retained subspace.

    a (b, k) weights, u (b, k, n) source offsets, resp (b, k, p) responses (kept whole), Ginv (b, n, n)
    the truncated pseudo-inverse of G = sum a u u^T.  Returns
        A      = (sum a resp u^T) G_T^+        (b, p, n): rows output, columns input
        total  = sum a |resp|^2
        fit    = tr(A G A^T)                    (Gram-weighted fitted energy, not |A|^2)
        res    = sum a |resp - A u|^2          (from the residual vectors themselves)
        Rm     = sum a r r^T                    (b, p, p) residual second moment.
    With G_T^+ built from the eigenpairs of the same G, A G A^T = M G_T^+ M^T, so the budget
    total = fit + res (and B = A G A^T + Rm) is exact up to rounding.
    """
    G = _weighted_moment(a, u)
    M = _weighted_moment(a, u, resp)
    A = M @ Ginv
    r = resp - u @ np.swapaxes(A, 1, 2)
    return dict(A=A, total=np.einsum("bk,bkp,bkp->b", a, resp, resp),
                fit=np.einsum("bpi,bij,bpj->b", A, G, A),
                res=np.einsum("bk,bkp,bkp->b", a, r, r),
                Rm=_weighted_moment(a, r))


def _eig_in_tangent(G: np.ndarray, E9: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    G9 = np.swapaxes(E9, 1, 2) @ G @ E9
    G9 = 0.5 * (G9 + np.swapaxes(G9, 1, 2))
    lam, V = np.linalg.eigh(G9)
    lam, V = lam[:, ::-1], V[:, :, ::-1]
    return lam, E9 @ V


def _weighted_moment(a: np.ndarray, u: np.ndarray, v: np.ndarray | None = None) -> np.ndarray:
    v = u if v is None else v
    return np.swapaxes(a[..., None] * v, 1, 2) @ u  # sum_j a_j v_j u_j^T  (rows: v, cols: u)


def retained_dim(lam: np.ndarray, rel_threshold: float, resolved: np.ndarray, lam_floor: np.ndarray) -> np.ndarray:
    """Number of eigenvalues >= rel_threshold * lambda_max that are also above the absolute floor
    (an eigenvalue lambda is a spread sqrt(2 tau lambda) in Fisher units; lam_floor = res^2 / (2 tau))."""
    lmax = lam[:, 0]
    keep = (lam >= rel_threshold * lmax[:, None]) & (lam > lam_floor[:, None])
    m = np.sum(keep, axis=1)
    return np.where(resolved & (lmax > lam_floor), m, 0)


def analyse_frame(op: Operator, rules: Rules | None = None, labels: np.ndarray | None = None,
                  resolution: np.ndarray | float | None = None, budget: int = 32768, n_jobs: int = 4) -> Frame:
    """Local second moments and tangent diagnostics for every point of the operator's cloud.

    resolution: per-point (or scalar) Fisher-distance resolution (default rules.resolution); see
    sphere.input_resolution.  A neighbourhood whose physical radius is not above it is unresolved (dim 0).
    """
    rules = rules or Rules()
    X = op.X
    N = X.shape[0]
    res = np.broadcast_to(np.asarray(rules.resolution if resolution is None else resolution,
                                     dtype=np.float64), (N,)).copy()
    eig = np.zeros((N, TDIM))
    basis = np.zeros((N, X.shape[1], TDIM))
    radius = np.zeros(N)
    n_eff = np.zeros(N)
    split_basis = [np.zeros((N, X.shape[1], TDIM)), np.zeros((N, X.shape[1], TDIM))]
    split_eig = [np.zeros((N, TDIM)), np.zeros((N, TDIM))]
    mu = np.zeros((N, X.shape[1]))
    eig_c = np.zeros((N, TDIM))

    rn = np.zeros(N)

    def batch(rows, idx, w):
        xi = X[rows]
        u = sphere.log_map(xi[:, None, :], X[idx])
        a = w / (2.0 * op.tau[rows])[:, None]
        G = _weighted_moment(a, u)
        E9 = sphere.tangent_basis(xi)
        lam, B = _eig_in_tangent(G, E9)
        eig[rows], basis[rows] = lam, B
        d = gather_distances(op, rows, idx.shape[1])
        radius[rows] = np.sqrt(np.sum(w * d**2, axis=1))
        n_eff[rows] = 1.0 / np.sum(w**2, axis=1)
        pos = np.arange(idx.shape[1])[None, :]
        for h in (0, 1):
            ah = a * ((pos % 2) == h)
            lh, Bh = _eig_in_tangent(_weighted_moment(ah, u), E9)
            split_eig[h][rows], split_basis[h][rows] = lh, Bh
        mu[rows] = np.einsum("bk,bki->bi", w, u)
        # centred second moment G - mu mu^T / (2 tau) (sum_j a_j = 1 / (2 tau)): the dispersion about the
        # neighbourhood mean, for the thick-mode check that a leading direction of G is not just the offset
        # of an anchor at the edge of its neighbourhood
        mur = mu[rows]
        Gc = G - mur[:, :, None] * mur[:, None, :] / (2.0 * op.tau[rows])[:, None, None]
        eig_c[rows] = _eig_in_tangent(Gc, E9)[0]
        rn[rows] = np.sqrt((w * (res[rows][:, None] + res[idx]) ** 2).sum(1))

    run_batches(op, batch, budget=budget, n_jobs=n_jobs)

    # resolution of the neighbourhood's offsets: P-weighted rms of r_i + r_j (anchor and neighbour both
    # count); a direction is retained only if its spread sqrt(2 tau lambda) exceeds it
    resolved = radius > rn
    lam_floor = rn**2 / (2.0 * op.tau)
    dim = retained_dim(eig, rules.rel_threshold, resolved, lam_floor)
    msk = np.arange(TDIM)[None, :] < dim[:, None]
    ar = np.arange(N)
    lam_m = np.where(dim > 0, eig[ar, np.maximum(dim - 1, 0)], np.nan)
    lam_next = np.where(dim < TDIM, eig[ar, np.minimum(dim, TDIM - 1)], 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        cond = np.where(dim > 0, eig[:, 0] / lam_m, np.nan)
        eigengap = np.where(dim == TDIM, np.inf,
                            np.where(lam_next > 0, lam_m / np.maximum(lam_next, 0.0), np.inf))
        eigengap = np.where(dim > 0, eigengap, np.nan)

    # split-sample subspace angle: sin(theta_max) = || (I - P_odd) U_even ||_2 on the top-m spaces
    Ue = split_basis[0] * msk[:, None, :]
    Uo = split_basis[1] * msk[:, None, :]
    resid = Ue - Uo @ (np.swapaxes(Uo, 1, 2) @ Ue)
    s = np.linalg.norm(resid, ord=2, axis=(1, 2))
    split_angle = np.where(dim > 0, np.arcsin(np.clip(s, 0.0, 1.0)), np.nan)
    # each half must itself support rank m, otherwise the comparison of top-m planes is meaningless
    for h in (0, 1):
        dh = retained_dim(split_eig[h], rules.rel_threshold, resolved, lam_floor)
        split_angle = np.where((dim > 0) & (dh < dim), np.pi / 2, split_angle)

    # one-sidedness inside the retained tangent: sqrt(sum_k (U_k . mu)^2 / (2 tau lambda_k))
    coords = np.einsum("nik,ni->nk", basis, mu)
    with np.errstate(invalid="ignore", divide="ignore"):
        z2 = np.where(msk, coords**2 / (2.0 * op.tau[:, None] * np.where(msk, eig, 1.0)), 0.0)
    one_sided = np.where(dim > 0, np.sqrt(z2.sum(1)), np.nan)
    drift = mu / op.tau[:, None]

    rcut = np.sqrt(op.params.get("cutoff", 25.0) * 4.0 * op.tau)
    r_trunc = np.minimum(op.knn_radius, rcut)
    with np.errstate(divide="ignore", invalid="ignore"):
        trunc_tail = np.where(dim > 0, chi2.sf(r_trunc**2 / (2.0 * op.tau), df=dim + 2), np.nan)

    competing = np.full(N, np.nan)
    if labels is not None:
        competing = radius / _distance_to_other_label(X, np.asarray(labels))

    # rank stability (thick mode): the same retained dimension at half and double the relative cutoff and
    # for the centred second moment
    rank_stable = ((retained_dim(eig, rules.rel_threshold / 2, resolved, lam_floor) == dim)
                   & (retained_dim(eig, min(2 * rules.rel_threshold, 1.0), resolved, lam_floor) == dim)
                   & (retained_dim(eig_c, rules.rel_threshold, resolved, lam_floor) == dim))

    n_need = np.maximum(rules.n_eff_min, 2.0 * (dim + 1))
    ok = (resolved & (dim > 0) & (n_eff >= n_need) & (eigengap >= rules.eigengap_min)
          & (split_angle <= rules.split_angle_max) & (one_sided <= rules.one_sided_max)
          & (trunc_tail <= rules.trunc_tail_max))
    if rules.mode == "flow":
        ok &= dim == TDIM            # nine individually resolved directions, or withheld
    if rules.mode == "thick":
        ok &= rank_stable
    if labels is not None:
        ok &= competing <= rules.competing_ratio
    return Frame(op=op, eig=eig, basis=basis, dim=dim, radius=radius, n_eff=n_eff, cond=cond,
                 eigengap=eigengap, split_angle=split_angle, one_sided=one_sided, drift=drift,
                 trunc_tail=trunc_tail, competing=competing, ok_tangent=ok, rules=rules, res=res,
                 split_U=(Ue, Uo), res_nb=rn, eig_c=eig_c, rank_stable=rank_stable)


def _distance_to_other_label(X: np.ndarray, labels: np.ndarray) -> np.ndarray:
    from sklearn.neighbors import NearestNeighbors

    out = np.full(X.shape[0], np.inf)
    for lab in np.unique(labels):
        inside = labels == lab
        if inside.all():
            continue
        nn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(X[~inside])
        _, j = nn.kneighbors(X[inside])
        out[inside] = sphere.fisher_distance(X[inside], X[~inside][j[:, 0]])
    return out


def rerule(frame: Frame, rules: Rules, labels: np.ndarray | None = None) -> Frame:
    """The same frame under another rule set with the same retained-dimension rule (rel_threshold and
    resolution): only the tangent release is recomputed, from the stored diagnostics (no new batch pass).
    Used to read one analysis under 'strict' and 'thick' without recomputing it."""
    from dataclasses import replace

    if rules.rel_threshold != frame.rules.rel_threshold or rules.mode == "flow" or frame.rules.mode == "flow":
        raise ValueError("rerule needs the same retained-dimension rule (use analyse_frame for flow)")
    dim = frame.dim
    resolved = frame.radius > (frame.res_nb if frame.res_nb is not None else frame.res)
    n_need = np.maximum(rules.n_eff_min, 2.0 * (dim + 1))
    with np.errstate(invalid="ignore"):
        ok = (resolved & (dim > 0) & (frame.n_eff >= n_need) & (frame.eigengap >= rules.eigengap_min)
              & (frame.split_angle <= rules.split_angle_max) & (frame.one_sided <= rules.one_sided_max)
              & (frame.trunc_tail <= rules.trunc_tail_max))
        if labels is not None:
            ok &= frame.competing <= rules.competing_ratio
    if rules.mode == "thick":
        ok &= frame.rank_stable
    return replace(frame, ok_tangent=ok, rules=rules)
