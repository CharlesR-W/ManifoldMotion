"""Operator-independent density estimates for the mass-balance check (SPEC item 9).

Estimator: cross-fitted k-nearest-neighbour density in a given (common) dimension m, with a half-space
boundary correction.

* Cross-fitting: points are split into two folds (a fixed random split; the same split must be used in
  both frames).  The density at a point of fold A is estimated from the points of fold B only, and vice
  versa, so a point never counts itself and the estimate shares no sample with the point's own weight.
* kNN estimator (Kozachenko-Leonenko form): with r_k the Fisher distance to the k-th other-fold neighbour
  and n the size of the other fold (the ball mass at r_k is Beta(k, n + 1 - k), E log = psi(k) - psi(n+1)),
      log rho = psi(k) - psi(n + 1) - log V_m - m log r_k - log f,
  where V_m is the volume of the unit m-ball and rho is a probability density w.r.t. the cloud's
  m-dimensional Fisher volume (geodesic-ball curvature corrections are O(r^2) and ignored).  psi(k) makes
  log rho unbiased for a locally uniform density (E log r_k^m = psi(k) - log(n V_m rho)).
* Boundary correction f: if the anchor is a distance delta from a locally flat boundary, the kNN ball is
  cut by a half-space and holds only the fraction f(t) = 1 - (1/2) I_{1-t^2}((m+1)/2, 1/2) (t = delta / r,
  I the regularised incomplete beta function) of a full ball's mass.  The centroid of the cut ball sits at
      c / r = (V_{m-1} / V_m) (1 - t^2)^((m+1)/2) / ((m + 1) f(t))
  from the anchor, pointing away from the boundary.  We measure the centroid of the k_b = 4k nearest
  other-fold neighbours (projected onto the given tangent basis).  Under a uniform interior ball each
  centroid coordinate has variance r^2 / (k_b (m + 2)), so |c|^2 k_b (m + 2) / r^2 is approximately chi2_m;
  only where it exceeds the chi2_m quantile 1 - boundary_alpha (default 1e-3) do we invert the formula for
  delta and divide the density at radius r_k by f(delta / r_k) (f = 1 when delta >= r_k).  Elsewhere f = 1.
  A centroid beyond the half-ball maximum (t = 0) is incompatible with a single flat face (a corner, e.g.
  near a simplex vertex where several faces are active): its density is withheld (NaN, flag
  `incompatible`).  A density gradient also shifts the centroid (by r^2 grad log rho / (m + 2)); that
  confound is not removed.  Fixed k is not consistent pointwise (the log-density noise stays ~ 1/sqrt(k));
  a convergence claim needs k -> infinity, k / n -> 0.  Zero r_k (ties, atoms) gives NaN, not a floor.
"""
from __future__ import annotations

import numpy as np
from scipy.special import betainc, digamma, gammaln
from scipy.stats import chi2
from sklearn.neighbors import NearestNeighbors

from . import sphere


def log_unit_ball_volume(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float64)
    return 0.5 * m * np.log(np.pi) - gammaln(0.5 * m + 1.0)


def cut_ball_fraction(t: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Fraction of the unit m-ball on the side {z_n >= -t} of a hyperplane at distance t (0 <= t <= 1)."""
    t = np.clip(t, 0.0, 1.0)
    return 1.0 - 0.5 * betainc(0.5 * (m + 1.0), 0.5, 1.0 - t**2)


def cut_ball_centroid(t: np.ndarray, m: np.ndarray) -> np.ndarray:
    """|centroid| / r of the cut unit ball (see module docstring)."""
    t = np.clip(t, 0.0, 1.0)
    m = np.asarray(m, dtype=np.float64)
    ratio = np.exp(gammaln(0.5 * m + 1.0) - gammaln(0.5 * (m + 1.0))) / np.sqrt(np.pi)  # V_{m-1} / V_m
    return ratio * (1.0 - t**2) ** (0.5 * (m + 1.0)) / ((m + 1.0) * cut_ball_fraction(t, m))


def invert_centroid(g: np.ndarray, m: np.ndarray, iters: int = 60) -> np.ndarray:
    """t in [0, 1] with cut_ball_centroid(t, m) = g (g decreasing in t); clipped at the ends."""
    lo = np.zeros_like(g, dtype=np.float64)
    hi = np.ones_like(g, dtype=np.float64)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        above = cut_ball_centroid(mid, m) > g
        lo = np.where(above, mid, lo)
        hi = np.where(above, hi, mid)
    return 0.5 * (lo + hi)


def fold_split(N: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    f = np.zeros(N, dtype=np.int64)
    f[rng.permutation(N)[: N // 2]] = 1
    return f


def knn_log_density(X: np.ndarray, dim: np.ndarray, basis: np.ndarray, folds: np.ndarray, k: int = 32,
                    boundary: bool = True, boundary_alpha: float = 1e-3, n_jobs: int = 4,
                    res: np.ndarray | float = 0.0) -> dict:
    """Cross-fitted kNN log-density in per-point dimension `dim` (see module docstring).

    basis: (N, n, >=max dim) tangent bases whose first dim[i] columns span the tangent at X[i] (used only
    for the boundary centroid).  Returns dict(logrho, r_k, f, boundary, incompatible, nbr) with NaN where
    dim == 0 or r_k is not above the per-point resolution `res`; nbr (N, k) are the global indices of the
    k other-fold neighbours that set r_k (used by the single-sheet check of the mass balance).
    """
    res = np.broadcast_to(np.asarray(res, dtype=np.float64), (X.shape[0],))
    X = np.asarray(X, dtype=np.float64)
    N = X.shape[0]
    dim = np.asarray(dim)
    kb = 4 * k
    logrho = np.full(N, np.nan)
    rk = np.full(N, np.nan)
    frac = np.ones(N)
    bflag = np.zeros(N, dtype=bool)
    incompat = np.zeros(N, dtype=bool)
    nbr = np.full((N, k), -1, dtype=np.int64)
    for fold in (0, 1):
        q = np.where(folds == fold)[0]
        o = np.where(folds != fold)[0]
        if q.size == 0 or o.size < k:
            continue
        kk = min(kb, o.size)
        nn = NearestNeighbors(n_neighbors=kk, algorithm="kd_tree", n_jobs=n_jobs).fit(X[o])
        _, j = nn.kneighbors(X[q])
        Xn = X[o][j]
        d = sphere.fisher_distance(X[q][:, None, :], Xn)
        srt = np.argsort(d, axis=1)
        d = np.take_along_axis(d, srt, 1)
        Xn = np.take_along_axis(Xn, srt[..., None], 1)
        m = dim[q].astype(np.float64)
        r = d[:, k - 1]
        rk[q] = r
        nbr[q] = o[np.take_along_axis(j, srt, 1)[:, :k]]
        f = np.ones(q.size)
        if boundary:
            u = sphere.log_map(X[q][:, None, :], Xn)                     # (nq, kk, n)
            B = basis[q] * (np.arange(basis.shape[2])[None, :] < dim[q][:, None])[:, None, :]
            c = np.einsum("bni,bn->bi", B, u.mean(axis=1))             # tangent coordinates of centroid
            rb = d[:, -1]
            mm_ = np.maximum(m, 1.0)
            with np.errstate(divide="ignore", invalid="ignore"):
                g = np.linalg.norm(c, axis=1) / rb
                stat = g**2 * kk * (m + 2.0)
                sig = (stat > chi2.isf(boundary_alpha, df=mm_)) & (m > 0) & (rb > 0)
            gmax = cut_ball_centroid(np.zeros_like(mm_), mm_)
            bad = sig & (g > gmax)
            t_b = invert_centroid(np.where(sig, np.minimum(g, gmax), 0.0), mm_)
            delta = t_b * rb
            with np.errstate(divide="ignore", invalid="ignore"):
                f = np.where(sig, cut_ball_fraction(np.where(r > 0, delta / r, 1.0), mm_), 1.0)
            f = np.where(bad, np.nan, f)
            bflag[q] = sig
            incompat[q] = bad
        frac[q] = f
        with np.errstate(divide="ignore", invalid="ignore"):
            lr = digamma(k) - digamma(o.size + 1) - log_unit_ball_volume(m) - m * np.log(r) - np.log(f)
        logrho[q] = np.where((m > 0) & (r > res[q]), lr, np.nan)
    return dict(logrho=logrho, r_k=rk, f=frac, boundary=bflag, incompatible=incompat, nbr=nbr)
