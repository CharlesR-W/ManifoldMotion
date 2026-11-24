"""Neighbour graphs and Markov operators on a sampled cloud of sphere points (SPEC Section 2, "Operator").

Fixed bandwidth (the baseline, kind="fixed")
--------------------------------------------
    K_ij   = exp(-d_ij^2 / (4 tau))                    d = Fisher distance, on a symmetric graph (below)
    q_i    = sum_j K_ij                                 (self loop included, K_ii = 1)
    K^a_ij = K_ij / (q_i q_j)^(1/2)                     diffusion-maps degree correction, alpha = 1/2
    P      = D^-1 K^a,  D_i = sum_j K^a_ij              row normalisation
    L      = (P - I) / tau
On a smooth interior patch of an m-dimensional cloud with density rho (w.r.t. its Fisher volume),
L f -> Delta f + grad log rho . grad f, and the carre du champ
Gamma(f, h)_i = (1 / 2 tau) sum_j P_ij (f_j - f_i)(h_j - h_i) -> <grad f, grad h>
(the Gaussian exp(-|z|^2 / 4 tau) has per-coordinate variance 2 tau; with alpha the drift is
2 (1 - alpha) grad log rho).  Reversible weights: pi_i = D_i tau_i (D P is symmetric).

Adaptive (kNN) bandwidth (kind="adaptive")
------------------------------------------
Exact construction:
    sigma_i  = sqrt(mean_{j in kNN_bw(i)} d_ij^2), the rms Fisher distance to the k_bw nearest neighbours
               other than i itself (no density pilot and no dimension are needed), floored at sigma_floor.
               Bandwidth units: Fisher distance; eps is dimensionless.
    K_ij     = exp(-d_ij^2 / (4 eps sigma_i sigma_j))                         symmetric in i, j
    q_i      = sum_j K_ij
    K^a_ij   = K_ij / ((q_i q_j)^(1/2) sigma_i sigma_j)                      degree correction + sigma weight
    P        = D^-1 K^a,  D_i = sum_j K^a_ij
    tau_i    = eps sigma_i^2                                                  row time
    L        = diag(1 / tau_i) (P - I)
Limit (smooth positive sigma, interior point, m-dimensional cloud, density rho).  Write z = y - x,
s = sigma(x), b = grad log sigma, S = 2 eps s^2 (per-coordinate variance of the leading Gaussian).  Then
1 / (sigma(x) sigma(y)) = s^-2 (1 - z.b + O(z^2)), so the kernel is
exp(-|z|^2 / 4 eps s^2) (1 + |z|^2 (z.b) / (4 eps s^2) + even terms); its first moment is
S (m + 2) b / 2 (Gaussian identity E[z |z|^2 (z.b)] = (m + 2) S^2 b).  The rest of the weight on y is
W(y) = rho(y) / (q(y)^(1/2) sigma(y)) with q(y) ∝ rho(y) sigma(y)^m (1 + O(eps)), whose first moment is
S grad log W = S (grad log rho / 2 - (m + 2) b / 2).  The mean displacement is therefore
mu = S grad log rho / 2, the second moment S I, and
    (P f - f)(x) = (S / 2) (Delta f + grad log rho . grad f) + O(eps^2),
so L f -> Delta f + grad log rho . grad f for every smooth sigma and every m: the same limit as the fixed
operator, with Gamma(f, h) -> <grad f, grad h> and G -> T under a_j = P_ij / (2 tau_i).  The 1/(sigma_i
sigma_j) factor is what makes the limit dimension-free: without it the drift is grad log(rho sigma^2)
(e.g. for sigma ∝ rho^(-1/m) the drift is (1 - 2/m) grad log rho, which tests detect).  O(eps) relative
corrections to q only enter at O(eps) in L, as usual.  The kNN sigma is a noisy estimate of a smooth
function; its fluctuations add variance (and only higher-order bias).  Reversible weights pi_i = D_i tau_i.

Graph and truncation (both kinds)
---------------------------------
The k nearest neighbours of every point in Fisher distance (self included) are found with a KD tree on the
chord coordinates (chord length is monotone in Fisher distance, and the tree computes coordinate
differences, so tiny distances are not lost to the cancellation of |a|^2 + |b|^2 - 2ab).  The graph is the
union of the kNN lists (j in kNN(i) or i in kNN(j)), so K is exactly symmetric; edges whose exponent
d^2 / (4 tau_ij) exceeds `cutoff` are dropped.  A radial cutoff at fixed dimensionless radius biases the
diffusion constant by the factor Pr(chi2_{m+2} <= 2 cutoff) / Pr(chi2_m <= 2 cutoff), which does not vanish
as tau -> 0; with the default cutoff = 25 (kernel < 1.4e-11) it is below 1e-6 for m <= 9 (it was 6e-4 at
m = 9 for a cutoff of 16).  Where the kNN radius is not large against the kernel width, the kernel is
truncated much more severely; the frame analysis reports the omitted second moment of a locally uniform
m-dimensional cloud, Pr(chi2_{m+2} > r^2 / (2 tau_i)) at r = min(kNN radius, cutoff radius), as
`trunc_tail` (a conservative containment diagnostic: a union-kNN row is not a ball).

Consistency regime: the limits above need tau -> 0 with n tau^(m/2) -> infinity (n tau^(m/2 + 1) -> infinity
for the pointwise drift) and kNN radius / sqrt(tau) -> infinity at interior points; for the adaptive
operator also k_bw -> infinity with k_bw / n -> 0 (a fixed-k_bw pilot is a noisy, not a smooth, sigma, and an
active floor is a kink).  The release diagnostics are finite-sample checks, not substitutes.

Containment versus neighbour count.  For a locally uniform m-dimensional cloud the kernel's second moment
stays inside the k-NN rows only if r_k^2 / (2 tau) exceeds the chi2_{m+2} upper quantile; with the kNN pilot
r_k / sigma ~ (k / k_bw)^(1/m) sqrt((m + 2) / m), so at fixed k the admissible eps falls quickly with m, and
the kernel's effective neighbour count n_eff ~ k_bw (8 pi eps m / (m + 2))^(m/2) / V_m falls with it.  With
k = 128, k_bw = 16, eps = 0.25 the kernel is contained for m <= 3; at m = 6..9 no eps gives both containment
and n_eff >~ 10 for k of a few hundred.  This is a property of the data's dimension, not a defect; the
trunc_tail and n_eff diagnostics report it.

Validation (tests/test_operator.py, tests/test_algebra.py): both kinds satisfy the exact identities
(row-stochastic, pi_i L_ij symmetric, Gamma = (1/2)[L(fh) - f Lh - h Lf], x.LX = -tr Gamma(X, X)) and the
continuum checks on a non-uniform great 2-sphere (Gamma on linear functions, L on a coordinate along the
density gradient, L X, G -> T); exp(-d^2 / 2 tau), alpha = 0 or 1, and the adaptive kernel without the
1 / (sigma_i sigma_j) weight are all caught.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits

from . import sphere


@dataclass
class Operator:
    """Row-stochastic P on a symmetric graph plus row times; L = diag(1/tau) (P - I)."""

    X: np.ndarray                 # (N, n) points
    P: sp.csr_matrix              # row-stochastic, rows sorted by distance (self first)
    D: sp.csr_matrix              # Fisher distances on the same sparsity pattern (explicit zeros kept)
    tau: np.ndarray               # (N,) row times
    degree: np.ndarray            # (N,) D_i = sum_j K^a_ij
    knn_radius: np.ndarray        # (N,) Fisher distance to the k-th neighbour (own kNN list)
    kind: str
    knn_idx: np.ndarray | None = None    # (N, k) own kNN lists (self included), sorted by distance
    knn_dist: np.ndarray | None = None   # (N, k) their Fisher distances
    params: dict = field(default_factory=dict)
    sigma: np.ndarray | None = None

    @property
    def N(self) -> int:
        return self.X.shape[0]

    @property
    def pi(self) -> np.ndarray:
        """Reversible weights d_i tau_i (for spectra)."""
        return self.degree * self.tau

    def L(self, f: np.ndarray) -> np.ndarray:
        """L f = (1 / tau_i) sum_j P_ij (f_j - f_i) for f of shape (N,) or (N, p).

        Computed from the differences, not as (P f - f) / tau: at the kernel times of near-vertex clouds
        (tau ~ 1e-16) the rounding of P f - f would otherwise be amplified to O(1)."""
        f = np.asarray(f, dtype=np.float64)
        one = f.ndim == 1
        f2 = f[:, None] if one else f
        out = np.empty_like(f2)
        for rows, idx, w in iter_rows(self, batch=4096):
            out[rows] = np.einsum("bk,bkp->bp", w, f2[idx] - f2[rows][:, None, :]) / self.tau[rows][:, None]
        return out[:, 0] if one else out

    def gamma(self, f: np.ndarray, h: np.ndarray | None = None) -> np.ndarray:
        """Carre du champ Gamma(f, h)_i = (1 / 2 tau_i) sum_j P_ij (f_j - f_i)(h_j - h_i)^T.

        f: (N, p), h: (N, q) -> (N, p, q).  1-D inputs are treated as p = 1.
        """
        f = np.asarray(f, dtype=np.float64)
        f = f[:, None] if f.ndim == 1 else f
        h = f if h is None else np.asarray(h, dtype=np.float64)
        h = h[:, None] if h.ndim == 1 else h
        out = np.empty((self.N, f.shape[1], h.shape[1]))
        for rows, idx, w in iter_rows(self, batch=4096):
            df = f[idx] - f[rows][:, None, :]
            dh = h[idx] - h[rows][:, None, :]
            out[rows] = np.einsum("bk,bkp,bkq->bpq", w, df, dh) / (2.0 * self.tau[rows])[:, None, None]
        return out


def tau_from_radius(radius: float, m: int) -> float:
    """Kernel time whose physical radius sqrt(sum_j P_ij d_ij^2) is `radius` on a locally uniform
    m-dimensional cloud with the kernel contained in the rows: radius^2 = 2 m tau (per-coordinate kernel
    variance 2 tau).  For choosing fixed bandwidths as physical radii (SPEC Section 4, "Scales")."""
    return float(radius) ** 2 / (2.0 * m)


def knn(X: np.ndarray, k: int, n_jobs: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """k nearest neighbours (self included) in Fisher distance; returns (indices, Fisher distances)."""
    X = np.asarray(X, dtype=np.float64)
    k = min(k, X.shape[0])
    nn = NearestNeighbors(n_neighbors=k, algorithm="kd_tree", n_jobs=n_jobs).fit(X)
    _, idx = nn.kneighbors(X)
    d = np.empty(idx.shape)
    chunk = max(1, 2_000_000 // (k * X.shape[1]))
    for s in range(0, X.shape[0], chunk):
        d[s : s + chunk] = sphere.fisher_distance(X[s : s + chunk, None, :], X[idx[s : s + chunk]])
    # Re-sort by the exact Fisher distance (ties / rounding in the tree's chord distance) and make sure
    # the point itself is in its own list (with many exact duplicates the tree may return others first).
    order = np.argsort(d, axis=1, kind="stable")
    idx = np.take_along_axis(idx, order, 1)
    d = np.take_along_axis(d, order, 1)
    self_in = (idx == np.arange(X.shape[0])[:, None]).any(1)
    if not self_in.all():
        bad = np.where(~self_in)[0]
        idx[bad, -1] = bad
        d[bad, -1] = 0.0
        o = np.argsort(d[bad], axis=1, kind="stable")
        idx[bad] = np.take_along_axis(idx[bad], o, 1)
        d[bad] = np.take_along_axis(d[bad], o, 1)
    return idx, d


def build_operator(
    X: np.ndarray,
    k: int = 128,
    tau: float | None = None,
    kind: str = "fixed",
    eps: float = 0.25,
    k_bw: int = 16,
    sigma_floor: float = 1e-9,
    sigma: np.ndarray | None = None,
    cutoff: float = 25.0,
    alpha: float = 0.5,
    kernel_denominator: float = 4.0,
    sigma_correction: bool = True,
    n_jobs: int = 4,
) -> Operator:
    """Build the fixed-bandwidth (tau required) or adaptive (eps, k_bw) operator of the module docstring.

    `sigma` (adaptive only) prescribes the bandwidth function instead of the kNN pilot (used by tests of the
    limit with a smooth bandwidth not tied to the density).  `alpha`, `kernel_denominator` (the 4 in
    exp(-d^2 / 4 tau)) and `sigma_correction` exist so that tests can run deliberately wrong conventions; do
    not change them for measurement.
    """
    X = np.asarray(X, dtype=np.float64)
    N = X.shape[0]
    idx, dk = knn(X, k, n_jobs=n_jobs)
    knn_radius = dk[:, -1].copy()
    # union graph with the distances already computed for the kNN lists (d_ij = d_ji)
    r0 = np.repeat(np.arange(N, dtype=np.int64), idx.shape[1])
    c0 = idx.ravel().astype(np.int64)
    keys = np.concatenate([r0 * N + c0, c0 * N + r0])
    dd = np.concatenate([dk.ravel(), dk.ravel()])
    order = np.argsort(keys, kind="stable")
    keys, dd = keys[order], dd[order]
    first = np.concatenate([[True], keys[1:] != keys[:-1]])
    keys, d = keys[first], dd[first]
    rows, indices = keys // N, keys % N

    sigma_given = sigma
    sigma = None
    if kind == "fixed":
        if tau is None:
            raise ValueError("fixed-bandwidth operator needs tau")
        tau_i = np.full(N, float(tau))
        scale = np.full(d.shape, float(tau))
    elif kind == "adaptive":
        if sigma_given is not None:
            sigma = np.asarray(sigma_given, dtype=np.float64).copy()
        else:
            # rms distance to the k_bw nearest neighbours other than self (drop one zero-distance entry);
            # the pilot has its own query when the graph's k is smaller than k_bw + 1
            dpil = dk if dk.shape[1] > k_bw else knn(X, k_bw + 1, n_jobs=n_jobs)[1]
            kb = min(k_bw, dpil.shape[1] - 1)
            sigma = (np.sqrt(np.mean(dpil[:, 1 : kb + 1] ** 2, axis=1)) if kb > 0
                     else np.zeros(N))
        sigma = np.maximum(sigma, sigma_floor)
        tau_i = eps * sigma**2
        scale = eps * sigma[rows] * sigma[indices]
    else:
        raise ValueError(kind)

    expo = d**2 / (kernel_denominator * scale)
    keep = (expo <= cutoff) | (rows == indices)
    rows, cols, d, expo = rows[keep], indices[keep], d[keep], expo[keep]
    Kv = np.exp(-expo)
    K = sp.csr_matrix((Kv, (rows, cols)), shape=(N, N))
    q = np.asarray(K.sum(1)).ravel()
    w = Kv / np.sqrt(q[rows] * q[cols]) ** (2 * alpha)
    if kind == "adaptive" and sigma_correction:
        w = w / (sigma[rows] * sigma[cols])
    Ka = sp.csr_matrix((w, (rows, cols)), shape=(N, N))
    deg = np.asarray(Ka.sum(1)).ravel()
    Pv = w / deg[rows]

    # sort each row by distance (self first) and store P and distances on the same pattern
    order = np.lexsort((d, rows))
    rows, cols, d, Pv = rows[order], cols[order], d[order], Pv[order]
    counts = np.bincount(rows, minlength=N)
    ptr = np.concatenate([[0], np.cumsum(counts)])
    P = sp.csr_matrix((Pv, cols, ptr), shape=(N, N))
    Dm = sp.csr_matrix((d, cols.copy(), ptr.copy()), shape=(N, N))
    params = dict(k=k, tau=tau, kind=kind, eps=eps, k_bw=k_bw, cutoff=cutoff, alpha=alpha,
                  kernel_denominator=kernel_denominator, sigma_correction=sigma_correction)
    return Operator(X=X, P=P, D=Dm, tau=tau_i, degree=deg, knn_radius=knn_radius, kind=kind,
                    knn_idx=idx, knn_dist=dk, params=params, sigma=sigma)


def iter_rows(op: Operator, batch: int = 256, budget: int = 65536, rows_subset: np.ndarray | None = None):
    """Yield (rows, idx, P) for batches of anchors: idx (b, kmax) neighbour indices padded with the anchor
    itself, P (b, kmax) weights padded with 0 (padding contributes nothing to any moment).  Rows keep the
    distance order of the operator (self first).  Batches are sized so that b * kmax <= budget.
    """
    indptr, indices, data = op.P.indptr, op.P.indices, op.P.data
    lens = np.diff(indptr)
    allrows = np.arange(op.N) if rows_subset is None else np.asarray(rows_subset)
    start = 0
    while start < allrows.size:
        stop = start + 1
        kmax = lens[allrows[start]]
        while stop < allrows.size and stop - start < batch:
            nk = max(kmax, lens[allrows[stop]])
            if nk * (stop - start + 1) > budget:
                break
            kmax = nk
            stop += 1
        rows = allrows[start:stop]
        b = rows.size
        idx = np.repeat(rows[:, None], kmax, axis=1)
        w = np.zeros((b, kmax))
        for t, r in enumerate(rows):
            s, e = indptr[r], indptr[r + 1]
            idx[t, : e - s] = indices[s:e]
            w[t, : e - s] = data[s:e]
        yield rows, idx, w
        start = stop


def gather_distances(op: Operator, rows: np.ndarray, kmax: int) -> np.ndarray:
    """Fisher distances in the same padded layout as iter_rows (padding: 0)."""
    indptr, data = op.D.indptr, op.D.data
    out = np.zeros((rows.size, kmax))
    for t, r in enumerate(rows):
        s, e = indptr[r], indptr[r + 1]
        out[t, : e - s] = data[s:e]
    return out


def run_batches(op: Operator, fn, budget: int = 32768, n_jobs: int = 4, rows_subset=None) -> None:
    """Call fn(rows, idx, w) on every batch of iter_rows, in n_jobs threads (numpy releases the GIL in the
    array kernels; BLAS is limited to one thread meanwhile).  fn must write only to its own rows."""
    batches = iter_rows(op, budget=budget, rows_subset=rows_subset)
    if n_jobs <= 1:
        for b in batches:
            fn(*b)
        return
    with threadpool_limits(1), ThreadPoolExecutor(max_workers=n_jobs) as ex:
        for _ in ex.map(lambda b: fn(*b), batches):
            pass
