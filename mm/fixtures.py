"""Synthetic clouds and motions on the radius-2 sphere S^9 with known answers (SPEC Section 4 fixtures).

All generators return float64 points of R^10 on the radius-2 sphere.  Embedded subspheres use an
orthonormal frame `frame` (10, m + 1 or m + 2) whose columns are e_0, e_1, ... of the construction.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import vonmises_fisher

from . import sphere

R = sphere.R


def axis_frame(ncols: int, n: int = 10) -> np.ndarray:
    return np.eye(n)[:, :ncols]


def random_frame(ncols: int, rng, n: int = 10) -> np.ndarray:
    Q, _ = np.linalg.qr(rng.standard_normal((n, ncols)))
    return Q


def barycenter_frame(ncols: int, rng=None, n: int = 10) -> np.ndarray:
    """Orthonormal frame whose first column is the orthant barycentre direction 1 / sqrt(n)."""
    rng = rng or np.random.default_rng(0)
    A = np.column_stack([np.ones(n) / np.sqrt(n), rng.standard_normal((n, ncols - 1))])
    Q, _ = np.linalg.qr(A)
    return Q * np.sign(Q[0, 0])


def unit_sphere(N: int, m: int, rng, kappa: float = 0.0, mu: np.ndarray | None = None) -> np.ndarray:
    """N points on the unit m-sphere in R^(m+1): uniform (kappa = 0) or von Mises-Fisher about mu (default
    e_0), density ∝ exp(kappa <mu, omega>)."""
    if kappa == 0.0:
        z = rng.standard_normal((N, m + 1))
        return z / np.linalg.norm(z, axis=1, keepdims=True)
    mu = np.eye(m + 1)[0] if mu is None else mu
    return vonmises_fisher(mu, kappa).rvs(N, random_state=rng)


def great_sphere(N: int, m: int, rng, kappa: float = 0.0, frame: np.ndarray | None = None):
    """Great m-subsphere of the radius-2 sphere with vMF density about frame[:, 0].

    Returns X and grad log rho (as a function of X): rho ∝ exp(kappa <x, c> / 2) w.r.t. Fisher volume, so
    grad_M log rho = (kappa / 2) T_x c with T_x the tangent projector of the great sphere.
    """
    frame = axis_frame(m + 1) if frame is None else frame
    om = unit_sphere(N, m, rng, kappa)
    X = R * om @ frame.T
    c = frame[:, 0]

    def grad_log_rho(Xq):
        Pf = frame @ frame.T
        cc = Pf @ c
        return 0.5 * kappa * (cc[None, :] - (Xq @ cc)[:, None] * Xq / R**2)

    return X, grad_log_rho


def cap_in_great_sphere(N: int, m: int, alpha_max: float, rng, frame: np.ndarray, kappa: float = 0.0):
    """Points of a geodesic cap (angular radius alpha_max around frame[:, 0]) of a great m-sphere, with
    density ∝ exp(kappa cos alpha) (rejection sampling from the vMF)."""
    out = []
    while sum(len(o) for o in out) < N:
        om = unit_sphere(4 * N, m, rng, kappa)
        out.append(om[om[:, 0] >= np.cos(alpha_max)])
    om = np.concatenate(out)[:N]
    return R * om @ frame.T


def latitude_sphere(N: int, m: int, theta: float, rng, kappa: float = 0.0, frame: np.ndarray | None = None,
                    omega: np.ndarray | None = None):
    """x = 2 (cos theta e_0 + sin theta omega), omega on the unit m-sphere in span(e_1, ..., e_{m+1}).

    A small sphere of Fisher radius 2 sin theta; returns (X, omega)."""
    frame = axis_frame(m + 2) if frame is None else frame
    om = unit_sphere(N, m, rng, kappa) if omega is None else omega
    return latitude_points(om, theta, frame), om


def latitude_points(omega: np.ndarray, theta: float | np.ndarray, frame: np.ndarray) -> np.ndarray:
    th = np.asarray(theta, dtype=np.float64)
    th = th[:, None] if th.ndim == 1 else th
    loc = np.column_stack([np.cos(th) * np.ones(len(omega)), np.sin(th) * omega])
    return R * loc @ frame.T


def conformal_flow(omega: np.ndarray, c: float, t: float) -> np.ndarray:
    """Exact flow for time t of xi = c (e_p - h omega) on the unit m-sphere (e_p = first coordinate,
    h = omega_0): the polar angle alpha from e_p obeys tan(alpha'/2) = tan(alpha/2) exp(-c t)."""
    h = np.clip(omega[:, 0], -1.0, 1.0)
    alpha = np.arccos(h)
    alpha2 = 2.0 * np.arctan(np.tan(alpha / 2.0) * np.exp(-c * t))
    perp = omega.copy()
    perp[:, 0] = 0.0
    nrm = np.linalg.norm(perp, axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        dirn = np.where(nrm > 0, perp / nrm, 0.0)
    out = np.sin(alpha2)[:, None] * dirn
    out[:, 0] = np.cos(alpha2)
    return out


def projective_map(X: np.ndarray, A: np.ndarray) -> np.ndarray:
    """x -> 2 A x / |A x| (a smooth one-to-one map of the sphere for invertible A)."""
    Y = X @ A.T
    return R * Y / np.linalg.norm(Y, axis=1, keepdims=True)


def projective_differential(X: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Exact differential (N, 10, 10) of projective_map: (2 / |A x|) (I - yhat yhat^T) A."""
    Y = X @ A.T
    nY = np.linalg.norm(Y, axis=1)
    yh = Y / nY[:, None]
    n = X.shape[1]
    return (R / nY)[:, None, None] * (np.eye(n)[None] - yh[:, :, None] * yh[:, None, :]) @ A


def great_sphere_tangent(X: np.ndarray, frame: np.ndarray) -> np.ndarray:
    """Orthonormal tangent basis (N, 10, m) of the great sphere spanned by `frame` at its points."""
    Z = X @ frame / R                          # local coordinates on the unit sphere
    N, k = Z.shape
    out = np.empty((N, X.shape[1], k - 1))
    for i in range(N):
        P = np.eye(k) - np.outer(Z[i], Z[i])
        w, V = np.linalg.eigh(P)
        out[i] = frame @ V[:, 1:]
    return out


def rotation(n: int, plane: tuple[int, int], angle: float, frame: np.ndarray | None = None) -> np.ndarray:
    """Rotation by `angle` in the coordinate plane (or the plane of two frame columns)."""
    A = np.zeros((n, n))
    i, j = plane
    A[i, j], A[j, i] = -1.0, 1.0
    if frame is not None:
        A = frame @ A[: frame.shape[1], : frame.shape[1]] @ frame.T
    from scipy.linalg import expm

    return expm(angle * A)


def random_skew(n: int, rng, scale: float = 1.0) -> np.ndarray:
    A = rng.standard_normal((n, n)) * scale
    return 0.5 * (A - A.T)


GOLDEN = (1.0 + 5.0**0.5) / 2.0


def fibonacci_sphere(N: int, kappa: float = 0.0) -> np.ndarray:
    """Deterministic low-discrepancy points on the unit 2-sphere in R^3 with density ∝ exp(kappa omega_0)
    (uniform for kappa = 0): omega_0 = F^-1((i + 1/2) / N) for the exact vMF marginal
    F(z) = (e^{kappa z} - e^{-kappa}) / (e^{kappa} - e^{-kappa}), longitude 2 pi i / golden ratio."""
    u = (np.arange(N) + 0.5) / N
    if kappa == 0.0:
        z = 1.0 - 2.0 * u
    else:
        z = -1.0 + np.log1p(u * np.expm1(2.0 * kappa)) / kappa
    phi = 2.0 * np.pi * np.arange(N) / GOLDEN
    r = np.sqrt(np.clip(1.0 - z**2, 0.0, None))
    return np.column_stack([z, r * np.cos(phi), r * np.sin(phi)])


# ----------------------------------------------------------------------------------------------------------
# matched-regime fixture: a 9-dimensional anisotropic blob on S^9 and a smooth one-to-one map of it

# local second-moment spectrum of the real clouds (median eigenvalues / largest, site b7, final checkpoint)
REAL_SPECTRUM = np.array([1.0, 0.79, 0.65, 0.50, 0.34, 0.23, 0.14, 0.08, 0.05])
# log-stretches of the linear part of the matched map (spread +1 .. -1.5, sum -0.6 ~ real median log J)
MATCHED_LOG_STRETCH = np.array([1.0, 0.6, 0.3, 0.1, 0.0, -0.1, -0.3, -0.7, -1.5])


class MatchedBlob:
    """Gaussian blob in exponential coordinates at the barycentre c = (2 / sqrt 10) 1 of S^9:
    x = exp_c(B z), z ~ N(0, scale^2 diag(spectrum)), B an orthonormal 10 x 9 basis of T_c S^9 (randomly
    rotated, so the blob's axes are not coordinate axes).  With spectrum = REAL_SPECTRUM^1.75 and scale 0.55,
    N = 10,000 and the default operator, the local spectrum at k = 128 has medians 1, .80, .72, .64, .55, .42,
    .24, .10, .04 (real b7: 1, .79, .65, .50, .34, .23, .14, .08, .05), radius 0.41 (k = 128) .. 0.47
    (k = 512) (real 0.40 .. 0.43) and one-sidedness 0.86 (real ~0.85).

    The map is f(x) = exp_c(B L g(z)), z = B^T log_c x, with
        g(z) = z + eps ell sin(W z / ell)     (W orthogonal; Lipschitz constant of the perturbation eps < 1,
                                               so g is one-to-one; its stretches vary by ~ +-eps over a
                                               length ell in z)
        L = P diag(exp(log_stretch)) Q^T      (P, Q random orthogonal)
    The exact differential (T_x -> T_f(x)) and the exact covariant derivative of the log displacement
    v(x) = log_x f(x) are computed by central differences along geodesics (h = 1e-5: truncation ~1e-10,
    rounding ~1e-11), so they are independent of the estimator."""

    def __init__(self, rng, scale: float, spectrum=REAL_SPECTRUM**1.75, log_stretch=MATCHED_LOG_STRETCH,
                 eps: float = 0.5, ell: float = 0.3, Q=None):
        n = 10
        self.c = np.full(n, R / np.sqrt(n))
        from . import sphere as _s
        E = _s.tangent_basis(self.c[None])[0]
        self.B = E @ np.linalg.qr(rng.standard_normal((9, 9)))[0]
        self.sd = scale * np.sqrt(np.asarray(spectrum, float))
        self.W = np.linalg.qr(rng.standard_normal((9, 9)))[0]
        P = np.linalg.qr(rng.standard_normal((9, 9)))[0]
        Q = np.linalg.qr(rng.standard_normal((9, 9)))[0] if Q is None else np.asarray(Q, float)
        self.P, self.Q = P, Q
        self.L = P @ np.diag(np.exp(np.asarray(log_stretch, float))) @ Q.T
        self.eps, self.ell = eps, ell

    def sample(self, N: int, rng) -> np.ndarray:
        from . import sphere as _s
        z = rng.standard_normal((N, 9)) * self.sd
        return _s.exp_map(np.broadcast_to(self.c, (N, 10)), z @ self.B.T)

    def f(self, X: np.ndarray) -> np.ndarray:
        from . import sphere as _s
        cb = np.broadcast_to(self.c, X.shape)
        z = _s.log_map(cb, X) @ self.B
        return _s.exp_map(cb, (self.g(z) @ self.L.T) @ self.B.T)

    def g(self, z: np.ndarray) -> np.ndarray:
        if self.eps == 0:
            return z
        return z + self.eps * self.ell * np.sin(z @ self.W.T / self.ell)

    def exact(self, X: np.ndarray, h: float = 1e-5) -> dict:
        """Exact differential J (N, 10, 9) in the tangent basis E_x of X, log-stretches (descending), and
        the covariant derivative K (N, 10, 10) of v(x) = log_x f(x) (acting on T_x, zero on x)."""
        from . import sphere as _s
        Y = self.f(X)
        v = _s.log_map(X, Y)
        E = _s.tangent_basis(X)
        J = np.zeros((X.shape[0], 10, 9))
        Kc = np.zeros((X.shape[0], 10, 9))
        for k in range(9):
            xp = _s.exp_map(X, h * E[:, :, k])
            xm = _s.exp_map(X, -h * E[:, :, k])
            J[:, :, k] = (_s.log_map(Y, self.f(xp)) - _s.log_map(Y, self.f(xm))) / (2 * h)
            vp = _s.parallel_transport(xp, X, _s.log_map(xp, self.f(xp)))
            vm = _s.parallel_transport(xm, X, _s.log_map(xm, self.f(xm)))
            Kc[:, :, k] = (vp - vm) / (2 * h)
        K = Kc @ np.swapaxes(E, 1, 2)
        s = np.linalg.svd(J, compute_uv=False)
        return dict(Y=Y, v=v, J=J, K=K, E=E, log_stretch=np.log(s))


class MapOnBlob(MatchedBlob):
    """A MatchedBlob source cloud with another map f (a callable on (N, 10) sphere points); the exact
    differential and covariant derivative come from the same central differences."""

    def __init__(self, blob: MatchedBlob, f):
        self.__dict__.update(blob.__dict__)
        self._f = f

    def f(self, X: np.ndarray) -> np.ndarray:
        return self._f(X)
