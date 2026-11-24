"""Geometry of the ambient space: the radius-2 sphere S^9 in R^10 with its round (Fisher-Rao) metric.

A probability vector p on the 10-simplex is represented by X = 2 sqrt(p), a point of the positive orthant of
the sphere of radius R = 2.  The round metric of that sphere is the Fisher-Rao metric, so the Fisher distance
is R times the angle between the two points.  Everything here is written for arbitrary points of the
radius-R sphere (fixtures use points outside the orthant), and is vectorised over leading axes.

Numerics (float64 throughout):

* Distance: d(x, y) = 2 R atan2(|x - y|, |x + y|) = 4 atan2(|s - s'|, |s + s'|) with s = sqrt(p).  Unlike
  arccos(<x, y>/R^2) it keeps full relative precision for nearly coincident points, and unlike arcsin it
  stays accurate up to antipodal points (which cannot occur inside the orthant: angles there are <= pi/2).
* Log map: log_x y = (theta / sin theta) Q_x (y - x), theta = d / R, Q_x = I - x x^T / R^2.  Using the
  difference y - x (rather than y) keeps precision when y is close to x; theta / sin theta is evaluated
  with np.sinc, which is exact at 0.
* Parallel transport along the short geodesic from x to y of a tangent vector xi at x:
      PT_{x->y} xi = xi - <y, xi> / (R^2 + <x, y>) (x + y).
  The denominator is >= R^2 in the orthant.  It is the exact Levi-Civita transport (the rotation in the
  plane of x and y taking x to y, identity on the orthogonal complement); tests check it against numerical
  integration of the transport ODE.
"""
from __future__ import annotations

import numpy as np

R = 2.0          # sphere radius: X = 2 sqrt(p)
R2 = R * R
AMBIENT = 10     # ambient dimension of the fixed lens space


def sphere_from_logp(logp: np.ndarray) -> np.ndarray:
    """Log-probabilities (..., n) -> sphere points X = 2 sqrt(softmax(logp)) in float64.

    The input is re-normalised with logsumexp in float64, so unnormalised logits are accepted too, and
    -inf entries give exact zeros.  Near a simplex vertex this is also the accurate form for float32
    input: a float32 log-softmax carries an absolute error ~6e-8 in log p_top (the rounding of
    log(1 + small)), but that error is common to the numerator and the renormalising sum, so p_top and the
    small p_c come out with relative errors ~6e-8 and the near-vertex geometry (set by the small
    coordinates) keeps the input's own relative precision (tests/test_sphere.py checks distances down to
    ~1e-9 against a float64 truth).
    """
    lp = np.asarray(logp, dtype=np.float64)
    mx = np.max(lp, axis=-1, keepdims=True)
    lse = mx + np.log(np.sum(np.exp(lp - mx), axis=-1, keepdims=True))
    return R * np.exp(0.5 * (lp - lse))


def input_resolution(logp: np.ndarray, factor: float = 4.0, floor: float = 1e-12) -> np.ndarray:
    """Per-point Fisher-distance resolution implied by the precision of the log-probability input.

    A change of one unit in the last place of log p_c moves X_c = 2 sqrt(p_c) by X_c ulp(log p_c) / 2, so
    the quantisation of the input alone puts an uncertainty of
        q_i = sqrt(sum_c (X_c ulp(log p_c) / 2)^2)
    on any displacement of point i.  The resolution is max(floor, factor * q_i).  `floor` (1e-12) is a
    conservative float64 resolution of the representation (coordinates of size ~2 carry rounding ~4e-16;
    near a vertex most of that rounding is radial and projected out, so the true floor is lower there).
    For float32 inputs this gives ~1e-6 for a generic point and the floor within ~1e-8 of a vertex, where
    all coordinates but one are tiny and known to their own relative precision.  It is a lower bound on
    the true noise (rounding in the network's own float32 forward pass is not modelled).
    """
    lp = np.asarray(logp)
    dt = lp.dtype if np.issubdtype(lp.dtype, np.floating) else np.dtype(np.float64)
    X = sphere_from_logp(lp)
    lp64 = lp.astype(np.float64)
    ulp = np.spacing(np.abs(lp.astype(dt))).astype(np.float64)
    ulp = np.where(np.isfinite(lp64), ulp, 0.0)
    q = np.sqrt(np.sum((X * ulp / 2.0) ** 2, axis=-1))
    return np.maximum(floor, factor * q)


def logp_from_sphere(X: np.ndarray) -> np.ndarray:
    """Inverse of sphere_from_logp for points of the closed positive orthant."""
    X = np.asarray(X, dtype=np.float64)
    with np.errstate(divide="ignore"):
        return 2.0 * np.log(np.abs(X) / R)


def project_to_sphere(X: np.ndarray) -> np.ndarray:
    """Radial projection onto the radius-R sphere."""
    X = np.asarray(X, dtype=np.float64)
    return R * X / np.linalg.norm(X, axis=-1, keepdims=True)


def _norm(a: np.ndarray) -> np.ndarray:
    return np.sqrt(np.einsum("...i,...i->...", a, a))


def angle(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Angle (radians) between points of the radius-R sphere, stable form 2 atan2(|x-y|, |x+y|)."""
    return 2.0 * np.arctan2(_norm(x - y), _norm(x + y))


def fisher_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fisher-Rao distance R * angle = 4 atan2(|x - y|, |x + y|) for R = 2 (broadcasts)."""
    return R * angle(x, y)


def fisher_distance_logp(logp_a: np.ndarray, logp_b: np.ndarray) -> np.ndarray:
    """Fisher distance between distributions given by log-probabilities: 4 atan2(|s - s'|, |s + s'|)."""
    return fisher_distance(sphere_from_logp(logp_a), sphere_from_logp(logp_b))


def tangent_project(x: np.ndarray, xi: np.ndarray) -> np.ndarray:
    """Q_x xi = xi - <x, xi> x / R^2: orthogonal projection onto the sphere's tangent space at x."""
    return xi - (np.einsum("...i,...i->...", x, xi) / R2)[..., None] * x


def tangent_projector(x: np.ndarray) -> np.ndarray:
    """Q_x = I - x x^T / R^2 as (..., n, n) matrices."""
    n = x.shape[-1]
    return np.eye(n) - x[..., :, None] * x[..., None, :] / R2


def _theta_over_sin(theta: np.ndarray) -> np.ndarray:
    # theta / sin(theta) = 1 / sinc(theta / pi); exact at 0, finite for theta < pi.
    return 1.0 / np.sinc(theta / np.pi)


def log_map(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """log_x y on the radius-R sphere, written in ambient coordinates (broadcasts over leading axes).

    |log_x y| = Fisher distance.  Undefined only at the antipode (not reachable in the orthant).
    """
    diff = y - x
    theta = 2.0 * np.arctan2(_norm(diff), _norm(x + y))
    # Q_x (y - x): since |x| = |y| = R, <x, y - x> = -|y - x|^2 / 2, but projecting explicitly also removes
    # any radial rounding error in the inputs.
    q = tangent_project(x, diff)
    return _theta_over_sin(theta)[..., None] * q


def exp_map(x: np.ndarray, xi: np.ndarray) -> np.ndarray:
    """exp_x xi on the radius-R sphere (xi tangent at x)."""
    nrm = _norm(xi)
    a = nrm / R
    # cos(a) x + R sin(a) xi / |xi| = cos(a) x + sinc(a) xi
    return np.cos(a)[..., None] * x + np.sinc(a / np.pi)[..., None] * xi


def parallel_transport(x: np.ndarray, y: np.ndarray, xi: np.ndarray) -> np.ndarray:
    """Parallel transport of xi (tangent at x) along the short geodesic from x to y.

    PT_{x->y} xi = xi - <y, xi> / (R^2 + <x, y>) (x + y).  Broadcasts; xi may carry an extra trailing
    matrix axis if passed as (..., n, k) via parallel_transport_matrix.
    """
    c = R2 + np.einsum("...i,...i->...", x, y)
    coef = np.einsum("...i,...i->...", y, xi) / c
    return xi - coef[..., None] * (x + y)


def parallel_transport_matrix(x: np.ndarray, y: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Transport the columns of A (..., n, k), each tangent at x, to y."""
    c = R2 + np.einsum("...i,...i->...", x, y)
    coef = np.einsum("...i,...ik->...k", y, A) / c[..., None]
    return A - (x + y)[..., :, None] * coef[..., None, :]


def tangent_basis(x: np.ndarray) -> np.ndarray:
    """Orthonormal basis (..., n, n-1) of the tangent space at x, from a Householder reflector.

    H = I - 2 v v^T / |v|^2 with v = xhat + s e_n (s = sign of the last coordinate) maps xhat to -s e_n,
    so its first n-1 columns are orthonormal and orthogonal to x.  No cancellation in v.
    """
    n = x.shape[-1]
    xh = x / _norm(x)[..., None]
    s = np.where(xh[..., -1] >= 0, 1.0, -1.0)
    v = xh.copy()
    v[..., -1] += s
    H = np.eye(n) - 2.0 * v[..., :, None] * v[..., None, :] / np.einsum("...i,...i->...", v, v)[..., None, None]
    return H[..., :, : n - 1]


def face_distance(logp: np.ndarray, k: int = 1) -> np.ndarray:
    """Fisher distance from each distribution to the nearest k-face of the simplex (k = 1: vertex, 2: edge).

    The nearest k-face is spanned by the k most probable classes I; the distance is
    2 atan2(sqrt(sum_{c not in I} p_c), sqrt(sum_{c in I} p_c))  (SPEC item 14).
    """
    X = sphere_from_logp(logp)
    p = (X / R) ** 2
    ps = np.sort(p, axis=-1)[..., ::-1]
    inside = ps[..., :k].sum(-1)
    outside = ps[..., k:].sum(-1)
    return 2.0 * np.arctan2(np.sqrt(outside), np.sqrt(inside))
