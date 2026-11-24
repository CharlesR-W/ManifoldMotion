"""Shared fixture helpers for the tests (exact answers of the synthetic clouds)."""
import numpy as np

from mm import fixtures as fx
from mm import graph, frame, sphere

R = sphere.R


def fib_great_sphere(N, kappa, fr):
    """Deterministic low-discrepancy great 2-sphere in R^10 (frame fr: 10 x 3), density on the
    radius-2 sphere prop. to exp(kappa (x . e) / 2), e = fr[:, 0]."""
    return R * fx.fibonacci_sphere(N, kappa) @ fr.T


def great_tangent(X, fr, c):
    """T c for the great sphere spanned by fr: Pf c - (x . c) x / R^2 (rows of X)."""
    Pc = fr @ (fr.T @ c)
    return Pc[None, :] - (X @ Pc)[:, None] * X / R**2


def grad_log_rho(X, fr, kappa):
    return 0.5 * kappa * great_tangent(X, fr, fr[:, 0])


def operator_errors(op, fr, kappa, m, rng=None, sel=None):
    """Relative errors of the operator against the exact continuum answers on a great m-sphere with
    density exp(kappa (x . e) / 2):
      gamma  rms (Gamma(f, f) - |T c|^2) / mean |T c|^2          (f = c . x, c in the span)
      L      rms (L f - (-m f / 4 + grad log rho . T c)) / rms(exact)
      drift  rms |L X - (-m x / 4 + grad log rho)| / rms |grad log rho|
      G      rms (lambda_k - 1) over the top m eigenvalues of G, and the mean of lambda_{m+1}
    Also the *mean* signed errors (bias) of gamma and L-drift projected on grad log rho."""
    rng = rng or np.random.default_rng(0)
    X = op.X
    sel = np.ones(len(X), bool) if sel is None else sel
    cs = fr @ rng.standard_normal((fr.shape[1], 3))
    cs /= np.linalg.norm(cs, axis=0)
    F = X @ cs
    gl = grad_log_rho(X, fr, kappa)
    g = op.gamma(F)
    L = op.L(F)
    eg, el, bias_g = [], [], []
    ref_g, ref_l = [], []
    for k in range(3):
        Tc = great_tangent(X, fr, cs[:, k])
        tg = np.sum(Tc**2, 1)
        tl = -m * F[:, k] / R**2 + np.sum(gl * Tc, 1)
        eg.append((g[:, k, k] - tg)[sel])
        ref_g.append(tg[sel])
        el.append((L[:, k] - tl)[sel])
        ref_l.append(tl[sel])
    eg, el = np.concatenate(eg), np.concatenate(el)
    out = dict(gamma=np.sqrt(np.mean(eg**2)) / np.mean(np.concatenate(ref_g)),
               gamma_bias=np.mean(eg) / np.mean(np.concatenate(ref_g)),
               L=np.sqrt(np.mean(el**2)) / np.sqrt(np.mean(np.concatenate(ref_l) ** 2)))
    LX = op.L(X)
    ex = LX - (-m * X / R**2 + gl)
    gn = np.sqrt(np.mean(np.sum(gl[sel] ** 2, 1)))
    out["drift"] = np.sqrt(np.mean(np.sum(ex[sel] ** 2, 1))) / gn
    # bias of the drift along grad log rho (mean over points): catches a wrong drift coefficient
    out["drift_coef"] = np.sum((LX + m * X / R**2)[sel] * gl[sel]) / np.sum(gl[sel] ** 2)
    # bias: errors averaged in 20 quantile bins of the coordinate along the density gradient (removes the
    # sampling noise of the pointwise estimates, keeps any position-dependent bias such as a wrong drift)
    pos = (X @ fr[:, 0])[sel]
    edges = np.quantile(pos, np.linspace(0, 1, 21))
    b = np.clip(np.searchsorted(edges, pos, side="right") - 1, 0, 19)
    el3 = el.reshape(3, -1)
    rl3 = np.concatenate(ref_l).reshape(3, -1)
    binned_l = np.array([[el3[k][b == i].mean() for i in range(20)] for k in range(3)])
    out["L_binned"] = np.sqrt(np.mean(binned_l**2)) / np.sqrt(np.mean(rl3**2))
    exs = ex[sel]
    binned_d = np.array([exs[b == i].mean(0) for i in range(20)])
    out["drift_binned"] = np.sqrt(np.mean(np.sum(binned_d**2, 1))) / gn
    fa = frame.analyse_frame(op)
    out["G"] = np.sqrt(np.mean((fa.eig[sel, :m] - 1.0) ** 2))
    out["G_next"] = np.mean(fa.eig[sel, m])
    out["dim_ok"] = np.mean(fa.dim[sel] == m)
    out["n_eff"] = np.median(fa.n_eff[sel])
    out["trunc"] = np.nanmedian(fa.trunc_tail[sel])
    return out


def sigma_field(X, fr, s0):
    """A smooth bandwidth function on the great sphere, varying by a factor e between poles and not
    tied to the density: s0 exp((x . fr[:, 1]) / 4)."""
    return s0 * np.exp(X @ fr[:, 1] / 4.0)
