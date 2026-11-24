"""Matched-regime fixture driver (used by tests/test_matched.py and by the fixture run on Modal).

Source cloud: a 9-dimensional anisotropic Gaussian blob on S^9 (fixtures.MatchedBlob) whose local
second-moment spectrum, neighbourhood radius (frame.radius ~ 0.4 at k = 128), one-sidedness (~0.85) and
kernel truncation resemble the real lens clouds at b7.  Maps with known differential (exact values by central
differences, independent of the estimator) and several contrasts, grouped so that every noisy variant has a
noiseless twin on the same points (its estimate is the "secant" truth: the finite-scale weighted slope under
the same anchor convention):

  smooth      the matched smooth one-to-one map (log-stretches ~ +1 .. -1.8, varying over ~ one radius);
              endpoint scatter at nu in NUS (independent noise on every point's endpoint: the anchor's own
              noise is shared by all its neighbour offsets); edge-noise contrast; shuffled targets
  weakstretch the least-sampled source axis carries the largest stretch
  weakcomp    the least-sampled source axis carries the strongest compression; scatter isotropic, along the
              image of that axis, or orthogonal to it
  rank8       one direction collapsed exactly (rank-8 map); nearsing: that stretch is exp(-5)
  vertex      piecewise collapse towards simplex vertices (t = 0.99 of the way), as at b8 -> b9
  fold        two branches, one direction reversed on one branch (a fold), along a random / the major axis
  curved      noiseless strongly curved map (variation length 0.1 instead of 0.3)
  controls    identity and a rotation of the sphere (zero strain; known spin)

    python tests/matched.py --group smooth --seed 0 --k 256 --eps 0.25 --out DIR
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mm import fixtures as fx, graph, sphere, step  # noqa: E402
from mm.frame import Rules, analyse_frame, regress  # noqa: E402

SCALE = 0.55
N_POINTS = 10_000
NUS = (0.3, 0.6, 1.0, 1.5)
R = sphere.R

# per-point arrays saved for every variant (estimates, gating statistics)
EST_KEYS = ["logJ", "ic_logJ", "fin_vol", "fin_shear", "fin_total", "E_trace", "ic_E_trace", "rate_vol",
            "rate_shear", "rate_total", "strain_sq", "strain_sq_db", "ic_strain_sq", "ic_strain_sq_db",
            "spin_sq", "spin_sq_db", "ic_spin_sq", "ic_spin_sq_db", "relmotion", "relmotion_fit",
            "relmotion_res", "trB", "trFGF", "explained", "heldout", "heldout_v", "heldout_cr", "heldout_identity",
            "heldout_const", "heldout_ic", "heldout_top3", "heldout_top6", "heldout_v_const", "heldout_v_ic",
            "intercept_rel", "intercept_v_rel", "cond", "cond_centred", "n_eff", "radius", "one_sided",
            "trunc_tail", "log_sigma_min_se", "logJ_se", "sigma_min", "travel", "speed", "ke", "ke_rigid",
            "ke_nonrigid", "cr_travel", "strain_sq_var", "spin_sq_var", "relmotion_shift", "relmotion_slope",
            "relmotion_icres", "ic_log_gain_max"]


def _sym(A):
    return 0.5 * (A + np.swapaxes(A, 1, 2))


def exact_fields(obj, X, G):
    """Exact per-point values (dt = 1, flow reading: T = whole tangent) from the analytic differential."""
    ex = obj.exact(X)
    ls = ex["log_stretch"]                                   # (N, 9) descending (may contain -inf)
    with np.errstate(invalid="ignore"):
        sl, sl2 = ls.sum(1), (ls**2).sum(1)
    K = ex["K"]
    E = _sym(K)
    W = 0.5 * (K - np.swapaxes(K, 1, 2))
    trE = np.trace(E, axis1=1, axis2=2)
    e2 = np.einsum("nij,nij->n", E, E)
    Fx = ex["J"] @ np.swapaxes(ex["E"], 1, 2)                 # (N, 10, 10) acting on T_x
    tru = {f"ls{k + 1}": ls[:, k] for k in range(9)}
    tru.update(logJ=sl, fin_vol=sl**2 / 9, fin_shear=sl2 - sl**2 / 9, fin_total=sl2, E_trace=trE,
               rate_vol=trE**2 / 9, rate_shear=e2 - trE**2 / 9, rate_total=e2, strain_sq=e2,
               spin_sq=np.einsum("nij,nij->n", W, W),
               relmotion_fit=np.einsum("nij,njk,nik->n", K, G, K),
               trFGF=np.einsum("nij,njk,nik->n", Fx, G, Fx))
    return tru, ex, Fx, K


def oracle_score(op, X, Y, Fx):
    """1 - sum_j P_ij |y_j - F u_j|^2 / sum_j P_ij |y_j|^2 with the exact differential: what a linear
    description with the true derivative scores on this neighbourhood."""
    num = np.zeros(len(X))
    den = np.zeros(len(X))
    for rows, idx, w in graph.iter_rows(op, batch=512):
        u = sphere.log_map(X[rows][:, None, :], X[idx])
        y = sphere.log_map(Y[rows][:, None, :], Y[idx])
        r = y - np.einsum("bij,bkj->bki", Fx[rows], u)
        num[rows] = (w * (r**2).sum(-1)).sum(1)
        den[rows] = (w * (y**2).sum(-1)).sum(1)
    return 1.0 - num / den


def edge_vs_endpoint(op, f0, X, Yc, sigma, rng, Fx_true):
    """Anchored and free-intercept fits of next-frame offsets under (a) endpoint noise (noise on each
    point's endpoint, shared by an anchor's neighbours through the anchor) and (b) independent noise on
    every edge response with the same per-edge variance.  Returns prediction-weighted relative errors of F
    and log J errors against the analytic differential."""
    Yn = sphere.exp_map(Yc, sphere.tangent_project(Yc, sigma * rng.standard_normal(Yc.shape)))
    N = len(X)
    out = {k: np.full(N, np.nan) for k in ("anch_endpoint", "ic_endpoint", "anch_edge", "ic_edge",
                                           "logJ_anch_endpoint", "logJ_ic_endpoint", "logJ_anch_edge",
                                           "logJ_ic_edge", "logJ_anch_clean", "logJ_ic_clean")}
    for rows, idx, w in graph.iter_rows(op, batch=512):
        b, k = idx.shape
        a = w / (2.0 * op.tau[rows])[:, None]
        u = sphere.log_map(X[rows][:, None, :], X[idx])
        y_end = sphere.log_map(Yn[rows][:, None, :], Yn[idx])
        yc = sphere.log_map(Yc[rows][:, None, :], Yc[idx])
        Yi = np.broadcast_to(Yc[rows][:, None, :], yc.shape)
        y_edge = yc + sphere.tangent_project(Yi, np.sqrt(2.0) * sigma * rng.standard_normal(yc.shape))
        nonself = idx != rows[:, None]
        y_edge = np.where(nonself[..., None], y_edge, 0.0)
        Gi = f0.ginv(rows)
        Bm = f0.basis[rows] * f0.mask[rows][:, None, :]
        pad = (~f0.mask[rows]) * f0.eig[rows, :1]
        G = Bm @ (f0.eig[rows][:, :, None] * np.swapaxes(Bm, 1, 2))
        Ft = Fx_true[rows]
        den = np.einsum("bij,bjk,bik->b", Ft, G, Ft)
        U = f0.basis[rows]
        for tag, yy in (("endpoint", y_end), ("edge", y_edge), ("clean", yc)):
            Aa = regress(a, u, yy, Gi)["A"]
            Ai = step._centred_fit(a * nonself, u, yy, Bm, pad)[0]
            for fit, A in (("anch", Aa), ("ic", Ai)):
                sv = np.linalg.svd(A @ U, compute_uv=False)
                with np.errstate(divide="ignore"):
                    out[f"logJ_{fit}_{tag}"][rows] = np.log(sv).sum(1)
                if tag != "clean":
                    D = A - Ft
                    out[f"{fit}_{tag}"][rows] = np.einsum("bij,bjk,bik->b", D, G, D) / den
    return out


def _clean_rotation(rng):
    A = fx.random_skew(10, rng)
    from scipy.linalg import expm
    Rm = expm(0.3 * A / np.linalg.norm(A, 2))
    return lambda X: X @ Rm.T


def build_group(group: str, seed: int):
    """Returns (X, base object with .f/.exact, list of (variant name, kind, params))."""
    rng = np.random.default_rng(1000 * seed + 7)
    if group in ("weakstretch", "weakcomp"):
        # blob axis 9 (smallest sd) -> stretch rank 1 (largest) or rank 9 (strongest compression)
        Q = np.eye(9)
        perm = [8, 0, 1, 2, 3, 4, 5, 6, 7] if group == "weakstretch" else list(range(9))
        Q = Q[:, perm]
        blob = fx.MatchedBlob(rng, SCALE, Q=Q, eps=0.25)
    elif group in ("rank8", "nearsing"):
        ls = fx.MATCHED_LOG_STRETCH.copy()
        ls[-1] = -np.inf if group == "rank8" else -5.0
        blob = fx.MatchedBlob(rng, SCALE, log_stretch=ls)
    elif group == "curved":
        blob = fx.MatchedBlob(rng, SCALE, ell=0.1)
    else:
        blob = fx.MatchedBlob(rng, SCALE)
    X = blob.sample(N_POINTS, rng)
    obj = blob
    if group == "vertex":
        V = R * np.eye(10)
        dirs = rng.standard_normal((4, 9)) @ blob.B.T          # 4 pieces of the blob, each to its vertex
        verts = rng.choice(10, 4, replace=False)

        def fv(Z, t=0.99):
            cb = np.broadcast_to(blob.c, Z.shape)
            piece = np.argmax(sphere.log_map(cb, Z) @ dirs.T, axis=1)
            tgt = V[verts[piece]]
            return sphere.exp_map(Z, t * sphere.log_map(Z, tgt))

        obj = fx.MapOnBlob(blob, fv)
    elif group == "fold":
        pass
    elif group == "controls":
        pass
    variants = {
        "smooth": [("clean", "clean", {}), *[(f"endpoint{nu}", "endpoint", dict(nu=nu)) for nu in NUS],
                   ("edge0.6", "edge", dict(nu=0.6)), ("shuffled", "shuffled", {})],
        "weakstretch": [("clean", "clean", {}), ("endpoint0.6", "endpoint", dict(nu=0.6))],
        "weakcomp": [("clean", "clean", {}), ("endpoint0.6", "endpoint", dict(nu=0.6)),
                     ("along0.6", "along", dict(nu=0.6)), ("orth0.6", "orth", dict(nu=0.6))],
        "rank8": [("clean", "clean", {}), ("endpoint0.6", "endpoint", dict(nu=0.6))],
        "nearsing": [("clean", "clean", {}), ("endpoint0.6", "endpoint", dict(nu=0.6))],
        "vertex": [("clean", "clean", {}), ("endpoint3", "endpoint", dict(nu=3.0)), ("endpoint10", "endpoint", dict(nu=10.0))],
        "fold": [("fold_random", "fold", dict(axis="random")), ("fold_major", "fold", dict(axis="major"))],
        "curved": [("clean", "clean", {})],
        "controls": [("identity", "identity", {}), ("rotation", "rotation", {}),
                     ("identity_noise0.6", "identity_noise", dict(nu=0.6))],
    }[group]
    return X, obj, variants, rng


def fold_obj(blob, axis, rng):
    d = rng.standard_normal(9) if axis == "random" else np.eye(9)[0]
    d /= np.linalg.norm(d)

    def ff(Z):
        cb = np.broadcast_to(blob.c, Z.shape)
        z = sphere.log_map(cb, Z) @ blob.B
        s = z @ d
        z = z - 2.0 * np.minimum(s, 0.0)[:, None] * d[None, :]         # reflect the negative half
        return sphere.exp_map(cb, (blob.g(z) @ blob.L.T) @ blob.B.T)

    return fx.MapOnBlob(blob, ff)


def run_group(group, seed, k, eps, outdir, n_jobs=4):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    X, obj, variants, rng = build_group(group, seed)
    rules = Rules.flow()
    opkw = dict(step.DEFAULT_OPERATOR, k=k, eps=eps)
    op0 = graph.build_operator(X, n_jobs=n_jobs, **opkw)
    f0 = analyse_frame(op0, rules, n_jobs=n_jobs)
    fs = analyse_frame(op0, Rules.strict(), n_jobs=n_jobs)
    ft = analyse_frame(op0, Rules.thick(), n_jobs=n_jobs)
    G = f0.basis @ (f0.eig[:, :, None] * np.swapaxes(f0.basis, 1, 2))
    regime = dict(
        group=group, seed=seed, k=k, eps=eps, N=len(X), scale=SCALE,
        spectrum_median=np.median(f0.eig / f0.eig[:, :1], 0).tolist(),
        centred_spectrum_median=np.median(f0.eig_c / f0.eig_c[:, :1], 0).tolist(),
        knn_radius_median=float(np.median(op0.knn_radius)), radius_quantiles=np.quantile(f0.radius, [.1, .5, .9]).tolist(),
        n_eff_median=float(np.median(f0.n_eff)), cond_quantiles=np.nanquantile(f0.cond, [.1, .5, .9]).tolist(),
        flow_dim9=float(np.mean(f0.dim == 9)), strict_dim=np.bincount(fs.dim, minlength=10).tolist(),
        one_sided_median=float(np.nanmedian(fs.one_sided)), trunc_tail_median=float(np.nanmedian(fs.trunc_tail)),
        strict_tangent=float(fs.ok_tangent.mean()), thick_tangent=float(ft.ok_tangent.mean()),
        thick_rank_stable=float(ft.rank_stable.mean()), tau_median=float(np.median(op0.tau)))
    results = dict(regime=regime, variants={})
    clean_est = None
    sigma_ref = None
    for name, kind, par in variants:
        vobj = obj
        if kind == "fold":
            vobj = fold_obj(obj, par["axis"], rng)
        if kind in ("identity", "identity_noise"):
            Y = X.copy()
            vobj = fx.MapOnBlob(obj, lambda Z: Z.copy())
            if kind == "identity_noise":
                sig = par["nu"] * float(np.median(f0.radius)) / np.sqrt(18.0)
                Y = sphere.exp_map(X, sphere.tangent_project(X, sig * rng.standard_normal(X.shape)))
        elif kind == "rotation":
            fr = _clean_rotation(rng)
            vobj = fx.MapOnBlob(obj, fr)
            Y = fr(X)
        else:
            Yc = vobj.f(X)
            if sigma_ref is None:
                op_c = graph.build_operator(Yc, n_jobs=n_jobs, **opkw)
                sigma_ref = float(np.median(analyse_frame(op_c, rules, n_jobs=n_jobs).radius)) / np.sqrt(18.0)
            sig = par.get("nu", 0.0) * sigma_ref
            if kind in ("clean", "fold"):
                Y = Yc
            elif kind == "endpoint":
                Y = sphere.exp_map(Yc, sphere.tangent_project(Yc, sig * rng.standard_normal(Yc.shape)))
            elif kind in ("along", "orth"):
                # image of the weakest source axis (blob axis 9) under the linear part, at the centre
                dimg = obj.B @ (obj.L @ np.eye(9)[8])
                dimg = dimg / np.linalg.norm(dimg)
                xi = sig * rng.standard_normal(Yc.shape)
                d_at = sphere.tangent_project(Yc, np.broadcast_to(dimg, Yc.shape))
                d_at /= np.linalg.norm(d_at, axis=1, keepdims=True)
                along = np.sum(xi * d_at, 1, keepdims=True) * d_at
                if kind == "along":
                    xi = 3.0 * along                      # same total variance as isotropic (9 coords)
                else:
                    xi = sphere.tangent_project(Yc, xi) - along
                    xi *= np.sqrt(9.0 / 8.0)
                Y = sphere.exp_map(Yc, xi)
            elif kind == "shuffled":
                Y = Yc[rng.permutation(len(Yc))]
            elif kind == "edge":
                tru, ex, Fx, Kx = exact_fields(vobj, X, G)
                ev = edge_vs_endpoint(op0, f0, X, Yc, sig, rng, Fx)
                np.savez_compressed(outdir / f"{group}_s{seed}_k{k}_e{eps}_{name}.npz",
                                    **{k_: v.astype(np.float32) for k_, v in ev.items()},
                                    logJ_true=tru["logJ"].astype(np.float32))
                results["variants"][name] = dict(kind=kind, sigma=sig, seconds=time.time() - t0)
                print(f"{group} s{seed} k{k} {name}: done t={time.time() - t0:.0f}s", flush=True)
                continue
        op1 = graph.build_operator(Y, n_jobs=n_jobs, **opkw)
        f1 = analyse_frame(op1, rules, n_jobs=n_jobs)
        o = step.measure_step(X, Y, 1.0, frame0=f0, frame1=f1, rules=rules, mask=False, n_jobs=n_jobs)
        arr = {key: np.asarray(o[key], np.float64) for key in EST_KEYS}
        for kk in range(9):
            arr[f"ls{kk + 1}"] = o["log_stretches"][:, kk]
            arr[f"ic_ls{kk + 1}"] = o["ic_log_stretches"][:, kk]
        arr["rank_loss"] = o["rank_loss"].astype(float)
        arr["dim"] = o["dim"].astype(float)
        for r in ("tangent", "strain", "strain_finite", "logJ", "relmotion", "relmotion_split", "strain_finite_ic",
                  "strain_ic", "gain_ic", "relmotion_fitsplit"):
            arr["rel_" + r] = o["release"][r].astype(float)
        if kind != "shuffled":
            tru, ex, Fx, Kx = exact_fields(vobj, X, G)
            for key, val in tru.items():
                arr["true_" + key] = val
            arr["true_heldout"] = oracle_score(op0, X, Y, Fx)
            Gc = G - 0.5 * op0.tau[:, None, None] * f0.drift[:, :, None] * f0.drift[:, None, :]
            arr["true_relmotion_slope"] = np.einsum("nij,njk,nik->n", Kx, Gc, Kx)
            D = o["F"] - Fx
            den = np.einsum("nij,njk,nik->n", Fx, G, Fx)
            arr["F_err_pw"] = np.einsum("nij,njk,nik->n", D, G, D) / den
            arr["F_err_fro"] = np.linalg.norm(D, axis=(1, 2)) / np.linalg.norm(Fx, axis=(1, 2))
            Di = o["F_ic"] - Fx
            arr["Fic_err_pw"] = np.einsum("nij,njk,nik->n", Di, G, Di) / den
            DK = o["K_d"] - Kx
            arr["K_err_pw"] = (np.einsum("nij,njk,nik->n", DK, G, DK)
                               / np.einsum("nij,njk,nik->n", Kx, G, Kx))
        if kind == "clean":
            clean_est = {key: arr[key].copy() for key in arr if not key.startswith("true_")}
        elif clean_est is not None and kind in ("endpoint", "along", "orth"):
            for key in ("logJ", "E_trace", "strain_sq", "spin_sq", "relmotion_fit", "trFGF", *[f"ls{j}" for j in range(1, 10)]):
                arr["secant_" + key] = clean_est[key]
        np.savez_compressed(outdir / f"{group}_s{seed}_k{k}_e{eps}_{name}.npz",
                            **{k_: np.asarray(v, np.float32) for k_, v in arr.items()})
        results["variants"][name] = dict(kind=kind, sigma=float(par.get("nu", 0.0) * (sigma_ref or 0.0)),
                                         heldout_median=float(np.nanmedian(o["heldout"])),
                                         seconds=time.time() - t0)
        print(f"{group} s{seed} k{k} {name}: heldout median {np.nanmedian(o['heldout']):.3f} "
              f"t={time.time() - t0:.0f}s", flush=True)
    (outdir / f"{group}_s{seed}_k{k}_e{eps}.json").write_text(json.dumps(results, indent=1, default=float))
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="smooth")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=256)
    ap.add_argument("--eps", type=float, default=0.25)
    ap.add_argument("--out", required=True)
    ap.add_argument("--N", type=int, default=10_000)
    a = ap.parse_args()
    N_POINTS = a.N
    run_group(a.group, a.seed, a.k, a.eps, a.out)
