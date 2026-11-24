"""Summarise the matched-regime fixture outputs (tests/matched.py) into tables and gate evaluations.

    python tests/matched_report.py DIR [--out report.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# per-field tolerance on |estimate - truth| (fixed before looking at seed 1; "rel" = relative to the median
# absolute truth over the cloud)
TOL = {
    "fitted_fraction": ("abs", 0.05),
    "heldout": ("abs", 0.05),
    "ls1": ("abs", 0.1),
    "ls5": ("abs", 0.1),
    "ls9": ("abs", 0.25),
    "logJ": ("abs", 0.3),
    "fin_shear": ("rel", 0.2),
    "fin_vol": ("rel", 0.3),
    "E_trace": ("relscale", 0.1),     # relative to the median |E| (sqrt rate_total) of the truth
    "strain_sq_db": ("rel", 0.2),
    "spin_sq_db": ("rel", 0.2),
    "relmotion_fit": ("rel", 0.1),
}
FIELDS_MAIN = ["fitted_fraction", "heldout", "ls1", "ls5", "ls9", "logJ", "fin_vol", "fin_shear", "fin_total",
               "E_trace", "rate_total", "strain_sq_db", "spin_sq", "spin_sq_db", "relmotion_fit"]


def load(path):
    z = np.load(path)
    d = {k: z[k].astype(np.float64) for k in z.files}
    if "relmotion" in d and "true_relmotion_fit" in d:
        d["fitted_fraction"] = d["relmotion_fit"] / d["relmotion"]
        d["true_fitted_fraction"] = d["true_relmotion_fit"] / d["relmotion"]
    if "true_strain_sq" in d:
        d["true_strain_sq_db"] = d["true_strain_sq"]
        d["true_spin_sq_db"] = d["true_spin_sq"]
        d["true_rate_total"] = d["true_rate_total"]
    return d


def tol_abs(d, key):
    kind, t = TOL[key]
    tru = d["true_" + key]
    if kind == "abs":
        return t
    if kind == "rel":
        return t * np.nanmedian(np.abs(tru[np.isfinite(tru)]))
    return t * np.nanmedian(np.sqrt(d["true_rate_total"]))


def stats(d, key, sel=None, secant=False):
    est = d.get(key)
    tru = d.get(("secant_" if secant else "true_") + key)
    if est is None or tru is None:
        return None
    ok = np.isfinite(est) & np.isfinite(tru)
    if sel is not None:
        ok &= sel
    if ok.sum() < 20:
        return dict(n=int(ok.sum()))
    e = est[ok] - tru[ok]
    scale = np.nanmedian(np.abs(tru[ok])) if np.any(tru[ok] != 0) else 1.0
    return dict(n=int(ok.sum()), bias=float(np.median(e)), scatter=float(1.4826 * np.median(np.abs(e - np.median(e)))),
                q90_abs=float(np.quantile(np.abs(e), 0.9)), mean_err=float(est[ok].mean() - tru[ok].mean()),
                true_median=float(np.median(tru[ok])), true_sd=float(tru[ok].std()), scale=float(scale))


def gate_eval(d, key, sel, tol):
    est, tru = d[key], d["true_" + key]
    ok = np.isfinite(est) & np.isfinite(tru)
    rs = sel & ok
    n = len(est)
    if rs.sum() == 0:
        return dict(coverage=0.0)
    err = np.abs(est[rs] - tru[rs])
    return dict(coverage=float(rs.sum() / n), false_release=float(np.mean(err > tol)),
                err_q=[float(q) for q in np.quantile(err, [0.5, 0.9, 0.99])],
                released_mean_bias=float(est[rs].mean() - tru[rs].mean()),
                released_true_mean=float(tru[rs].mean()), all_true_mean=float(tru[ok].mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--out")
    a = ap.parse_args()
    D = Path(a.dir)
    rep = {}

    def F(group, seed, var, k=256, eps=0.25):
        p = D / f"{group}_s{seed}_k{k}_e{eps}_{var}.npz"
        return load(p) if p.exists() else None

    # 1. regime
    rep["regime"] = {p.stem: json.loads(p.read_text())["regime"] for p in sorted(D.glob("*.json"))
                     if p.stem.startswith(("smooth_s0", "controls_s0"))}
    # 2. accuracy by variant (all resolved points; secant truth for noisy variants)
    acc = {}
    for seed in (0, 1):
        for var in ("clean", "endpoint0.3", "endpoint0.6", "endpoint1.0", "endpoint1.5"):
            d = F("smooth", seed, var)
            if d is None:
                continue
            row = dict(heldout_median=float(np.nanmedian(d["heldout"])),
                       oracle_median=float(np.nanmedian(d["true_heldout"])))
            for key in FIELDS_MAIN + ["ic_logJ", "ic_ls9", "ic_E_trace", "ic_spin_sq_db"]:
                kk = key
                if key.startswith("ic_"):
                    base = key[3:]
                    d["true_" + key] = d.get("true_" + base)
                    if "secant_" + base in d:
                        d["secant_" + key] = d["secant_" + base]
                row[kk] = stats(d, kk)
                if "secant_" + kk in d:
                    row[kk + "@secant"] = stats(d, kk, secant=True)
            row["F_err_pw_median"] = float(np.nanmedian(d["F_err_pw"]))
            row["Fic_err_pw_median"] = float(np.nanmedian(d["Fic_err_pw"]))
            row["F_err_fro_median"] = float(np.nanmedian(d["F_err_fro"]))
            row["K_err_pw_median"] = float(np.nanmedian(d["K_err_pw"]))
            acc[f"s{seed}:{var}"] = row
    rep["accuracy_smooth_k256"] = acc
    # 3. baselines
    base = {}
    for group, var in (("smooth", "clean"), ("smooth", "endpoint0.6"), ("smooth", "endpoint1.5"),
                       ("smooth", "shuffled"), ("vertex", "clean"), ("vertex", "endpoint10"), ("fold", "fold_random"),
                       ("curved", "clean"), ("controls", "identity"), ("controls", "rotation"), ("rank8", "clean")):
        d = F(group, 0, var)
        if d is None:
            continue
        base[f"{group}:{var}"] = {k: float(np.nanmedian(d[k])) for k in
                                  ("heldout", "heldout_ic", "heldout_const", "heldout_identity", "heldout_top3",
                                   "heldout_top6", "heldout_v", "heldout_v_ic", "heldout_v_const", "intercept_rel",
                                   "intercept_v_rel", "explained", "cond", "cond_centred") if k in d}
    rep["baselines_median"] = base
    # 4. edge vs endpoint
    for seed in (0, 1):
        p = D / f"smooth_s{seed}_k256_e0.25_edge0.6.npz"
        if p.exists():
            z = np.load(p)
            t = z["logJ_true"]
            rep[f"edge_vs_endpoint_s{seed}"] = {
                **{k: float(np.nanmedian(z[k])) for k in ("anch_endpoint", "ic_endpoint", "anch_edge", "ic_edge")},
                **{f"logJ_bias_{k[5:]}": float(np.nanmedian(z[k] - t)) for k in z.files if k.startswith("logJ_") and k != "logJ_true"}}
    # 5. contrasts
    con = {}
    for group, vars_ in (("weakstretch", ("clean", "endpoint0.6")), ("weakcomp", ("clean", "endpoint0.6", "along0.6", "orth0.6")),
                         ("rank8", ("clean", "endpoint0.6")), ("nearsing", ("clean", "endpoint0.6")),
                         ("vertex", ("clean", "endpoint3", "endpoint10")), ("fold", ("fold_random", "fold_major")),
                         ("curved", ("clean",)), ("controls", ("identity", "rotation"))):
        for var in vars_:
            d = F(group, 0, var)
            if d is None:
                continue
            row = dict(heldout_median=float(np.nanmedian(d["heldout"])), oracle=float(np.nanmedian(d["true_heldout"])),
                       heldout_v_median=float(np.nanmedian(d["heldout_v"])))
            for key in ("ls1", "ls9", "logJ", "ic_logJ", "E_trace", "strain_sq_db", "spin_sq_db", "fitted_fraction"):
                if key == "ic_logJ":
                    d["true_ic_logJ"] = d["true_logJ"]
                row[key] = stats(d, key)
            rl = d["rank_loss"] > 0
            true_ls9 = d["true_ls9"]
            row["rank_loss_frac"] = float(rl.mean())
            row["logJ_finite_frac"] = float(np.isfinite(d["logJ"]).mean())
            row["true_ls9_median"] = float(np.nanmedian(true_ls9))
            row["est_ls9_median"] = float(np.nanmedian(d["ls9"]))
            row["est_ls9_q"] = [float(x) for x in np.nanquantile(d["ls9"], [0.1, 0.9])]
            if group in ("controls",):
                row["strain_sq_raw_median"] = float(np.nanmedian(d["strain_sq"]))
                row["strain_sq_db_median"] = float(np.nanmedian(d["strain_sq_db"]))
                row["spin_sq_true_median"] = float(np.nanmedian(d["true_spin_sq"]))
            con[f"{group}:{var}"] = row
    rep["contrasts"] = con
    # 6. scale
    sc = {}
    for k, eps in ((128, 0.25), (256, 0.25), (512, 0.25), (512, 1.0)):
        for var in ("clean", "endpoint0.6"):
            d = F("smooth", 0, var, k, eps)
            if d is None:
                continue
            sc[f"k{k}_e{eps}:{var}"] = dict(radius=float(np.nanmedian(d["radius"])),
                                            heldout=float(np.nanmedian(d["heldout"])),
                                            **{key: stats(d, key) for key in ("ls1", "ls9", "logJ", "E_trace", "fitted_fraction")})
    rep["scale"] = sc
    if a.out:
        def js(x):
            if isinstance(x, dict):
                return {k: js(v) for k, v in x.items()}
            if isinstance(x, list):
                return [js(v) for v in x]
            return None if isinstance(x, float) and not np.isfinite(x) else x
        Path(a.out).write_text(json.dumps(js(rep), indent=1, allow_nan=False))
    print(json.dumps(rep, indent=1)[:200])


if __name__ == "__main__":
    main()
