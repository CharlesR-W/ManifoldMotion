"""Release table of the matched fixture under the frozen flow gates (mm.frame.FLOW_GATES, Rules.flow) as
implemented in the code that produced the fixture run (tests/matched.py, tag v4: release masks saved by the
library itself).  Writes final_eval.json:

  [f"s{seed}:{group}:{variant}"][field] (top-level keys, the final release table) = {coverage, false_release,
      err_q50, err_q90, bias_median, released_mean_bias, released_true_mean, all_true_mean}, plus
      heldout_ic_median and oracle_median per variant
      coverage       fraction of all 10,000 points released for this field
      false_release  fraction of released points with |estimate - analytic truth| > tolerance
  pre_identity_gate[...]["ic_log_gain_max"]  the same for the largest log gain released by the free-intercept
      fit gates alone, i.e. before the '0.3 above the no-deformation score' gate (masks: strain_finite_ic)
  tolerances, gates, masks, provenance

    python final_eval.py FIXTURE_DIR OUT_JSON
"""
import json, sys
import numpy as np
sys.path.insert(0, "/home/ubuntu/code/ManifoldMotion")
from mm.frame import FLOW_GATES

D, OUT = sys.argv[1], sys.argv[2]


def L(g, s, v, k=256, e=0.25):
    try:
        z = np.load(f"{D}/{g}_s{s}_k{k}_e{e}_{v}.npz")
    except FileNotFoundError:
        return None
    d = {a: z[a].astype(float) for a in z.files}
    with np.errstate(invalid="ignore", divide="ignore"):
        d["slope_fraction"] = d["relmotion_slope"] / d["relmotion"]
        if "true_relmotion_slope" in d:
            d["true_slope_fraction"] = d["true_relmotion_slope"] / d["relmotion"]
    return d


ref = L("smooth", 0, "clean")
Escale = float(np.nanmedian(np.sqrt(ref["true_rate_total"])))
# field: (truth key, tolerance, library release mask saved by tests/matched.py)
TOL = {"ic_log_gain_max": ("true_ls1", 0.1, "rel_gain_ic"),
       "ic_E_trace": ("true_E_trace", 0.2 * Escale, "rel_strain_ic"),
       "ic_strain_sq": ("true_strain_sq", 0.2 * float(np.nanmedian(ref["true_strain_sq"])), "rel_strain_ic"),
       "ic_spin_sq": ("true_spin_sq", 0.2 * float(np.nanmedian(ref["true_spin_sq"])), "rel_strain_ic"),
       "slope_fraction": ("true_slope_fraction", 0.05, "rel_relmotion_fitsplit"),
       "heldout_ic": ("true_heldout", 0.05, "rel_relmotion_fitsplit")}
VARS = [("smooth", "clean"), ("smooth", "endpoint0.3"), ("smooth", "endpoint0.6"), ("smooth", "endpoint1.0"),
        ("smooth", "endpoint1.5"), ("weakstretch", "clean"), ("weakstretch", "endpoint0.6"), ("weakcomp", "clean"),
        ("weakcomp", "endpoint0.6"), ("weakcomp", "along0.6"), ("weakcomp", "orth0.6"), ("rank8", "clean"),
        ("rank8", "endpoint0.6"), ("nearsing", "clean"), ("nearsing", "endpoint0.6"), ("vertex", "clean"),
        ("vertex", "endpoint3"), ("vertex", "endpoint10"), ("fold", "fold_random"), ("fold", "fold_major"),
        ("curved", "clean"), ("controls", "identity"), ("controls", "rotation"), ("controls", "identity_noise0.6"),
        ("smooth", "shuffled")]


def evaluate(d, f, tk, tol, rule):
    sel = d[rule] > 0
    est = d[f]
    cov = float(np.mean(sel & np.isfinite(est)))
    if tk not in d:
        return dict(coverage=cov)
    ok = sel & np.isfinite(est) & np.isfinite(d[tk])
    if ok.sum() < 10:
        return dict(coverage=cov, n_released=int(ok.sum()))
    e = est[ok] - d[tk][ok]
    return dict(coverage=cov, n_released=int(ok.sum()), false_release=float(np.mean(np.abs(e) > tol)),
                err_q50=float(np.quantile(np.abs(e), .5)), err_q90=float(np.quantile(np.abs(e), .9)),
                bias_median=float(np.median(e)), released_mean_bias=float(e.mean()),
                released_true_mean=float(d[tk][ok].mean()), all_true_mean=float(d[tk][np.isfinite(d[tk])].mean()))


def _jsonable(x):
    """inf caps (disabled gates) -> None; NaN -> None (allow_nan=False output)."""
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


out = dict(provenance="tests/matched.py fixture run tag v4 (code after the five review fixes; masks = the "
                      "library's own release rules), k = 256, eps = 0.25, N = 10,000, seeds 0 and 1",
           gates=dict(FLOW_GATES, identity_exact_min=0.95), tolerances={k: v[1] for k, v in TOL.items()},
           masks={k: v[2] for k, v in TOL.items()}, pre_identity_gate={})
for seed in (0, 1):
    for g, v in VARS:
        d = L(g, seed, v)
        if d is None:
            continue
        key = f"s{seed}:{g}:{v}"
        row = dict(heldout_ic_median=float(np.nanmedian(d["heldout_ic"])),
                   oracle_median=float(np.nanmedian(d["true_heldout"])) if "true_heldout" in d else None)
        for f, (tk, tol, rule) in TOL.items():
            row[f] = evaluate(d, f, tk, tol, rule)
        out[key] = row
        out["pre_identity_gate"][key] = {"ic_log_gain_max": evaluate(d, "ic_log_gain_max", "true_ls1", 0.1,
                                                                     "rel_strain_finite_ic")}
out["notes"] = ("Final release table: the top-level keys 's<seed>:<group>:<variant>'. Released fields with a frozen "
    "tolerance: ic_log_gain_max, ic_E_trace, ic_strain_sq, ic_spin_sq. slope_fraction and heldout_ic are an in-sample "
    "budget and an out-of-sample score, released (mask relmotion_fitsplit) as measured quantities; their rows compare "
    "them with values built from the analytic derivative (slope part tr(K Gc K^T); oracle score with the true "
    "differential) and are characterisation, not release criteria. pre_identity_gate holds the largest log gain "
    "released by the free-intercept gates alone (mask strain_finite_ic), i.e. before the 'beat the no-deformation "
    "score by 0.3' gate (final mask gain_ic).")
idn = [out[f"s{sd}:controls:identity_noise0.6"]["ic_log_gain_max"] for sd in (0, 1)]
out["notes"] += (" Gate exception (Rules.flow, identity_exact_min = 0.95): the 0.3 margin over the no-deformation "
    "score is waived where that score is itself >= 0.95 (essentially undeformed, near noise-free neighbourhoods, e.g. "
    "a rigid rotation, which scores 0.97 and cannot clear a 0.3 margin; in this fixture the rotation control's gains are "
    "released through it, all within tolerance); no released value of the lens grid used it. "
    f"Noisy-identity control (controls:identity_noise0.6): after the gate the largest gain is released on "
    f"{idn[0]['n_released']} (seed 0) and {idn[1]['n_released']} (seed 1) of 10,000 points (0.4%), of which "
    f"{round(idn[0]['n_released'] * idn[0]['false_release'])} and {round(idn[1]['n_released'] * idn[1]['false_release'])} "
    "exceed the 0.1 tolerance: coverage is negligible and the accuracy of the few released points is poor. The "
    "statement 'at most 2-10% false releases under scatter' refers to the deforming-map variants (smooth, weak-axis, "
    "rank-8, near-singular, fold), not to this control.")
out["gates_null_means"] = ("a gate value of null is a disabled cap (no upper limit), e.g. cond_max_strain: flow mode "
                           "uses the centred-design cap cond_centred_max instead of the uncentred one")
json.dump(_jsonable(out), open(OUT, "w"), indent=1, allow_nan=False)
for key, row in [(k, v) for k, v in out.items() if k.startswith(("s0:", "s1:"))]:
    pre = out["pre_identity_gate"][key]["ic_log_gain_max"]
    parts = [f"{f[:9]} c{r['coverage']:.3f}" + (f" f{r['false_release']:.2f}" if "false_release" in r else "")
             for f, r in row.items() if isinstance(r, dict)]
    print(f"{key:32s} ho{row['heldout_ic_median']:.2f} | gain pre c{pre['coverage']:.3f}"
          + (f" f{pre['false_release']:.2f}" if 'false_release' in pre else '') + " | " + " | ".join(parts))
