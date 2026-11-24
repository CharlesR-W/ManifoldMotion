"""Per-step totals (SPEC item 13): means over images (equal mass 1/N) of the per-point fields, each with
the mass fraction on which it is defined and the median physical radius of that set.

A total is the mean over the points where the field's release rule holds (and the value is finite); its
`mass_fraction` is that set's share of all images (or of the group, for grouped totals).  Additive splits
are also given on a common set so that their parts add up exactly:
  kinetic energy    ke = ke_tan + ke_norm on the 'normal' set; ke = ke_rigid + ke_nonrigid + ke_cross on
                    every set (the rigid fit is global: ke_cross averages to 0 over all images only, so
                    within a group the fitted and residual energies are components of the global fit)
  relative motion   relmotion = relmotion_fit + relmotion_res on the 'relmotion_split' set
  fitted gradient   grad_sq = strain_sq + spin_sq + tilt_sq on the 'tilt_rate' set (m < 9), and
                    grad_sq = strain_sq + spin_sq on the 'grad_full' set (m = 9: tilt not applicable)
  strain (rate)     rate_total = rate_vol + rate_shear on the 'strain' set
  strain (finite)   fin_total = fin_vol + fin_shear on the 'logJ' set
Kinetic energy is (1/2)|V|^2 per unit mass; squared rates are per dt^2.
"""
from __future__ import annotations

import numpy as np

from .step import measure_step, release_mask

TOTAL_FIELDS = [
    "ke", "ke_tan", "ke_norm", "ke_rigid", "ke_nonrigid", "ke_cross", "relmotion", "relmotion_fit", "relmotion_res",
    "grad_sq", "strain_sq", "spin_sq", "tilt_sq", "rate_vol", "rate_shear", "rate_total", "fin_vol",
    "fin_shear", "fin_total", "logJ", "travel", "cr_ke_tan", "cr_ke_norm", "cr_relmotion", "cr_strain_sq",
    "cr_spin_sq", "cr_tilt_sq", "tilt_max", "cr_tilt_max", "E_trace", "E_flow_trace", "E_normal_trace",
    "strain_sq_db", "spin_sq_db", "explained", "heldout", "heldout_v", "heldout_identity", "heldout_const",
    "heldout_ic", "heldout_top3", "heldout_top6", "heldout_v_const", "heldout_v_ic", "intercept_rel",
    "intercept_v_rel", "tan_fraction", "relmotion_shift", "relmotion_slope", "relmotion_icres", "ic_log_gain_max",
    "ic_E_trace", "ic_strain_sq", "ic_spin_sq",
]

SPLITS = {
    "ke_tangential_normal": ("normal", ["ke", "ke_tan", "ke_norm"]),
    "ke_rigid_nonrigid": ("always", ["ke", "ke_rigid", "ke_nonrigid", "ke_cross"]),
    "relmotion_fit_res": ("relmotion_split", ["relmotion", "relmotion_fit", "relmotion_res"]),
    # free-intercept budget (flow reading): shared shift + linear slope + unexplained, exact
    "relmotion_shift_slope_res": ("relmotion_fitsplit",
                                  ["relmotion", "relmotion_shift", "relmotion_slope", "relmotion_icres"]),
    "gradient_strain_spin_tilt": ("tilt_rate", ["grad_sq", "strain_sq", "spin_sq", "tilt_sq"]),
    # full rank (flow reading): tilt is not applicable (zero by construction, not evidence of no tilt)
    "gradient_strain_spin_fullrank": ("grad_full", ["grad_sq", "strain_sq", "spin_sq"]),
    "strain_rate_vol_shear": ("strain", ["rate_total", "rate_vol", "rate_shear"]),
    "strain_finite_vol_shear": ("logJ", ["fin_total", "fin_vol", "fin_shear"]),
    "strain_attribution": ("attribution", ["E_trace", "E_flow_trace", "E_normal_trace"]),
}


def _summary(vals: np.ndarray, sel: np.ndarray, radius: np.ndarray, denom: int) -> dict:
    s = sel & np.isfinite(vals)
    n = int(s.sum())
    return dict(value=float(np.mean(vals[s])) if n else float("nan"),
                mass_fraction=n / max(denom, 1),
                radius=float(np.median(radius[s])) if n else float("nan"))


def step_totals(o: dict, subset: np.ndarray | None = None) -> dict:
    """Totals of one measure_step result (masked or raw).  Returns {name: {value, mass_fraction, radius}}
    for TOTAL_FIELDS, the additive SPLITS on common sets, the full-rank fraction and the global rigid fits."""
    rel = o["release"]
    N = len(o["travel"])
    sub = np.ones(N, dtype=bool) if subset is None else np.asarray(subset, dtype=bool)
    denom = int(sub.sum())
    radius = o["radius"]
    out = {}
    for f in TOTAL_FIELDS:
        out[f] = _summary(np.asarray(o[f], dtype=np.float64), sub & release_mask(o, f), radius, denom)
    for name, (rule, fields) in SPLITS.items():
        sel = sub & rel[rule]
        for f in fields:
            sel &= release_mask(o, f) & np.isfinite(np.asarray(o[f], dtype=np.float64))
        out["split:" + name] = {f: _summary(np.asarray(o[f], dtype=np.float64), sel, radius, denom)
                                for f in fields}
    out["full_rank_fraction"] = float(np.mean(o["full_rank"][sub])) if denom else float("nan")
    # global fits (all images), whatever the subset
    out["rigid_fraction_global"] = o["rigid"]["rigid_fraction"]
    out["procrustes_fraction_global"] = o["rigid"]["procrustes_fraction"]
    return out


def diag_totals(o: dict, names, subset: np.ndarray | None = None) -> dict:
    """Totals of fields under their release *rule only*, ignoring the mode allowlist: for diagnostics of
    quantities a mode withholds (e.g. the anchored fit in flow mode).  Same structure as step_totals."""
    from .step import FIELD_RULE
    rel = o["release"]
    N = len(o["travel"])
    sub = np.ones(N, dtype=bool) if subset is None else np.asarray(subset, dtype=bool)
    out = {}
    for f in names:
        out[f] = _summary(np.asarray(o[f], dtype=np.float64), sub & rel[FIELD_RULE.get(f, "always")], o["radius"],
                          int(sub.sum()))
    return out


def grouped_totals(o: dict, groups: np.ndarray) -> dict:
    """step_totals within each group (class, correct/incorrect, distance-to-vertex bin, ...); the mass
    fractions are within the group."""
    groups = np.asarray(groups)
    return {g.item() if hasattr(g, "item") else g: step_totals(o, groups == g) for g in np.unique(groups)}


def subsample_totals(X0: np.ndarray, X1: np.ndarray, dt: float = 1.0, n_rep: int = 4, frac: float = 0.5,
                     seed: int = 0, **kw) -> dict:
    """Paired-subsample variation of the totals: n_rep subsets of round(frac N) images drawn without
    replacement, the same images in both frames, operators rebuilt per subset (per-point res0 / res1 /
    labels in kw are subset too).  Returns {name: {mean, std, values, mass_fraction, radius}} for the
    TOTAL_FIELDS and, under 'split:<name>', for every part of the additive splits."""
    rng = np.random.default_rng(seed)
    N = X0.shape[0]
    size = int(round(frac * N))
    reps = []
    for _ in range(n_rep):
        idx = np.sort(rng.choice(N, size=size, replace=False))
        kwi = dict(kw)
        for key in ("res0", "res1", "labels"):
            if kwi.get(key) is not None and np.ndim(kwi[key]) > 0:
                kwi[key] = np.asarray(kwi[key])[idx]
        o = measure_step(X0[idx], X1[idx], dt, mask=True, **kwi)
        reps.append(step_totals(o))
    out = {}

    def summ(vals, mf, rad):
        vals = np.asarray(vals)
        return dict(mean=float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan"),
                    std=float(np.nanstd(vals, ddof=1)) if np.isfinite(vals).sum() > 1 else float("nan"),
                    values=vals, mass_fraction=np.asarray(mf), radius=np.asarray(rad))

    for f in TOTAL_FIELDS:
        out[f] = summ([r[f]["value"] for r in reps], [r[f]["mass_fraction"] for r in reps],
                      [r[f]["radius"] for r in reps])
    for name in SPLITS:
        key = "split:" + name
        out[key] = {f: summ([r[key][f]["value"] for r in reps], [r[key][f]["mass_fraction"] for r in reps],
                            [r[key][f]["radius"] for r in reps]) for f in reps[0][key]}
    return out
