"""Kinematic fields of the lens clouds (mm library) on Modal, CPU only, through a detached coordinator.

    fields_modal.py deploy
    fields_modal.py launch {pilot,grid,extras,all}
    fields_modal.py status {pilot,grid,extras,all} [--wait SECONDS]
    fields_modal.py fetch [--run r20s0]
    fields_modal.py manifest [--run r20s0]     (move speed-only placeholders aside, write + validate checkpoints.json)

Units (one CPU container each, 4 cores, 8 GB; idempotent: a unit whose done-marker exists is skipped):
  ckpt:<step>    seed 0, checkpoint <step>, k = 256: the depth clock (9 steps, per-point fields + totals under
                 the strict, flow and thick rule sets) and, if <step> is not the first checkpoint, the training
                 clock from the previous included checkpoint at each of the 10 sites (totals only, dt = the
                 difference in training steps)
  scale:<k>:<eps> seed 0, final checkpoint, depth-clock totals at another neighbourhood size / bandwidth
  subsample      seed 0, final checkpoint, depth-clock totals on 5 paired subsets of 5,000 images
                 (operators rebuilt)
  seed1          seed 1, final checkpoint, depth-clock totals (and per-point fields)
The pilot is ckpt:64000.  Volume mm-manifoldmotion: inputs /lens/r20s<seed>/, outputs /fields/r20s<seed>/.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from uuid import uuid4

import modal

HERE = Path(__file__).resolve().parent
APP = "mm-manifoldmotion-fields"
VOLUME = "mm-manifoldmotion"
RUN_RECORDS = HERE / "runs"
STEPS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 32, 64, 128, 256, 512, 1024, 1280, 2560, 3840,
         5120, 6400, 7680, 8960, 11520, 14080, 17920, 21760, 25600, 26880, 33280, 42240, 51200, 64000]
FINAL = 64000
SITES = ["stem", *(f"b{i}" for i in range(1, 10))]
K_GRID = 256
EPS = 0.25
MODES = ("strict", "flow", "thick")
UNIT_TIMEOUT, COORD_TIMEOUT, STARTUP, IDLE = 2700, 4 * 3600, 60, 2   # units took 16-29 min in the first run
MAX_CONTAINERS = 48
# On-demand USD/s from https://modal.com/pricing (CPU per core, memory per GiB), as recorded in lens_modal.py
CPU_RATE, RAM_RATE = 0.0000131, 0.00000222
WORKER_RATE = 4 * CPU_RATE + 8 * RAM_RATE
COORD_RATE = 0.25 * CPU_RATE + 0.5 * RAM_RATE
BUDGET = 15.0
STAGES = {
    "pilot": [f"ckpt:{FINAL}"],
    "grid": [f"ckpt:{s}" for s in STEPS],
    "extras": ["scale:128:0.25", "scale:512:0.25", "scale:512:1.0", "subsample", "seed1"],
}
STAGES["all"] = STAGES["grid"] + STAGES["extras"]
# units of the second run whose containers were preempted (and then skipped by the first claim guard)
STAGES["rest"] = ["ckpt:9", "ckpt:5120", "ckpt:11520", "ckpt:21760"]

app = modal.App(APP)
vol = modal.Volume.from_name(VOLUME)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy==2.5.5", "scipy==1.18.1", "scikit-learn==1.9.1")
    .env({"PYTHONPATH": "/root/proj"})
    .add_local_dir(str(HERE / "mm"), remote_path="/root/proj/mm", ignore=["__pycache__"])
)

# ----------------------------------------------------------------------------------------------------------
# exported per-point fields (row l of each array is the step from site l to site l + 1)

FIELDS = [
    # Motion
    dict(name="speed", label="Speed", group="Motion", scale="sequential",
         units="Fisher-Rao distance per block (vertex to vertex = π)",
         description="Fisher-Rao distance between the image's output distribution read at this site and at the "
                     "next (the length of the displacement). Defined at every point; no neighbourhood involved."),
    dict(name="nonrigid_speed", label="Non-rigid speed", group="Motion", scale="sequential",
         units="Fisher-Rao distance per block",
         description="Speed after removing the single finite rotation of the whole sphere that best carries this site's cloud onto the next (orthogonal Procrustes over all 10,000 images): the part of the image's move that is not a rigid turn of the whole cloud. The headline rigid fraction of kinetic energy uses a different fit (the least-squares instantaneous rotation field), so the two are not complements."),
    dict(name="path_length", label="Path length so far", group="Motion", scale="sequential",
         units="Fisher-Rao distance",
         description="Sum of the image's speeds over the steps from the stem up to and including this step "
                     "(so row l is the distance travelled from the stem to site l + 1 along the depth path)."),
    dict(name="toward_label", label="Move toward label vertex", group="Motion", scale="diverging",
         units="Fisher-Rao distance per block",
         description="Component of the image's displacement along the shortest path to the vertex of its true "
                     "class (one-hot distribution): positive = moving toward the correct class, negative = away. "
                     "Defined at every point."),
    # Neighbourhood
    dict(name="radius", label="Neighbourhood radius", group="Neighbourhood", scale="sequential",
         units="Fisher-Rao distance (rms over the neighbourhood weights)",
         description="Size of the neighbourhood every local quantity below is measured at: the root-mean-square "
                     "Fisher-Rao distance from the image to its neighbours under the graph weights (k = 256 "
                     "nearest neighbours, adaptive Gaussian kernel). Local quantities are at this scale, not "
                     "pointwise derivatives."),
    dict(name="local_dim", label="Local dimension (participation ratio)", group="Neighbourhood",
         scale="sequential", units="dimensions (1 to 9)",
         description="(sum of eigenvalues)^2 / (sum of squared eigenvalues) of the weighted second moment of "
                     "the offsets to the neighbours, in the 9-dimensional tangent space: about 9 for an "
                     "isotropic neighbourhood, about 1 for a line. A smooth count of the directions the "
                     "neighbourhood spreads in, with no cutoff."),
    dict(name="one_sided", label="One-sidedness", group="Neighbourhood", scale="sequential",
         units="neighbourhood mean / spread",
         description="Distance from the image to the weighted mean of its neighbours, in units of the "
                     "neighbourhood's own spread (all 9 directions). Near 0 in the middle of a cloud; near 1 "
                     "at an edge. For a 9-dimensional blob most points are at an edge in this sense."),
    dict(name="vertex_distance", label="Distance to nearest vertex", group="Neighbourhood", scale="sequential",
         units="Fisher-Rao distance (0 = one-hot)",
         description="Fisher-Rao distance from the image's distribution at this site to the nearest one-hot "
                     "distribution (simplex vertex): small = confident prediction of some class."),
    # Relative motion
    dict(name="relmotion", label="Relative motion", group="Relative motion", scale="sequential",
         units="(Fisher-Rao distance per block)^2 per unit kernel time",
         description="Weighted mean squared difference between the image's displacement and its neighbours' "
                     "displacements (each neighbour's carried over to this image along the shortest path), "
                     "divided by 2τ: how much the neighbourhood is torn apart or reshaped in this step. Directly "
                     "measured, no fit."),
    dict(name="nn_log_ratio", label="Log change of neighbour distances", group="Relative motion",
         scale="diverging", units="natural log (0 = unchanged, -1 = distances shrink by e)",
         description="Mean over the image's 16 nearest neighbours (at this site) of log(distance at the next "
                     "site / distance at this site). Direct, fit-free measure of local contraction (negative) or "
                     "expansion (positive) at the smallest scale; NaN where a distance is below numerical "
                     "resolution."),
    dict(name="shift_fraction", label="Relative motion: shared shift", group="Relative motion",
         scale="sequential", units="fraction (in-sample)",
         description="Fraction of the relative motion that is a shift common to the whole neighbourhood: the "
                     "weighted mean of the neighbours' displacement differences (carried to this image), i.e. the "
                     "image moving relative to its neighbourhood as a block. From the free-intercept fit of the "
                     "displacement differences on the neighbours' offsets in all 9 directions; shown only where "
                     "all 9 directions are resolved. Shift + linear + not-linear fractions = 1."),
    dict(name="slope_fraction", label="Relative motion: linear deformation", group="Relative motion",
         scale="sequential", units="fraction (in-sample)",
         description="Fraction of the relative motion reproduced by the linear (slope) part of the same "
                     "free-intercept fit: displacement differences that vary linearly with the neighbour's "
                     "offset, i.e. a linear deformation of the neighbourhood at its radius. Shown only where all "
                     "9 directions are resolved."),
    dict(name="residual_fraction", label="Relative motion: not linear", group="Relative motion",
         scale="sequential", units="fraction (in-sample)",
         description="Fraction of the relative motion left after the shared shift and the linear part: "
                     "separation not explained by this fit (nonlinear deformation, tearing, scatter). Shown only "
                     "where all 9 directions are resolved."),
    dict(name="heldout", label="Held-out prediction score", group="Relative motion", scale="diverging",
         units="1 - test error / test energy",
         description="Out-of-sample score (5 folds of neighbours) of the linear predictor, with a free constant "
                     "term, of next-site neighbour offsets from this-site offsets: 1 - squared prediction error / "
                     "squared offset, i.e. the improvement over predicting zero offset. Not a centred R²; read it "
                     "against the two baseline scores below. Shown only where all 9 directions are resolved."),
    dict(name="heldout_identity", label="Score of 'no deformation'", group="Relative motion", scale="diverging",
         units="1 - error / energy",
         description="The same score for the baseline that predicts each next-site offset as the this-site "
                     "offset carried along unchanged (no deformation). Close to the held-out score = the fit adds "
                     "little over 'nothing changes shape'."),
    dict(name="heldout_const", label="Score of a constant offset", group="Relative motion", scale="diverging",
         units="1 - error / energy",
         description="The same score for the baseline that predicts every next-site offset by one common "
                     "vector (the training neighbours' mean). High = the next-site offsets are dominated by a "
                     "shared shift (e.g. the image's own move relative to its neighbours), not by positions."),
    # Strain (flow reading): free-intercept fits, gated (see Rules.flow and tests/test_matched.py)
    dict(name="log_gain_max", label="Largest log gain", group="Strain (flow reading)", scale="sequential",
         units="natural log (0 = no change)",
         description="Log of the largest singular value of the fitted linear predictor (free intercept, all 9 directions, at the neighbourhood radius): the strongest stretching gain of the best graph-weighted linear prediction of next-site offsets. A statistic of the fitted slope, not the stretch of any particular pair of images. Released where the fit passes the flow gates; the calibration fixture's tolerance for this field is 0.1."),
    dict(name="E_trace", label="Expansion rate", group="Strain (flow reading)", scale="diverging",
         units="per block",
         description="Trace of the symmetric part of the fitted gradient of the displacement field (free intercept, all 9 directions, at the neighbourhood radius): positive where the fitted slope spreads neighbours apart, negative where it draws them together. A statistic of the fitted slope, not a measured volume change. Released where the displacement-difference fit passes the flow gates."),
    dict(name="strain_rate", label="Strain rate", group="Strain (flow reading)", scale="sequential",
         units="per block, squared",
         description="Squared Frobenius norm of the symmetric part of the fitted displacement gradient (free intercept, all 9 directions, at the neighbourhood radius). A statistic of the fitted slope of neighbours' displacement differences, not a measured deformation of the cloud. Released where the displacement-difference fit passes the flow gates."),
    dict(name="spin", label="Spin", group="Strain (flow reading)", scale="sequential",
         units="per block, squared",
         description="Squared Frobenius norm of the antisymmetric part of the fitted displacement gradient (free intercept, all 9 directions, at the neighbourhood radius). A statistic of the fitted slope of neighbours' displacement differences. Released where the displacement-difference fit passes the flow gates."),
    # Thick/thin (secondary)
    dict(name="leading_fraction", label="Displacement along leading directions", group="Thick/thin (secondary)",
         scale="sequential", units="fraction of squared displacement",
         description="Fraction of the squared displacement lying in the leading eigenspace of the source "
                     "offsets' second moment at this radius (eigenvalues >= 0.1 of the largest), released only "
                     "where that eigenspace is stable (same rank at cutoffs 0.05 and 0.2 and for the centred "
                     "covariance, stable under split samples, clear eigengap). Not a tangent/normal split of a "
                     "manifold."),
    dict(name="tilt", label="Turn of leading directions", group="Thick/thin (secondary)", scale="sequential",
         units="radians",
         description="Largest principal angle between the leading eigenspace at this site (carried along the "
                     "displacement) and the leading eigenspace at the next site: change of orientation of the "
                     "cloud's thick directions. Same release condition as the previous field, at both sites."),
]


def _np():
    import numpy as np
    return np


def save_json(path: Path, data) -> None:
    tmp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    tmp.write_text(json.dumps(data, indent=1, allow_nan=False) + "\n")
    tmp.replace(path)


def clean(x):
    """JSON-safe: NaN / inf -> None, numpy scalars -> Python."""
    np = _np()
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, (np.floating, float)):
        return float(x) if np.isfinite(x) else None
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


# ----------------------------------------------------------------------------------------------------------
# computation (runs in the container; importable locally for checks)

def site_frames(logp_site, k, eps, n_jobs=4):
    """Operator and frames of one site: flow (all directions), strict, thick (strict analysis re-ruled)."""
    from mm import graph, sphere, step
    from mm.frame import Rules, analyse_frame, rerule

    X = sphere.sphere_from_logp(logp_site)
    res = sphere.input_resolution(logp_site)
    op = graph.build_operator(X, n_jobs=n_jobs, **dict(step.DEFAULT_OPERATOR, k=k, eps=eps))
    fs = analyse_frame(op, Rules.strict(), resolution=res, n_jobs=n_jobs)
    ff = analyse_frame(op, Rules.flow(), resolution=res, n_jobs=n_jobs)
    return dict(X=X, strict=fs, thick=rerule(fs, Rules.thick()), flow=ff)


def mode_rules(mode):
    from mm.frame import Rules
    return dict(strict=Rules.strict, flow=Rules.flow, thick=Rules.thick)[mode]()


def flat_totals(totals_by_mode):
    """{mode: step_totals} -> {"mode:name": {value, mass_fraction}} (splits as mode:split:name:field)."""
    out = {}
    for mode, tot in totals_by_mode.items():
        for name, val in tot.items():
            if name.startswith("split:"):
                for f, part in val.items():
                    out[f"{mode}:{name}:{f}"] = dict(value=part["value"], mass_fraction=part["mass_fraction"])
            elif isinstance(val, dict):
                out[f"{mode}:{name}"] = dict(value=val["value"], mass_fraction=val["mass_fraction"])
            else:
                out[f"{mode}:{name}"] = dict(value=val, mass_fraction=1.0)
    return out


def step_all_modes(fa, fb, dt, labels=None, n_jobs=4):
    """measure_step under each mode for frames fa -> fb; returns ({mode: raw output}, {mode: masked})."""
    from mm import step
    raw, masked = {}, {}
    for mode in MODES:
        raw[mode] = step.measure_step(fa["X"], fb["X"], dt, frame0=fa[mode], frame1=fb[mode],
                                      rules=mode_rules(mode), mask=False, n_jobs=n_jobs)
        masked[mode] = step.apply_release(raw[mode])
    return raw, masked


MEDIANS = ("heldout_ic", "heldout_identity", "heldout_const", "heldout_v_ic", "heldout_v_const", "intercept_rel",
           "slope_fraction", "shift_fraction", "residual_fraction", "cond_centred")


def _med(arr):
    np = _np()
    ok = np.isfinite(arr)
    return dict(value=float(np.median(arr[ok])) if ok.any() else None, mass_fraction=float(ok.mean()))


def fractions(o):
    """shared-shift / slope / residual fractions of the relative motion (free-intercept budget), per point."""
    np = _np()
    with np.errstate(invalid="ignore", divide="ignore"):
        return {"slope_fraction": o["relmotion_slope"] / o["relmotion"],
                "shift_fraction": o["relmotion_shift"] / o["relmotion"],
                "residual_fraction": o["relmotion_icres"] / o["relmotion"]}


def step_row(fa, fb, dt, n_jobs=4):
    """All three modes for one step: public totals row (flat names), masked outputs, coverage.

    Public flow names carry free-intercept estimators only: 'flow:heldout' and 'median:flow:heldout' are the
    free-intercept held-out score (the same estimator as the per-point field 'heldout'), likewise
    'flow:heldout_v'.  Anchored-fit quantities, which flow mode withholds, appear only as
    'diag:anchored:flow:<name>' (and 'diag:anchored:median:flow:heldout')."""
    np = _np()
    from mm import step, totals as tt
    raw, masked = step_all_modes(fa, fb, dt, n_jobs=n_jobs)
    row = flat_totals({m: tt.step_totals(o) for m, o in raw.items()})
    row["flow:heldout"] = dict(row["flow:heldout_ic"])
    row["flow:heldout_v"] = dict(row["flow:heldout_v_ic"])
    of = masked["flow"]
    fr = fractions(of)
    for key in MEDIANS:
        row[f"median:flow:{key}"] = _med(fr[key] if key in fr else of[key])
    row["median:flow:heldout"] = dict(row["median:flow:heldout_ic"])
    rf = raw["flow"]
    for name, val in tt.diag_totals(rf, step.ANCHORED_DIAG).items():
        row[f"diag:anchored:flow:{name}"] = dict(value=val["value"], mass_fraction=val["mass_fraction"])
    sel = rf["release"]["relmotion_split"].copy()
    for f in ("relmotion", "relmotion_fit", "relmotion_res"):
        sel &= np.isfinite(rf[f])
    for f in ("relmotion", "relmotion_fit", "relmotion_res"):
        v = rf[f][sel]
        row[f"diag:anchored:flow:split:relmotion_fit_res:{f}"] = dict(
            value=float(v.mean()) if v.size else None, mass_fraction=float(sel.mean()))
    row["diag:anchored:median:flow:heldout"] = _med(
        np.where(rf["release"][step.FIELD_RULE["heldout"]], rf["heldout"], np.nan))
    cov = {m: {r: float(v.mean()) for r, v in o["release"].items()} for m, o in raw.items()}
    return row, masked, cov


def depth_clock(logp, labels, k, eps, want_points=True, n_jobs=4, frames=None):
    """9 depth steps of one checkpoint.  Returns (per-point arrays [9, N] or None, totals list,
    diagnostics, coverage, frames)."""
    np = _np()
    from mm import sphere, totals as tt

    S, N = logp.shape[0], logp.shape[1]
    frames = frames or [site_frames(logp[s], k, eps, n_jobs) for s in range(S)]
    pts = {f["name"]: np.full((S - 1, N), np.nan, np.float32) for f in FIELDS} if want_points else None
    tots, cov, diag = [], [], []
    for l in range(S - 1):
        fa, fb = frames[l], frames[l + 1]
        row, res, cv = step_row(fa, fb, 1.0, n_jobs=n_jobs)
        nnr = nn_log_ratio(fa, fb)
        row.update(direct_totals(fa, fb, logp[l], logp[l + 1], nnr))
        tots.append(clean(row))
        cov.append(cv)
        of = res["flow"]
        eig = fa["flow"].eig
        diag.append(dict(
            site=SITES[l], radius_q=np.quantile(fa["flow"].radius, [.1, .5, .9]),
            n_eff_q=np.quantile(fa["flow"].n_eff, [.1, .5, .9]),
            cond_flow_q=np.nanquantile(fa["flow"].cond, [.1, .5, .9]),
            spectrum_median=np.median(eig / eig[:, :1], 0),
            strict_dim=np.bincount(fa["strict"].dim, minlength=10),
            one_sided_strict_median=np.nanmedian(fa["strict"].one_sided),
            trunc_tail_strict_median=np.nanmedian(fa["strict"].trunc_tail),
            thick_rank_stable=float(fa["thick"].rank_stable.mean()),
            flow_heldout_q=(np.nanquantile(of["heldout_ic"], [.1, .5, .9]) if np.isfinite(of["heldout_ic"]).any()
                            else None),
        ))
        if want_points:
            Xa, Xb = fa["X"], fb["X"]
            v = sphere.log_map(Xa, Xb)
            pts["speed"][l] = of["speed"]
            pts["nonrigid_speed"][l] = of["cr_travel"]
            lab = R_VERTEX(labels)
            tgt = sphere.log_map(Xa, lab)
            nt = np.linalg.norm(tgt, axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                pts["toward_label"][l] = np.where(nt > 0, np.sum(v * tgt, 1) / nt, np.nan)
            pts["radius"][l] = fa["flow"].radius
            pts["local_dim"][l] = eig.sum(1) ** 2 / (eig**2).sum(1)
            pts["one_sided"][l] = fa["flow"].one_sided
            pts["vertex_distance"][l] = sphere.face_distance(logp[l], 1)
            pts["relmotion"][l] = of["relmotion"]
            pts["nn_log_ratio"][l] = nnr
            for key, arr in fractions(of).items():
                pts[key][l] = arr
            pts["heldout"][l] = of["heldout_ic"]
            for key in ("heldout_identity", "heldout_const"):
                pts[key][l] = of[key]
            pts["log_gain_max"][l] = of["ic_log_gain_max"]
            pts["E_trace"][l] = of["ic_E_trace"]
            pts["strain_rate"][l] = of["ic_strain_sq"]
            pts["spin"][l] = of["ic_spin_sq"]
            ot = res["thick"]
            pts["leading_fraction"][l] = ot["tan_fraction"]
            pts["tilt"][l] = ot["tilt_max"]
    if want_points:
        pts["path_length"] = np.cumsum(np.nan_to_num(pts["speed"], nan=0.0), axis=0).astype(np.float32)
    return pts, tots, clean(diag), clean(cov), frames


def nn_log_ratio(fa, fb, k=16):
    """Per point: mean over its k nearest source neighbours of log(d(Y_i, Y_j) / d(X_i, X_j)); NaN where a
    distance is not above the pair's resolution."""
    np = _np()
    from mm import sphere
    op = fa["flow"].op
    idx = op.knn_idx[:, 1 : k + 1]
    d0 = sphere.fisher_distance(fa["X"][:, None, :], fa["X"][idx])
    d1 = sphere.fisher_distance(fb["X"][:, None, :], fb["X"][idx])
    r0 = fa["flow"].res[:, None] + fa["flow"].res[idx]
    r1 = fb["flow"].res[:, None] + fb["flow"].res[idx]
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = np.where((d0 > r0) & (d1 > r1), np.log(d1 / d0), np.nan)
    ok = np.isfinite(lr).all(1)
    return np.where(ok, np.nanmean(lr, 1), np.nan)


def direct_totals(fa, fb, lp0, lp1, nnr):
    """Fit-free statistics of the step: distance to the nearest vertex at both ends, the fraction of images
    within 0.05 (Fisher) of a vertex, and the mean log change of nearest-neighbour distances."""
    np = _np()
    from mm import sphere
    v0, v1 = sphere.face_distance(lp0, 1), sphere.face_distance(lp1, 1)
    ok = np.isfinite(nnr)
    return {"direct:vertex_distance_from": dict(value=float(v0.mean()), mass_fraction=1.0),
            "direct:vertex_distance_to": dict(value=float(v1.mean()), mass_fraction=1.0),
            "direct:near_vertex_from": dict(value=float(np.mean(v0 < 0.05)), mass_fraction=1.0),
            "direct:near_vertex_to": dict(value=float(np.mean(v1 < 0.05)), mass_fraction=1.0),
            "direct:nn_log_ratio": dict(value=float(nnr[ok].mean()) if ok.any() else None,
                                        mass_fraction=float(ok.mean()))}


def R_VERTEX(labels):
    np = _np()
    from mm import sphere
    return sphere.R * np.eye(10)[labels]


def training_clock(frames_a, frames_b, dt, n_jobs=4):
    """Totals per site for the step checkpoint a -> b at each site (same images), dt = training steps."""
    np = _np()
    from mm import totals as tt
    rows = []
    for fa, fb in zip(frames_a, frames_b):
        row, _, _ = step_row(fa, fb, dt, n_jobs=n_jobs)
        nnr = nn_log_ratio(fa, fb)
        ok = np.isfinite(nnr)
        row["direct:nn_log_ratio"] = dict(value=float(nnr[ok].mean()) if ok.any() else None,
                                          mass_fraction=float(ok.mean()))
        rows.append(clean(row))
    return rows


# ----------------------------------------------------------------------------------------------------------
# Modal functions

def _lens(seed, step):
    np = _np()
    root = Path("/v/lens") / f"r20s{seed}"
    return np.load(root / f"logp_{step}.npy"), np.load(root / "labels.npy")


def _out(seed):
    p = Path("/v/fields") / f"r20s{seed}"
    (p / "extra").mkdir(parents=True, exist_ok=True)
    return p


def _done(unit):
    return Path("/v/fields/_done") / (unit.replace(":", "_") + ".json")


def _claim(unit):
    return Path("/v/fields/_claim") / (unit.replace(":", "_") + ".json")


@app.function(image=image, cpu=(4, 4), memory=(8192, 8192), timeout=UNIT_TIMEOUT, startup_timeout=STARTUP,
              scaledown_window=IDLE, max_containers=MAX_CONTAINERS, retries=0, volumes={"/v": vol})
def unit(name: str, token: str = ""):
    np = _np()
    t0 = time.time()
    vol.reload()
    if _done(name).is_file():
        return dict(unit=name, skipped=True, seconds=time.time() - t0)
    # a restarted coordinator re-dispatches pending units under a new token: do not run a unit that another
    # coordinator attempt's container is running.  A preempted unit is restarted by Modal with the same
    # input (same token) and must run again, so a claim with this unit's own token does not block.
    c = _claim(name)
    if c.is_file():
        cl = json.loads(c.read_text())
        if cl.get("token") != token and time.time() - cl["t"] < UNIT_TIMEOUT:
            return dict(unit=name, skipped=True, claimed=True, seconds=time.time() - t0)
    c.parent.mkdir(parents=True, exist_ok=True)
    save_json(c, dict(t=time.time(), token=token))
    vol.commit()
    kind, *par = name.split(":")
    rec = dict(unit=name)
    if kind == "ckpt":
        stp = int(par[0])
        logp, labels = _lens(0, stp)
        out = _out(0)
        pts, tots, diag, cov, frames = depth_clock(logp, labels, K_GRID, EPS)
        tmp = out / f".depth_{stp}.{uuid4().hex}.npz"
        with tmp.open("wb") as fh:
            np.savez(fh, **pts)
        tmp.replace(out / f"depth_{stp}.npz")
        save_json(out / f"totals_{stp}.json", tots)
        save_json(out / "extra" / f"diag_{stp}.json", diag)
        save_json(out / "extra" / f"coverage_{stp}.json", cov)
        i = STEPS.index(stp)
        if i > 0:
            prev = STEPS[i - 1]
            lp, _ = _lens(0, prev)
            fprev = [site_frames(lp[s], K_GRID, EPS) for s in range(lp.shape[0])]
            rows = training_clock(fprev, frames, float(stp - prev))
            save_json(out / "extra" / f"train_{prev}_{stp}.json",
                      dict(step_from=prev, step_to=stp, dt=stp - prev, sites=SITES, totals=rows))
        if stp == FINAL:
            save_json(out / "fields_meta.json", [dict(f, clock="depth") for f in FIELDS])
    elif kind == "scale":
        k, eps = int(par[0]), float(par[1])
        logp, labels = _lens(0, FINAL)
        _, tots, diag, cov, _ = depth_clock(logp, labels, k, eps, want_points=False)
        save_json(_out(0) / "extra" / f"scale_k{k}_eps{eps}.json",
                  dict(k=k, eps=eps, step=FINAL, totals=tots, diag=diag, coverage=cov))
    elif kind == "subsample":
        logp, labels = _lens(0, FINAL)
        rng = np.random.default_rng(0)
        reps = []
        for r in range(5):
            idx = np.sort(rng.choice(logp.shape[1], 5000, replace=False))
            _, tots, diag, cov, _ = depth_clock(logp[:, idx], labels[idx], K_GRID, EPS, want_points=False)
            reps.append(dict(rep=r, totals=tots, coverage=cov))
        save_json(_out(0) / "extra" / "subsample.json",
                  dict(step=FINAL, n_subsets=5, size=5000, k=K_GRID, eps=EPS, replicates=reps))
    elif kind == "seed1":
        logp, labels = _lens(1, FINAL)
        out = _out(1)
        pts, tots, diag, cov, _ = depth_clock(logp, labels, K_GRID, EPS)
        tmp = out / f".depth_{FINAL}.{uuid4().hex}.npz"
        with tmp.open("wb") as fh:
            np.savez(fh, **pts)
        tmp.replace(out / f"depth_{FINAL}.npz")
        save_json(out / f"totals_{FINAL}.json", tots)
        save_json(out / "extra" / f"diag_{FINAL}.json", diag)
        save_json(out / "extra" / f"coverage_{FINAL}.json", cov)
        save_json(out / "fields_meta.json", [dict(f, clock="depth") for f in FIELDS])
    else:
        raise ValueError(name)
    rec["seconds"] = time.time() - t0
    _done(name).parent.mkdir(parents=True, exist_ok=True)
    save_json(_done(name), rec)
    vol.commit()
    return dict(rec, skipped=False)


@app.function(image=image, cpu=(0.25, 0.25), memory=(512, 512), timeout=COORD_TIMEOUT, startup_timeout=STARTUP,
              scaledown_window=IDLE, max_containers=1, retries=0, volumes={"/v": vol})
def coordinate(stage: str, reserved_usd: float):
    t0 = time.time()
    vol.reload()
    names = [n for n in STAGES[stage] if not _done(n).is_file()]
    ceiling = len(names) * (UNIT_TIMEOUT + STARTUP + IDLE) * WORKER_RATE + (COORD_TIMEOUT + STARTUP) * COORD_RATE
    if ceiling > reserved_usd + 1e-9:
        raise RuntimeError(f"ceiling ${ceiling:.2f} exceeds reservation ${reserved_usd:.2f}")
    token = uuid4().hex                      # new per coordinator attempt (a restarted coordinator gets a new one)
    results = list(unit.map(names, [token] * len(names), return_exceptions=True)) if names else []
    # units skipped because another container holds their claim: wait for their done-markers
    waiting = [r["unit"] for r in results if isinstance(r, dict) and r.get("claimed")]
    while waiting and time.time() - t0 < COORD_TIMEOUT - 120:
        time.sleep(60)
        vol.reload()
        waiting = [u for u in waiting if not _done(u).is_file()]
    failures = {n: repr(r) for n, r in zip(names, results) if isinstance(r, Exception)}
    failures.update({u: "claimed unit never finished" for u in waiting})
    worker_s = sum(r["seconds"] for r in results if isinstance(r, dict))
    wall = time.time() - t0
    out = dict(stage=stage, units=names, failures=failures, worker_seconds=worker_s, wall_seconds=wall,
               unit_seconds={r["unit"]: r["seconds"] for r in results if isinstance(r, dict)},
               estimated_usd=worker_s * WORKER_RATE + wall * COORD_RATE,
               spend_with_startup_reserve_usd=(worker_s + len(names) * (STARTUP + IDLE)) * WORKER_RATE
               + (wall + STARTUP) * COORD_RATE)
    if failures:
        raise RuntimeError(json.dumps(out))
    return out


# ----------------------------------------------------------------------------------------------------------
# local CLI

def records(stage="*"):
    return sorted(RUN_RECORDS.glob(f"fields-{stage}-*.json"))


def launch(stage):
    RUN_RECORDS.mkdir(exist_ok=True)
    prev = [json.loads(p.read_text()) for p in records()]
    if any("result" not in r for r in prev):
        raise RuntimeError("a previous launch has no recorded result; run status first")
    used = sum(r["result"].get("spend_with_startup_reserve_usd", r["reserved_usd"]) for r in prev)
    n = len(STAGES[stage])
    reservation = n * (UNIT_TIMEOUT + STARTUP + IDLE) * WORKER_RATE + (COORD_TIMEOUT + STARTUP) * COORD_RATE
    if used + reservation > BUDGET:
        raise RuntimeError(f"cumulative ceiling ${used + reservation:.2f} exceeds ${BUDGET:.2f}")
    t = time.time()
    call = modal.Function.from_name(APP, "coordinate").spawn(stage, reservation)
    rec = dict(stage=stage, app=APP, call_id=call.object_id, launched_at=t,
               launched_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t)), reserved_usd=reservation)
    save_json(RUN_RECORDS / f"fields-{stage}-{int(t * 1e6)}.json", rec)
    print(json.dumps(rec, indent=1))


def status(stage, wait):
    path = records(stage)[-1]
    rec = json.loads(path.read_text())
    if "result" not in rec:
        try:
            rec["result"] = modal.FunctionCall.from_id(rec["call_id"]).get(timeout=wait)
        except TimeoutError:
            print(f"running: {rec['call_id']}")
            return
        except Exception as exc:  # the coordinator's own exception, re-raised here
            if isinstance(exc, (ConnectionError, OSError)):
                raise
            rec["result"] = {"error": repr(exc)}
        save_json(path, rec)
    print(json.dumps(rec, indent=1))


def fetch(run):
    dest = HERE / "data" / "fields" / run
    (dest / "extra").mkdir(parents=True, exist_ok=True)
    v = modal.Volume.from_name(VOLUME)
    names = []
    for e in v.listdir(f"/fields/{run}", recursive=True):
        if e.type == modal.volume.FileEntryType.FILE and "/." not in "/" + e.path:
            names.append(e.path)
    from concurrent.futures import ThreadPoolExecutor

    def get(p):
        rel = Path(p).relative_to(f"fields/{run}")
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
        with tmp.open("wb") as fh:
            for chunk in v.read_file(p):
                fh.write(chunk)
        tmp.replace(target)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(get, names))
    print(f"fetched {len(names)} files to {dest}")


MANIFEST_NOTES = [
    "Strict-mode tilt totals (strict:tilt_max, strict:cr_tilt_max) are not exported in this release (null, mass "
    "fraction 0 in every totals file, training-clock and extra files included): the grid was computed while the "
    "endpoint-plane tilt rule meant for thick mode was applied in strict mode too, dropping strict mode's "
    "held-out prediction gate; the library now keeps that gate in strict mode (mm/step.py, rule 'tilt_angles'). "
    "Strict tilt coverage under the correct rule is near zero (e.g. 14 images at checkpoint 15, b5->b6). "
    "Thick-mode tilt (thick:tilt_max, per-point field 'tilt') is unaffected.",
    "JSON files contain no NaN/Infinity literals: undefined values are null, with mass_fraction 0 for totals.",
]


def manifest(run):
    """Move speed-only placeholder archives (checkpoints this run does not measure) to speed_only/, write
    checkpoints.json (measured checkpoints and the fields each archive holds) and validate it against the
    archives, fields_meta.json and the totals files.  Raises on any mismatch."""
    import numpy as np

    d = HERE / "data" / "fields" / run
    meta = json.loads((d / "fields_meta.json").read_text())
    names = [f["name"] for f in meta]
    steps = STEPS if run == "r20s0" else [FINAL]
    so = d / "speed_only"
    for p in sorted(d.glob("depth_*.npz")) + sorted(d.glob("totals_*.json")):
        stp = int(p.stem.split("_")[1])
        if stp not in steps:
            so.mkdir(exist_ok=True)
            p.replace(so / p.name)
    entries = []
    for stp in steps:
        with np.load(d / f"depth_{stp}.npz") as z:
            fields = sorted(z.files)
            for n in fields:
                if z[n].shape != (len(SITES) - 1, 10_000) or z[n].dtype != np.float32:
                    raise ValueError(f"depth_{stp}.npz:{n} has shape {z[n].shape} dtype {z[n].dtype}")
        rows = json.loads((d / f"totals_{stp}.json").read_text())
        if len(rows) != len(SITES) - 1:
            raise ValueError(f"totals_{stp}.json has {len(rows)} rows")
        entries.append(dict(step=stp, depth=f"depth_{stp}.npz", totals=f"totals_{stp}.json", fields=fields,
                            n_totals=len(rows[0])))
    man = dict(run=run, fields_meta="fields_meta.json", shape=[len(SITES) - 1, 10_000], dtype="float32",
               speed_only_dir="speed_only" if so.is_dir() else None, notes=MANIFEST_NOTES, checkpoints=entries)
    save_json(d / "checkpoints.json", man)                     # allow_nan=False
    # validation, last step: manifest == archives == fields_meta, nothing else left in the directory
    man = json.loads((d / "checkpoints.json").read_text())
    for e in man["checkpoints"]:
        with np.load(d / e["depth"]) as z:
            if sorted(z.files) != e["fields"] or sorted(e["fields"]) != sorted(names):
                raise ValueError(f"manifest / archive / fields_meta mismatch at {e['step']}")
    left = {int(p.stem.split("_")[1]) for p in d.glob("depth_*.npz")}
    if left != {e["step"] for e in man["checkpoints"]}:
        raise ValueError(f"unexpected archives: {sorted(left ^ {e['step'] for e in man['checkpoints']})}")
    print(f"{run}: manifest of {len(entries)} checkpoints x {len(names)} fields validated")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("deploy")
    for c in ("launch", "status"):
        p = sub.add_parser(c)
        p.add_argument("stage", choices=STAGES)
        if c == "status":
            p.add_argument("--wait", type=float, default=0)
    p = sub.add_parser("fetch")
    p.add_argument("--run", default="r20s0")
    p = sub.add_parser("manifest")
    p.add_argument("--run", default="r20s0")
    a = ap.parse_args()
    if a.cmd == "deploy":
        subprocess.run([sys.executable, "-m", "modal", "deploy", __file__], check=True)
    elif a.cmd == "launch":
        launch(a.stage)
    elif a.cmd == "status":
        status(a.stage, a.wait)
    elif a.cmd == "manifest":
        manifest(a.run)
    else:
        fetch(a.run)


if __name__ == "__main__":
    main()
