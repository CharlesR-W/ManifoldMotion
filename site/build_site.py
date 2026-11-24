"""Build the static ManifoldMotion site into site/dist/ (the GitHub Pages deploy root).

Reads data/lens/<run>/ (lens log-probabilities per checkpoint) and, when present, data/fields/<run>/ (per-point
fields from the kinematics library), and writes compact binary assets that the browser loads per checkpoint and
site. Usage (from the repository root):

    uv run --with numpy --with pillow python site/build_site.py [--run r20s0] [--n-steps 40] [--all-steps]

One run per deploy: its assets go to dist/data/, and meta.json and the page text name it. The build stops with an
error, before touching dist/, if the estimated deploy size exceeds --max-mb (default 200).

Binary formats (all little-endian, row i = CIFAR-10 test image i), described again in dist/data/meta.json:

  labels.bin               uint8  [N]          true class
  images.webp              sprite atlas, 100 images per row, 32 x 32 px each, WebP quality 80
  x/<step>_<site>.bin      uint16 [N, 10]      the lens's top class (argmax of log p) holds the code 65535; every
                                               other class holds q = round(min(-log p, L) / L * 65534), with
                                               L = meta.L[step index][site] = min(40, -min log p over that frame),
                                               rounded up to 1e-3. Decode sqrt(p_c) = exp(-q L / 131068) for the
                                               others, then sqrt(p_top) = sqrt(1 - sum of their squares), so every
                                               decoded point lies exactly on the sphere and the lens prediction is
                                               exact. Relative error of sqrt(p_c) <= L / 262136 (<= 1.6e-4); p below
                                               e^-40 is stored as e^-40. At near ties a runner-up's code is raised
                                               until it decodes at least 1e-6 below the top class, so the decoded
                                               ranking never contradicts the prediction (checked for every row).
  f/<field>/<step>.bin     uint8  [9, N]       [lo, hi] = the field's 0.5 and 99.5 percentiles over all included
                                               checkpoints and depth steps ([-m, m], centred on zero, for diverging
                                               fields). Code 0 = below lo, 254 = above hi, k in 1..253 = bin k of
                                               253 equal bins of [lo, hi] (centre lo + (k - 0.5) / 253 (hi - lo)),
                                               255 = undefined (NaN).
  totals.json              {steps: [...], totals: {name: {value: [[9] per step], mass_fraction: [[9] per step]}}}
"""
import argparse
import datetime as dt
import json
import math
import re
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REQUIRED = [0, 64, 256, 1024, 2560, 6400, 25600, 64000]
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
STATIC = ["index.html", "css", "js", "vendor", "experiments"]
LMAX = 40.0


def pick_steps(steps, n, dense_until=1280):
    """Every checkpoint up to `dense_until`, then log-spaced later ones (always the REQUIRED ones) up to n in all."""
    if n >= len(steps):
        return list(steps)
    early = [s for s in steps if s <= dense_until]
    late = [s for s in steps if s > dense_until]
    k = max(2, n - len(early))
    while True:
        lo, hi = math.log(late[0]), math.log(late[-1])
        targets = [lo + (hi - lo) * i / (k - 1) for i in range(k)]
        sel = set(early) | (set(REQUIRED) & set(late)) | {min(late, key=lambda s: abs(math.log(s) - t)) for t in targets}
        if len(sel) >= n or k > len(late):
            return sorted(sel)
        k += 1


def decode(q, L):
    """The browser's decoder (js/data.js), for the build's own checks."""
    top = q == 65535
    s = np.where(top, 0.0, np.exp(-q.astype(np.float64) * (L / 131068.0)))
    s[top] = np.sqrt(np.clip(1 - (s ** 2).sum(1), 0, 1))
    return s


def rank_violations(q, L, top):
    """Boolean [N, 10]: non-top classes whose decoded sqrt p is not at least 1e-6 below the top class's."""
    s = decode(q, L)
    rows = np.arange(len(s))
    st = s[rows, top].copy()
    s[rows, top] = -1
    return s > (st - 1e-6)[:, None]


def encode(logp):
    """uint16 codes for one frame [N, 10] of log-probabilities, and its L (see the module docstring)."""
    L = math.ceil(min(LMAX, -float(logp.min())) * 1000) / 1000 or 1e-3
    q = np.round(np.clip(-logp, 0, L) / L * 65534)
    top = logp.argmax(1)
    q[np.arange(len(q)), top] = 65535
    for _ in range(2000):                 # near ties: push runners-up down until the top class decodes largest
        bad = rank_violations(q, L, top)
        if not bad.any():
            return q.astype("<u2"), L
        q[bad] = np.minimum(q[bad] + 1, 65534)
    raise RuntimeError("could not separate near ties")


def size_of(p):
    p = Path(p)
    return p.stat().st_size if p.is_file() else sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


class Build:
    def __init__(self, args):
        self.run = args.run
        self.src = ROOT / "data" / "lens" / args.run
        self.fsrc = Path(args.fields) if args.fields else ROOT / "data" / "fields" / args.run
        self.out = Path(args.out) if args.out else HERE / "dist"
        self.d = self.out / "data"
        if not (self.src / "meta.json").exists():
            raise SystemExit(f"error: no lens data at {self.src}")
        lens_meta = json.loads((self.src / "meta.json").read_text())
        if not str(lens_meta.get("run", "")).startswith("resnet20"):
            raise SystemExit(f"error: {self.src} is not a ResNet-20 run, which the site text describes")
        self.all_steps = lens_meta["steps"]
        self.sites = lens_meta["sites"]
        self.N = lens_meta["N"]
        self.steps = list(self.all_steps) if args.all_steps else pick_steps(self.all_steps, args.n_steps)
        missing = [s for s in REQUIRED if s in self.all_steps and s not in self.steps]
        assert not missing, missing
        need = ["labels.npy", "images.npy"] + [f"{k}_{s}.{e}" for s in self.steps for k, e in [("logp", "npy"), ("checks", "json")]]
        absent = [n for n in need if not (self.src / n).exists()]
        if absent:
            raise SystemExit(f"error: {self.src} lacks {len(absent)} input files (e.g. {', '.join(absent[:3])}); "
                             f"dist/ was not changed")
        self.meta = {
            "run": args.run, "model": lens_meta.get("run"), "seed": lens_meta.get("seed"), "N": self.N,
            "sites": self.sites, "classes": CLASSES, "steps": self.steps, "all_steps": self.all_steps,
            "built": dt.date.today().isoformat(),
        }
        self.max_mb = args.max_mb
        self.check_budget()

    def check_budget(self):
        """Estimate the deploy size before anything is written; stop with an error if it is over the limit."""
        S, N = len(self.sites), self.N
        coords = len(self.steps) * S * N * 20
        fields = 0
        mf = self.fsrc / "fields_meta.json"
        if mf.exists():
            names = {f["name"] for f in json.loads(mf.read_text())}
            for s in self.steps:
                p = self.fsrc / f"depth_{s}.npz"
                if p.exists():
                    with np.load(p) as z:
                        fields += len(names & set(z.files)) * (S - 1) * N
        static = sum(size_of(HERE / n) for n in STATIC if (HERE / n).exists())
        est = coords + fields + static + 3.4e6             # + atlas, labels, meta and totals
        print(f"estimated deploy size {est / 1e6:.1f} MB (coords {coords / 1e6:.1f}, fields {fields / 1e6:.1f}, "
              f"static {static / 1e6:.1f}); limit {self.max_mb:g} MB")
        if est > self.max_mb * 1e6:
            raise SystemExit(f"error: the deploy would be about {est / 1e6:.0f} MB, over the {self.max_mb:g} MB limit. "
                             f"Include fewer checkpoints (--n-steps, now {len(self.steps)}) or fewer fields. "
                             f"dist/ was not changed.")

    def static(self):
        if self.out.exists():
            shutil.rmtree(self.out)
        self.out.mkdir()
        for name in STATIC:
            p = HERE / name
            if p.is_dir():
                shutil.copytree(p, self.out / name, ignore=shutil.ignore_patterns("src") if name == "experiments" else None)
            elif p.exists():
                shutil.copy2(p, self.out / name)
        (self.out / ".nojekyll").write_text("")
        idx = self.out / "index.html"
        sub = {"{{BUILT}}": self.meta["built"], "{{N_STEPS}}": str(len(self.steps)), "{{SEED}}": str(self.meta["seed"]),
               "{{N_ALL_STEPS}}": str(len(self.all_steps)), "{{LAST_STEP}}": f"{self.all_steps[-1]:,}"}
        html = idx.read_text()
        for k, v in sub.items():
            html = html.replace(k, v)
        assert "{{" not in html.replace("{{f:", ""), "unfilled placeholder in index.html"
        idx.write_text(html)

    def images(self):
        labels = np.load(self.src / "labels.npy")
        assert labels.shape == (self.N,)
        (self.d / "labels.bin").write_bytes(labels.astype(np.uint8).tobytes())
        self.labels = labels
        imgs = np.load(self.src / "images.npy")
        cols = 100
        rows = -(-self.N // cols)
        sheet = np.zeros((rows * 32, cols * 32, 3), np.uint8)
        for i in range(self.N):
            r, c = divmod(i, cols)
            sheet[r * 32:(r + 1) * 32, c * 32:(c + 1) * 32] = imgs[i]
        Image.fromarray(sheet).save(self.d / "images.webp", "WEBP", quality=80, method=6)
        self.meta["sprite"] = {"file": "images.webp", "cols": cols, "px": 32}

    def coords(self):
        xd = self.d / "x"
        xd.mkdir()
        L_all, acc, worst, mism, raised = [], [], 0.0, 0, 0
        rng = np.random.default_rng(0)
        pool = []
        pca_steps = set(self.steps[::3]) | {self.steps[-1]}
        for step in self.steps:
            lp = np.load(self.src / f"logp_{step}.npy")              # [sites, N, 10] float32
            checks = json.loads((self.src / f"checks_{step}.json").read_text())
            acc.append([round(float(a), 4) for a in checks["lens_top1_accuracy"]])
            Ls = []
            for s in range(len(self.sites)):
                x = lp[s].astype(np.float64)
                q, L = encode(x)
                (xd / f"{step}_{s}.bin").write_bytes(q.tobytes())
                Ls.append(L)
                top = x.argmax(1)
                # round trip: the decoded ranking agrees with the prediction in every row
                assert not rank_violations(q, L, top).any(), (step, s)
                dec = decode(q, L)
                assert (dec.argmax(1) == top).all(), (step, s)
                plain = np.round(np.clip(-x, 0, L) / L * 65534)
                raised += int((q.astype(np.float64) != np.where(q == 65535, 65535, plain)).sum())
                exact = np.exp(x / 2)
                worst = max(worst, float(np.abs(dec - exact).max()))
                mism += int(abs((top == self.labels).mean() - acc[-1][s]) > 1e-9)
                if step in pca_steps:
                    pool.append(2 * exact[rng.choice(self.N, 1500, replace=False)])
            L_all.append(Ls)
        self.meta["L"] = L_all
        self.meta["acc"] = acc
        self.meta["encoding"] = {
            "x": "uint16 [N, 10]; top class = 65535, others q = round(min(-log p, L) / L * 65534), raised at near "
                 "ties so the top class decodes largest; sqrt(p) = exp(-q L / 131068); sqrt(p_top) = sqrt(1 - sum "
                 "of squares of the others)",
            "field": "uint8 [9, N]; 0 = below lo, 254 = above hi, k in 1..253 = bin k of [lo, hi] (centre lo + "
                     "(k - 0.5) / 253 (hi - lo)), 255 = undefined",
            "max_abs_error_sqrt_p": worst, "frames_where_accuracy_differs_from_checks": mism,
            "codes_raised_for_near_ties": raised,
        }
        print(f"coords: {len(self.steps)} checkpoints x {len(self.sites)} sites; max |sqrt p error| {worst:.2e}; "
              f"frames whose argmax accuracy differs from checks: {mism}; codes raised at near ties: {raised}")
        P = np.concatenate(pool)
        mean = P.mean(0)
        C = np.cov((P - mean).T)
        w, V = np.linalg.eigh(C)
        order = np.argsort(w)[::-1]
        w, V = w[order], V[:, order]
        axes = V[:, :3].T.copy()
        for a in axes:                                    # deterministic signs: largest-magnitude entry positive
            if a[np.abs(a).argmax()] < 0:
                a *= -1
        self.meta["pca"] = {
            "mean": mean.round(6).tolist(), "axes": axes.round(6).tolist(),
            "explained": (w[:3] / w.sum()).round(4).tolist(), "n_points": len(P),
            "steps": sorted(pca_steps),
            "note": "PCA of X = 2 sqrt(p), centred, pooled over all 10 sites at every third included checkpoint "
                    "and the last, 1,500 random images per frame",
        }
        print(f"pca: explained {self.meta['pca']['explained']}")

    def fields(self):
        self.meta["fields"] = []
        mf = self.fsrc / "fields_meta.json"
        if not mf.exists():
            print("fields: none")
            return
        fmeta = json.loads(mf.read_text())
        have = [s for s in self.steps if (self.fsrc / f"depth_{s}.npz").exists()]
        for f in fmeta:
            name = f["name"]
            if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
                raise SystemExit(f"error: field name {name!r} must be letters, digits, '_', '.' or '-'")
            arrs = {}
            for s in have:
                with np.load(self.fsrc / f"depth_{s}.npz") as z:
                    if name in z.files:
                        a = z[name].astype(np.float32)
                        assert a.shape == (len(self.sites) - 1, self.N), (name, s, a.shape)
                        arrs[s] = a
            if not arrs:
                continue
            vals = np.concatenate([a[np.isfinite(a)] for a in arrs.values()])
            if vals.size == 0:
                continue
            lo, hi = np.percentile(vals, [0.5, 99.5]).astype(float)
            if f.get("scale") == "diverging":
                m = max(abs(lo), abs(hi)) or 1e-12
                lo, hi = -m, m
            if hi <= lo:
                hi = lo + max(abs(lo), 1) * 1e-9
            fd = self.d / "f" / name
            fd.mkdir(parents=True)
            for s, a in arrs.items():
                with np.errstate(invalid="ignore"):
                    k = np.clip(np.floor((a - lo) / (hi - lo) * 253) + 1, 1, 253)
                    code = np.where(a < lo, 0, np.where(a > hi, 254, k))
                code = np.where(np.isfinite(a), code, 255).astype(np.uint8)
                (fd / f"{s}.bin").write_bytes(code.tobytes())
            entry = {k: f.get(k) for k in ["name", "label", "units", "description", "scale", "clock", "group"]}
            entry["label"] = entry["label"] or name
            entry.update(lo=float(lo), hi=float(hi), vmin=float(vals.min()), vmax=float(vals.max()),
                         steps=sorted(arrs), defined=float(np.mean([np.isfinite(a).mean() for a in arrs.values()])))
            self.meta["fields"].append(entry)
            print(f"field {name}: {len(arrs)} checkpoints, colour range [{lo:.4g}, {hi:.4g}], data range "
                  f"[{vals.min():.4g}, {vals.max():.4g}], defined on {entry['defined']:.1%}")
        self.totals()

    # the explorer's totals card: a short fixed selection, each with a one-line reading
    TOTALS = [
        ("ke", "Kinetic energy", "Mean over images of |v/Δt|²/2 with Δt = one block: how far images move in this block (Fisher–Rao distance² per block²).",
         "bar", None, [("flow:ke", "kinetic energy")]),
        ("rigid", "Rigid fraction", "One number for the whole cloud: the share of the kinetic energy carried by the best rotation field of the sphere; 1 = the cloud only turns.",
         "bar", [0, 1], [("flow:rigid_fraction_global", "rigid fraction")]),
        ("split", "Relative motion: shift, slope, residual", "How neighbours move relative to each other, split into a shift they share, "
         "a slope a local linear map explains, and the rest (shares of the mean, over the images whose nine directions are resolved).", "stack", [0, 1],
         [("flow:split:relmotion_shift_slope_res:relmotion_shift/flow:split:relmotion_shift_slope_res:relmotion", "shift"), ("flow:split:relmotion_shift_slope_res:relmotion_slope/flow:split:relmotion_shift_slope_res:relmotion", "slope"), ("flow:split:relmotion_shift_slope_res:relmotion_icres/flow:split:relmotion_shift_slope_res:relmotion", "residual")]),
        ("scores", "Held-out score and its baselines", "Median over images of how well a local linear map predicts held-out neighbours; "
         "it means something only where it beats both baselines.", "lines", [0, 1],
         [("median:flow:heldout_ic", "linear fit"), ("median:flow:heldout_const", "constant offset"), ("median:flow:heldout_identity", "no deformation")]),
        ("nn", "Neighbour distance change", "Mean over images of log(distance at the next site / distance here) to the 16 nearest "
         "neighbours: positive = neighbourhoods spread, negative = they contract.", "bar", None, [("direct:nn_log_ratio", "log change")]),
        ("vertex", "Share near a vertex", "Share of images within Fisher–Rao distance 0.05 of a one-hot distribution at the next site.",
         "bar", [0, 1], [("direct:near_vertex_to", "near a vertex")]),
    ]

    def totals(self):
        tsteps = [s for s in self.steps if (self.fsrc / f"totals_{s}.json").exists()]
        if not tsteps:
            return
        rows = {}
        for s in tsteps:
            t = json.loads((self.fsrc / f"totals_{s}.json").read_text())
            rows[s] = t["steps"] if isinstance(t, dict) else t

        def get(r, key):
            if "/" in key:
                a, b = key.split("/")
                va, vb = get(r, a), get(r, b)
                ok = va[0] is not None and vb[0] not in (None, 0)
                return (va[0] / vb[0] if ok else None, min(va[1] or 0, vb[1] or 0))
            e = r.get(key)
            return (None, 0.0) if e is None else (e.get("value"), e.get("mass_fraction"))

        views = []
        for vid, label, reading, kind, rng, series in self.TOTALS:
            ser = []
            for key, name in series:
                base = key.split("/")[0]
                if not all(base in rows[s][0] for s in tsteps):
                    print(f"totals: {key} missing in some totals files; left out of the explorer card")
                    continue
                vals = [[get(r, key) for r in rows[s]] for s in tsteps]
                ser.append(dict(name=name, value=[[v for v, _ in row] for row in vals], mass=[[m for _, m in row] for row in vals]))
            if ser:
                views.append(dict(id=vid, label=label, reading=reading, kind=kind, range=rng, series=ser))
        (self.d / "totals.json").write_text(json.dumps({"steps": tsteps, "views": views}, separators=(",", ":")))
        self.meta["totals"] = "totals.json"
        print(f"totals: {len(views)} views for the explorer card")

    def experiments(self):
        """Experiment pages and the facts the Overview prints (make_experiments.py)."""
        import make_experiments
        try:
            facts = make_experiments.build(self.out)
        except FileNotFoundError as e:
            raise SystemExit(f"error: experiment pages need the measurement files ({e})")
        idx = self.out / "index.html"
        idx.write_text(facts.fill(idx.read_text(), "index.html"))
        (HERE / "facts.tsv").write_text(facts.tsv())
        print(f"experiments: {len(facts.d)} facts; table in site/facts.tsv")

    def manifest(self):
        json.loads((HERE / "experiments" / "manifest.json").read_text())     # must parse

    def run_all(self):
        self.static()
        self.d.mkdir(parents=True)
        self.images()
        self.coords()
        self.fields()
        self.manifest()
        self.experiments()
        (self.d / "meta.json").write_text(json.dumps(self.meta, separators=(",", ":")))
        mb = lambda p: size_of(p) / 1e6
        print(f"steps ({len(self.steps)}): {self.steps}")
        print(f"dist: {mb(self.out):.1f} MB (coords {mb(self.d / 'x'):.1f}, fields "
              f"{mb(self.d / 'f') if (self.d / 'f').exists() else 0:.1f}, atlas {mb(self.d / 'images.webp'):.2f}, "
              f"vendor {mb(self.out / 'vendor'):.1f})")
        if size_of(self.out) > self.max_mb * 1e6:
            raise SystemExit(f"error: dist/ is {mb(self.out):.0f} MB, over the {self.max_mb:g} MB limit")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="r20s0")
    ap.add_argument("--n-steps", type=int, default=40, help="checkpoints to include (default 40)")
    ap.add_argument("--all-steps", action="store_true", help="include all checkpoints (about 2 MB each)")
    ap.add_argument("--fields", help="fields directory (default data/fields/<run>)")
    ap.add_argument("--out", help="output directory (default site/dist)")
    ap.add_argument("--max-mb", type=float, default=200, help="largest allowed deploy size in MB (default 200)")
    Build(ap.parse_args()).run_all()
