"""Write the one per-point field that needs no neighbourhood, `speed`, in the fields interface the site reads.

speed[l, i] = Fisher-Rao distance between image i's lens points at site l and site l + 1 (same checkpoint),
d(X, Y) = 4 atan2(|X - Y|, |X + Y|) with X = 2 sqrt(p). This is SPEC item 1, the length |v| of v = log_X Y.
It stands in for the kinematics library's output until that exists; the format is the one the library will use:

  data/fields/<run>/fields_meta.json   list of {name, label, units, description, scale, clock}
  data/fields/<run>/depth_<step>.npz   one float32 array per field name, shape [9 depth steps, N], NaN = undefined
  data/fields/<run>/totals_<step>.json list of 9 dicts (one per depth step): {total name: {value, mass_fraction}}

Usage: uv run --with numpy python site/make_speed_field.py [--run r20s0]
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

META = [{
    "name": "speed",
    "label": "Speed",
    "units": "Fisher-Rao distance per block (vertex to vertex = π)",
    "description": "Fisher-Rao distance between the image's lens point at this site and at the next site: the length "
                   "of the displacement v = log_X Y. Needs no neighbourhood, so it is defined at every point.",
    "scale": "sequential",
    "clock": "depth",
}]


def fisher(a, b):
    """Fisher-Rao distance between rows of sphere points a, b (radius 2), float64."""
    return 4 * np.arctan2(np.linalg.norm(a - b, axis=-1), np.linalg.norm(a + b, axis=-1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="r20s0")
    args = ap.parse_args()
    src = ROOT / "data" / "lens" / args.run
    out = ROOT / "data" / "fields" / args.run
    out.mkdir(parents=True, exist_ok=True)
    steps = json.loads((src / "meta.json").read_text())["steps"]
    (out / "fields_meta.json").write_text(json.dumps(META, indent=1, ensure_ascii=False) + "\n")
    for step in steps:
        X = 2 * np.exp(np.load(src / f"logp_{step}.npy").astype(np.float64) / 2)   # [sites, N, 10]
        speed = np.stack([fisher(X[l], X[l + 1]) for l in range(X.shape[0] - 1)]).astype(np.float32)
        np.savez(out / f"depth_{step}.npz", speed=speed)
        totals = [{"speed_mean": {"value": float(np.nanmean(row)), "mass_fraction": float(np.isfinite(row).mean())}}
                  for row in speed]
        (out / f"totals_{step}.json").write_text(json.dumps(totals) + "\n")
    print(f"wrote speed for {len(steps)} checkpoints to {out}")


if __name__ == "__main__":
    main()
