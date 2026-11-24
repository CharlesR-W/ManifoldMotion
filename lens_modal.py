"""Extract ResNet-20 skip-path lenses using a detached Modal coordinator.

    lens_modal.py deploy
    lens_modal.py launch {pilot,seed0,seed1}
    lens_modal.py status {pilot,seed0,seed1} [--wait SECONDS]
    lens_modal.py fetch {pilot,seed0,seed1}

The pilot writes the same seed-0 step-64000 files used by the full sweep.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from uuid import uuid4

import modal

HERE = Path(__file__).resolve().parent
ZOO_SRC = Path.home() / "code/ModelZoo/src"
APP = "mm-manifoldmotion-lens"
OUT_VOLUME = "mm-manifoldmotion"
RUN_RECORDS = HERE / "runs"
OUT = Path("/out/lens")
STAGES = {"pilot": 0, "seed0": 0, "seed1": 1}
SITES = ["stem", *(f"b{i}" for i in range(1, 10))]
N, CLASSES, BATCH = 10_000, 10, 256
EXPLICIT_N = N
FINAL_STEP, N_STEPS = 64000, 73
ATOL, RTOL = 1e-8, 1e-8
SCHEMA = 2
GPU = ["L4", "T4"]  # Prefer L4; both fit the budget even at the L4 rate.
GPU_TIMEOUT, STARTUP, IDLE, COORD_TIMEOUT = 100, 20, 2, 900
MAX_CONTAINERS = 16
# On-demand USD/s from https://modal.com/pricing (checked 2026-11-24 UTC).
CPU_RATE, RAM_RATE, GPU_RATE = 0.0000131, 0.00000222, 0.000222
WORKER_RATE = GPU_RATE + CPU_RATE + 2 * RAM_RATE
COORD_RATE = 0.25 * CPU_RATE + 0.5 * RAM_RATE
UNIT_RESERVE = (GPU_TIMEOUT + STARTUP + IDLE) * WORKER_RATE
COORD_RESERVE = (COORD_TIMEOUT + STARTUP + IDLE) * COORD_RATE
BUDGET = 5.0

app = modal.App(APP)
ckpt_vol = modal.Volume.from_name("cm-cifar-coherent").read_only()
data_vol = modal.Volume.from_name("crw-modelzoo-train").read_only()
out_vol = modal.Volume.from_name(OUT_VOLUME, create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.8.0", "numpy==2.1.3")
    .env({"PYTHONPATH": "/root/src", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
    .add_local_dir(str(ZOO_SRC), remote_path="/root/src")
)


def run_name(seed):
    return f"resnet20-relu-seed{seed}"


def run_root(seed):
    return Path("/checkpoints/runs") / run_name(seed)


def save_json(path, data):
    tmp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def save_npy(path, data):
    import numpy as np

    tmp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with tmp.open("wb") as f:
            np.save(f, data, allow_pickle=False)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def shortcut(h, block):
    import torch.nn.functional as F

    if block.stride != 1:
        h = h[:, :, ::2, ::2]
    if block.pad_channels > 0:
        h = F.pad(h, (0, 0, 0, 0, 0, block.pad_channels))
    return h


def skip_pool(h, blocks, site):
    for i in range(site, len(blocks)):
        h = shortcut(h, blocks[i])
    return h.mean(dim=(2, 3))


def measure_batch(model, x, explicit_n):
    """Capture the real model's post-ReLU sites and check three lens identities."""
    import torch
    import torch.nn.functional as F

    blocks = tuple(model.blocks)
    activations = []
    with ExitStack() as hooks:
        for module in (model.act, *blocks):
            hooks.enter_context(module.register_forward_hook(
                lambda _m, _inp, out: activations.append(out)))
        ordinary = model(x)
    if len(activations) != len(SITES):
        raise ValueError(f"Expected {len(SITES)} activations, got {len(activations)}")
    lens = torch.stack([model.fc(skip_pool(h, blocks, i)) for i, h in enumerate(activations)])

    increments = torch.stack([
        F.linear(skip_pool(activations[i + 1] - shortcut(activations[i], block), blocks, i + 1),
                 model.fc.weight, bias=None)
        for i, block in enumerate(blocks)
    ])
    errors = [error_pair(lens[-1:], ordinary.unsqueeze(0)),
              error_pair(lens[1:] - lens[:-1], increments)]

    if explicit_n:
        explicit = []
        with ExitStack() as hooks:
            for block in blocks:
                hooks.enter_context(block.bn2.register_forward_hook(
                    lambda _m, _inp, out: torch.zeros_like(out)))
            for i, h in enumerate(activations):
                tail = h[:explicit_n]
                for block in blocks[i:]:
                    tail = block(tail)
                explicit.append(model.fc(tail.mean(dim=(2, 3))))
        errors.append(error_pair(lens[:, :explicit_n], torch.stack(explicit)))
    return lens, ordinary, errors


def error_pair(actual, reference):
    import torch

    absolute = (actual - reference).abs().amax(dim=-1)
    # Relative infinity error per image, with a one-logit absolute scale for
    # near-zero rows (notably for small increments with cancellation).
    scaled = absolute / reference.abs().amax(dim=-1).clamp_min(1.)
    return torch.stack((absolute.amax(dim=-1), scaled.amax(dim=-1)), dim=-1)


def error_record(pair, n, name):
    max_abs, max_relative = pair
    if (not math.isfinite(max_abs) or not math.isfinite(max_relative)
            or max_abs > ATOL or max_relative > RTOL):
        raise ValueError(f"{name}: check failed: {max_abs=:.4g}, {max_relative=:.4g}, "
                         f"{ATOL=:.4g}, {RTOL=:.4g}")
    return dict(n=n, max_abs=max_abs, max_relative=max_relative)


def shared_ready(root, seed, steps):
    path = root / "meta.json"
    if not path.is_file() or not (root / "labels.npy").is_file():
        return False
    if seed == 0 and not (root / "images.npy").is_file():
        return False
    meta = json.loads(path.read_text())
    return (meta.get("schema") == SCHEMA and meta.get("seed") == seed
            and meta.get("steps") == steps)


def completed(root, seed, step):
    marker = root / f"checks_{step}.json"
    if not all(p.is_file() for p in (marker, root / f"logits_{step}.npy", root / f"logp_{step}.npy")):
        return False
    rec = json.loads(marker.read_text())
    return (rec.get("schema") == SCHEMA and rec.get("seed") == seed
            and rec.get("step") == step and rec.get("passed") is True)


# Reuse the fixed test tensors within a worker container, not between workers.
_DATA = {}


@app.function(image=image, gpu=GPU, cpu=(1, 1), memory=(2048, 2048), timeout=GPU_TIMEOUT,
              startup_timeout=STARTUP, scaledown_window=IDLE, max_containers=MAX_CONTAINERS,
              retries=0, volumes={"/checkpoints": ckpt_vol, "/data": data_vol, "/out": out_vol})
def checkpoint(seed: int, step: int, steps: list[int], for_shared: bool, archived_accuracy: float):
    start = time.monotonic()
    import numpy as np
    import torch
    from zoo.data import build_data
    from zoo.io import load_checkpoint
    from zoo.models import build_model

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")
    torch.set_num_threads(1)
    out_vol.reload()
    root = OUT / f"r20s{seed}"
    if completed(root, seed, step) and (not for_shared or shared_ready(root, seed, steps)):
        return dict(step=step, skipped=True, seconds=time.monotonic() - start)

    ck = load_checkpoint(run_root(seed) / "checkpoints" / f"step_{step}.pt")
    cfg = ck["config"]
    if (ck["step"] != step or cfg["dataset"] != "cifar10" or cfg["seed"] != seed
            or cfg["model"] != {"arch": "resnet20", "activation": "relu", "num_classes": CLASSES}):
        raise ValueError(f"Unexpected checkpoint metadata for seed {seed}, step {step}")
    model = build_model(cfg)
    model.load_state_dict(ck["model_state"])
    model.eval().to(device="cuda", dtype=torch.float64)
    stage_changes = [(i + 1, b.stride, b.pad_channels) for i, b in enumerate(model.blocks)
                     if b.stride != 1 or b.pad_channels != 0]
    if len(model.blocks) != 9 or stage_changes != [(4, 2, 16), (7, 2, 32)]:
        raise ValueError(f"Unexpected blocks: {len(model.blocks)=}, {stage_changes=}")
    key = (cfg["dataset"], cfg["batch_size"])
    if key not in _DATA:
        from tempfile import TemporaryDirectory

        # ModelZoo unconditionally calls mkdir, which can fail even when the
        # directory already exists on a read-only Volume. Keep the cache root
        # writable in this container but read the archival tarball in place.
        mounted = Path("/data/data")  # the archive lives under data/ in the mounted Volume
        archive = mounted / "cifar10/cifar-10-python.tar.gz"
        hashes = (mounted / "DATA_HASHES.json").read_text()
        if not archive.is_file() or archive.name not in json.loads(hashes):
            raise ValueError("Mounted CIFAR-10 archive or its recorded hash is missing")
        with TemporaryDirectory(prefix="mm-cifar-") as cache:
            local = Path(cache)
            (local / "cifar10").mkdir()
            (local / "cifar10" / archive.name).symlink_to(archive)
            (local / "DATA_HASHES.json").write_text(hashes)
            _DATA[key] = build_data(cfg, local, "cpu")
    data = _DATA[key]
    if (tuple(data.test_x.shape) != (N, 3, 32, 32) or data.test_x.dtype != torch.uint8
            or tuple(data.test_y.shape) != (N,) or data.test_y.dtype != torch.int64):
        raise ValueError("Unexpected CIFAR-10 test array shapes or dtypes")
    labels = data.test_y.numpy()
    y = data.test_y.cuda()
    setup_s = time.monotonic() - start

    shape = (len(SITES), N, CLASSES)
    output = np.empty(shape, dtype=np.float32)
    logp = np.empty_like(output)
    a = torch.zeros((1, 2), device="cuda", dtype=torch.float64)
    b = torch.zeros((10, 2), device="cuda", dtype=torch.float64)
    c = torch.zeros((9, 2), device="cuda", dtype=torch.float64)
    correct = torch.zeros(10, device="cuda", dtype=torch.int64)
    ordinary_correct = torch.zeros((), device="cuda", dtype=torch.int64)
    with torch.inference_mode():
        for lo in range(0, N, BATCH):
            hi = min(lo + BATCH, N)
            # Match ModelZoo's normalization (float32), then promote before model inference.
            x = data.normalize(data.test_x[lo:hi]).to(device="cuda", dtype=torch.float64)
            z, ordinary, errors = measure_batch(model, x, max(0, min(hi, EXPLICIT_N) - lo))
            a = torch.maximum(a, errors[0])
            c = torch.maximum(c, errors[1])
            if len(errors) == 3:
                b = torch.maximum(b, errors[2])
            correct += (z.argmax(dim=-1) == y[lo:hi]).sum(dim=1)
            ordinary_correct += (ordinary.argmax(dim=-1) == y[lo:hi]).sum()
            output[:, lo:hi] = z.float().cpu().numpy()
            logp[:, lo:hi] = z.log_softmax(dim=-1).float().cpu().numpy()
    checks_a = error_record(a.tolist()[0], N, "a b9 vs model")
    checks_b = {site: error_record(pair, EXPLICIT_N, f"b {site} vs real zero-branch tail")
                for site, pair in zip(SITES, b.tolist())}
    checks_c = {f"{src}->{dst}": error_record(pair, N, f"c {src}->{dst} affine increment")
                for src, dst, pair in zip(SITES, SITES[1:], c.tolist())}
    counts = correct.tolist()
    ordinary_count = ordinary_correct.item()
    if counts[-1] != ordinary_count or not np.isfinite(output).all() or not np.isfinite(logp).all():
        raise ValueError(f"Inconsistent accuracy or non-finite outputs for seed {seed}, step {step}")
    inference_s = time.monotonic() - start - setup_s

    if for_shared and not shared_ready(root, seed, steps):
        save_npy(root / "labels.npy", labels)
        if seed == 0:
            save_npy(root / "images.npy", data.test_x.permute(0, 2, 3, 1).contiguous().numpy())
        save_json(root / "meta.json", dict(
            schema=SCHEMA, run=run_name(seed), seed=seed, steps=steps, sites=SITES, N=N,
            torch_version=torch.__version__, row_order="Row i is CIFAR-10 test image i (zero-based original dataset order).",
            normalization="ModelZoo VisionData.normalize (float32), then float64 eval inference and log-softmax",
        ))

    save_npy(root / f"logits_{step}.npy", output)
    save_npy(root / f"logp_{step}.npy", logp)
    seconds = time.monotonic() - start
    save_json(root / f"checks_{step}.json", dict(
        schema=SCHEMA, seed=seed, step=step, passed=True, atol=ATOL, rtol=RTOL,
        error_definition="max_abs: max component error; max_relative: max over images of component infinity error divided by max(1, reference logit/step infinity norm)",
        explicit_rows=f"first {EXPLICIT_N} CIFAR-10 test images (0 through {EXPLICIT_N - 1})",
        a=checks_a, b=checks_b, c=checks_c, lens_top1_accuracy=[n / N for n in counts],
        model_top1_accuracy=ordinary_count / N, archived_model_top1_accuracy=archived_accuracy,
        gpu=torch.cuda.get_device_name(), setup_s=setup_s, inference_s=inference_s, seconds_before_commit=seconds,
    ))
    out_vol.commit()
    return dict(step=step, skipped=False, seconds=time.monotonic() - start)


@app.function(cpu=(0.25, 0.25), memory=(512, 512), timeout=COORD_TIMEOUT, startup_timeout=STARTUP,
              scaledown_window=IDLE, max_containers=1, retries=0,
              volumes={"/checkpoints": ckpt_vol, "/out": out_vol})
def coordinate(stage: str, reserved_usd: float):
    start = time.monotonic()
    launched_at = time.time()
    seed = STAGES[stage]
    root = OUT / f"r20s{seed}"
    manifest = json.loads((run_root(seed) / "manifest.json").read_text())
    expected = {row["step"]: row["val_acc"] for row in manifest["checkpoints"]}
    checkpoints = run_root(seed) / "checkpoints"
    steps = sorted(int(m[1]) for p in checkpoints.iterdir()
                   if p.is_file() and (m := re.fullmatch(r"step_(\d+)\.pt", p.name)))
    if len(steps) != N_STEPS or steps != sorted(expected) or steps[0] != 0 or steps[-1] != FINAL_STEP:
        raise ValueError(f"Checkpoint inventory disagrees with the manifest or expected range: {steps}")
    out_vol.reload()
    if stage != "pilot" and not completed(OUT / "r20s0", 0, FINAL_STEP):
        raise RuntimeError("Run and inspect the pilot before sweeping")
    if stage == "seed1" and not all(completed(OUT / "r20s0", 0, s) for s in steps_for_seed0()):
        raise RuntimeError("Finish seed 0 before seed 1")
    if not root.is_dir():
        root.mkdir(parents=True)
        out_vol.commit()
    want = [FINAL_STEP] if stage == "pilot" else steps
    shared_ok = shared_ready(root, seed, steps)
    pending = [s for s in want if not completed(root, seed, s) or (s == FINAL_STEP and not shared_ok)]
    ceiling = len(pending) * UNIT_RESERVE + COORD_RESERVE
    if ceiling > reserved_usd + 1e-9:
        raise RuntimeError(f"Estimated ceiling ${ceiling:.3f} exceeds launch reservation ${reserved_usd:.3f}")
    args = [(seed, s, steps, s == FINAL_STEP, expected[s]) for s in pending]
    results = list(checkpoint.starmap(args, return_exceptions=True)) if args else []
    failures = {str(s): repr(result) for s, result in zip(pending, results) if isinstance(result, Exception)}
    if failures:
        raise RuntimeError(f"Checkpoint failures for seed {seed}: {failures}")
    out_vol.reload()
    if not shared_ready(root, seed, steps) or not all(completed(root, seed, s) for s in want):
        raise RuntimeError(f"Missing committed outputs for {stage}")
    worker_seconds = sum(r["seconds"] for r in results)
    wall_s = time.monotonic() - start
    return dict(stage=stage, seed=seed, steps=want, seconds=wall_s, worker_seconds=worker_seconds,
                computed=len(pending) - sum(r["skipped"] for r in results),
                skipped=len(want) - len(pending) + sum(r["skipped"] for r in results),
                coordinator_started_at=launched_at, finished_at=time.time(),
                estimated_usd=worker_seconds * WORKER_RATE + wall_s * COORD_RATE,
                spend_with_startup_reserve_usd=(worker_seconds + len(pending) * (STARTUP + IDLE)) * WORKER_RATE
                + (wall_s + STARTUP + IDLE) * COORD_RATE)


def steps_for_seed0():
    manifest = json.loads((run_root(0) / "manifest.json").read_text())
    return [row["step"] for row in manifest["checkpoints"]]


def records(stage="*"):
    return sorted(RUN_RECORDS.glob(f"lens-{stage}-*.json"))


def launch(stage):
    RUN_RECORDS.mkdir(exist_ok=True)
    previous = [json.loads(f.read_text()) for f in records()]
    if any("result" not in r for r in previous):
        raise RuntimeError("A previous call has no result in its launch record; run status first")
    if stage != "pilot" and not any(r["stage"] == "pilot" and "error" not in r["result"] for r in previous):
        raise RuntimeError("Complete and inspect the pilot before fan-out")
    seed = STAGES[stage]
    tag = f"lens/r20s{seed}"
    vol = modal.Volume.from_name(OUT_VOLUME)
    try:
        files = {Path(entry.path).name for entry in vol.listdir(f"/{tag}")}
    except modal.exception.NotFoundError:
        files = set()
    done = set()
    for name in files:
        m = re.fullmatch(r"checks_(\d+)\.json", name)
        if m and {f"logits_{m[1]}.npy", f"logp_{m[1]}.npy"} <= files:
            check = json.loads(b"".join(vol.read_file(f"{tag}/{name}")))
            if (check.get("schema") == SCHEMA and check.get("seed") == seed
                    and check.get("step") == int(m[1]) and check.get("passed") is True):
                done.add(int(m[1]))
    if not {"meta.json", "labels.npy", *(["images.npy"] if seed == 0 else [])} <= files:
        done.discard(FINAL_STEP)
    n = int(FINAL_STEP not in done) if stage == "pilot" else N_STEPS - len(done)
    reservation = n * UNIT_RESERVE + COORD_RESERVE
    used = sum(r["result"].get("spend_with_startup_reserve_usd", r["reserved_usd"]) for r in previous)
    if used + reservation > BUDGET:
        raise RuntimeError(f"Cumulative ceiling ${used + reservation:.3f} exceeds ${BUDGET:.2f}")
    t = time.time()
    call = modal.Function.from_name(APP, "coordinate").spawn(stage, reservation)
    rec = dict(stage=stage, app=APP, call_id=call.object_id, gpu=GPU,
               launched_at=t, launched_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t)),
               reserved_usd=reservation)
    path = RUN_RECORDS / f"lens-{stage}-{int(t * 1e6)}.json"
    save_json(path, rec)
    print(json.dumps(rec, indent=2))


def status(stage, wait):
    matches = records(stage)
    if not matches:
        raise RuntimeError(f"No launch record for {stage}")
    path = matches[-1]
    rec = json.loads(path.read_text())
    if "result" not in rec:
        try:
            rec["result"] = modal.FunctionCall.from_id(rec["call_id"]).get(timeout=wait)
        except TimeoutError:
            print(f"running: {rec['call_id']}")
            return
        except (modal.exception.RemoteError, modal.exception.InputCancellation) as exc:
            rec["result"] = {"error": repr(exc)}
        except Exception as exc:
            # Modal re-raises the worker/coordinator exception in the caller.
            # Leave transport failures pending; polling again is safe.
            if not isinstance(exc, (RuntimeError, ValueError, modal.exception.FunctionTimeoutError)):
                raise
            rec["result"] = {"error": repr(exc)}
        save_json(path, rec)
    print(json.dumps(rec, indent=2))
    if "error" in rec["result"]:
        raise SystemExit(1)


def fetch(stage):
    tag = f"r20s{STAGES[stage]}"
    dest = HERE / "data/lens" / tag
    dest.mkdir(parents=True, exist_ok=True)
    vol = modal.Volume.from_name(OUT_VOLUME)
    remote = f"lens/{tag}"
    meta = json.loads(b"".join(vol.read_file(f"{remote}/meta.json")))
    names = ["meta.json", "labels.npy"] + (["images.npy"] if STAGES[stage] == 0 else [])
    for s in ([FINAL_STEP] if stage == "pilot" else meta["steps"]):
        names.extend([f"{p}_{s}.{ext}" for p, ext in (("checks", "json"), ("logits", "npy"), ("logp", "npy"))])
    from concurrent.futures import ThreadPoolExecutor

    def download(name):
        target = dest / name
        tmp = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
        try:
            with tmp.open("wb") as f:
                for chunk in vol.read_file(f"{remote}/{name}"):
                    f.write(chunk)
            tmp.replace(target)
        finally:
            tmp.unlink(missing_ok=True)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(download, names))
    print(f"fetched {len(names)} files to {dest}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("deploy")
    for cmd in ("launch", "status", "fetch"):
        p = sub.add_parser(cmd)
        p.add_argument("stage", choices=STAGES)
        if cmd == "status":
            p.add_argument("--wait", type=float, default=0)
    args = parser.parse_args()
    if args.command == "deploy":
        subprocess.run([sys.executable, "-m", "modal", "deploy", __file__], check=True)
    elif args.command == "launch":
        launch(args.stage)
    elif args.command == "status":
        status(args.stage, args.wait)
    else:
        fetch(args.stage)


if __name__ == "__main__":
    main()
