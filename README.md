# ManifoldMotion

How the forward pass of a network moves the data through an ambient Fisher space, measured as kinematics of a
point cloud: speed, rigid and non-rigid motion, relative motion of neighbours, and (where it can be estimated)
strain.

Site, with the explorer and the experiments: https://charlesr-w.github.io/ManifoldMotion/

## What is here

- `SPEC.md`: what is measured and why. Section 1 defines the space (logit lens along the skip path, then
  `X = 2 sqrt(p)` on the radius-2 sphere, whose geometry is Fisher-Rao); Sections 2-4 the calculus, fields and
  release rules; Section 4a what the real cloud allowed and which estimands are released.
- `lens_modal.py`: lens extraction on Modal, with three exact checks per checkpoint.
- `mm/`: the measurement library (sphere geometry, diffusion operators, local moments and regressions, fields,
  totals, fixtures). `tests/`: the fixture suite (`uv run pytest`; slow sweeps with `uv run pytest -m slow`).
- `fields_modal.py`: the measurement run on Modal.
- `site/`: the site (`site/build_site.py` builds `site/dist/`).

Subject: the CIFAR-10 ResNet-20 trained in CoherentModes (Modal volume `cm-cifar-coherent`,
`/runs/resnet20-relu-seed{0,1}`, 73 checkpoints each), read on the 10,000 test images. Outputs live on the Modal
volume `mm-manifoldmotion` (`/lens/<run>/`, `/fields/<run>/`) and locally under `data/` (not in git).

## Status (2026-11-24)

- Lens extraction: both seeds, all 146 checkpoints; every check passes (worst increment-identity error 6.75e-14).
  Modal app `mm-manifoldmotion-lens`; about $2.
- Measurement run: seed 0, 40 checkpoints, depth and training clocks, k = 256; scale, subsample and seed-1 checks
  at the final checkpoint. Modal app `mm-manifoldmotion-fields`, calls `fc-01M86XACE89ZX5SPFQCBCFBN8V` (pilot) and
  `fc-01M8709487F7ST3E7MQHPRFYGJ` (main); about $6. Launch records in `runs/`.
- Outcome in one line: at sites stem..b8 the lens cloud is a 9-dimensional blob in the 9-dimensional Fisher
  sphere, not a thin manifold, so the intrinsic / extrinsic split and strain energy the project set out to measure
  are not defined or not recoverable at 10,000 points; travel, the rigid fraction, and the split of neighbours'
  relative motion into shared shift, slope and residual are measured on every image.
- Not done: training images (deliberately deferred); second-order quantities (SPEC Section 3 item 12).

## Rebuild

    uv run --with modal python lens_modal.py deploy     # then launch / status / fetch, see its docstring
    uv run --with modal python fields_modal.py deploy   # likewise
    uv run --with numpy --with pillow python site/build_site.py
