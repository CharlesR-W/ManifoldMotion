# ManifoldMotion: instructions for agents

The owner's global instructions in `~/.claude/CLAUDE.md` apply and take precedence.

- Read `SPEC.md` first. Its Section 0 is the owner's fixed brief: raw activations, graph versions of manifold
  quantities, the data as a weighted (metric-measure) space with mass from sampling, Fisher geometry of the
  output as the reference. Do not reintroduce the CoherentModes kernel / doubly stochastic construction.
- The sampled image of a ReLU network is not a smooth embedded manifold. Report tangent-based or
  curvature-like quantities only where the scale audit (SPEC Section 3) finds a stable local scale, and name
  graph quantities as graph quantities (e.g. "coarse transport curvature", not "Ricci").
- Every estimator is run on the calibration fixtures (SPEC Section 3) before it is used on the network.
- Changes to what is measured go into `SPEC.md` and get a GPT critique before code.
- GPU or long runs: Modal (the `modal` skill). Checkpoints live on the volume `cm-cifar-coherent`; never write
  into `/runs` there. This project's outputs go to its own volume.
- Reuse before rewriting: ModelZoo `zoo.models`, CoherentModes `explore/cifar_coherent`, DADApy, POT.
- Commit and push only when the owner asks.
