"""ManifoldMotion measurement library: kinematics of a moving point cloud on the Fisher-Rao sphere.

Modules: sphere (geometry), graph (kNN, fixed and adaptive Markov operators, carre du champ), frame
(local second moment G, tangent spaces, release diagnostics, truncated regression), step (per-point fields
of SPEC Section 3 items 1-11 and their release rules), density (cross-fitted kNN density for item 9),
totals (item 13), fixtures (synthetic clouds with known answers).

Main entry points
-----------------
measure_step_logp(logp0, logp1, dt, **kw)   log-probabilities (N, 10) of the same points at two times ->
                                            dict of per-point fields (keys and meanings in FIELD_DOC),
                                            'release' (boolean rule per point), 'rigid', 'meta'.
measure_step(X0, X1, dt, v=None, **kw)      the same on sphere points (or with a supplied displacement).
step_totals(o) / grouped_totals(o, groups) / subsample_totals(X0, X1, dt, ...)   item 13.

Defaults and why
----------------
operator     adaptive, k = 128, k_bw = 16, eps = 0.25 (graph.py).  The adaptive operator passes the same
             operator tests as the fixed one and, unlike it, keeps the kernel inside the kNN rows when the
             density varies by orders of magnitude, as in the real clouds (neighbourhood scales span
             ~1e-7 .. 1 in Fisher distance at the last site).  eps = 0.25 keeps the kernel's second moment
             inside k = 128 rows for local dimension m <= 3 with n_eff ~ 16 at m = 2; k = 128 keeps a
             10,000-point step at ~10-20 s and ~1.3 GB on 4 cores.  The fixed operator (kind='fixed',
             tau=...) is the spec's baseline and must be given an explicit tau.
rules        frame.Rules: retained eigenvalues >= 0.1 lambda_max; tangent rule n_eff >= max(10, 2(m + 1)),
             eigengap >= 2, split-sample angle <= 0.35 rad, one-sidedness <= 0.5, truncated kernel second
             moment <= 1%; strain rules add n_eff >= 5 (m + 1), condition number <= 5, held-out score
             >= 0.5; log J adds a jackknife s.e. of log sigma_min <= 0.1.  Heuristics meant to be swept
             (SPEC Section 4).
resolution   per point: max(1e-12, 4 x the Fisher displacement of one ulp in each input log-probability)
             (sphere.input_resolution); 1e-12 is a conservative float64 floor.
density_k    32 nearest other-fold neighbours, boundary correction on.
"""
from .sphere import (R, sphere_from_logp, logp_from_sphere, fisher_distance, fisher_distance_logp, log_map,
                     exp_map, parallel_transport, parallel_transport_matrix, tangent_project, tangent_projector,
                     face_distance, input_resolution)
from .graph import Operator, build_operator, knn, tau_from_radius
from .frame import Frame, Rules, analyse_frame, regress, truncated_pinv
from .step import (measure_step, measure_step_logp, apply_release, killing_fit, procrustes_rotation,
                   path_length, principal_angles, FIELD_RULE, FIELD_DOC, DEFAULT_OPERATOR)
from .totals import step_totals, grouped_totals, subsample_totals

__all__ = [
    "R", "sphere_from_logp", "logp_from_sphere", "fisher_distance", "fisher_distance_logp", "log_map",
    "exp_map", "parallel_transport", "parallel_transport_matrix", "tangent_project", "tangent_projector",
    "face_distance", "input_resolution", "Operator", "build_operator", "knn", "tau_from_radius", "Frame", "Rules",
    "analyse_frame", "regress", "truncated_pinv", "measure_step", "measure_step_logp", "apply_release",
    "killing_fit", "procrustes_rotation", "path_length", "principal_angles", "FIELD_RULE", "FIELD_DOC",
    "DEFAULT_OPERATOR",
    "step_totals", "grouped_totals", "subsample_totals",
]
