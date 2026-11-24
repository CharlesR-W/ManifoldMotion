# ManifoldMotion: measurement spec (draft 5: kinematics of the lens cloud by carré-du-champ calculus)

Draft 5 follows the owner's direction of 2026-11-23 (Section 0): continuum-mechanics quantities of the moving
point cloud, computed with Bakry-Emery (carré du champ) calculus. Earlier drafts and their critiques are
summarised in Section 9. GPT-6.2 Astra checked the formulas of draft 5 over two rounds; its corrections
are incorporated and it found nothing blocking stage 1 or the start of stage 2. Nothing has been run.

## 0. Owner's brief (fixed)

- Subject: activations of the CIFAR-10 ResNet-20 trained in CoherentModes (Modal volume `cm-cifar-coherent`,
  `/runs/resnet20-relu-seed{0,1}`, 73 training checkpoints each).
- A picture of the data manifold moving over "time" (depth) through an ambient Fisher space.
- Fisher space is reached by the logit lens along the skip path, not by pulling the output Fisher back.
- Raw Euclidean geometry of activations is dropped. That the lens is a poor readout early does not matter.
- What to measure: intrinsic and extrinsic components of velocity, strains and strain energy, and quantities of
  that kind: local speed, local relative motion, local deformation.
- Method: Bakry-Emery calculus or similar, treating the points as a weighted Riemannian manifold with the
  probability as density, in graph form.
- No Wasserstein / optimal-transport quantities. The cloud is studied on its own; the loss is a side interest.
- No CoherentModes-style operator construction (cosine kernels scaled to doubly stochastic, SEBA).

## 1. The space and the moving body

**Lens.** Sites l = 0..9: stem output and outputs of blocks b1..b9 (post-ReLU). z_l = A_l h_l + b, where A_l is
the skip-only continuation (subsample by 2 from index 0 and append zero channels at each later stage change,
global average pool, final linear layer): the network's output if every later residual branch were zero. With Q
the next shortcut, z_{l+1} - z_l = A_{l+1}[relu(Q h_l + F(h_l)) - Q h_l]. Evaluation-mode BatchNorm.

**Ambient space.** X_l(x) = 2 sqrt(softmax(z_l(x))) in the positive orthant of the radius-2 sphere S^9 in R^10.
The sphere's induced metric is Fisher-Rao, and it is the same space at every site and checkpoint, so the
ambient geometry does not change with time: every change of the cloud's induced metric is material deformation.
Computations are done with the R^10 coordinates; "normal" means normal to the cloud within the sphere, and the
radial direction is projected out.

**Body.** The material points are the images; X_l is the configuration at time l. The cloud at time l carries
mass 1/N per point, and is treated as a weighted manifold (M_l, g_l, rho_l dvol) with g_l induced from the
sphere. Two clocks: depth (10 configurations) and training (73 checkpoints, rates per actual step interval).

**Displacement.** v_l(x) = log_{X_l(x)} X_{l+1}(x), a tangent vector of the sphere, written in R^10. A block is
a finite step, not an infinitesimal one; quantities are given in a finite form (exact for finite deformation)
and a rate form (first order in the step) and the two are compared.

## 2. The calculus

Conventions: column vectors; in any moment matrix the row is the output component and the column the input
direction. At a point x on the sphere, Q_x = I - x x^T / 4 projects onto the sphere's tangent space; T is the
cloud's tangent projector inside it and N = Q_x - T the normal projector within the sphere. tau is kernel
time, Delta t the physical step (one block, or the actual number of training steps), E strain.

**Operator.** Baseline, with a proven limit: fixed-bandwidth kernel exp(-d^2 / 4 tau) in Fisher distance,
symmetric, diffusion-maps degree correction alpha = 1/2, row normalisation to P, L = (P - I) / tau. On a smooth
interior patch L -> Delta + grad log rho . grad. An adaptive (kNN-bandwidth) version is used only once its
kernel, bandwidth units, density pilot, degree correction, row time, symmetrisation and truncation are written
down with the limit of that exact construction, and it passes the same tests. Two tests are needed because
the first cannot detect a wrong drift: Gamma on linear functions, and L on a coordinate along a known density
gradient. The operator's reversible weights are d_i tau_i (corrected degree times row time), used for spectra;
material totals use equal image weights.

**Carré du champ**, exact on the graph for any row-stochastic P:

    Gamma(f, h)_i = (1/2)[L(fh) - f Lh - h Lf]_i = (1 / 2 tau) sum_j P_ij (f_j - f_i)(h_j - h_i)  ->  <grad f, grad h>.

**Local moments at an anchor i** (weights a_j = P_ij / 2 tau). Offsets are sphere log maps, which removes the
radial part of chords: u_j = log_{x_i} x_j at time l, y_j = log_{Y_i} Y_j at time l+1, and
w_j = PT_{j->i} v_j - v_i, the neighbour's displacement parallel-transported to the anchor minus the anchor's.

    G = sum a_j u_j u_j^T        M = sum a_j y_j u_j^T        B = sum a_j y_j y_j^T        C_v = sum a_j w_j u_j^T

- G is the local second moment of the cloud. In the calibrated limit it is the tangent projector T and its
  trace the dimension m; at finite bandwidth it is not a projector (density curvature, the cloud's curvature
  and one-sided neighbourhoods all enter at order tau or worse), so it is used as a regression design, never
  assumed to be the identity. The tangent space is the retained eigenspace of G (relative threshold); its
  eigenvalues, the physical radius, the effective neighbour count 1 / sum_j P_ij^2 and the conditioning are
  reported. Only the source predictor u_j is truncated to the retained subspace; the responses y_j and w_j stay
  in the full sphere tangent (projecting them would force the tilt to zero and hide unresolved separation),
  and fitted matrices are lifted back to ambient coordinates before symmetric and skew parts are taken.
- Deformation gradient F = M G^+ (weighted least squares of next-time offsets on current offsets). Exactly,
  B = F G F^T + R with R = sum a_j (y_j - F u_j)(y_j - F u_j)^T >= 0.
- Displacement gradient K_d = C_v G^+, the Fisher-space covariant derivative of the displacement field, with
  its own residual R_v; the rate is K = K_d / Delta t for V = v / Delta t (squared rates divide by Delta t^2;
  this matters on the training clock, where intervals differ). Exact graph budget of relative motion:
  sum a_j |w_j|^2 = tr(K_d G K_d^T) + tr R_v.
- "Fraction of next-frame neighbour separation explained by a source-linear fit" = tr(F G F^T) / tr B, with a
  bandwidth sweep and a separate held-out score 1 - SSE_test / sum_test a_j |y_j|^2 (which can be negative). The residual mixes smooth nonlinearity at finite radius,
  dropped directions, sampling noise and genuine unpredictability (the lens at l does not determine the lens
  at l+1); it is reported, not interpreted as one of these.

**Second order** (stage 4). The identifications below hold in the continuum limit; on the graph the iterated
expressions define estimators. Iterated Gamma gives the Hessian,
Hess f(grad a, grad b) = (1/2)[Gamma(a, Gamma(f, b)) + Gamma(b, Gamma(f, a)) - Gamma(f, Gamma(a, b))]. Applied to
the R^10 coordinates it returns the Euclidean second fundamental form; on the radius-2 sphere
II_R(u, w) = II_S(u, w) - <u, w> x / 4 and L X = H_S - m x / 4 + grad_M log rho, with H_S = trace of II_S. So
N L X = H_S (mean curvature vector in Fisher space) and T L X = grad log rho. Exact implementation check at
every bandwidth: x_i . (L X)_i = - tr Gamma(X, X)_i (coordinate chords, not the sphere-log design G). Bochner: Gamma_2(f, f) = (1/2) L Gamma(f, f) - Gamma(f, L f)
= |Hess f|^2 + (Ric - Hess log rho)(grad f, grad f); the Bakry-Emery Ricci is Gamma_2 minus the Hessian term,
per unit gradient (a uniform great m-sphere has Ric = (m - 1) g / 4, the sign-and-scale test).

## 3. Fields measured per point and per step

v = log_x Y is a displacement over Delta t; rates divide by Delta t. It is the initial velocity of the geodesic
joining the two frames, not the network's instantaneous velocity.

**Speed (motion); no neighbourhood needed for item 1.**
1. Fisher travel distance |v| per step; accumulated path length.
2. Intrinsic / extrinsic split: v = T v + N v. Tangential: the point slides along the cloud. Normal: the cloud
   itself moves through Fisher space. (Motion along a fixed curved support has a normal part of order
   Delta t^2. Where the cloud is locally 9-dimensional there is no normal space and the split is trivially
   all-tangential; such points are reported as full-rank, not as "no extrinsic motion".)

**Relative motion.**
3. q = sum a_j |w_j|^2: how differently neighbours move (the carré du champ of the displacement field in
   Fisher space, its Dirichlet energy density).
4. Two separate budgets, with different denominators. Measured energy = fitted part tr(K_d G K_d^T) + residual
   tr R_v. Squared norm of the fitted gradient:
   |K T|^2 = |sym(T K T)|^2 + |skew(T K T)|^2 + |N K T|^2 = strain rate + in-plane spin + tilt.
   Strain is intrinsic deformation; spin changes nothing; tilt is reorientation of the tangent plane (not by
   itself a change of curvature).

**Intrinsic deformation (strain).**
5. Rate form: E = sym(T K T), which represents (1/2) g^{-1} dg/dt. Trace = volume expansion rate; trace-free
   part = shear; top eigenvalue and eigenspace = maximal stretching.
6. Finite form: singular values sigma_k of F on the retained tangent space; Hencky strain (1/2) log(F^T F),
   principal log-stretches lambda_k = log sigma_k with their source directions; log J = sum lambda_k. A
   vanishing singular value is reported as rank loss and log J withheld, not floored.
7. Kinematic strain measures (unit-weight; not a stored elastic energy, which would need a material law):
   volumetric (sum lambda_k)^2 / m, shear sum lambda_k^2 - (sum lambda_k)^2 / m, total sum lambda_k^2; rate
   analogues (tr E)^2 / m, |E|^2 - (tr E)^2 / m, |E|^2.
8. Where the strain comes from: E(u, w) = sym grad_M V_T (u, w) - <II_S(u, w), V_N>: stretching by flow within
   the cloud, plus stretching caused by normal motion of a curved cloud. Computed by differentiating V_T and
   V_N with each neighbour's own projector. Released only where the projector field is stable; the two terms
   are tested individually on fixtures (they can be large and opposite).
9. Mass balance, where the map is locally one-to-one with constant dimension and the target is occupied by a
   single sheet (no distant patch arriving at the same place): rho_{l+1}(Y) J = rho_l(x). Tested
   against a density estimated independently of the operator (cross-fitted, common dimension, boundary
   correction). At a merge of sheets densities add, so a mismatch there is not an error.

**Extrinsic deformation.**
10. Finite tilt: principal angles between PT_{x->Y}(T_x M_l) and T_Y M_{l+1}, the tangent planes compared at the
    same point of the sphere. (The angle between F(T_x M_l) and T_Y M_{l+1} is a separate map-consistency check.)
11. Rigid part. Rate form: least-squares Killing field v = Omega x, Omega antisymmetric (normal equation
    Omega C + C Omega = D - D^T, C = sum x x^T, D = sum v x^T), with conditioning reported. Finite form:
    origin-preserving SO(10) Procrustes on the paired positions (chordal loss). Reported: rigid fraction of
    kinetic energy, 1 - RSS / sum |V|^2, from the rate-form fit (the additive split); items 2, 4, 10 again in
    the co-rotating frame of the finite fit (a diagnostic, not an additive complement).
12. Second order, stage 4: H_S and |II_S|^2 per time; normal displacement against H_S; bending = change of II_S
    with inputs pulled back by F and normal outputs compared after parallel transport, in the co-rotating
    frame (whether as sphere-tangent vectors or after aligning the two normal planes is fixed before stage 4);
    Bakry-Emery curvature.

**Totals per step.**
13. Mean over images of: kinetic energy, tangential / normal and rigid / non-rigid; relative motion, fitted /
    residual and strain / spin / tilt; strain, volumetric / shear. Each curve with its physical radius, the mass
    fraction on which it is defined, and paired-subsample variation. Over depth; over training for each depth
    step; and with the training clock.

**Secondary.**
14. Each scalar field by class, by correct / incorrect, and by distance to the nearest simplex vertex and edge
    (2 atan2(sqrt(sum_{c not in I} pi_c), sqrt(sum_{c in I} pi_c))). Deferred: spectra over eigenfunctions of L
    (reversible inner product, near-degenerate eigenspaces grouped) and autocorrelations.
15. Side interest: signed component of v on the unit geodesic to the label vertex, and the finite decrease in
    distance to it.

## 4. Estimation

- Release rules. Item 1, total kinetic energy and its rigid fit need no neighbourhood and are always reported
  (the tangential / normal kinetic split does need the tangent rule). Tangent-
  dependent fields need a stable retained subspace (eigengap, split-sample subspace angle, effective neighbour
  count, radius small against the distance to competing clusters). Strain, tilt and log J need stable,
  well-populated directions and held-out prediction. Where a rule fails the field is undefined at that point;
  the point keeps its mass, and every aggregate states the mass fraction it covers, by class and by distance
  to the vertex.
- Scales: two or three bandwidths chosen as physical radii; sensitivity to the eigenvalue threshold.
- Step size: displacement angle |v| / 2 and Delta t |K| per block; agreement of rate and finite forms.
- Fixtures on the same sphere at matched radii, anisotropy and occupancy. Exact-algebra tests pass to numerical
  precision; continuum tests pass over a bandwidth and sample-size sweep.
  1. Algebra only: anisotropic design, a dropped direction, a known nonlinear residual (transpose convention,
     truncated fit, residual closure, Gram-weighted versus Frobenius energy).
  2. Identity map on a non-uniformly sampled great-subsphere patch: zero motion and strain, unit stretches,
     correct diffusion scale, non-zero tangential L X where the density varies; run with a deliberately wrong
     bandwidth convention to confirm the test catches it.
  3. Killing field supplied directly (zero strain), and finite rotations at several angles (unit stretches;
     log-displacement strain tends to zero only as a rate); a full-rank patch where normal quantities vanish;
     one endpoint pair assigned two different time intervals (finite strain unchanged, rates scale as
     1 / Delta t, squared rates as 1 / Delta t^2).
  4. Latitude subsphere moved from angle theta to theta' at fixed longitude: every stretch sin theta' /
     sin theta, expansion rate m theta-dot cot theta, all of it from the normal-motion term (checks sign, radius,
     trace convention and mass balance together). Shear from an explicit map with known induced metric.
  5. Flow along a fixed curved support, and a mixed tangential / normal flow where the two strain terms partly
     cancel.
  6. Smooth nonlinear one-to-one map; two sheets that merge; shuffled or noisy target (held-out prediction
     against a permutation distribution).
  7. Asymmetric patch against an orthant face, controlled thin dimensions, a cluster shrinking towards a vertex
     (drift and rank bias, coverage, rank loss reported as such).
- Variation: paired subsampling of images without replacement, operators rebuilt per subset.

## 4a. What the real cloud allowed, and the estimands released (frozen 2026-11-24)

Recorded after the matched fixture and before the grid run. Sources: `tests/matched.py`, `data/fixture_matched/`,
`mm/frame.py` (`Rules.flow`, `Rules.thick`), `mm/step.py` (`MODE_FIELDS`, `FLOW_GATES`).

- At sites stem..b8 the local second moment of the cloud has no eigengap (b7, relative to the largest
  eigenvalue: 1, .80, .67, .50, .33, .22, .14, .08, .05). At every scale 10,000 points resolve, the cloud is a
  9-dimensional anisotropic blob in S^9, not a thin submanifold. Every point sits at the edge of its own
  neighbourhood (one-sidedness about 0.85). The strict rules of Section 4 release tangent-dependent fields on
  under 1% of points. That is the result for the strict reading: no submanifold, so no normal space, and the
  intrinsic / extrinsic split of Section 3 item 2 is not defined at these sites.
- Flow reading (`Rules.flow`): the cloud is treated as full-dimensional (all nine directions resolved, or the
  point is withheld). Normal quantities are not applicable. Every regression quantity is the gain of a
  graph-weighted linear predictor of neighbour offsets at the neighbourhood radius, not a pointwise derivative.
- A fixture matched to b7 (same spectrum, radius, one-sidedness, truncation; known map; scatter at four levels;
  shuffled, rank-deficient, collapsing, folded and curved variants; two seeds) decided the following.
  - A fit forced through the anchor absorbs any offset shared by an anchor's neighbours (6.2% error against 1.0%
    for a free intercept under endpoint scatter). Released regression fields use the free-intercept fit.
  - The held-out score (improvement over predicting zero offset) is optimistic and is reported with two
    baselines: no deformation (parallel transport) and a constant offset.
  - Not recoverable, and not released in any form: the smallest log-stretches, Hencky measures, log J, the
    volumetric / shear split, and everything from the anchored fit. A rank-8 map reads as full rank.
  - Released everywhere: travel; rigid fit; relative motion with its exact split into shared shift + slope +
    residual (free-intercept fit); the scores and baselines; a fit-free neighbour log-distance change.
  - Released where gates pass (held-out >= 0.5; held-out minus constant-offset score >= 0.5; centred-design
    condition number <= 50; effective neighbours >= 2(m+1); for the largest gain also 0.3 above the
    no-deformation score): largest log gain (tolerance 0.1), expansion rate tr sym K (0.2 of median |E|),
    |sym K|^2 and |skew K|^2 (20% of cloud median). The 0.3 margin is waived when the no-deformation score
    is itself at least 0.95 (a cloud that is carried without deforming; no released value on the real grid
    used this). False releases among released points, on the deforming-map variants: 0-10% under scatter up
    to level 1.0, up to 21% at level 1.5 (on at most 24% of points), 5-29% on a strongly curved map (the gap
    between a slope at radius r and a derivative). On the control where nothing deforms but endpoints scatter,
    the largest gain is released on 0.4% of points and almost all of those exceed the tolerance: there the gate
    protects by coverage, not by accuracy.
- Thick reading (`Rules.thick`), secondary: "displacement projected onto the leading eigenspace of the source
  second moment at radius r" and "change in orientation of the transported leading eigenspace", only where the
  retained rank is stable under nearby cutoffs and for the centred covariance.
- Grid: k = 256, adaptive operator (eps 0.25), 40 checkpoints, depth and training clocks; scale, subsample and
  seed-1 checks at the final checkpoint. Outputs in `data/fields/`.
- After the first run an independent code review found five release-mask and labelling mismatches in the
  export (anchored-fit quantities in the public flow totals; fit quantities published on rows with fewer than
  nine resolved directions; tilt gated by a regression rule; a mislabelled intercept field; speed-only
  placeholder files beside measured checkpoints). All were fixed and every export regenerated; the files under
  `data/fields/` and on the site are the regenerated ones. Flow fit quantities are released only on rows with
  nine resolved directions (rule `relmotion_fitsplit`); anchored-fit quantities exist only under `diag:anchored:`.

## 5. Data and stages

- Points: the 10,000 CIFAR-10 test images (the reference population, which sets the density of the weighted
  manifold). Training images are a later follow-up.
- Stored per image, site and checkpoint: 10 log probabilities.
- Stage 1: one checkpoint end to end. Lens extraction with three checks (last lens = ordinary inference; every
  lens = explicit forward pass with later branches zeroed; the increment identity across both stage changes).
- Stage 2: fixtures; items 1-11 and 13 at the final checkpoint; viewer. Gates: first-order
  results after fixtures 1-4; the two-term strain attribution (item 8) after fixture 5; predictability and
  mass balance (item 9) after fixture 6; rank and normal-motion claims near vertices after fixture 7.
- Stage 3: all 73 checkpoints, both clocks; seed 1 for headline curves.
- Stage 4: second-order quantities (item 12) and deferred spectra, where fixtures show they are estimable.

## 6. Viewer

One fixed projection for all frames (three chosen classes plus "rest" as a tetrahedron, switchable; or three
principal directions of all frames pooled). Depth and training sliders with playback, trails. Colour by any
field of Section 3. Displacement arrows split into tangential and normal parts; strain shown as principal
stretch axes at selected points. Click a point for its image and the same neighbourhood carried to the next
frame. Curves of item 13 alongside. Cut down from the CoherentModes explorer.

## 7. Owner decisions (2026-11-23)

- Go ahead and measure.
- Reference population: the 10,000 test images only. Training images, to compare their motion, are a later
  follow-up, not part of this run.
- Site: public GitHub repo with a Pages site.

## 8. Code to reuse

ModelZoo `zoo.models`, `zoo.io`, `zoo.data`; CoherentModes `explore/cifar_coherent` (`measure.py` extraction,
Modal coordination, explorer) but not its operator; Jones's diffusion-geometry code (carré du champ, tangent
spaces, Hessians, curvature; github.com/Iolo-Jones/ManifoldDiffusionGeometry), to be validated on the fixtures;
datafold (diffusion maps with alpha normalisation and variable bandwidth); geomstats (sphere log map).

## 9. History

Drafts 1-2: raw Euclidean geometry against the pullback of the output Fisher (dropped by the owner). Drafts
3-4: lens Fisher space with a broad menu including transport, graph-path and coarse-curvature items (dropped
by the owner in favour of the kinematic quantities here). GPT-6.2 Astra reviewed drafts 1-4; points carried
over: the affine lens and its increment identity, the stable distance and face formulas, masking of undefined
directions, fixtures before interpretation, subsampling without replacement, and the caution that the pairing
of points across frames is a coupling rather than a map (here measured as the unresolved term R).
