# EPS-Bench v0 Specification

## 1. Benchmark question

Can a compact ecological state predict surface persistence and visibility transformations under observer movement more robustly than pixel, generic-latent, and metric baselines?

Version 0 is intentionally not a general robotics benchmark. It isolates one disputed representational choice.

## 2. Environment families

All scenes are procedurally generated from simple opaque surfaces. The camera/agent moves on a horizontal support plane.

1. **Single occluder** — a foreground panel partially hides one or more background surfaces.
2. **Corridor** — textured side surfaces, floor, end surface, and optional openings.
3. **Doorway/opening** — a bounded aperture reveals a second enclosure.
4. **T-junction** — forward movement changes which branch surfaces are disclosed.
5. **Nested enclosure** — openings within openings and multiple occlusion orders.
6. **Moving panel** — deferred until static self-motion experiments pass.

## 3. Action space

The v0 action is a small egomotion command:

\[
u_t=(\Delta f,\Delta s,\Delta \theta)
\]

where \(\Delta f\) is forward displacement, \(\Delta s\) lateral displacement, and \(\Delta\theta\) yaw rotation. Initial experiments may discretise each component.

## 4. Recorded modalities

Every transition records:

- RGB frame before and after the action;
- action command;
- simulator surface segmentation;
- depth image for metric baseline and evaluation only;
- camera transform for instrumentation only;
- analytic or geometry-derived optical flow;
- boundary map and boundary ownership;
- surface visibility fractions;
- occlusion-order graph;
- region correspondence across the transition;
- visibility events: stable, accretion, deletion, appearance, disappearance, split, merge.

The dataset loader must enforce modality permissions per model family.

## 5. Ecological state v0

For time \(t\), define:

\[
G_t=(V_t,E_t,B_t,F_t,Q_t)
\]

where:

- \(V_t\): visible and remembered surface hypotheses;
- \(E_t\): adjacency, containment, and occlusion edges;
- \(B_t\): oriented boundary structure and ownership;
- \(F_t\): region-level or dense optical transformation statistics;
- \(Q_t\): confidence and visibility state.

### Surface-node attributes

- temporary/persistent experimental identifier;
- visible fraction;
- normalised image centroid;
- angular or image-plane extent;
- boundary length and boundary-type counts;
- mean and covariance of optical flow;
- currently visible / remembered / newly revealed / being deleted;
- confidence.

Simulator identifiers are permitted only to generate and evaluate oracle correspondences. They must be remapped or hidden from learned models.

### Edge types

- adjacent;
- contains / inside;
- occludes / occluded-by;
- temporally-continues;
- shares-boundary.

## 6. Tasks

### Stage A: prescribed-motion optical prediction

**A1 — Visibility-event prediction**  
Given \(G_t\) and \(u_t\), predict which surface regions undergo accretion, deletion, appearance, or disappearance.

**A2 — Surface persistence**  
Maintain identity of a surface through partial and complete occlusion and predict whether it will be revealed by a subsequent movement.

**A3 — Occlusion-order prediction**  
Predict changes in front–back relations and boundary ownership.

**A4 — Ecological next-state prediction**  
Predict \(G_{t+1}\) rather than the next RGB image.

### Stage B: active disclosure

**B1 — Reveal**  
Choose a short action sequence that maximises newly revealed target surface area.

**B2 — Re-identify place**  
Recognise an equivalent surface-layout transition under novel appearance.

**B3 — Navigate to disclosed opening**  
Reach an opening using ecological state planning. This is introduced only after Stage A passes.

### Stage C: later affordance extension

Vary camera height, body width, turning radius, and speed. Test body-conditioned passability and reachability. This is outside v0.

## 7. Model families

### P — Pixel predictor

A compact ConvLSTM or video autoencoder predicts the future RGB frame. Planning uses predicted images or a learned image-space task head.

### L — Generic latent predictor

A frozen compact visual encoder produces patch features; an action-conditioned transformer predicts future features. This is structurally comparable to DINO-WM but kept small enough for local experiments.

### M — Metric predictor

A model predicts depth plus egomotion or a local occupancy representation. It may use metric labels during training but receives the same RGB/action histories at test time.

### E — Ecological predictor

A graph network or set transformer predicts ecological graph/state transitions. In the first experiment it receives oracle ecological state. A later model learns the state extractor from RGB.

### E+M — Optional local metric augmentation

The ecological state can request a local metric estimate for an explicitly precision-sensitive task. It must not receive a full persistent metric scene by default.

## 8. Dataset splits

1. **ID:** familiar geometry and appearance distributions.
2. **Appearance OOD:** unseen textures, colours, illumination, and texture frequencies.
3. **Geometry OOD:** unseen corridor widths, occluder sizes, and opening ratios.
4. **Camera OOD:** moderate changes in field of view and camera height.
5. **Dynamics OOD:** moving surfaces, added only after static scenes pass.

The final test seeds and appearance assets must be frozen before comparative training.

## 9. Metrics

### Optical/ecological prediction

- boundary F1;
- visibility-event macro F1;
- occlusion-edge precision/recall/F1;
- surface correspondence IDF1;
- visible-fraction mean absolute error;
- rollout consistency across multiple steps.

### Active tasks

- success rate;
- collision rate;
- newly revealed target fraction;
- actions to success;
- regret against an oracle planner.

### Robustness and efficiency

- absolute and relative ID-to-OOD performance drop;
- area under the learning curve versus number of trajectories;
- parameter count, training FLOPs estimate, inference latency, and state size;
- performance under texture replacement while geometry remains fixed.

## 10. Primary comparison

The primary endpoint is the difference in appearance-OOD robustness loss between E and each baseline, aggregated over pre-registered tasks and random seeds. Report bootstrap confidence intervals and all individual seeds.

A persuasive result requires:

- no material ID collapse;
- lower appearance-OOD degradation;
- matched or transparently reported model/data budgets;
- no access by E to hidden metric variables;
- replication across more than one scene family.

## 11. Critical ablations

1. Remove optical flow.
2. Remove explicit occlusion edges.
3. Remove temporal surface correspondence.
4. Replace ecological graph with an equally sized unstructured latent vector.
5. Give the latent baseline the same oracle surface tokens but no ecological edge types.
6. Add local metric information to E only at decision time.
7. Train with passive trajectories versus action-balanced exploratory trajectories.

## 12. Leakage tests

- Randomise simulator surface identifiers per episode.
- Verify that ecological inputs cannot decode camera world coordinates above chance beyond what is visually specified.
- Ensure metric arrays are absent from ecological batches.
- Train a probe to detect texture identity; ecological representations should not require high texture-identification accuracy.
- Re-render identical trajectories with different appearance and verify target ecological labels are invariant.
