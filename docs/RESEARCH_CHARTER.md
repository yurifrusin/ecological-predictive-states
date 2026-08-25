# Research Charter

## 1. Working proposition

The world available to an embodied perceiver need not be represented first as a complete observer-independent Cartesian reconstruction. A useful predictive state may instead encode lawful relations among:

- persistent surfaces;
- optical regions and boundaries;
- adjacency, nesting, and occlusion order;
- accretion and deletion at occluding boundaries;
- optical expansion, contraction, shear, and flow;
- actions or changes in point of observation.

The first project tests whether this organisation is computationally useful. It does not assume the result.

## 2. Ontological discipline

### 2.1 Cartesian space is instrumentation, not prohibited knowledge

The simulator may use coordinates, meshes, camera matrices, and depth buffers. These are measurement instruments and sources of controlled ground truth. They are not automatically the state presented to, predicted by, or optimised by the ecological model.

### 2.2 Pixels are sensory samples, not the world target

RGB frames may be inputs. The project rejects the assumption that reproducing every future pixel is the privileged criterion of a good world model.

### 2.3 Surfaces precede semantic objects in v0

The first state contains surface regions, boundaries, persistence relations, and visibility events. It does not require labels such as `chair`, `wall`, or `cup`.

### 2.4 Affordances are deferred, not denied

The optical substrate is tested before body-scaled possibilities for action are added. Later work may condition the same ecological state on body geometry and action repertoire.

### 2.5 Local metric computation remains available

If a downstream task genuinely requires metric precision, a local metric module may be invoked. The research question is whether universal persistent metric reconstruction is necessary, not whether measurement is ever useful.

## 3. Scientific rather than doctrinal use of Gibson

The project is Gibson-inspired, not an attempt to settle textual disputes in ecological psychology through software. A learned internal predictive state may be philosophically unacceptable to a strict theory of direct perception. We therefore distinguish:

1. claims about human or animal perception;
2. engineering hypotheses inspired by ecological optics;
3. empirical performance of a particular architecture.

A successful model would show that ecological optical variables are useful computational inductive biases. It would not by itself prove that biological perception contains the same representation.

## 4. Primary falsifiable hypothesis

Under matched data and model budgets, an action-conditioned ecological predictive state will show a smaller loss of task performance under appearance changes than future-RGB, generic visual-latent, or universal metric-state baselines.

Formally, for model family \(m\), let \(R_m^{ID}\) and \(R_m^{OOD}\) be in-distribution and appearance-out-of-distribution task performance. Define robustness loss:

\[
\Delta_m = R_m^{ID} - R_m^{OOD}.
\]

The primary directional hypothesis is:

\[
\Delta_{eco} < \Delta_{pixel},\; \Delta_{latent},\; \Delta_{metric}
\]

without materially inferior in-distribution task success.

## 5. Secondary hypotheses

1. The ecological model reaches a fixed OOD success level with fewer trajectories.
2. It preserves surface identity and visibility relations through occlusion better than generic latent prediction.
3. It transfers across texture, colour, illumination, and moderate camera changes while retaining layout-sensitive distinctions.
4. Metric augmentation improves only those tasks that demand precise measurement, rather than all tasks uniformly.
5. Action-labelled exploration is more data-efficient than passive video for learning the relevant state transitions.

## 6. Negative results that count as progress

The project should stop or change direction if any of the following is robustly observed:

- the oracle ecological state is insufficient for the proposed prediction and planning tasks;
- its advantage disappears when baselines are matched fairly;
- it only works because simulator surface identifiers leak hidden geometry;
- learned extraction from RGB destroys the oracle-state advantage;
- gains are explained solely by output dimensionality or easier supervision;
- metric state consistently dominates at equal compute and data budgets, including appearance-OOD tests.

## 7. Forbidden shortcuts

- Do not feed world coordinates, camera pose, depth, object pose, or simulator IDs to the ecological learner unless an ablation explicitly allows them.
- Do not evaluate only on the same textures and lighting used for training.
- Do not call a representation “Gibsonian” merely because it includes optical flow or the word affordance.
- Do not use unequal model capacity or substantially different training data without reporting it.
- Do not optimise benchmark definitions after seeing final test results.
- Do not make claims about biological perception from simulated engineering performance alone.

## 8. Terminology

**Ecological predictive state (EPS):** a compact, action-conditioned state sufficient to predict task-relevant transformations in the observer–environment optical relation.

**Ecological graph:** a relational representation whose nodes correspond to temporally persistent surface hypotheses and whose edges encode adjacency, containment, occlusion, and continuity.

**Visibility event:** appearance, disappearance, accretion, deletion, split, merge, or stable persistence of an optical/surface region under movement.

**Metric instrument:** coordinate, depth, pose, occupancy, or geometric calculation used for controlled generation, evaluation, or a local precision task.
