# Milestone 0 — Ontology Before Scale

## Objective

Produce a reproducible, low-resource benchmark that can falsify the claim that an ecological predictive state offers a useful inductive bias for visual prediction under appearance change.

Milestone 0 is infrastructure and scientific hygiene. It does not include a physical robot or a publication claim.

## Gate 0A — Repository foundation

Deliverables:

- Python project using `uv`;
- Windows-compatible setup instructions;
- lint, type-check, and test commands;
- deterministic random seed utility;
- configuration hierarchy;
- CI-ready test suite;
- experiment metadata and version capture.

Exit criteria:

- clean install in a fresh environment;
- all tests pass;
- one command generates a tiny synthetic dataset;
- one command validates that dataset.

## Gate 0B — Simulator and data contract

Deliverables:

- procedurally generated single-occluder and corridor scenes;
- kinematic camera agent;
- RGB, segmentation, depth, and camera instrumentation;
- exact action log;
- derived boundary, flow, visibility, and occlusion annotations;
- versioned dataset manifest;
- visual inspection utility.

Exit criteria:

- deterministic regeneration from seed;
- re-rendering with new textures preserves ecological labels;
- segmentation and depth align pixel-for-pixel;
- visibility-event labels pass analytic unit tests;
- metric modalities are permission-gated by loader.

## Gate 0C — Oracle ecological state

Deliverables:

- conversion from simulator annotations to ecological graph/state;
- persistent-surface bookkeeping across transitions;
- graph visualiser;
- state serialisation and schema validation;
- trivial oracle and majority baselines.

Exit criteria:

- surface persistence is correct across controlled occlusion cases;
- graph transitions reconstruct all benchmark visibility labels;
- no world coordinates or depth appear in ecological tensors;
- simulator IDs are randomised and not used as model features.

## Gate 0D — Matched baseline interfaces

Deliverables:

- common `WorldModel` protocol;
- common trainer/evaluator interface;
- pixel, generic-latent, metric, and ecological model stubs;
- parameter-budget report;
- shared logging and checkpoint format.

Exit criteria:

- every model consumes only authorised modalities;
- all models can overfit a tiny batch;
- rollouts share identical action sequences and evaluation tasks;
- parameter counts and training examples are reported automatically.

## Gate 0E — First comparative experiment

Deliverables:

- Stage A1 and A2 experiments on at least two scene families;
- ID and appearance-OOD results across pre-registered seeds;
- learning curves and robustness-loss table;
- ablation without occlusion edges;
- failure-case gallery.

Decision rule:

Proceed to learned ecological extraction only if the oracle ecological state is sufficient and exhibits a credible robustness or data-efficiency advantage. If it does not, revise the state definition before adding scale or hardware.

## Gate 0F — Learned extractor feasibility

Deliverables:

- small RGB-to-ecological-state extractor;
- teacher-forced and end-to-end variants;
- comparison with oracle-state ceiling;
- texture-randomisation training ablation.

Exit criteria:

- enough of the oracle advantage survives extraction to justify Stage B active disclosure;
- errors are localised to measurable components rather than hidden in a single opaque score.

## Explicit non-goals

- large video foundation-model pretraining;
- photorealism;
- semantic object recognition;
- natural-language control;
- manipulation;
- real-time deployment;
- claims about biological neural implementation;
- purchasing robotic hardware.
