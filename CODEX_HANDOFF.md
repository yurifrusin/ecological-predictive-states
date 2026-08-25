# Codex Handoff — Milestone 0A and 0B

You are implementing the foundation of a research repository called **Ecological Predictive States**. Read `README.md`, `AGENTS.md`, `docs/RESEARCH_CHARTER.md`, `docs/EPS_BENCH_V0.md`, and `docs/MILESTONE_0.md` in full before changing code.

## Goal

Complete only Gates 0A and the smallest vertical slice of 0B:

1. establish a clean, Windows-compatible Python/`uv` project;
2. implement deterministic configuration and experiment metadata;
3. implement one procedurally generated **single-occluder** MuJoCo scene;
4. move a kinematic monocular camera using a discrete action;
5. render RGB, depth, and segmentation for the before/after frames;
6. derive and serialise a minimal transition record;
7. implement validation and visual-inspection commands;
8. add strong tests for determinism, modality permissions, and alignment.

Do not implement machine-learning models yet.

## Scientific constraints

- MuJoCo coordinates, depth, and camera transforms are instrumentation.
- Do not place metric arrays inside the ecological-state fields.
- Surface/geom IDs must be remapped randomly per episode before dataset serialisation.
- A second render of the same geometry and trajectory with changed textures must preserve the ecological-label hash while changing the RGB hash.
- The data loader must require an explicit modality permission set and reject unauthorised access.
- Keep resolution and scene complexity small.

## Required package shape

Create or complete:

```text
src/epsbench/
  annotations/
  cli/
  config/
  data/
  evaluation/
  sim/
  utils/
```

Retain `src/epsbench/schema.py` as the public schema module, but revise it where justified.

## Required CLI commands

Provide commands equivalent to:

```powershell
epsbench generate --config configs/benchmark_v0.yaml --episodes 2 --output data/smoke
epsbench validate data/smoke
epsbench inspect data/smoke --episode 0 --output artifacts/episode_0.png
```

The exact CLI library is your choice, but keep it lightweight and typed.

## Required tests

At minimum:

1. same seed produces byte-identical manifest and annotation files;
2. RGB, depth, and segmentation dimensions align;
3. texture randomisation changes RGB hash but not ecological-label hash;
4. ecological-only loader cannot access depth or camera pose;
5. surface IDs differ between two serialised episodes even when geometry is identical;
6. all transition records round-trip through serialisation;
7. invalid visibility fractions, duplicate IDs, and malformed occlusion edges fail validation;
8. generated smoke dataset validates successfully.

## Deliverables

- implementation;
- updated `pyproject.toml` and console entry point;
- tests;
- concise architecture note in `docs/IMPLEMENTATION_NOTES.md`;
- reproducible PowerShell commands in README;
- final report stating changed files, commands run, test results, known limitations, and next recommended gate.

## Boundaries

Do not:

- implement RGB, latent, metric, or ecological neural models;
- import DINO-WM or Sensorimotor World Model code;
- add ROS, Isaac, Habitat, iGibson, or Blender;
- add semantic object labels;
- add cloud services;
- purchase or assume a robot;
- claim scientific success from the data generator.

Make sensible implementation decisions without asking routine questions. Stop only if a genuine blocker prevents a correct implementation, and document the blocker precisely.
