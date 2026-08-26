# ecological-predictive-states

Ecological Predictive States is a research repository for testing whether an action-conditioned state built around persistent surfaces, boundaries, occlusion, visibility change, and optical transformation is a useful inductive bias under appearance change. The hypothesis is falsifiable and is not assumed to be true.

## Current status

Gate 0A (repository foundation), the reviewed single-occluder Gate 0B slice, and a separately reviewable corridor apparatus slice are implemented on the current development branch. The corridor slice adds a deterministic four-surface scene, prescribed forward monocular-camera motion, scene-specific privileged instrumentation, and scene-aware generation, validation, inspection, and corruption tests while preserving the single-occluder contract.

**No scientific result exists yet.** Successful generation and tests establish infrastructure only. The corridor slice is pending independent review. Full Gate 0B completion is not claimed: dense optical flow and full oriented boundary ownership remain incomplete. Gate 0C and all model work are not authorised.

The scientific authority is [the Research Charter](docs/RESEARCH_CHARTER.md), followed by [EPS-Bench v0](docs/EPS_BENCH_V0.md) and [Milestone 0](docs/MILESTONE_0.md). Architecture and current limits are recorded in [the implementation notes](docs/IMPLEMENTATION_NOTES.md); implementation, review, ownership, and closeout are separated by [the review protocol](docs/review-protocol.md).

## Windows PowerShell setup

Install `uv` if it is not already available, then let `uv` install the pinned Python 3.11 runtime and the exact locked dependency set:

```powershell
winget install --id=astral-sh.uv -e
git clone https://github.com/yurifrusin/ecological-predictive-states.git
Set-Location ecological-predictive-states
uv sync --locked
```

No GPU, cloud service, API key, model weight, or model download is required.

## Generate, validate, and inspect

```powershell
uv run epsbench generate `
  --config configs/benchmark_v0.yaml `
  --episodes 2 `
  --output data/smoke

uv run epsbench validate data/smoke

uv run epsbench inspect `
  data/smoke `
  --episode 0 `
  --output artifacts/episode_0.png

uv run epsbench generate `
  --config configs/corridor_v0.yaml `
  --episodes 2 `
  --output data/corridor-smoke

uv run epsbench validate data/corridor-smoke

uv run epsbench inspect `
  data/corridor-smoke `
  --episode 0 `
  --output artifacts/corridor-episode-0.png
```

Generation refuses to overwrite a non-empty output directory. Inspection output must remain outside the dataset so it cannot change dataset identity.

## Quality checks

```powershell
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy src tests
git diff --check
```

Use `uv run ruff format .` to apply formatting intentionally.

## Modality boundaries

| Class | Current contents |
| --- | --- |
| Sensory | RGB and executed action |
| Ecological oracle | Opaque surface regions, boundary contacts, visibility fractions/events, correspondence, and occlusion relation |
| Metric baseline | Depth |
| Instrumentation only | Camera world transforms, raw MuJoCo geom IDs, raw coordinates, sampled corridor geometry, and generation records |
| Control metadata | Compound transition and scene-family records; never treated as learner inputs |

`DatasetLoader` requires an explicit `ModalityPermissionSet`. Ecological-only access fails before depth, camera pose, raw IDs, or world-coordinate artifacts are opened.

## Repository map

```text
configs/             Versioned benchmark configuration
docs/                Authoritative research documents and implementation notes
src/epsbench/        Typed schemas, simulator, annotations, data tools, and CLI
tests/               Unit, integration, and regression contract tests
data/                Ignored generated datasets (policy file tracked)
artifacts/           Ignored inspection output (policy file tracked)
.github/workflows/   Locked CPU/offscreen continuous integration
```

The package intentionally contains no neural model, LLM, planner, semantic object recogniser, robot integration, or external research-code import.
