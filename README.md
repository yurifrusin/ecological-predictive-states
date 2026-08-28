# ecological-predictive-states

Ecological Predictive States is a research repository for testing whether an action-conditioned state built around persistent surfaces, boundaries, occlusion, visibility change, and optical transformation is a useful inductive bias under appearance change. The hypothesis is falsifiable and is not assumed to be true.

## Current status

Gate 0A is canonical. Gate 0B single-occluder Slice 1, corridor Slice 2, and analytic-transport Slice 3 are canonical. Gate 0B oriented-boundary and visibility-event Slice 4 is also canonical after exact-head engineering and scientific review. Its exact reviewed implementation head was `24deb074d4ba0cf3e1044e466ecbbb6e0b2a4cc4`, which received `ENGINEERING_PASS` and `SCIENTIFIC_PASS`. Its final boundary method is `analytic_oriented_boundary_ownership_v4`, its final projected attachment method is `projected_compiled_contact_locus_v3`, and its final visibility-event method is `analytic_transport_boundary_causal_events_v2`. The method, review evidence, and bounded limits are documented in [the Slice 4 contract](docs/GATE_0B_SLICE_4_BOUNDARY_EVENTS.md).

Gate 0B Slice 5 procedural appearance candidates and a bounded scientific-review correction are
implemented on the current branch pending verification, engineering review, scientific re-review,
and owner approval. The strict ten-profile registry, protected dataset-bound eight-seed candidate
registry, private appearance identities, versioned portable-versus-renderer-local roots, complete
audit command, and admission packet are described in
[the Slice 5 candidate contract](docs/GATE_0B_SLICE_5_APPEARANCE_CANDIDATES.md). This is candidate
apparatus, not a benchmark freeze; the current admitted-profile set is empty.

**No scientific result exists yet.** Successful generation, validation, tests, and Slice 4 closeout establish infrastructure only. Full Gate 0B completion is not claimed: broader appearance and final-seed decisions remain open, and the remaining Gate 0B exit-criterion evidence has not received owner approval. Gate 0C and all model work remain unauthorised.

The scientific authority is [the Research Charter](docs/RESEARCH_CHARTER.md), followed by [EPS-Bench v0](docs/EPS_BENCH_V0.md) and [Milestone 0](docs/MILESTONE_0.md). Architecture and current limits are recorded in [the implementation notes](docs/IMPLEMENTATION_NOTES.md). Review workflow v2 is canonical after PR #7 and its linked closeout; the exact reviewed implementation head was `741e31882cb484fa5858630be83d2f08e483585f`, which received `ENGINEERING_PASS` and `SCIENTIFIC_PASS`. Active review records remain external to implementation PRs, and final immutable records enter through a separately authorised linked closeout. `MERGE_CONVERGENCE` and the terminal `RECORD_ONLY_CLOSEOUT` path remain distinct. The orchestrator coordinates state without inheriting review authority. This governance closeout has no Gate 0B or Gate 0C effect, implements no machine-readable review automation or next slice, and establishes no scientific result. See [the review protocol](docs/review-protocol.md) and [review-record directory policy](docs/reviews/README.md).

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

uv run epsbench appearance-audit `
  --registry configs/appearance_candidates_v0.yaml `
  --seeds configs/evaluation_seed_candidates_v0.yaml `
  --single-config configs/benchmark_v0.yaml `
  --corridor-config configs/corridor_v0.yaml `
  --output artifacts/gate-0b-slice-5-candidates
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
| Ecological oracle | Opaque surface regions, boundary contacts, projected-image fractions, correspondence, neutral mask changes, typed analytic optical transport, sparse oriented boundary ownership, bounded ecological visibility-event maps/summaries, and typed occlusion availability/relations |
| Metric baseline | Depth |
| Instrumentation only | Camera world transforms, raw MuJoCo geom IDs, raw coordinates, sampled corridor geometry, generation records, and permission-gated appearance profile/slot/texture/light and evaluation-seed-registry control evidence |
| Control metadata | Compound transition and scene-family records; never treated as learner inputs |

`DatasetLoader` requires an explicit `ModalityPermissionSet`. Ecological-only access may read the complete image-plane analytic transport, oriented-boundary, and visibility-event bundles; access without the relevant modality fails before its transition or artifacts are opened. Ecological-only access still fails before depth, camera pose, raw IDs, world-coordinate geometry, attachment evidence, counterfactual ray evidence, appearance instrumentation, or the dataset-bound evaluation-seed-registry snapshot is opened.

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
