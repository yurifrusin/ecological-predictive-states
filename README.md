# ecological-predictive-states

Ecological Predictive States is a research repository for testing whether an action-conditioned state built around persistent surfaces, boundaries, occlusion, visibility change, and optical transformation is a useful inductive bias under appearance change. The hypothesis is falsifiable and is not assumed to be true.

## Current status

Gate 0A and Gate 0B single-occluder Slice 1, corridor Slice 2, analytic-transport Slice 3, and oriented-boundary and visibility-event Slice 4 remain canonical. Slice 4's exact reviewed implementation head was `24deb074d4ba0cf3e1044e466ecbbb6e0b2a4cc4`; its final methods and bounded limits are documented in [the Slice 4 contract](docs/GATE_0B_SLICE_4_BOUNDARY_EVENTS.md).

Gate 0B procedural appearance-candidate Slice 5 is also canonical after exact-head engineering and scientific review. Its exact reviewed implementation head was `2ea05867e7c8cb04ee4f9d0138394602b6c3e53a`, which received `ENGINEERING_PASS` and `SCIENTIFIC_PASS`. Slice 5 provides a reproducible appearance-candidate audit apparatus and preserved negative evidence: all 160 cells succeeded, 44 cells were admitted, 116 rejected cells were retained, all ten profiles were rejected at profile level, and the admitted-profile set is empty. The strict registry, protected dataset-bound candidate seeds, portable-versus-renderer-local roots, complete audit command, and admission packet are documented in [the Slice 5 candidate contract](docs/GATE_0B_SLICE_5_APPEARANCE_CANDIDATES.md).

Appearance Candidate Revision 1 is canonical after exact-head dual review and owner-approved
closeout. Its exact reviewed implementation head was
`da37a729bc4af00ea83c9460c3849307171bba69`, and its immutable prospective definition-lock commit
was `914550ce4e3a819dcbcd0bd5390e3c6034af5bf6`. The canonical Slice 5 baseline remains 160
successful cells, 44 admitted cells, and 116 retained rejected cells, with all ten baseline profiles
rejected and an empty admitted-profile set. Revision 1 admits exactly
`revision1_balanced_reference_v1`, `revision1_colour_shift_v1`,
`revision1_checker_low_v1`, `revision1_checker_high_v1`, and
`revision1_combined_stress_v1` across both locked renderers. `revision1_stripes_low_v1` and
`revision1_illumination_shift_v1` remain rejected, and the WGL/OSMesa design-cell differences remain
preserved as renderer-local negative evidence. This admitted set is candidate evidence only: no
profile has been assigned or frozen into a final development/OOD role, and no final evaluation seed
has been frozen. The protocol and locked definitions are documented in
[the Revision 1 contract](docs/GATE_0B_APPEARANCE_CANDIDATE_REVISION_1.md).

PR #17 progressed through historical exact head
`62e09a920c10d50c62643543f6ebcbd26570b1d4`, second-correction exact head
`dc17d8453bed2583c085aa7441255ca476d3e1f0`, and third-correction exact head
`53b988372634e8a7d2d006021af5b5aadbf96d3c`. Those exact-head dispositions remain historical and
do not apply forward. Final fourth correction head
`4d47595ede0f1690f1d9d96075df9a3ff7ab60d9` and tree
`ba112bcc4be4396648305bb1d1529cc002755ea1` received independent `SCIENTIFIC_PASS` and
`ENGINEERING_PASS`; `EPS-SR17-0001` and `EPS-ER17-0001` through `EPS-ER17-0008` were recorded as
verified by their respective reviewers. Yuri Frusin acting as `OWNER_PI` issued exact-head approval,
and PR #17 was merged by normal merge commit `4851bfc195a977f1bf9f29005b2579341dc75c38`.

The owner-authorised linked closeout makes Appearance Benchmark Input Freeze v0 canonical when its
commit is present on `main` and final canonical-main readback succeeds. The unchanged authoritative
replacement-lock commit is `d4072f912cc58bbc1ca41ceb2652e41783dbf3e1`, with root
`28acd2c340b3ef1e2b8b2b7b31ca69356cd9baa35b94ff8a598e020099883435`. It freezes five selected
appearance profiles, retains two excluded profiles, and binds exactly sixteen final evaluation
roots across the two scene families. Both locked renderer apparatus qualifications are complete.
Evidence class `PUBLIC_REPOSITORY_ONLY` permits repository/CI evidence through public or connected
GitHub records, not unpublished off-platform material. Repository visibility is private; current
artifact access requires connected authenticated GitHub Actions access.
Final receipts and credential-free metadata are preserved in `docs/reviews/`; the four raw ZIPs are
recorded by exact identity and remain Actions-retained rather than committed. The scope and
limitations are documented in [the freeze contract](docs/GATE_0B_APPEARANCE_BENCHMARK_FREEZE_V0.md)
and [the closeout record](docs/reviews/pr-0017-closeout.md).

Historically, before exact-head CI, renewed review, owner issuance, merge, and linked closeout, the
source-only table recorded `EXACT_HEAD_REQUALIFICATION_PENDING`, `EXTERNAL_TO_SOURCE_COMMIT`,
`FOURTH_CORRECTION_SOURCE_CANDIDATE_PENDING_INDEPENDENT_REVIEW`,
`RENEWED_EXACT_HEAD_DUAL_REVIEW_REQUIRED`,
`FOURTH_CORRECTION_AUTHORISED; IMPLEMENTATION_MERGE_APPROVAL_NOT_GIVEN`, `NOT_PERFORMED`,
`NOT_FROZEN`, `NOT_ADVANCED`, and `NONE`. Those values are retained as historical pre-closeout
evidence, not current authority.

### PR #17 canonical closeout status

| Boundary | Current source-record status |
| --- | --- |
| Apparatus status | `QUALIFICATION_COMPLETE_BOTH_LOCKED_RENDERERS` |
| Post-commit exact-head CI evidence | `SUCCESS_AT_APPROVED_HEAD` |
| Implementation status | `MERGED_BY_NORMAL_MERGE_COMMIT` |
| Review status | `EXACT_HEAD_DUAL_REVIEW_RECORDED` |
| Owner status | `EXACT_HEAD_APPROVED_FOR_MERGE_AND_BENCHMARK_FREEZE_CLOSEOUT` |
| Benchmark-freeze status | `FROZEN_BY_LINKED_CLOSEOUT_ON_CANONICAL_MAIN` |
| Model-protocol status | `NOT_FROZEN` |
| Gate status | `NOT_ADVANCED` |
| Scientific result | `NONE` |

**No scientific result exists yet.** This closes and freezes only the authorised appearance
benchmark-input scope. The model protocol is not frozen; the primary model-result renderer, the
other-renderer classification, and any renderer aggregation rule remain unselected; comparative
model-result access remains unauthorised. Full Gate 0B remains incomplete, Gate 0C and Gate 0D are
not authorised, no model or empirical-gate work was performed, no tag or Release was created, and
the next work package has not begun.

The scientific authority is [the Research Charter](docs/RESEARCH_CHARTER.md), followed by [EPS-Bench v0](docs/EPS_BENCH_V0.md) and [Milestone 0](docs/MILESTONE_0.md). Architecture and current limits are recorded in [the implementation notes](docs/IMPLEMENTATION_NOTES.md). Review workflow v2 is canonical after PR #7 and its linked closeout; the exact reviewed implementation head was `741e31882cb484fa5858630be83d2f08e483585f`, which received `ENGINEERING_PASS` and `SCIENTIFIC_PASS`. Active review records remain external to implementation PRs, and final immutable records enter through a separately authorised linked closeout. `MERGE_CONVERGENCE` and the terminal `RECORD_ONLY_CLOSEOUT` path remain distinct. The orchestrator coordinates state without inheriting review authority. This governance closeout has no Gate 0B or Gate 0C effect, implements no machine-readable review automation or next slice, and establishes no scientific result. See [the review protocol](docs/review-protocol.md) and [review-record directory policy](docs/reviews/README.md).

## Design explorations

[Design Note 001 — Percept, Affect, Concept](docs/design-notes/001_DELEUZE_GUATTARI_EPS_DESIGN_NOTE.md)
is canonical after exact-head `SCIENTIFIC_PASS` at
`4e37f9cc25a33927135077a76cf72eae9d191392` and owner-approved closeout, solely as a
non-authoritative design exploration. This status does not make its hypotheses scientifically
supported, implemented, or authorised. Design notes do not define scientific-contract, benchmark,
or gate authority.

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
