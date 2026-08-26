# PR #1 correction response

This is the `CORRECTION_IMPLEMENTER` response to the scientific review of exact head `d4c3c4cc0ec9422d4e224ba430341b65b4de641e`. It records implementation evidence, not independent verification. All responses require renewed engineering and scientific review of the eventual correction head.

## Response identity

- Repository: `yurifrusin/ecological-predictive-states`
- Pull request: `#1 — Establish EPS Milestone 0 foundation`
- Original reviewed head: `d4c3c4cc0ec9422d4e224ba430341b65b4de641e`
- Correction role: `CORRECTION_IMPLEMENTER`
- Correction head: `PENDING CORRECTION COMMIT`
- Engineering review: `PENDING`
- Scientific re-review: `PENDING`
- Owner approval: `PENDING`
- Merge, tag, release, or gate advancement: `NOT AUTHORISED`
- Scientific result: `NONE`

## Finding responses

### `EPS-SR-0001` — Exact action–camera correspondence

- Accepted interpretation: unsupported or non-finite actions must fail, and persisted camera evidence must independently agree with the configured action.
- Files changed: `configs/benchmark_v0.yaml`, `src/epsbench/config/models.py`, `src/epsbench/schema.py`, `src/epsbench/data/validate.py`.
- Tests added: invalid forward/yaw/sign/zero/NaN/infinity configurations; non-finite and non-orthonormal camera schemas; hash-consistent persisted position and rotation tampering; positive persisted action/camera correspondence.
- Commands run: `uv run pytest tests/unit/test_config.py tests/unit/test_schema.py tests/integration/test_generation.py tests/integration/test_scientific_corrections.py` and the full validation suite below.
- Implementation status: `IMPLEMENTED_PENDING_VERIFICATION`.
- Unresolved limitations: forward and yaw simulation remain intentionally unsupported; tolerance and apparatus generalisation require review before later scene families.

### `EPS-SR-0002` — Derived and independently validated occlusion relation

- Accepted interpretation: frame membership and relation direction require controlled visual evidence rather than a fixed semantic assertion.
- Files changed: `src/epsbench/sim/single_occluder.py`, `src/epsbench/data/generate.py`, `src/epsbench/data/validate.py`, `src/epsbench/schema.py`, `docs/IMPLEMENTATION_NOTES.md`.
- Tests added: positive counterfactual evidence/frame derivation plus rejection of reversed, omitted, invented, wrong-frame, and unsupported-frame relations.
- Commands run: the targeted scientific-correction tests and full validation suite below.
- Implementation status: `IMPLEMENTED_PENDING_VERIFICATION`.
- Unresolved limitations: `counterfactual_occluder_exclusion_v1` is deliberately limited to the current single-occluder apparatus and is not general-scene reasoning or Gate 0C.

### `EPS-SR-0003` — Dataset-wide episode-local surface identifiers

- Accepted interpretation: opaque surface-ID sets must be pairwise disjoint across all episodes; numeric labels need only be local.
- Files changed: `src/epsbench/data/validate.py`, `docs/IMPLEMENTATION_NOTES.md`.
- Tests added: hash-consistent copying of one episode's opaque ID into another with rejection by whole-dataset validation.
- Commands run: the targeted scientific-correction tests and full validation suite below.
- Implementation status: `IMPLEMENTED_PENDING_VERIFICATION`.
- Unresolved limitations: numeric segmentation labels may intentionally repeat between episodes and remain scoped by episode.

### `EPS-SR-0004` — Exact source and governing-document provenance

- Accepted interpretation: source identity must represent Git availability and dirty state truthfully, bind exact governing hashes, and remain separate from content identity without recursion.
- Files changed: `src/epsbench/data/provenance.py`, `src/epsbench/data/identity.py`, `src/epsbench/data/generate.py`, `src/epsbench/data/validate.py`, `src/epsbench/schema.py`, `docs/IMPLEMENTATION_NOTES.md`.
- Tests added: clean, dirty, unavailable-Git, changed-lock, changed-governing-document, and tampered-inline-provenance cases.
- Commands run: `Get-FileHash -Algorithm SHA256` over all four governing documents, targeted provenance tests, and the full validation suite below.
- Implementation status: `IMPLEMENTED_PENDING_VERIFICATION`.
- Unresolved limitations: this correction records clean versus engineering-dirty runs but intentionally does not create a scientific-run or benchmark-freeze gate.

### `EPS-SR-0005` — Scientific schema semantics and modality classification

- Accepted interpretation: names must state their actual denominators and correspondence semantics; the compound transition is neutral control metadata; the unreleased schema may be corrected in place.
- Files changed: `configs/benchmark_v0.yaml`, `src/epsbench/annotations/derive.py`, `src/epsbench/config/models.py`, `src/epsbench/data/generate.py`, `src/epsbench/data/identity.py`, `src/epsbench/schema.py`, `docs/IMPLEMENTATION_NOTES.md`, `README.md`, and affected tests.
- Tests added: strict schema round trips, ecological projection/permission boundaries, and the updated cross-platform ecological identity regression.
- Commands run: the full validation suite below.
- Implementation status: `IMPLEMENTED_PENDING_VERIFICATION`.
- Unresolved limitations: persistent opaque identity provides region correspondence, while dense flow and region-level optical transformation remain unfinished Gate 0B work.

### `EPS-SR-0006` — Array, numeric and structural validation hardening

- Accepted interpretation: declared role, dtype, shape, value domain, ordering, and unique storage identity must all be enforced independently.
- Files changed: `src/epsbench/schema.py`, `src/epsbench/data/validate.py`, `tests/dataset_mutations.py`, `tests/integration/test_scientific_corrections.py`.
- Tests added: non-finite and negative depth, wrong depth dtype, undeclared segmentation label, duplicate path, hardlink alias, wrong frame index, wrong logical modality, and non-contiguous episode indices.
- Commands run: the targeted scientific-correction tests and full validation suite below.
- Implementation status: `IMPLEMENTED_PENDING_VERIFICATION`.
- Unresolved limitations: this is apparatus validation, not a general archival or benchmark-freeze subsystem.

### `EPS-SR-0007` — Review-role and approval protocol

- Accepted interpretation: durable, exact-head role separation is required and must not manufacture any approval.
- Files changed: `docs/review-protocol.md`, `AGENTS.md`, `.github/pull_request_template.md`, `docs/reviews/pr-0001-scientific-review.md`, this correction response.
- Tests added: no executable test substitutes for independent role separation; document inspection and `git diff --check` cover repository form only.
- Commands run: full validation suite below and direct inspection of all protocol artifacts.
- Implementation status: `IMPLEMENTED_PENDING_VERIFICATION`.
- Unresolved limitations: engineering review, scientific re-review, owner approval, and closeout must occur in the required separate roles and exact-head sessions.

### `EPS-SR-0008` — PR evidence and reporting consistency

- Accepted interpretation: the existing PR must report its new exact head and evidence while preserving the historical request-changes record.
- Files changed: `.github/pull_request_template.md`, `docs/reviews/pr-0001-scientific-review.md`, this correction response; PR #1 body update remains a post-commit action.
- Tests added: none; this is evidence and reporting consistency.
- Commands run: final Git, GitHub PR, and CI inspection commands recorded after the correction commit.
- Implementation status: `IMPLEMENTED_PENDING_VERIFICATION`.
- Unresolved limitations: correction head, CI URL/conclusion, and PR body evidence remain pending until commit and push; renewed review remains pending afterward.

## Local implementation evidence before correction commit

```text
uv run ruff check .             all checks completed successfully
uv run ruff format --check .    49 files already formatted
uv run mypy src tests           no issues in 34 source files
uv run pytest                   61 tests completed successfully; 89% total coverage
git diff --check                no whitespace errors
```

The required clean-commit smoke dataset, inspection image, final status, push, exact-head CI result, and PR body update are recorded after the correction commit. None of the local evidence above constitutes independent engineering or scientific verification.
