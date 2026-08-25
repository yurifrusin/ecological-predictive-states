# PR #1 scientific review record

This file preserves the scientific review of one exact revision. It is not a correction response or a renewed review.

## Review identity

- Repository: `yurifrusin/ecological-predictive-states`
- Pull request: `#1 — Establish EPS Milestone 0 foundation`
- Reviewed head: `d4c3c4cc0ec9422d4e224ba430341b65b4de641e`
- Reviewer role: `SCIENTIFIC_REVIEWER`
- Disposition: `SCIENTIFIC_REQUEST_CHANGES`
- Method: direct inspection of the GitHub pull-request source, tests, and CI evidence
- Execution limitation: no independent local execution was performed in the reviewer’s analysis shell
- Repository changes by reviewer: none

## Findings

### `EPS-SR-0001` — Exact action–camera correspondence

The configuration checked only the lateral displacement scalar. It did not reject unsupported or non-finite action forms, nor independently prove from persisted camera instrumentation that the recorded forward, lateral, height, and rotation changes matched the declared action.

### `EPS-SR-0002` — Derived and independently validated occlusion relation

The foreground-to-background relation and frame membership were written as a fixed semantic assertion. No controlled counterfactual, ray, or geometry evidence was persisted and independently checked.

### `EPS-SR-0003` — Dataset-wide episode-local surface identifiers

Generation produced distinct episode identifiers, but whole-dataset validation did not enforce pairwise disjoint opaque surface IDs across episodes.

### `EPS-SR-0004` — Exact source and governing-document provenance

The manifest lacked a strict source-provenance record binding Git state, dirty-state evidence, lock-file identity, governing-document hashes, package version, and Python version to the generated content.

### `EPS-SR-0005` — Scientific schema semantics and modality classification

Frame-area fractions and same-coordinate mask overlap used names that could be mistaken for physical-surface visibility and flow correspondence. The compound transition artifact was classified as a visibility-event modality rather than neutral control metadata.

### `EPS-SR-0006` — Array, numeric and structural validation hardening

Validation did not completely constrain scientific array dtypes/shapes, non-finite or negative values, camera matrices, contiguous episode indices, duplicate/aliased artifact paths, or all persisted structural claims.

### `EPS-SR-0007` — Review-role and approval protocol

The repository lacked a durable separation protocol for owner/PI, implementation, engineering review, scientific review, correction, and closeout roles, including exact-head approval invalidation.

### `EPS-SR-0008` — PR evidence and reporting consistency

PR evidence needed to identify the corrected exact head, current test count, stable finding responses, renewed review status, CI for that head, remaining Gate 0B work, and the absence of scientific, merge, release, or gate claims.

## Interpretation boundary

The reviewer judged the broad architecture sound enough for correction without redesign. The disposition above applies only to the reviewed head. This record does not verify later corrections, grant engineering or scientific approval, record owner approval, authorise merge, or advance a gate.
