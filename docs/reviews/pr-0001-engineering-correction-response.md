# PR #1 engineering-correction response

This is the bounded `CORRECTION_IMPLEMENTER` response to engineering findings on exact head `04038655be0b3974af96287c6ff6a871e5f96f35`. It is implementation evidence only. The new correction head requires renewed engineering review, and `EPS-ER-0001` plus `EPS-ER-0003` require renewed scientific attention.

## Response identity

- Repository: `yurifrusin/ecological-predictive-states`
- Pull request: `#1 — Establish EPS Milestone 0 foundation`
- Engineering-reviewed head: `04038655be0b3974af96287c6ff6a871e5f96f35`
- Role: `CORRECTION_IMPLEMENTER`
- New correction head: `PENDING CORRECTION COMMIT`
- Engineering re-review: `PENDING`
- Scientific attention for `EPS-ER-0001` and `EPS-ER-0003`: `PENDING`
- Owner approval: `PENDING`
- Merge, tag, release, or gate advancement: `NOT AUTHORISED`
- Scientific result: `NONE`

## Finding responses

### `EPS-ER-0001`

- Accepted interpretation: `counterfactual_occluder_exclusion_v1` requires an exact three-surface apparatus bijection, complete occluder absence, changed pixels exactly equal to the ordinary occluder footprint, and reveal evidence only where ordinary occluder pixels become candidate background pixels.
- Files changed: `src/epsbench/schema.py`, `src/epsbench/data/generate.py`, `src/epsbench/data/validate.py`, `tests/dataset_mutations.py`, `tests/integration/test_scientific_corrections.py`, `docs/IMPLEMENTATION_NOTES.md`.
- Tests: fully hash-rebuilt occluder-presence, outside-footprint-change, fabricated-reveal, extra-apparatus-surface, and non-bijective-mapping corruptions.
- Status: `IMPLEMENTED_PENDING_VERIFICATION`.
- Remaining limitation: the rule remains intentionally specific to the single-occluder apparatus and is not general-scene occlusion reasoning.

### `EPS-ER-0002`

- Accepted interpretation: serialized repository references must not contain URL credentials, query/fragment credentials, SSH user information, file URLs, Windows paths, or relative local paths.
- Files changed: `src/epsbench/schema.py`, `src/epsbench/data/provenance.py`, `tests/unit/test_provenance.py`, `docs/IMPLEMENTATION_NOTES.md`.
- Tests: credential-bearing HTTPS, credential-bearing SSH URL, SCP-style SSH, file URL, Windows local path, and relative local path origins.
- Status: `IMPLEMENTED_PENDING_VERIFICATION`.
- Remaining limitation: unclassified origin syntaxes are deliberately replaced with a generic redaction marker rather than preserved.

### `EPS-ER-0003`

- Accepted interpretation: stable renderer/execution provenance requires its own identity and must participate in the content/provenance binding without entering `dataset_logical_sha256` or ecological-label identity.
- Files changed: `src/epsbench/schema.py`, `src/epsbench/data/identity.py`, `src/epsbench/data/generate.py`, `src/epsbench/data/validate.py`, `tests/dataset_mutations.py`, `tests/integration/test_scientific_corrections.py`, `docs/IMPLEMENTATION_NOTES.md`.
- Tests: direct renderer provenance tampering, renderer hash rebuilding without binding rebuilding, and proof that renderer changes leave scientific content identity unchanged while changing the provenance binding.
- Status: `IMPLEMENTED_PENDING_VERIFICATION`.
- Remaining limitation: volatile timestamp and hostname remain outside the manifest and binding so deterministic content regeneration is preserved.

### `EPS-ER-0004`

- Accepted interpretation: inspection must complete whole-dataset validation before reading panels or creating output, and failure must leave a requested absent output absent.
- Files changed: `src/epsbench/data/inspect.py`, `tests/integration/test_validation_and_inspection.py`, `tests/integration/test_cli.py`, `docs/IMPLEMENTATION_NOTES.md`.
- Tests: corrupt-dataset API inspection and CLI inspection both return failure without producing an image.
- Status: `IMPLEMENTED_PENDING_VERIFICATION`.
- Remaining limitation: inspection does not delete or replace an unrelated pre-existing output when validation fails.

### `EPS-ER-0005`

- Accepted interpretation: artifact paths beginning with a Windows drive prefix must fail schema validation regardless of the host operating system.
- Files changed: `src/epsbench/schema.py`, `tests/unit/test_schema.py`, `docs/IMPLEMENTATION_NOTES.md`.
- Tests: drive-qualified and drive-relative paths with upper- and lower-case drive letters.
- Status: `IMPLEMENTED_PENDING_VERIFICATION`.
- Remaining limitation: accepted artifact paths remain POSIX-style dataset-relative paths; this correction does not create a general URI subsystem.

## Evidence boundary

Commands, results, new exact head, and CI URL are added to PR #1 after commit and push. This response does not independently verify any correction or create owner approval.
