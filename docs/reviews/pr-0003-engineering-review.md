# PR #3 engineering review record

This file preserves the engineering review supplied for one exact revision. It is not a
correction response or renewed review.

## Review identity

- Repository: `yurifrusin/ecological-predictive-states`
- Pull request: `#3 — Gate 0B Slice 2: add corridor scene-family apparatus`
- Reviewed head: `5451f21c6fd7df99c41758bdea647954b556c18a`
- Reviewer role: `ENGINEERING_REVIEWER`
- Disposition: `ENGINEERING_REQUEST_CHANGES`
- Correction role authorised by the handoff: `CORRECTION_IMPLEMENTER`
- Engineering execution or repository-mutation method: not asserted by the supplied handoff
- Owner approval, merge, tag, release, or gate advancement: none

## Findings

### `EPS-ER3-0001`

Independently bind semantic apparatus names, raw geom IDs, world positions, and compiled geom
dimensions for both scene families. Reject fully rehashed raw-ID permutations and
finite-but-false coordinates.

### `EPS-ER3-0002`

Require all frame and scene-specific raster dimensions to equal the resolved render
configuration. Add fully rehashed mismatch tests for both scene families.

### `EPS-ER3-0003`

Remove the full public manifest/control-metadata exposure from ecological-only `DatasetLoader`
instances. Retain explicit permission-gated metadata access and add public-surface tests. Do not
describe the loader as a hostile-code sandbox.

### `EPS-ER3-0004`

Reject a root manifest whose resolved target escapes the dataset root or is an unsafe
alias/special file. Apply the same rule in validation and loading. Add an Ubuntu/POSIX
external-symlink regression and applicable Windows tests.

### `EPS-ER3-0005`

Replace `PENDING CORRECTION COMMIT` with non-self-referential committed evidence. Distinguish
failed CI run `32929169048` from successful exact-head run `32929628787` and update stale
review-status prose without manufacturing approval.

### `EPS-ER3-0006`

Add compiled-MuJoCo contract tests binding scene-content constants to actual geom positions/sizes
and camera field of view/orientation.

## Interpretation boundary

The disposition applies only to reviewed head
`5451f21c6fd7df99c41758bdea647954b556c18a`. This record does not verify a later correction,
provide scientific review, record owner approval, authorise merge/tag/release, advance a gate, or
establish a scientific result.
