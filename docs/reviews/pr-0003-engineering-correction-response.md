# PR #3 engineering correction response

This is the `CORRECTION_IMPLEMENTER` response to engineering review of exact head
`5451f21c6fd7df99c41758bdea647954b556c18a`. It records implementation evidence, not independent
verification. The containing commit is intentionally not self-referenced; PR #3 records the new
exact correction head after commit and push.

## Response identity

- Repository: `yurifrusin/ecological-predictive-states`
- Pull request: `#3 — Gate 0B Slice 2: add corridor scene-family apparatus`
- Engineering-reviewed head: `5451f21c6fd7df99c41758bdea647954b556c18a`
- Correction role: `CORRECTION_IMPLEMENTER`
- Prior failed CI evidence: run `32929169048` on
  `446934899e1e814697e79f3d8b87506194176012`
- Prior successful CI evidence: run `32929628787` on engineering-reviewed head
  `5451f21c6fd7df99c41758bdea647954b556c18a`
- Renewed engineering review: required on the eventual new exact head
- Scientific re-review: `PENDING`
- Owner approval: `PENDING`
- Merge, tag, release, or gate advancement: `NOT AUTHORISED`
- Scientific result: `NONE`

## Finding responses

### `EPS-ER3-0001`

- Implementation: both generators persist compiled geom sizes alongside semantic-name/raw-ID and
  world-position evidence. Validation independently recompiles the resolved scene and requires
  exact semantic-name bindings plus numeric agreement for world positions and sizes.
- Regression coverage: hash-consistent raw-ID permutations, finite false positions, and false
  compiled sizes for both scene families.
- Status: `IMPLEMENTED_PENDING_VERIFICATION`.

### `EPS-ER3-0002`

- Implementation: public RGB/depth/segmentation frame dimensions and privileged
  counterfactual/raw-segmentation dimensions must equal resolved render height/width.
- Regression coverage: fully rehashed resolved-config mismatches and privileged-raster mismatches
  for both scene families.
- Status: `IMPLEMENTED_PENDING_VERIFICATION`.

### `EPS-ER3-0003`

- Implementation: the loader manifest is private; full manifest access is an explicit method
  requiring control-metadata, scene-family, and privileged-generation permissions.
- Regression coverage: ecological loader public-surface inspection and denied full-manifest
  access, plus positive privileged access.
- Limitation: this is a cooperative typed API boundary, not a hostile-code sandbox.
- Status: `IMPLEMENTED_PENDING_VERIFICATION`.

### `EPS-ER3-0004`

- Implementation: validation and loading share one root-manifest resolver that rejects escapes,
  symbolic aliases, hard-link aliases, special files, and missing/non-directory roots before
  parsing.
- Regression coverage: POSIX external symlink and special-file tests plus a cross-platform
  external hard-link test.
- Status: `IMPLEMENTED_PENDING_VERIFICATION`.

### `EPS-ER3-0005`

- Implementation: the scientific correction record now names its two committed correction SHAs,
  the engineering disposition, failed run `32929169048`, and successful run `32929628787` without
  treating CI as approval. README and bounded Slice 2 documentation reflect the current review
  state.
- Status: `IMPLEMENTED_PENDING_VERIFICATION`.

### `EPS-ER3-0006`

- Implementation: compiled-scene regression tests compare scene-content constants with actual
  MuJoCo geom world positions, compiled sizes, camera field of view, and compiled orientation for
  both scene families.
- Status: `IMPLEMENTED_PENDING_VERIFICATION`.

## Evidence boundary

Local validation, inspection, commit/push state, and renewed exact-head CI are reported in PR #3
after the correction is committed. None of that evidence independently verifies these responses.
