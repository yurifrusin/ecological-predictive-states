# PR #3 scientific correction response

This is the `CORRECTION_IMPLEMENTER` response to the scientific review of exact head
`5129612540a6520aea1c6bf62c6d2a89cc704534`. It records implementation evidence, not independent
verification. The original findings remain unchanged in
`docs/reviews/pr-0003-scientific-review.md`.

## Response identity

- Repository: `yurifrusin/ecological-predictive-states`
- Pull request: `#3 — Gate 0B Slice 2: add corridor scene-family apparatus`
- Original reviewed head: `5129612540a6520aea1c6bf62c6d2a89cc704534`
- Correction role: `CORRECTION_IMPLEMENTER`
- Committed scientific-correction evidence: `446934899e1e814697e79f3d8b87506194176012`
  and `5451f21c6fd7df99c41758bdea647954b556c18a`
- Engineering review: `ENGINEERING_REQUEST_CHANGES` on
  `5451f21c6fd7df99c41758bdea647954b556c18a`
- Scientific re-review: `PENDING`
- Owner approval: `PENDING`
- Merge, tag, release, or gate advancement: `NOT AUTHORISED`
- Scientific result: `NONE`

## Finding responses

### `EPS-SR3-0001` — Separate raster mask change from ecological accretion/deletion

- Files changed: `src/epsbench/schema.py`, `src/epsbench/annotations/derive.py`,
  `src/epsbench/annotations/__init__.py`, `src/epsbench/data/generate.py`,
  `src/epsbench/data/loader.py`, `src/epsbench/data/validate.py`,
  `src/epsbench/data/identity.py`, `tests/unit/test_annotations.py`, affected integration and
  regression tests, `README.md`, `docs/IMPLEMENTATION_NOTES.md`, and
  `docs/GATE_0B_SLICE_2_CORRIDOR.md`.
- Tests: translated and expanding masks produce only neutral gained/lost image-pixel records;
  generation, ecological projection, hashing, and independent rederivation use the new records;
  both scene families carry the typed unavailable ecological-visibility-event status.
- Implementation status: `IMPLEMENTED_PENDING_VERIFICATION`.
- Limitations: no dense optical flow, optical transport, accretion/deletion inference, or oriented
  boundary ownership is implemented. Region appearance/disappearance remains a raster support
  fact, not a causal occlusion claim.

### `EPS-SR3-0002` — Distinguish unavailable occlusion annotation from known-empty

- Files changed: `src/epsbench/schema.py`, `src/epsbench/data/generate.py`,
  `src/epsbench/data/loader.py`, `src/epsbench/data/validate.py`,
  `src/epsbench/data/identity.py`, `tests/integration/test_generation.py`,
  `tests/integration/test_corridor.py`, `tests/integration/test_corridor_corruption.py`,
  `tests/integration/test_scientific_corrections.py`, and
  `tests/regression/test_identity_domains.py`.
- Tests: single-occluder relations are available and must equal counterfactual evidence;
  corridor unavailable is accepted; hash-consistently fabricated available-empty and fabricated
  available-relation corridor records fail; available-empty and unavailable hash differently.
- Implementation status: `IMPLEMENTED_PENDING_VERIFICATION`.
- Limitations: no corridor occlusion oracle or full boundary-ownership method is implemented;
  available-empty remains a reserved representable state rather than a conclusion emitted by the
  current generator.

### `EPS-SR3-0003` — Advance the changed ecological transition schema honestly

- Files changed: `src/epsbench/schema.py`, `src/epsbench/data/generate.py`, affected schema and
  identity regression tests, `README.md`, `docs/IMPLEMENTATION_NOTES.md`, and
  `docs/GATE_0B_SLICE_2_CORRIDOR.md`.
- Tests: transition round-trip requires `0.1.0-dev.2`; the historical Slice 1
  `0.1.0-dev.1` hash remains recorded; the new single-occluder identity and backend-specific
  Windows/WGL and Ubuntu/OSMesa corridor identities are exact regressions.
- Implementation status: `IMPLEMENTED_PENDING_VERIFICATION`.
- Limitations: the migration is intentionally not byte compatible. It does not offer a V1/V2
  loader union because these contracts are unreleased. Corridor ecological identity is not
  cross-platform; exact values are selected by recorded renderer backend. Successful renewed
  exact-head CI confirmation is required before re-review can accept the pinned values.

### `EPS-SR3-0004` — Correct scene-content identity semantics

- Files changed: `src/epsbench/data/identity.py`, `src/epsbench/data/generate.py`,
  `src/epsbench/data/validate.py`, `tests/regression/test_scene_content_identity.py`, existing
  appearance-invariance tests, `docs/IMPLEMENTATION_NOTES.md`, and
  `docs/GATE_0B_SLICE_2_CORRIDOR.md`.
- Tests: fixed content is invariant to appearance and seed/remapping history; sampled width,
  length, camera, field of view, and action alter corridor content identity; the
  single-occluder domain asserts its exact apparatus and trajectory; independently rebuilt
  manifest hashes cannot hide scene-content tampering from either scene-family validator.
- Implementation status: `IMPLEMENTED_PENDING_VERIFICATION`.
- Limitations: `scene_content_sha256` describes controlled non-appearance layout and trajectory,
  not pixel/container identity or provenance. Episode seed remains separately recorded; no
  separate seed-sensitive `scene_instance_sha256` is needed by the current project.

## Local and CI evidence

Run `32929169048` failed on intermediate head
`446934899e1e814697e79f3d8b87506194176012` when it exposed unpinned Ubuntu/OSMesa corridor
identities; its dataset-command step was skipped. Run `32929628787` passed all checks and both
dataset-command paths on exact scientific-correction head
`5451f21c6fd7df99c41758bdea647954b556c18a`. Engineering review of that exact head requested the
separate corrections recorded in the engineering review cycle. Until independent re-review,
every scientific finding remains `IMPLEMENTED_PENDING_VERIFICATION`.
