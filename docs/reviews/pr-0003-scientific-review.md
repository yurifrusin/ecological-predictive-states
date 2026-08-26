# PR #3 scientific review record

This file preserves the scientific review of one exact revision. It is not a correction response
or renewed review.

## Review identity

- Repository: `yurifrusin/ecological-predictive-states`
- Pull request: `#3 — Gate 0B Slice 2: add corridor scene-family apparatus`
- Reviewed head: `5129612540a6520aea1c6bf62c6d2a89cc704534`
- Reviewer role: `SCIENTIFIC_REVIEWER`
- Disposition: `SCIENTIFIC_REQUEST_CHANGES`
- Execution limitation: no independent local execution was performed by the reviewer
- Repository changes by reviewer: none
- Engineering review, owner approval, merge, or gate advancement: none

## Findings

### `EPS-SR3-0001` — Separate raster mask change from ecological accretion/deletion

The current derivation labels same-coordinate segmentation differences as ACCRETING and DELETING.
Under forward corridor motion this conflates optical transport/expansion with accretion or
deletion at occluding boundaries.

Implement a neutral record for what is actually available, for example:

```text
RegionMaskChange
MaskChangeKind:
  GAINED_IMAGE_PIXELS
  LOST_IMAGE_PIXELS
  REGION_APPEARED
  REGION_DISAPPEARED
  MASK_UNCHANGED
```

Naming may differ, but it must not imply an occlusion cause.

Preserve exact:

- before and after visible pixel counts;
- projected-image fractions;
- same-image-coordinate overlap;
- gained and lost image-coordinate pixel counts.

Add a typed ecological-visibility-event annotation status. Until flow and oriented boundary
ownership are implemented, it must be:

```text
status: unavailable
reason/category: optical transport and boundary ownership not yet available
```

Do not fabricate actual accretion/deletion values. Tests must demonstrate that a translated or
expanding mask produces neutral gain/loss records and is not asserted to be
accretion/deletion. Do not implement dense optical flow in this correction.

### `EPS-SR3-0002` — Distinguish unavailable occlusion annotation from known-empty

Replace the bare occlusion tuple with a typed availability contract.

A suitable shape is:

```text
AvailableOcclusionAnnotation:
  status: available
  oracle_rule
  relations

UnavailableOcclusionAnnotation:
  status: unavailable
  reason/category
```

Use:

- available plus counterfactual relations for `single_occluder`;
- unavailable for corridor, with a reason explaining that oriented corridor occlusion awaits a
  controlled boundary-ownership oracle.

Reserve `status: available, relations: []` for a future oracle-supported conclusion that no
relation exists.

Update ecological hashes and validators so that:

- available relations are checked against the single-occluder oracle;
- unavailable corridor annotation cannot contain relations;
- a fabricated available-empty corridor record fails;
- a fabricated corridor relation fails;
- known-empty and unavailable cannot share identity.

Do not implement a corridor occlusion oracle or full boundary ownership.

### `EPS-SR3-0003` — Advance the changed ecological transition schema honestly

The transition contract changed when forward actions and the corridor annotation posture were
introduced. It cannot remain `0.1.0-dev.1`.

Advance the ecological transition schema explicitly, preferably to `0.1.0-dev.2`.

The dataset/config/instrumentation development version may remain dev.2 if the resulting
contracts are coherent; these PR changes are unreleased.

Do not preserve the old single-occluder ecological hash by retaining a false schema version.

Document:

- historical Slice 1 transition version and hash;
- new transition contract;
- reason for migration;
- expected new single-occluder hash;
- expected corridor hashes;
- lack of byte compatibility;
- preservation of scientific meaning for unchanged Slice 1 observations.

Pin new regression identities only after local and exact-head CI evidence.

A versioned V1/V2 union is permitted only if it is cleaner than migration. Do not introduce
unnecessary compatibility machinery merely to preserve a development hash.

### `EPS-SR3-0004` — Correct scene-content identity semantics

Revise `scene_content_sha256` so it represents content rather than generation history.

Exclude episode seed from the scene-content domain.

For corridor content include at least:

- scene family;
- sampled width and length;
- wall height;
- camera start and end;
- camera height;
- field of view;
- executed action;
- any other non-appearance variable that determines scene layout or trajectory.

For single-occluder content include at least:

- scene family or apparatus version;
- scene-defining parameters;
- camera before/after configuration;
- field of view;
- executed action.

Exclude:

- episode seed;
- appearance;
- opaque surface IDs;
- segmentation labels;
- timestamps;
- renderer and source provenance.

Episode seed remains separately present in the manifest/provenance.

If the project needs a seed-sensitive identity, add or rename a separate
`scene_instance_sha256`. Do not call a seed-sensitive value scene content.

Add tests proving:

1. appearance-only change preserves scene-content identity;
2. remapping-only or seed-history-only change preserves scene-content identity;
3. width, length, camera, field of view or action change changes content identity;
4. single-occluder scene content includes its actual fixed apparatus and trajectory rather than
   only family plus seed;
5. validators independently recompute the domain.

## Interpretation boundary

The disposition applies only to reviewed head
`5129612540a6520aea1c6bf62c6d2a89cc704534`. This record does not verify later corrections,
provide engineering review, record owner approval, authorise merge/tag/release, advance Gate 0B,
authorise Gate 0C, or establish a scientific result.
