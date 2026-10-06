# Boundary observation v1

Review profile: DUAL_REVIEW. Evidence class: PUBLIC_REPOSITORY_ONLY.
Phase-gate effect: NONE. This is a prospective, native-free contract and finite
software construct check, with no experiment or model authorization.

`epsbench.diagnostics.boundary_observation` defines a separate observation
alternative. It does not replace, adapt, rescore or reinterpret legacy oracle
ownership, causal v1 evidence or its INCONCLUSIVE disposition.

## Admitted evidence and chronology

The only provider method is `raster(sequence_index) -> VisibleRaster`. Its
nonempty int32 segmentation uses zero for visible background and positive local
raster labels for visible regions. Local labels have no simulator-ID meaning.
The supplied association covers exactly visible nonzero labels with unique
opaque `surface-<16 hexadecimal digits>` tokens. The caller must supply episode
randomized, opaque associations; syntax cannot certify their provenance or
randomness. Identity continuity remains an admitted oracle association privilege.
No hidden inventory or raw-to-opaque mapping is accepted.

`BoundaryObservationView.observe(i)` requires typed `ModalityPermissionSet`
permission for `SURFACE_REGIONS` and `0 <= i <= decision_index`, checked before
provider access. The decision index is an explicit caller-established causal
cutoff, not an independently verified clock. Provider output must have the exact
requested index and type. Payloads are validated and owned immutable snapshots.
The permission set is snapshotted. No global loader permissions change.

The segmentation and visible opaque association are an oracle-segmentation /
association ceiling. There is no claim of extraction from RGB, sensor-only
access, learned identity recovery, physical boundary ownership or sufficiency.
RGB, completed flows and commands are not admitted or fetched by this feature
API. Tests hold these past controls fixed across an evaluator-private synthetic
intervention; their existence in the test does not authorize their use by this API.
The synthetic controls satisfy the `CompletedFlow` payload/validity/reason contract.
No native or analytic admission is tested; they are past-only and not RGB-derived.

## Boundary semantics

For each unequal pair of adjacent raster samples, retain one oriented internal
lattice edge. `horizontal` at `(row, column)` compares `(row, column)` with
`(row, column + 1)`; `vertical` compares it with `(row + 1, column)`. The axis
names describe the direction of comparison, following existing lattice usage,
not the geometric tangent of the contour. Negative and positive sides follow
that comparison direction. Zero maps to `None`, denoting visible background,
not an unseen surface. Image exterior edges are omitted. A one-pixel or uniform
raster has no internal edges; nonpositive raster dimensions are rejected.

Every edge has `ownership = "unknown"`. Raster discontinuity alone does not
establish occlusion, depth ordering, attachment, counterfactual ownership or
front/back. No geometry-informed guess is made. Changed visible structure must
change the resulting edge description; unknown ownership does not erase it.

Output order is axis string, row, column. Serialization is UTF-8 canonical JSON
with sorted keys, compact separators, no NaN and explicit
`version = "boundary-observation-v1"`, sequence index, raster shape and edges.
It contains no raster labels, volatile run metadata, geometry or oracle target.
Renaming opaque tokens only renames the corresponding neighbor fields.

## Finite validation and limits

`uv run --locked python scripts/check_a1_source.py --boundary-observation-only`
runs only these synthetic tests with native/generation imports blocked, ambient
pytest plugins disabled and repository conftest excluded, then constructs,
serializes, parses and prints one small example for inspection. It performs no
rendering, capture, numeric fixture, study, native SDK import or data access.

The synthetic two-variant check shares every declared past optical array,
visible association, RGB array, completed flow and command, while evaluator-only
hidden continuation and oracle ownership differ. The observational output must
remain identical and retain a nonempty visible boundary. A changed visible
boundary is a positive control. Additional checks cover token and local-label
renaming, pre-fetch denial, chronology, immutable ownership of inputs/outputs,
canonical serialization and degenerate raster edges.

Passing demonstrates these finite implementation and construct properties. It
provides no predictive-sufficiency, state-loss, future-prediction, EPS advantage,
empirical-gate, benchmark-freeze or RGB extractor evidence. A later prospective
diagnostic requires separate design, fixture and experiment authority.
