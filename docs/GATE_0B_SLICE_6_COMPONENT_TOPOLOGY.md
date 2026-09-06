# Gate 0B Slice 6 — Component topology and split/merge visibility oracle v0

Owner issuance authorises implementation from canonical main
`0e9e0593cc7a0f445718575152eca5fc6651edc4`, tree
`ec98e8bcc8ed62686ccc0f98a0fe8956690be698`, on
`codex/gate-0b-slice-6-component-topology`. The profile is `DUAL_REVIEW`, evidence
class `PUBLIC_REPOSITORY_ONLY`, closeout boundary `WORK_PACKAGE`. This document is
the prospective scientific definition. It gives neither independent review nor
owner merge approval. Current execution evidence and exact identities belong in
the implementation PR and worker report, outside this locked definition.

## Public component domain

Every frame independently partitions the public opaque int32 segmentation map
into maximal four-neighbour components of equal nonzero label. Zero is excluded.
Every nonempty component is retained, including isolated one-pixel components.
Diagonal contact does not connect components. Declared opaque surface IDs and
their segmentation labels must be unique; undeclared labels are rejected.

Components are ordered by opaque surface ID, then their minimum `(row, column)`
in row-major order. A coordinate tie is an error. Each surface's components have
zero-based ranks in that order. The frame map has consecutive positive int32
labels in the complete canonical component order; zero denotes background. The
map label is only a frame-map lookup key. The component ID is `component-` followed
by SHA-256 of the `component_id` domain envelope containing frame index, opaque
surface ID and surface-local rank. It is frame-local and does not claim persistent
object identity. Different surfaces may both have rank zero without sharing IDs.

Each component records its frame, ID, opaque surface ID, surface rank, map label,
exact pixel count, minimum coordinate, `(top,left,bottom,right)` integer bounds
with exclusive bottom/right, and a mask root. Masks are full-frame row-major uint8
membership maps containing only zero and one. The retained component map is
lossless membership evidence for every mask. No size threshold, metric extent,
floating-point tolerance or appearance-based grouping is introduced.

## Exact transport and graph

Inputs are opaque segmentation, the six retained public analytic-transport arrays
and two public pixel-event code arrays. Generation passes those arrays explicitly;
the topology function accepts no raw hit-assignment, geometry, camera, RGB,
depth, semantic apparatus, remapping or appearance object.

The input transport contract is `analytic_static_scene_transport_v3`, scale 1024,
signed int32 `(dx,dy)` displacements. A source pixel contributes only when validity
is one, reason is zero, and its segmentation label is nonzero. Invalid vectors
must be canonical zero; unknown validity/reason codes fail closed. Boundary-band,
occluded, frame-exit and no-controlled-surface samples never contribute support.

For a source pixel `(r,c)`, the target cell is exactly:

```text
target_column = floor((1024*c + 512 + dx_fixed) / 1024)
target_row    = floor((1024*r + 512 + dy_fixed) / 1024)
```

Integer floor is toward negative infinity. Cells are half-open; an exact grid-line
tie belongs to the cell on the positive side. Targets outside the retained raster
are excluded without clipping or clamping. Targets on background or another
opaque surface contribute nothing. No aligned-mask overlap shortcut is used.

Every before/after same-surface Cartesian pair is retained, including pairs with
both counts zero. Counts are numbers of eligible source pixels, not weights or
rounded fractions. Forward and backward counts are separate. An undirected
bipartite edge exists if **either count is positive**. One-direction-only support
is admissible evidence under this v1 rule, remains numerically explicit, and does
not claim reciprocal transport. No minimum count beyond one is used.

Support records sort uniquely by surface ID, before component ID, then after
component ID. Event members are uniquely sorted component IDs; events sort
uniquely by surface ID, before-ID tuple, then after-ID tuple.

Connected subgraphs, including isolated nodes, partition every component exactly
once. One before and one after gives `one_to_one_continuation`; one before and
multiple after gives `one_to_many_split`; multiple before and one after gives
`many_to_one_merge`; multiple on both sides gives `complex_many_to_many`.

An isolated component gives appearance/disappearance only if its surface has no
component at all in the other frame, or **every** pixel in that component has
public directional code 1 (accretion/deletion) or 2 (entry/exit). Otherwise it gives
`indeterminate_insufficient_transport_support`. Boundary ambiguity, missing valid
support, and a changed component count alone never establish split or merge.
Pixel accretion/deletion and whole-surface lifecycle remain separate annotations.
Whole-surface appearance may yield several component appearances.

## Capability and identities

The available contract has schema `component_topology_annotation_v1` and explicit
literal versions for connectivity, coordinates, target assignment, support,
classification, acceptance and all annotation domains. Unknown fields, versions,
enums, domains or noncanonical memberships are rejected. A complete oracle with
any indeterminate event has capability status `indeterminate`; evidence remains
present, while qualified component-topology capability is withheld. An annotation
without indeterminate events has status `available`. Complex events are retained
and permitted; they never become a split or merge by relabelling.

Every root uses a distinct envelope with `envelope_version:
logical_domain_envelope_v1`, `domain: epsbench.component_topology.v1.<kind>`, and
`payload`. Kinds include `component_id`, `component_mask`, `component_map`,
`component_set`, `event_id`, `supports`, `events`, `annotation`, `portable_graph`,
`definition`, `scientific_sources`, `qualification_membership`, `lock`,
`portable_matrix`, and `qualification_packet`. Logical roots exclude artifact
paths, file-container byte hashes and volatile run metadata. File hashes, byte
counts and safe ownership checks independently protect retained arrays.

The complete renderer-local annotation binds segmentation hashes, transport hash,
public event-code hashes, complete maps, component records, all support counts,
events and the portable root. Visibility-event method
`analytic_transport_boundary_causal_events_with_component_topology_v3` binds this
annotation and its root; transition `0.1.0-dev.10` binds the visibility event in
the ecological-label domain. Dataset versions `0.1.0-dev.9` (legacy appearance
registry) and `0.1.0-dev.10` (Revision 1 registry) require this transition contract.

The explicit `COMPONENT_TOPOLOGY` modality is ecological-oracle data.
`read_component_topology` requires it before opening the transition. Public event
and ecological-transition projections containing topology require this permission
too. The dedicated loader exposes no profile, role, final-root registry, source
provenance, raw identifier, metric or generation metadata. Qualification packets
are separate privileged control evidence, never returned by the ecological loader.

Legacy `generate` remains the canonical Slice 4/5/freeze generator: transition
dev.9, visibility method v2, and typed unavailable topology remain byte-compatible
at the logical-domain level. Only `component_topology=True` or the new audit
command opts into Slice 6. Historical transport, boundary and event identities and
the existing appearance-freeze apparatus are not rewritten.

## Independent validation and inspection

Generation uses flood fill; validation independently scans and unions north/west
neighbours. Target assignment uses vectorised int64 arithmetic for generation and
Python integers for reconstruction. Graph grouping uses traversal for generation
and set-equivalence closure for reconstruction. Both apply the exact declared
classification. Closed-form tests separately assert split, merge, complex,
lifecycle, boundary, zero/one-direction support and rounding semantics.

The whole-dataset validator consumes owned, hashed public artifacts, reconstructs
every component/map/count/edge/event and compares complete equality. It separately
validates the existing privileged apparatus; privileged results do not enter the
public topology derivation. Fully rehashed false topology, maps, counts, IDs or
events fail reconstruction even when all outer identities have been resealed.

Inspection validates the complete dataset before opening panels. It publishes a
PNG atomically outside the dataset, refuses overwrite, and does not change dataset
identity. It displays both component maps and labels, complete IDs and opaque
surface IDs, supported edges with both counts, zero/one-direction pair totals,
all topology events, qualification posture and both topology hashes. Full zero
pair records remain in the retained annotation.

```powershell
uv run epsbench component-topology-audit --config configs/benchmark_v0.yaml --episodes 2 --output artifacts/component-topology-single-smoke
uv run epsbench component-topology-audit --config configs/corridor_v0.yaml --episodes 2 --output artifacts/component-topology-corridor-smoke
uv run epsbench inspect artifacts/component-topology-single-smoke --output artifacts/component-topology-single.png
uv run epsbench inspect artifacts/component-topology-corridor-smoke --output artifacts/component-topology-corridor.png
```

## Prospective lock and frozen-root execution

`configs/component_topology_v0.yaml` is checked against the complete versioned
definition, then `configs/component_topology_v0_lock.json` binds it, all Python
source/tests/scripts, configuration snapshots, dependency lock, CI, this document,
exact frozen matrix membership and both renderer fingerprints. Text source hashes
normalise CRLF to LF, matching repository Git attributes. This is explicit source
normalisation; protected appearance-freeze files are additionally checked for byte
identity against the authorised base. No source-self or Git-commit hash is embedded
in its own lock root.

The implementation, tests and complete definition must be committed together.
Before any final rendering, the first-addition lock commit and tree, lock root,
clean implementation head/tree, ancestry from the authorised base, original lock
bytes, and exact remote authorised-branch readback must be verified. The original
lock cannot be replaced after execution starts. Phase A uses synthetic fixtures,
ordinary seeds and non-final regressions only; the existing final-root rendering
test is deferred until this publication boundary.

The dependency remains Appearance Benchmark Input Freeze v0 replacement lock
`d4072f912cc58bbc1ca41ceb2652e41783dbf3e1`, root
`28acd2c340b3ef1e2b8b2b7b31ca69356cd9baa35b94ff8a598e020099883435`.
Per renderer, qualification uses exactly its 160 selected cells (five roles, two
scenes, sixteen roots) plus its 32 retained legacy controls, once each. Excluded
Revision 1 stripes and illumination remain historical negative evidence and are
not evaluated on final roots. Existing canonical freeze jobs still run separately.

Required fingerprints are MuJoCo 3.12.0, NumPy 2.4.6 and `mujoco.Renderer`, with
Windows/`wgl-default` and Linux/`osmesa`, exactly as the canonical freeze defines.
There is no backend-specific scientific method or threshold selection.

Every matched scene/root group must have equal complete topology roots across
all six appearances within each renderer. Across renderers, portable roots and
exact event summaries must match for all 192 cells. The portable domain binds
frame/surface/rank component IDs, node membership, supported edge existence,
capability and exact event records. It deliberately excludes raster counts,
bounds, masks and direction counts, which are retained in complete renderer-local
annotation roots. Portable equality is a tested requirement, not an assumption;
different four-connected raster topology fails it and must remain visible.

Any cell reconstruction failure, indeterminate capability, missing cell,
appearance-invariance failure or portable renderer mismatch withholds overall
qualification. Complex events alone are permissible retained evidence. A CLI/CI
success can mean that a complete **failed-closed evidence packet** was successfully
produced and verified; it does not mean the capability qualified or a gate passed.
All statuses and counts by scene/profile/renderer are retained. No component,
failed cell, difficult root or negative outcome is removed or replaced.

After first final-root execution, any required scientific-source change returns
`OUT_OF_SCOPE_PENDING_OWNER`; no final evidence may guide code/rule/threshold/test
fixture changes. Operational repairs that leave the locked definition and source
semantics untouched require a new exact implementation head, with the failed run
and repair reported. Unfinished runs retain their partial sources and error logs.
Generated datasets and ordinary inspection/report artifacts remain untracked.

Private connected GitHub publication retains complete source packets, raw archive
digest/size, live repository/run/job/artifact identities and reconstructive
validation receipts at the exact PR head. Cross-renderer comparison independently
reconstructs downloaded WGL sources using the existing safe archive controls.
These are engineering evidence receipts, not active review-state records.

## Limits and non-authority

This oracle describes image-plane components in two static scene families. It
does not define ecological graph/state conversion, semantic objects, persistent
identity, dynamics, learned inference or task performance. Quantised transport,
four-neighbour raster connectivity and excluded boundary samples can leave small
components indeterminate; retaining them is required. Portable frame/surface/rank
IDs assert canonical image-plane ordering only, and cannot prove a metric or
semantic cross-renderer correspondence.

Independent scientific review and independent engineering review remain required
at the exact implementation PR head. Owner implementation approval is not given;
merge is not performed. Full Gate 0B completion is not claimed; Gate 0C and Gate 0D
are not authorised. Model protocol freeze is not performed; comparative model
result access is not authorised; model work is not performed. No empirical gate
is evaluated, tag created, release published or scientific result established.
The next work package is not begun.
