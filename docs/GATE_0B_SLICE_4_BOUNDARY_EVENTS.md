# Gate 0B Slice 4: oriented boundaries and visibility events

## Status and scientific purpose

This document describes implementation work on `codex/gate-0b-slice-4-boundary-events` under review profile `DUAL_REVIEW`, evidence class `PUBLIC_REPOSITORY_ONLY`, and closeout boundary `WORK_PACKAGE`. The work is implemented pending independent engineering review, independent scientific review, and owner approval. It is not canonical and is not a scientific result.

Slice 4 adds an analytic image-plane oracle for oriented boundary ownership and uses that oracle with the canonical analytic optical transport to distinguish stable transport, accretion/deletion at supported occluding contours, frame entry/exit, and whole-surface visibility states. It does not infer those events from aligned mask differences, RGB, rendered depth, or renderer segmentation.

## Edge lattice and owner side

The public method uses `four_neighbour_sample_edge_lattice_v1` over pixel centres.

- A horizontal coordinate `(r, c)` separates the negative-axis sample `(r, c)` on the left from the positive-axis sample `(r, c + 1)` on the right. Its lattice shape is `(H, W - 1)`.
- A vertical coordinate `(r, c)` separates the negative-axis sample `(r, c)` on the top from the positive-axis sample `(r + 1, c)` on the bottom. Its lattice shape is `(H - 1, W)`.
- `NEGATIVE_AXIS_SIDE` therefore means left on a horizontal coordinate and top on a vertical coordinate. `POSITIVE_AXIS_SIDE` means right or bottom respectively.
- `NONE` states that no unilateral owner was established.

The representation is a canonical sparse tuple of `OrientedBoundaryElement` records. A same-surface neighbour pair is `NO_BOUNDARY` and is represented by the absence of a sparse record. The schema rejects an explicit `NO_BOUNDARY` record, duplicate or non-canonical coordinates, out-of-range coordinates, unknown kinds, unknown surface IDs, and an owner that does not equal the surface on the declared side.

Each record contains only frame index, edge axis and coordinate, opaque episode-local neighbouring `SurfaceId` values or null, boundary kind, owner side, and opaque owner ID or null. It contains no semantic apparatus name, raw MuJoCo ID, depth, metric position, contact point, camera pose, or counterfactual distance.

## Boundary kinds

`oriented_boundary_kind_domain_v1` preserves six states:

- `NO_BOUNDARY`: both analytic neighbouring samples have the same assignment; represented implicitly by no sparse record.
- `OCCLUDING_CONTOUR`: two non-attached controlled surfaces meet and exactly one counterfactual exclusion establishes continuation of the other surface. The nearer terminating surface owns the edge.
- `ATTACHED_JUNCTION`: the pair is declared by the scene contract and compiled geometry verifies its contact or intersection. It has no owner.
- `CONTROLLED_SILHOUETTE`: exactly one side is a controlled surface and the other is uncontrolled space. The controlled side owns the edge; uncontrolled space is not assigned a synthetic surface.
- `MULTI_SURFACE_JUNCTION_AMBIGUOUS`: the deterministic incident neighbourhood contains more than two controlled assignments. It has no owner.
- `UNRESOLVED_BOUNDARY`: the bounded analytic method cannot choose a supported state. It has no owner, and strict canonical generation rejects it.

The ambiguity rule is `edge_incident_3x2_or_2x3_multi_surface_v1`: a horizontal edge examines the clipped three-row by two-column neighbourhood incident to its samples, while a vertical edge examines the clipped two-row by three-column neighbourhood. More than two controlled assignments makes the edge multi-surface ambiguous before owner testing.

## Privileged attachment contract

`compiled_axis_aligned_plane_box_and_box_contact_v1` independently recompiles each apparatus and verifies the complete controlled-surface contact graph. It accepts only compiled MuJoCo planes and boxes whose world rotations are axis aligned within `1e-12`. Plane optical thickness is zero; box half-extents and plane finite visual extents define world-axis intervals. Two controlled geoms contact when all three closed interval gaps are at most `1e-12`.

The verifier fails closed for unsupported geom types or rotations, missing declared contacts, undeclared observed contacts, duplicate declarations, bad semantic/raw-ID bindings, and out-of-range raw IDs. Compiled types, rotations, interval gaps, tolerances, semantic names, raw IDs, and the complete pair evidence remain privileged instrumentation. The public annotation carries only the method/version and a numerical-contract hash.

The single-occluder contract declares support-to-foreground-panel and support-to-background-panel contact. It does not declare foreground-to-background contact. The corridor declares floor-to-left-wall, floor-to-right-wall, floor-to-end-wall, left-wall-to-end-wall, and right-wall-to-end-wall contact. It does not declare left-wall-to-right-wall contact.

## Counterfactual ownership

For a controlled-controlled non-attached edge, `counterfactual_nearest_surface_continuation_v1` excludes the surface visible at each neighbouring pixel centre in turn and analytically finds the next nearest controlled hit on that same ray. Ownership is assigned only under `exactly_one_side_continues_v1`:

- if excluding the negative-side surface reveals the positive-side surface, only the negative side owns;
- if excluding the positive-side surface reveals the negative-side surface, only the positive side owns;
- if both or neither continuation test succeeds, the edge is unresolved.

This computation uses compiled metric geometry as privileged apparatus evidence. The public result is only the image-relative side and opaque surface IDs. Renderer depth, semantic name, and colour are not owner rules.

## Visibility-event contract

`analytic_transport_boundary_causal_events_v1` provides two lossless `uint8` directional maps plus aligned `int32` affected-surface and owner-surface label maps. The labels map back to the transition's opaque `SurfaceId` records; zero is the canonical no-surface value. Owner and affected labels are non-zero only for accretion/deletion code `1`.

The before-frame fate domain is:

| Code | Meaning |
| --- | --- |
| `0` | stable transport |
| `1` | deletion at an occluding boundary |
| `2` | frame exit |
| `3` | analytic-boundary ambiguous |
| `4` | no controlled surface |
| `5` | unresolved occlusion |

The after-frame origin domain uses the same numeric layout, with code `1` meaning accretion at an occluding boundary and code `2` meaning frame entry.

Stable transport means the same visible static surface point remains visible in the other frame. Frame exit and entry are field-of-view outcomes and are not deletion or accretion. Deletion requires forward transport to be blocked by a controlled owner/affected pair supported by an `OCCLUDING_CONTOUR` in the target frame. Accretion applies the symmetric backward analysis against an owner/affected pair supported in the prior frame. Occlusions between globally verified attached pairs remain analytic-boundary ambiguous. Unsupported occluder pairs remain unresolved, and strict canonical generation rejects any such pixel.

`OccludingVisibilityEventSummary` aggregates exact non-zero counts by event kind and opaque affected/owner pair and must agree with the dense maps. `WholeSurfaceVisibilityEvent` uses exact analytic controlled-surface visibility counts:

- zero before and positive after is `APPEARANCE`;
- positive before and zero after is `DISAPPEARANCE`;
- positive in both is `PERSISTENTLY_VISIBLE`;
- zero in both is `PERSISTENTLY_HIDDEN_OR_ABSENT`.

Partial accretion or deletion does not become whole-surface appearance or disappearance. Transport-causal pixel events and whole-surface events are available. Component split/merge remains explicitly unavailable with reason `canonical Slice 4 does not define a component-topology oracle`; zero split/merge counts are not fabricated.

`RegionMaskChange` remains a neutral aligned-raster observation and is not an event oracle.

## Corridor occlusion posture

For the current closed corridor, independent analytic boundary recomputation finds only verified attached controlled-controlled junctions, plus bounded multi-surface ambiguity at their local meetings. It finds no controlled-controlled occluding contour. Corridor occlusion is therefore `status: available`, `relations: []`, with oracle rule `oriented_boundary_ownership_v1`. This known-empty state is distinct from the historical unavailable posture.

The single-occluder retains its separately derived `counterfactual_occluder_exclusion_v1` relation. Validation requires every declared relation frame to have a matching opaque owner/affected `OCCLUDING_CONTOUR` pair.

## Public and privileged boundaries

`ORIENTED_BOUNDARY_OWNERSHIP` and `ECOLOGICAL_VISIBILITY_EVENTS` are ecological-oracle modalities. Their dedicated loader methods check permission before opening the transition or event artifacts. The ecological transition includes the full typed boundary coordinate/kind contract, not a bare owner array, and event loading returns metadata, six aligned maps, and the episode-local label-to-opaque-ID mapping.

Semantic apparatus names, raw IDs, compiled positions/sizes/types/rotations, contact interval gaps, tolerances, camera transforms, counterfactual next-hit evidence, raw analytic assignments, and raw event owner/affected IDs remain in privileged instrumentation. The ordinary ecological view exposes none of those fields.

## Identity domains and schema versions

`oriented_boundary_sha256` covers the method, raster shape, complete coordinate convention, kind and owner domains, attachment rule and public contract versions, separately hashed numerical constants, counterfactual and tie rules, junction and silhouette rules, frame indices, and ordered public sparse records.

`visibility_event_sha256` covers the method, code-domain versions, logical hashes for all six directional artifacts, deterministic owner/affected summaries, whole-surface events, the explicit component-topology capability, frame indices, `oriented_boundary_sha256`, and `analytic_transport_sha256`.

Both domains exclude appearance, RGB, rendered depth, renderer backend, semantic names, raw IDs, world coordinates, contact coordinates, paths, timestamps, and provenance. Opaque surface IDs and episode-local labels are included only where they form the public records. The ecological-label identity binds both separate identities, and the dataset scientific-content identity binds both episode fields.

The wire-version matrix is:

| Contract | Version |
| --- | --- |
| Single-occluder and corridor configuration | `0.1.0-dev.2` (unchanged) |
| Transition record | `0.1.0-dev.6` |
| Dataset and episode manifest | `0.1.0-dev.4` |
| Privileged instrumentation | `0.1.0-dev.6` |

## Validation, tests, and inspection

Whole-dataset validation independently recompiles the exact scene, verifies its complete attachment graph, recomputes analytic assignments and counterfactual next hits, reconstructs every boundary record, and recomputes both event directions, summaries, whole-surface states, and identities. It also checks the privileged compiled-geometry and raw diagnostic evidence. Renderer segmentation remains a non-authoritative exact-interior diagnostic inherited from Slice 3.

Closed-form tests constrain foreground/background continuation, support attachment, perpendicular attachment, silhouettes, multi-surface ambiguity, unresolved both/neither continuation, horizontal/vertical owner sides, static transport, frame entry/exit, exact whole-surface zero crossings, and strict unresolved occlusion. Scene tests constrain counterfactual consistency, action reversal, corridor known-empty truthfulness, appearance invariance, opaque remapping, capability status, permissions, and semantic/metric leakage. Fully rehashed mutations of boundary kind/owner/method/identity, event code/owner/summary/method/identity, and attachment evidence must fail.

Inspection validates the entire dataset before publication and then shows RGB, opaque segmentation, forward/backward transport and reasons, oriented-boundary overlays with kind and owner legends, before/after event maps, opaque accretion/deletion pair summaries, and occlusion/capability status. Output remains atomic, no-replace, outside the immutable dataset, and outside scientific identity.

## Determinism and cross-platform evidence

The locked Windows/WGL two-episode identities are:

| Scene | Episode | Boundary | Visibility events |
| --- | ---: | --- | --- |
| Single occluder | 0 | `ea99e38626142a7c3b1845c234e5b577493fbf8ff55e40169a96bffaafe84f66` | `c35fe87f68dde0b282f8069a6700895d135e956a3b5ba2190f3b606f259b8ebb` |
| Single occluder | 1 | `c690400c1c1496da23b4d8ee91af6b52d7e163a415fb7929ffa4494061becbef` | `c9032eb0250598e95902b3c4d5001ea724a48abf6cd66158fbd5374b2d312f2b` |
| Corridor | 0 | `77e4aba8aa763827ce3c3a1e20ebc03f134c0c44edb127235f071b32c769f503` | `2db60ab41fe0f93f1fd574c0d1218d374727199b51eb3bf15625f6264cefbf0a` |
| Corridor | 1 | `fd5c4360c0cc03d6be11bd9917de4909aa6f26129f95d69364d793c42e8e9ab3` | `480e283e5eecc7fc21e3a21f56e114f44c69460681e167b9292c45fa8e7b5a15` |

These exact values are regression assertions in the full test suite. The same assertions run under locked Ubuntu/OSMesa CI; its exact-head outcome belongs to the public PR evidence and must be evaluated rather than assumed. Complete dataset and RGB identities remain renderer-specific.

## Limitations and remaining work

- The method is bounded to static controlled MuJoCo planes and axis-aligned boxes in the two canonical scene families. Moving surfaces, arbitrary rotations/types, openings, T-junction scene semantics, and nested enclosures are not supported.
- Multi-surface junctions are intentionally ambiguous at the declared local lattice neighbourhood. No subpixel contour or component topology is inferred.
- Attached-pair occlusion during transport is conservatively boundary-ambiguous; it is not coerced to causal accretion/deletion.
- Event ownership is a pair-level consequence of exact analytic transport and the presence of a supported contour pair; this slice does not trace a continuous boundary component to each transported ray.
- Cross-platform evidence is limited to the locked Windows/WGL and Ubuntu/OSMesa environments and does not imply stability under arbitrary drivers or dependency versions.
- Broader appearance assets, final evaluation seeds, full Gate 0B exit-criterion evidence, and owner decisions remain future work. Independent dual review and linked closeout are still required before Slice 4 can become canonical.

This work does not implement a component split/merge oracle, ecological graph or state conversion, model interface, learned method, navigation, robot integration, Gate 0C, or Gate 0D model. Full Gate 0B completion is not claimed, Gate 0C is not authorised, and no scientific result is claimed.
