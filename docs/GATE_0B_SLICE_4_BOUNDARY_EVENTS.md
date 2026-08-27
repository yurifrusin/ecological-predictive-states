# Gate 0B Slice 4: oriented boundaries and visibility events

## Status and scientific purpose

Slice 4 is canonical after exact-head engineering and scientific review under review profile `DUAL_REVIEW`, evidence class `PUBLIC_REPOSITORY_ONLY`, and closeout boundary `WORK_PACKAGE`. The exact reviewed implementation head was `24deb074d4ba0cf3e1044e466ecbbb6e0b2a4cc4`; it received `ENGINEERING_PASS`, renewed `SCIENTIFIC_PASS`, and exact-head owner approval before merge. This apparatus closeout has no phase-gate effect and is not a scientific result.

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
- `OCCLUDING_CONTOUR`: two controlled surfaces meet away from a projected attachment locus and exactly one counterfactual exclusion establishes continuation of the other surface. The terminating surface owns the edge, even when that pair contacts elsewhere in the apparatus.
- `ATTACHED_JUNCTION`: the sampled image edge is locally associated with the analytic projection of that pair's compiled 3D contact cell. It has no owner. Scene-level contact membership alone is insufficient.
- `CONTROLLED_SILHOUETTE`: exactly one side is a controlled surface and the other is uncontrolled space. The controlled side owns the edge; uncontrolled space is not assigned a synthetic surface.
- `MULTI_SURFACE_JUNCTION_AMBIGUOUS`: the deterministic incident neighbourhood contains more than two assignments, including uncontrolled space at a contact or silhouette endpoint. It has no owner.
- `UNRESOLVED_BOUNDARY`: the bounded analytic method cannot choose a supported state. It has no owner, and strict canonical generation rejects it.

The ambiguity rule is `edge_incident_3x2_or_2x3_multi_assignment_v2`: a horizontal edge examines the clipped three-row by two-column neighbourhood incident to its samples, while a vertical edge examines the clipped two-row by three-column neighbourhood. More than two assignments makes the edge ambiguous before local attachment or owner testing. This precedence gives contact endpoints and controlled/uncontrolled meetings a typed, deterministic posture.

## Privileged attachment contract

`projected_compiled_contact_locus_v3` independently recompiles each apparatus, verifies the complete controlled-surface contact graph, derives each observed pair's exact axis-aligned intersection cell, and associates that cell with individual image-lattice edges. It accepts only compiled MuJoCo planes and boxes whose world rotations are axis aligned within `1e-12`. Plane optical thickness is zero; box half-extents and plane finite visual extents define world-axis intervals. Two controlled geoms contact when all three closed interval gaps are at most `1e-12`.

The privileged numerical and type contract binds:

- contact-cell derivation `compiled_axis_aligned_intersection_cell_v1`;
- supported cell types `point`, `axis_aligned_segment`, `axis_aligned_rectangle`, and `axis_aligned_overlap_volume`;
- projection convention `analytic_pinhole_pixel_centre_v1`;
- strict in-front rule `strict_forward_distance_greater_than_epsilon_v1`, with forward distance required to be greater than `1e-12`;
- feasibility rule `image_constraints_only_slack_strict_front_and_cell_bounds_exact_v1`, with `1e-12` slack only on inclusive image-association constraints and zero slack on both the strict in-front constraint and contact-cell parameter bounds;
- association rule `sample_connection_segment_intersects_projected_contact_cell_v1`;
- a `0.5` image-pixel band perpendicular to the sample-to-sample edge segment;
- inclusive contact endpoints; and
- `multi_surface_ambiguity_precedes_attachment_v1`.

Association is a bounded linear-feasibility test in camera coordinates, including an explicit zero-dimensional branch for point contact manifolds. It asks whether an actual point in the closed compiled contact cell projects onto the relevant centre-to-centre edge segment within the declared perpendicular band while satisfying the same strict in-front epsilon as analytic transport. Contact-cell parameters are constrained exactly to `[0, 1]`; image feasibility slack cannot create a witness outside that domain. Camera-plane points and points exactly at the epsilon are excluded, while the next representable greater forward distance is included. Segment, rectangle, and volume threshold regressions enforce this distinction. It does not use rendered depth or renderer segmentation. A globally attached pair outside that local projected locus proceeds to ordinary counterfactual ownership, allowing a resting panel's base to be attached while its lateral edge owns support-surface occlusion.

The verifier fails closed for unsupported geom types or rotations, missing declared contacts, undeclared observed contacts, duplicate declarations, bad semantic/raw-ID bindings, out-of-range raw IDs, malformed contact cells, and unsupported projection conditions. The exported projected-attachment entry point validates the complete analytic camera pose and FOV before scanning edges, including the no-boundary case; malformed, non-finite, non-orthonormal, and improper rotations are rejected. Compiled types, rotations, interval gaps, contact-cell bounds/types, tolerances, semantic names, raw IDs, and per-edge locus booleans remain privileged instrumentation. The public annotation carries only typed non-metric method versions and a numerical-contract hash.

The single-occluder contract declares support-to-foreground-panel and support-to-background-panel contact. It does not declare foreground-to-background contact. The corridor declares floor-to-left-wall, floor-to-right-wall, floor-to-end-wall, left-wall-to-end-wall, and right-wall-to-end-wall contact. It does not declare left-wall-to-right-wall contact.

## Counterfactual ownership

For a controlled-controlled edge that is not on its pair's local projected contact locus, `counterfactual_nearest_surface_continuation_v1` excludes the surface visible at each neighbouring pixel centre in turn and analytically finds the next nearest controlled hit on that same ray. Ownership is assigned only under `exactly_one_side_continues_v1`:

- if excluding the negative-side surface reveals the positive-side surface, only the negative side owns;
- if excluding the positive-side surface reveals the negative-side surface, only the positive side owns;
- if both or neither continuation test succeeds, the edge is unresolved.

This computation uses compiled metric geometry as privileged apparatus evidence. The public result is only the image-relative side and opaque surface IDs. Renderer depth, semantic name, and colour are not owner rules.

## Visibility-event contract

`analytic_transport_boundary_causal_events_v2` provides two lossless `uint8` directional maps plus aligned `int32` affected-surface and owner-surface label maps. The labels map back to the transition's opaque `SurfaceId` records; zero is the canonical no-surface value. Owner and affected labels are non-zero only for accretion/deletion code `1`.

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

Stable transport means the same visible static surface point remains visible in the other frame. Frame exit and entry are field-of-view outcomes and are not deletion or accretion. Code `3`, `ANALYTIC_BOUNDARY_AMBIGUOUS`, is copied only from the transport oracle's actual source or projected-target analytic boundary band. It is never inferred merely because two surfaces contact elsewhere.

Deletion requires an interior forward-transport sample to be blocked by a controlled owner/affected pair supported by an `OCCLUDING_CONTOUR` in the target frame. Accretion applies the symmetric backward analysis against a supported pair in the prior frame. Thus the locally supported side contours of the panel/support pairs produce causal support-surface events, while samples on the local attached seam retain the analytic boundary-band code and cannot become causal. Unsupported interior occluder pairs remain `UNRESOLVED_OCCLUSION`, and strict canonical generation rejects any such pixel.

`OccludingVisibilityEventSummary` aggregates exact non-zero counts by event kind and opaque affected/owner pair and must agree with the dense maps. `WholeSurfaceVisibilityEvent` uses exact analytic controlled-surface visibility counts:

- zero before and positive after is `APPEARANCE`;
- positive before and zero after is `DISAPPEARANCE`;
- positive in both is `PERSISTENTLY_VISIBLE`;
- zero in both is `PERSISTENTLY_HIDDEN_OR_ABSENT`.

Partial accretion or deletion does not become whole-surface appearance or disappearance. Transport-causal pixel events and whole-surface events are available. Component split/merge remains explicitly unavailable with reason `canonical Slice 4 does not define a component-topology oracle`; zero split/merge counts are not fabricated.

`RegionMaskChange` remains a neutral aligned-raster observation and is not an event oracle.

## Complete public occlusion graph

For both scene families the public occlusion annotation is exactly the complete canonical set of oriented owner/affected pairs and their frame memberships. Validation requires equality, not subset membership, so omitted, invented, reversed, unsupported, or incorrectly framed relations fail after independent recomputation.

The single-occluder uses `oriented_boundary_ownership_with_counterfactual_crosscheck_v1`. Its public graph includes every supported foreground/background and panel/support relation. The privileged `counterfactual_occluder_exclusion_v1` foreground/background evidence remains an independent cross-check: its exact frame membership must agree with the corresponding member of the complete public graph, but it no longer limits the graph to that relation.

For the current closed corridor, independent analytic boundary recomputation finds only locally projected attached controlled-controlled junctions, plus bounded multi-surface ambiguity at their meetings. It finds no controlled-controlled occluding contour. Corridor occlusion is therefore `status: available`, `relations: []`, with oracle rule `oriented_boundary_ownership_complete_v2`. This oracle-supported known-empty result is distinct from an unavailable posture.

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
| Transition record | `0.1.0-dev.9` |
| Dataset and episode manifest | `0.1.0-dev.4` |
| Privileged instrumentation | `0.1.0-dev.9` |

The manifest remains `0.1.0-dev.4` because its wire shape is unchanged. Transition and instrumentation advance because the public attachment projection rule and privileged image-slack field changed. Boundary ownership advances to `analytic_oriented_boundary_ownership_v4`; attachment projection advances to `projected_compiled_contact_locus_v3` and public attachment contract v4. Visibility events remain `analytic_transport_boundary_causal_events_v2` because event derivation and code semantics are unchanged, although event identities change through their required boundary-identity binding. The six numeric event codes remain truthful under their existing v1 domains, so those domains do not change.

## Validation, tests, and inspection

Whole-dataset validation independently recompiles the exact scene, verifies its complete attachment graph and contact cells, recomputes projected attachment-locus membership, analytic assignments, and counterfactual next hits, reconstructs every boundary record, complete occlusion graph, both event directions, summaries, whole-surface states, and identities. It also checks the privileged compiled-geometry and raw diagnostic evidence and the designated counterfactual cross-check. Renderer segmentation remains a non-authoritative exact-interior diagnostic inherited from Slice 3.

Closed-form tests constrain local versus global attachment, foreground/background continuation, support attachment, perpendicular attachment, silhouettes, multi-surface ambiguity, unresolved both/neither continuation, horizontal/vertical owner sides, actual boundary-band ambiguity, interior pair-level occlusion, attached-seam non-causality, static transport, frame entry/exit, exact whole-surface zero crossings, and strict unresolved occlusion. Scene tests constrain base attachment plus a panel-owned side contour for the same pair, deterministic contact projection, counterfactual consistency, complete graph equality, action reversal, corridor known-empty truthfulness, appearance invariance, opaque remapping, capability status, permissions, and semantic/metric leakage. Fully rehashed mutations of boundary kind/owner/method/identity, event code/owner/summary/method/identity, contact-locus bounds, and every public graph posture must fail.

Inspection validates the entire dataset before publication and then shows RGB, opaque segmentation, forward/backward transport and reasons, oriented-boundary overlays with kind and owner legends, before/after event maps, opaque accretion/deletion pair summaries, and occlusion/capability status. Output remains atomic, no-replace, outside the immutable dataset, and outside scientific identity.

## Determinism and cross-platform evidence

The scientifically reviewed head `c82ac9d68463fea82ad132194007549ed955adf3` was rejected for the three local-attachment, ambiguity, and graph-completeness findings. Its v1 boundary/event identities remain historical evidence and must not be treated as corrected results:

| Rejected head scene | Episode | Boundary | Visibility events |
| --- | ---: | --- | --- |
| Single occluder | 0 | `ea99e38626142a7c3b1845c234e5b577493fbf8ff55e40169a96bffaafe84f66` | `c35fe87f68dde0b282f8069a6700895d135e956a3b5ba2190f3b606f259b8ebb` |
| Single occluder | 1 | `c690400c1c1496da23b4d8ee91af6b52d7e163a415fb7929ffa4494061becbef` | `c9032eb0250598e95902b3c4d5001ea724a48abf6cd66158fbd5374b2d312f2b` |
| Corridor | 0 | `77e4aba8aa763827ce3c3a1e20ebc03f134c0c44edb127235f071b32c769f503` | `2db60ab41fe0f93f1fd574c0d1218d374727199b51eb3bf15625f6264cefbf0a` |
| Corridor | 1 | `fd5c4360c0cc03d6be11bd9917de4909aa6f26129f95d69364d793c42e8e9ab3` | `480e283e5eecc7fc21e3a21f56e114f44c69460681e167b9292c45fa8e7b5a15` |

The first scientific-correction head `66f23ccc88fb1003411069e50ed6b5051754bbf8` supplied the following dev.7 identities. They are now historical engineering-review input, not the active corrected-head evidence:

| Historical dev.7 scene | Episode | Boundary | Visibility events |
| --- | ---: | --- | --- |
| Single occluder | 0 | `da74e273f5e6786c9093709c07147ea2666db58bb64d687665dbf563e5f9e717` | `dfd7fddd2a460ed1d3147bfca55c9e9034269c4ad0cecdc5de4751b1678cd3d6` |
| Single occluder | 1 | `1918d7616cdb39ebdd5a01d6b6b6eed82667bbfff33b06b55f59536c20e99538` | `70809b9c2febd343ec459ae79be667892d6aa170a3dcc4c5b7bf0c914665392a` |
| Corridor | 0 | `24b330294cf15e014d25ae9eb0b883728e2e45dd999eb62b53f86c86a77aab07` | `129635aab9379aa49593210c5dabdbe0ff2a206f665f54f5c9c8fe931fd9af54` |
| Corridor | 1 | `a263206271e0126c0ec060b5ed7320ac76d848759980bce29dc8e1a37c350803` | `9a49722f9ccdade242d9e72b6707544a7b293e5d9850c0ab8bacaa526d75292b` |

The not-verified EPS-ER9-0002 head `ec71e2c9bfba2c2e00c9a11b5ed8d05370dda1d8` supplied these dev.8 identities. They are historical engineering-review input rather than active corrected-head evidence:

| Not-verified dev.8 scene | Episode | Boundary | Visibility events |
| --- | ---: | --- | --- |
| Single occluder | 0 | `203c074e745c47180b2be1c385712b2d4046319e1ba0251c8cc37957c85b57a7` | `b29978dbee23c77a5b05402f39b85650bdfa6149238a3220ef58dc91b47a62fc` |
| Single occluder | 1 | `db4a039507db60e6baf16e3f09a47ba0fb059b30836440fb7ed346c413be301b` | `6fb12d8fe4355fccb29d45befdc138fb5e20bf686d787f0c6d241a094c18b12d` |
| Corridor | 0 | `a24fb5965418145b8abb9ee9ec7f344fb514895a492a4902f6cc0b327077423b` | `ab89d8eb8329b634a35de470337a552baaff8b25ef2344fed28e57f44eb5fb4a` |
| Corridor | 1 | `02cca96eb7aa47dfbbae87a8c90eebe5c99b7d9686b6420c5050978f6d31f328` | `a9a5271000a73c2298ccc933c448ef1cc3ebbbace2cd4d14710091ae43230e44` |

The final reviewed dev.9 identities are:

| Final reviewed scene | Episode | Boundary | Visibility events |
| --- | ---: | --- | --- |
| Single occluder | 0 | `5a6a4792a3554d3b7670c73ac3e2c4d1ed7c438e12f2dabe8065f9dbc230a2c8` | `c41807b1245faa9fe1027584ca2dcc2e050956e6a4086bad783e74381abb611a` |
| Single occluder | 1 | `485aadab9af42e8ee5ca2194da02eff26626d2b887050675de4fcb192cad3aba` | `d9d2c91fe5fd10bf90fcff316dbf6c7956ef8e34baccb3b185971c0aa16192f9` |
| Corridor | 0 | `ddbb23daf1f7e17c936e74fd68ad63075385dcfb7c3b60f810acc835e346b235` | `4ff67bac73f9c01abe6fecd8263ff8d2eda1636b9222dd9f6fc7916c7e414d33` |
| Corridor | 1 | `df0c356d09e3cd601a0c896c5b4b2df34fb52b21a9dccaaefea2d7a4f4570448` | `9c8a8e1a57d6d5c43b56690812ceafe8a25c662a1e2abfd3d1c2ef035ca0cbb3` |

These final exact values are regression assertions rather than renderer-backend dispatches. Exact-head Ubuntu/OSMesa CI run `33047664727` at `24deb074d4ba0cf3e1044e466ecbbb6e0b2a4cc4` reproduced all four Windows/WGL boundary and visibility-event identity pairs. Complete dataset and RGB identities remain renderer-specific; disagreement in another environment must be preserved as a failing result rather than hidden behind backend-specific boundary/event expectations.

The final two-episode canonical counts, matched under locked Windows/WGL and Ubuntu/OSMesa, are:

| Scene/episode | Boundary kinds | Before-event codes | After-event codes |
| --- | --- | --- | --- |
| Single occluder 0 and 1 | attached 186; silhouette 504; junction-ambiguous 24; occluding 324 | stable 10382; deletion 229; exit 1523; boundary-ambiguous 839; no-surface 6227; unresolved 0 | stable 10382; accretion 229; entry 1523; boundary-ambiguous 839; no-surface 6227; unresolved 0 |
| Corridor 0 | attached 760; junction-ambiguous 12; occluding 0 | stable 11232; deletion 0; exit 7422; boundary-ambiguous 546; unresolved 0 | stable 18234; accretion 0; entry 0; boundary-ambiguous 966; unresolved 0 |
| Corridor 1 | attached 752; junction-ambiguous 12; occluding 0 | stable 11046; deletion 0; exit 7624; boundary-ambiguous 530; unresolved 0 | stable 18346; accretion 0; entry 0; boundary-ambiguous 854; unresolved 0 |

For each single-occluder episode and direction, the 229 causal pixels decompose into 203 foreground-panel-over-background pixels, 13 background-panel-over-support pixels, and 13 foreground-panel-over-support pixels. The support events are newly visible after localising attachment; all 839 code-3 pixels come from the analytic transport boundary band rather than pair-level contact.

## Final review evidence

The final method/schema posture is `analytic_oriented_boundary_ownership_v4`, `projected_compiled_contact_locus_v3`, `scene_attachment_public_contract_v4`, `compiled_axis_aligned_intersection_cell_v1`, `strict_forward_distance_greater_than_epsilon_v1`, `image_constraints_only_slack_strict_front_and_cell_bounds_exact_v1`, and `analytic_transport_boundary_causal_events_v2`, with transition and privileged instrumentation `0.1.0-dev.9`, dataset/episode manifest `0.1.0-dev.4`, and configuration `0.1.0-dev.2`.

The final strict witness contract requires an actual witness inside the closed compiled contact cell. Contact-cell parameters have exact `[0, 1]` bounds; the `1e-12` feasibility slack applies only to inclusive image-association inequalities, and neither the strict in-front plane nor the contact-cell bounds receive slack. Point, segment, rectangle, and overlap-volume cases are accepted only where an actual supported in-cell witness exists. Segment, rectangle, and overlap-volume cells with no strictly in-front in-cell point are rejected. Camera validation remains fail-closed.

Local Windows/WGL review evidence recorded 299 passed and 3 platform skips. Exact-head Ubuntu/OSMesa CI run `33047664727` passed locked installation, lint, formatting, typing, all 302 tests, and both two-episode generation, validation, and inspection workflows. Inspection composites were coherent. The exact identities and counts above matched across the two locked environments. The final correction deliberately changed boundary/event identities because the formal numerical contract changed, while leaving canonical boundary/event records and counts unchanged from the preceding dev.8 correction.

The accepted posture retains projected local attachment, ambiguity only at the actual analytic transport boundary band, equality of the public graph with all supported oriented owner/affected pairs and frame memberships, typed fail-closed permissions, and the privileged historical foreground/background counterfactual cross-check. Engineering and scientific review concern apparatus and construct validity; they do not establish that an ecological predictive state outperforms a baseline.

## Limitations and remaining work

- The method is bounded to static controlled MuJoCo planes and axis-aligned boxes in the two canonical scene families. Moving surfaces, arbitrary rotations/types, openings, T-junction scene semantics, and nested enclosures are not supported.
- Multi-surface junctions are intentionally ambiguous at the declared local lattice neighbourhood. No subpixel contour or component topology is inferred.
- Contact-locus association is bounded to compiled axis-aligned intersection cells and the declared half-pixel lattice band; it is not a general arbitrary-mesh contact oracle.
- Event ownership for an interior occluded sample is a consequence of exact analytic transport plus a supported owner/affected contour pair in the relevant target frame. Actual source/projected-target boundary-band samples take precedence. This slice does not trace a continuous boundary component to every transported ray.
- Cross-platform evidence is limited to the locked Windows/WGL and Ubuntu/OSMesa environments and does not imply stability under arbitrary drivers or dependency versions.
- Broader appearance assets, final evaluation seeds, and remaining full Gate 0B exit-criterion evidence and owner decisions remain future work.

This work does not implement a component split/merge oracle, ecological graph or state conversion, model interface, learned method, navigation, robot integration, Gate 0C, or Gate 0D model. Full Gate 0B completion is not claimed, Gate 0C is not authorised, and no scientific result is claimed.
