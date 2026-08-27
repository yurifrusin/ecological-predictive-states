# Milestone 0 implementation notes

## Scope and architecture

This implementation completes the repository foundation in Gate 0A and preserves the canonical single-occluder Gate 0B Slice 1, corridor Slice 2, analytic optical-transport Slice 3, and oriented-boundary and ecological visibility-event Slice 4. Slice 4 became canonical after exact-head dual independent review and owner approval at `24deb074d4ba0cf3e1044e466ecbbb6e0b2a4cc4`. It does not satisfy the full Gate 0B deliverable list in `MILESTONE_0.md` and makes no scientific claim.

The path is intentionally short:

```text
strict scene-family-discriminated YAML config
  -> single-occluder or sampled corridor MuJoCo XML
  -> lateral or forward fixed kinematic camera transition
  -> RGB/depth/raw segmentation instrumentation
  -> scene-specific privileged instrumentation
  -> single-occluder counterfactual occluder-exclusion segmentation
  -> episode-local opaque segmentation remap
  -> minimal segmentation-derived ecological annotations
  -> analytic forward/backward image-plane transport from compiled geometry
  -> analytic oriented edge-lattice ownership from local projected contact loci and counterfactual rays
  -> transport-causal ecological visibility-event maps and opaque pair summaries
  -> strict transition and manifest schemas
  -> permission-gated loader, validator, and inspection image
```

`src/epsbench/schema.py` is the public contract. Unknown schema and configuration fields are rejected. Dataset-relative POSIX paths are validated against traversal, Windows drive-qualified or drive-relative forms, and generation refuses to overwrite a non-empty directory. Inspection performs complete dataset validation before reading panels or creating an output image.

## Scenes and actions

`src/epsbench/sim/single_occluder.py` generates a small MuJoCo XML model using the official `mujoco` package. It contains a support plane, one opaque foreground panel, one opaque background surface, a directional light, and one fixed monocular camera. The executed action is a discrete lateral translation from `x=-0.35` to `x=0.35`; forward displacement and yaw are zero. MuJoCo world coordinates and camera matrices are retained only as privileged instrumentation.

The `base` and `alternate` appearance variants change procedurally specified surface colours while preserving geometry, camera trajectory, action, correspondence, neutral mask-change records, and the single-occluder counterfactual occlusion relation. Configuration validation rejects non-finite values, forward or yaw actions, zero lateral displacement, and action names with the wrong lateral sign. Whole-dataset validation separately checks the persisted before/after camera positions and proper, approximately orthonormal rotations against the configured action and the compiled MuJoCo camera orientation and field of view.

`src/epsbench/sim/corridor.py` generates a parametrically sized four-surface corridor containing the floor, left surface, right surface, and end surface. Width and length vary deterministically per episode; wall height, camera start, camera height, field of view, and positive forward displacement are strictly configured. The corridor is open above, but the configuration contract requires the walls to cover the complete optical field for the longest permitted scene. The camera translates only on the forward axis and must remain clear of the side and end walls with unchanged height and rotation. Whole-dataset validation independently resamples the geometry, recompiles the scene, binds each semantic surface name to its exact compiled raw geom ID, world position, and three compiled size dimensions, reconstructs public opaque segmentation from privileged raw renderer arrays, and compares the action with both camera artifacts and duplicated scene instrumentation. The same compiled contract is enforced for the single-occluder apparatus.

The scene configuration is a strict discriminator over `single_occluder` and `corridor`. Scene-specific geometry and action types make corridor fields invalid for the single occluder, single-occluder camera fields invalid for the corridor, and forward/lateral mixed actions invalid for both.

## Modality boundaries and loader permissions

- `SENSORY`: RGB and executed action.
- `ECOLOGICAL_ORACLE`: remapped surface regions, segmentation-derived boundary contacts, frame-normalised projected-image fractions, region correspondence, neutral raster mask changes, typed analytic optical transport with validity/reasons, sparse oriented boundary ownership, bounded ecological visibility-event maps/summaries, and typed occlusion availability/relations.
- `METRIC_BASELINE`: depth.
- `INSTRUMENTATION_ONLY`: camera world transform, raw MuJoCo geom IDs, raw simulator coordinates, sampled corridor geometry, and privileged generation records.
- `CONTROL_METADATA`: the compound transition and scene-family records that index explicitly typed records; neither is a learner input.

Every `DatasetLoader` is constructed with an explicit `ModalityPermissionSet`. It checks permissions before resolving or opening the requested transition or artifact. Analytic optical transport is a public ecological-oracle modality, and its dedicated loader returns the complete fixed-point forward/backward vectors, validity and reason masks, method metadata, and decoded-flow conveniences only after checking that modality. The boundary loader returns the complete typed sparse edge-lattice records. The event loader returns its typed capability/identity annotation, two code arrays, four aligned opaque-label arrays, and the label-to-opaque-`SurfaceId` mapping. The ecological view returns no `FrameRecord`, depth path, camera record, raw identifier, world-coordinate value, sampled corridor dimension, semantic apparatus name, privileged attachment/counterfactual evidence, scene-family field, or full manifest. Full manifest access is an explicit method requiring control-metadata, scene-family, and privileged-generation permissions. This is a typed cooperative API boundary, not a sandbox for hostile Python code; callers that deliberately bypass private attributes are outside its threat model.

Loading and validation share one fail-closed root-manifest resolver. `manifest.json` must be a single-link regular file whose resolved target remains inside the resolved dataset root; external or internal symbolic aliases, hard-link aliases, special files, and missing manifests are rejected before parsing. Every ordinary RGB, depth, and segmentation frame and every scene-specific counterfactual or raw-segmentation raster must also equal the resolved render width and height.

## Surface identifiers

MuJoCo geom IDs are read only during generation. A namespaced deterministic random generator produces a different random order, opaque `surface-<token>` identifier, and non-raw integer segmentation label for every episode. Ordinary segmentation arrays contain only zero (unlabelled renderer background) or those declared episode-local labels. The raw-ID-to-opaque-ID mapping exists only in `instrumentation.json`, whose modality is privileged generation data.

The same root seed and episode index reproduce the mapping exactly, but independently serialised episode indices use disjoint opaque identifiers even with identical geometry. Corridor episodes derive independent geometry-sampling, surface-remapping, and appearance namespaces from the episode seed. Whole-dataset validation enforces that surface-ID sets are disjoint across episodes. Numeric segmentation labels remain episode-local and may be reused by different episodes.

## Annotation derivation

Visible pixels and projected-image fractions are measured from remapped segmentation. `before_projected_image_fraction` and `after_projected_image_fraction` mean surface-labelled pixels divided by all image pixels; they are not fractions of a physical surface. Correspondence uses persistent opaque surface identity before and after and records exact before count, after count, and `same_image_coordinate_overlap_pixels`. That overlap is mask intersection at unchanged image coordinates, not optical-flow correspondence or an estimate of optical transformation.

`RegionMaskChange` reports only what aligned raster masks establish: gained image-coordinate pixels, lost image-coordinate pixels, region appearance, region disappearance, or an unchanged mask. A translation or projected expansion can therefore produce neutral gain/loss records without being called ecological accretion or deletion. Exact gain and loss counts are derived from same-coordinate mask differences. Slice 4 ecological visibility events remain a separate analytic contract: accretion/deletion requires canonical optical transport plus a supported opaque owner/affected pair from oriented boundary ownership; frame entry/exit, ambiguity, no-surface, and unresolved occlusion remain distinct codes.

Legacy `BoundaryStructure` records still count renderer-segmentation four-neighbour label transitions and aggregate contacts by opaque identifier. They remain a separate non-authoritative diagnostic and are not relabelled as oriented ownership. `OrientedBoundaryElement` instead uses the analytic four-neighbour pixel-centre edge lattice. Same-assignment edges are implicit `NO_BOUNDARY`; controlled-controlled edges are classified as attached, occluding, or multi-surface ambiguous; a controlled-to-uncontrolled edge is a controlled silhouette; and unresolved edges fail canonical generation. The exact attachment, continuation, coordinate, event, and limitation contracts are in `docs/GATE_0B_SLICE_4_BOUNDARY_EVENTS.md`.

The public occlusion graph is the complete canonical set of supported oriented owner/affected pairs and frame memberships. Single-occluder transitions use `oriented_boundary_ownership_with_counterfactual_crosscheck_v1`; corridor transitions use `oriented_boundary_ownership_complete_v2`, including available-empty when recomputation finds no occluding contours. For each single-occluder frame, generation also renders privileged `counterfactual_occluder_exclusion_v1` evidence with the candidate foreground occluder's MuJoCo render group disabled. This evidence independently cross-checks the designated foreground/background member of the complete graph; it does not limit graph completeness. Generation stores the counterfactual array, revealed-pixel count, and logical reveal-mask hash only in privileged instrumentation. Validation requires exactly the three named apparatus surfaces and a bijection among semantic apparatus names, raw IDs, and opaque surface IDs. It reconstructs ordinary raw segmentation from the public labels and privileged bijection, verifies the exact counterfactual footprint and reveal evidence, recomputes the complete graph, and requires equality plus designated-relation agreement.

Analytic optical transport uses centre-of-pixel rays and explicit intersections against compiled MuJoCo plane and oriented-box geometry. Under `finite_plane_visual_extent_v2`, each local plane axis uses `limit = geom_size + 16.0 * binary64_epsilon * max(1.0, geom_size)` and an inclusive comparison. The complete comparison policy and constants are typed and enter analytic identity. MuJoCo's infinite collision plane is not the optical surface. Public raster dimensions, FOV, camera position/rotation, and controlled geom identifiers are validated before allocation or indexing. Each direction records exact `int32` vectors at 1024 units per image pixel plus mandatory `uint8` validity and reason masks. RGB, depth, and rendered segmentation do not define the vectors or their identity. Full equations and limitations are in `docs/GATE_0B_SLICE_3_ANALYTIC_FLOW.md`.

Whole-dataset validation independently recompiles each scene and recomputes both directions, then compares every vector component, validity bit, and reason code. Privileged renderer diagnostics compare analytic assignments with raw renderer segmentation only away from analytic boundaries and require zero unexplained interior disagreement. The rejected 0.90 acceptance threshold is removed. These diagnostics are exact consistency alarms, not inputs to the analytic annotation or identity.

Inverse correspondence is tested at the transported location: `F(p) + B(p + F(p))`, not `F(p) + B(p)`. An exactly one-pixel lateral case indexes the discrete backward array at the destination and has zero fixed-point residual. Radial expansion is tested as a continuous-coordinate inverse at the expanded correspondence. No interpolation rule is declared for arbitrary subpixel dense-array destinations, so no discrete-resampling inverse statistic is claimed.

Attachment is local rather than a scene-pair label. `projected_compiled_contact_locus_v3` derives a privileged axis-aligned 3D intersection cell for every compiled declared contact and tests individual sample-to-sample image edges against its analytic pinhole projection, including zero-dimensional point cells. Only associated edges are attached; other edges for the same globally touching pair undergo counterfactual continuation and may be panel-owned occluding contours. Projection requires forward distance strictly greater than `1e-12`; the `1e-12` inclusive slack applies only to image-association constraints, while the strict front plane and closed contact-cell parameter bounds have zero slack. A feasibility witness must therefore be an actual member of the compiled contact cell. The exported projection entry validates camera pose/FOV even when the edge raster is empty. Actual analytic source/projected-target boundary-band samples alone receive `ANALYTIC_BOUNDARY_AMBIGUOUS`; global pair contact does not create event ambiguity. The exact method, numerical, endpoint, graph-completeness, and limitation contracts are in `docs/GATE_0B_SLICE_4_BOUNDARY_EVENTS.md`.

## Hash domains

All hashes are SHA-256 and lowercase hexadecimal.

- Each image or array `logical_sha256` covers canonical `{dtype, shape}`, a zero separator, and contiguous semantic array bytes.
- Each JSON `logical_sha256` covers sorted, compact, UTF-8 canonical JSON without the file's final newline.
- Every `file_sha256` covers the complete container file bytes. It is separate from logical array or JSON identity.
- `analytic_transport_sha256` separately covers method/version, image-coordinate convention, fixed-point quantisation, reason domain, the typed surface-intersection/finite-plane/target-visibility rules and all numerical tolerance values, analytic boundary rule, ordered frame indices, and the logical hashes of both vector, validity, and reason triplets. It excludes seed, appearance, opaque IDs, segmentation, RGB, depth, renderer backend, provenance, paths, and file-container hashes.
- `oriented_boundary_sha256` separately covers the image edge-lattice convention, boundary/owner domains, attachment and counterfactual method contracts, numerical-rule hash, frame indices, and ordered public opaque records. It excludes appearance, RGB/depth, renderer, semantic/raw/metric state, paths, and provenance.
- `visibility_event_sha256` separately covers the directional event code domains and logical array hashes, exact opaque owner/affected summaries, whole-surface events, explicit component-topology unavailability, boundary identity, analytic transport identity, and frame indices, with the same privileged and appearance exclusions.
- `ecological_label_sha256` covers schema version, executed action values, opaque surface references, logical segmentation hashes, visibility states, region correspondence, neutral region-mask changes, separate oriented-boundary and visibility-event domains/identities, typed occlusion availability/relations, boundary structures, and the available analytic transport annotation while retaining its separate identity. Known-empty and unavailable occlusion therefore have different identities.
- The ecological-label domain excludes RGB, depth, colour/appearance identity, camera transforms, raw simulator IDs, raw/world coordinates, paths, timestamps, and hostnames.
- `scene_content_sha256` covers non-appearance scene layout and camera/action trajectory. The single-occluder domain contains the scene family, fixed apparatus version, exact support/background/occluder shapes and poses, before/after camera positions and orientation rule, field of view, and action. The corridor domain contains the scene family, apparatus version and surface membership, sampled width and length, wall height and fixed thicknesses, before/after camera positions and orientation rule, camera height, field of view, and action. Both domains exclude episode seed, appearance, opaque IDs, segmentation labels, timestamps, and source/renderer provenance. Episode seed remains separately recorded in the manifest and privileged generation evidence.
- `dataset_logical_sha256` covers schema/generator versions, scene family, root seed, resolved-config logical hash, appearance variant, and each episode's stable identifier, derived seed, scene-content hash, ecological-label hash, analytic-transport hash, and RGB logical hashes. It excludes renderer provenance, file-container hashes, timestamps, hostname, and absolute paths.
- `source_provenance_sha256` independently covers the typed inline source record: sanitized repository reference, exact commit and truthful dirty state, dirty-diff hash when applicable, lock-file identity, governing-document hashes, package version, and Python version. HTTP credentials and query/fragment data are removed; SSH user information is removed; file and local-path origins are replaced with a non-path redaction marker. Git unavailability is recorded as unavailable with a reason and never represented as clean.
- `renderer_execution_provenance_sha256` independently covers stable renderer/execution facts: MuJoCo and NumPy versions, renderer, backend, and operating system. These facts remain outside the scientific content domain.
- `content_provenance_binding_sha256` hashes the already-computed dataset logical hash together with the already-computed source-provenance and renderer/execution-provenance hashes. This binds content and both provenance domains without making any domain self-referential. The existing scientific content identity deliberately continues to exclude provenance.
- Volatile time, hostname, and Python build text live only in unreferenced `run.json`; they do not participate in manifest or scientific identity.

The final Slice 3 schema and method matrix advances only changed wire contracts:

| Contract | Final version or method |
| --- | --- |
| Single-occluder config | `0.1.0-dev.2` |
| Corridor config | `0.1.0-dev.2` |
| Dataset manifest | `0.1.0-dev.3` |
| Transition | `0.1.0-dev.5` |
| Single-occluder privileged instrumentation | `0.1.0-dev.5` |
| Corridor privileged instrumentation | `0.1.0-dev.5` |
| Analytic transport | `analytic_static_scene_transport_v3` |
| Compiled intersection | `compiled_plane_and_oriented_box_nearest_hit_v3` |
| Finite-plane extent | `finite_plane_visual_extent_v2` |

The Slice 3 dataset manifest stayed at `0.1.0-dev.3` because its wire shape was unchanged. Transition and both scene-specific privileged-instrumentation schemas advanced for method v3 and the complete finite-edge comparison contract. Slice 1 used transition `0.1.0-dev.1`; its pinned episode-0 ecological hash was `8f7c7a7e8f70f9e84bf2f256ecf93f8d94f5327abe536c01a6b8f7e87aef3df0`. Slice 2 used `0.1.0-dev.2`; the scientifically rejected infinite-plane PR #5 revision used transition and instrumentation `0.1.0-dev.3`; the finite-plane engineering-reviewed head `eee0eaa4b1de50b4dea1e391dba6f32b588e763c` used `0.1.0-dev.4`. The first Slice 4 scientific correction at `66f23ccc88fb1003411069e50ed6b5051754bbf8` used transition and instrumentation `0.1.0-dev.7`. The not-verified EPS-ER9-0002 head `ec71e2c9bfba2c2e00c9a11b5ed8d05370dda1d8` used `0.1.0-dev.8`. The bounded follow-up advances transition and instrumentation to `0.1.0-dev.9`, attachment method v3, boundary method v4, and an explicit image-only feasibility-slack field/rule while retaining manifest `0.1.0-dev.4`; configuration and manifest wire shapes did not change.

The final Slice 4 method/schema matrix is:

| Contract | Final version or method |
| --- | --- |
| Single-occluder and corridor configuration | `0.1.0-dev.2` |
| Dataset and episode manifest | `0.1.0-dev.4` |
| Transition | `0.1.0-dev.9` |
| Privileged instrumentation | `0.1.0-dev.9` |
| Oriented boundary | `analytic_oriented_boundary_ownership_v4` |
| Projected attachment | `projected_compiled_contact_locus_v3` |
| Public attachment contract | `scene_attachment_public_contract_v4` |
| Contact manifold | `compiled_axis_aligned_intersection_cell_v1` |
| Projection in-front rule | `strict_forward_distance_greater_than_epsilon_v1` |
| Feasibility rule | `image_constraints_only_slack_strict_front_and_cell_bounds_exact_v1` |
| Visibility events | `analytic_transport_boundary_causal_events_v2` |

Neither meaningful migration is byte compatible: canonical transition JSON, ecological identity domains, artifact hashes, dataset hashes, and schema validation change. Under historical `0.1.0-dev.2`, the expected single-occluder episode-0 ecological hash is `0210bdbce412c6cf199c1cad58b5e8831b97d8a112a0e82f285b6ea1c20df4a3` in both tested environments. Historical corridor raster boundaries differ between Windows/WGL and Ubuntu/OSMesa, so exact regression values remain recorded by backend: Windows/WGL episode 0/1 are `92e89e4e0eb23b2a50a39cb3803c490654899531a000a3c1ef139e875177f2f8` and `ea231fe400fabfeb1afea6f9ba58450700734d6b539cfd0a3d540b7ad3345980`; Ubuntu/OSMesa episode 0/1 are `8ca92d6161acc2029421f1182491a96837f01058486fb5ee0c0189a1c2ff9a22` and `65ed70d5ad141313070978217d84a73e8504c17be95be193568d569940e71189`. No public `0.1.0` scientific dataset is implied.

The scientifically rejected PR #5 v1 analytic identities remain historical evidence: `e6d5fc664957c4e68b3c249ce5bd32113755b1c86d5fff8dbeaafc75b550c7cd` for both canonical single-occluder episodes, `12ea54fea0b8d9716d12189fcba89397156f7c8111fcb7a488702ffb8240f3bf` for corridor episode 0, and `9b41e6780bbf89654cda8c5f6d5f4d6326d64bac2594a1afb12db883f91395b6` for corridor episode 1. Their earlier Windows/WGL and Ubuntu/OSMesa equality does not rescue the rejected infinite-plane method.

The reviewed v2 analytic identities remain historical evidence for exact head `eee0eaa4b1de50b4dea1e391dba6f32b588e763c`: `479d4540835dcc5d204e530766edfcc4bd74b971cc8efe9cbe391317d4ca6740` for both single-occluder episodes, `ddb4dff0fba18d89cd6c15eb988e672c93988d3d4eda2ae625ab317d36631015` for corridor episode 0, and `a276190abe142bd6859cd0e29983964cbdd17c2cbbf291141ed6f32c6c3c5007` for corridor episode 1. Method v3 deliberately changes analytic identity by binding the complete finite-edge policy, while all 24 canonical v2/v3 vector, validity, and reason arrays remained byte-identical.

The final v3 identities are `77821c734e4a5316851b9e57417a014e8caf292f568314da05e2493608960832` for both single-occluder episodes, `ffa9b31e91da7a4cdb68ce938beba901beecb3973684bc303d3e362fa01fa09a` for corridor episode 0, and `64706a77347aa2a98e52d6da023ba80f7a04d1b7a94eaa809124c9b7fe9fca0a` for corridor episode 1.

The v1 boundary/event identities at scientifically rejected PR #9 head `c82ac9d68463fea82ad132194007549ed955adf3` remain historical evidence only. Single-occluder episode 0/1 boundary identities were `ea99e38626142a7c3b1845c234e5b577493fbf8ff55e40169a96bffaafe84f66` and `c690400c1c1496da23b4d8ee91af6b52d7e163a415fb7929ffa4494061becbef`; their event identities were `c35fe87f68dde0b282f8069a6700895d135e956a3b5ba2190f3b606f259b8ebb` and `c9032eb0250598e95902b3c4d5001ea724a48abf6cd66158fbd5374b2d312f2b`. Corridor episode 0/1 boundary identities were `77e4aba8aa763827ce3c3a1e20ebc03f134c0c44edb127235f071b32c769f503` and `fd5c4360c0cc03d6be11bd9917de4909aa6f26129f95d69364d793c42e8e9ab3`; their event identities were `2db60ab41fe0f93f1fd574c0d1218d374727199b51eb3bf15625f6264cefbf0a` and `480e283e5eecc7fc21e3a21f56e114f44c69460681e167b9292c45fa8e7b5a15`. The later dev.7 head `66f23ccc88fb1003411069e50ed6b5051754bbf8` produced single-occluder boundary identities `da74e273f5e6786c9093709c07147ea2666db58bb64d687665dbf563e5f9e717` and `1918d7616cdb39ebdd5a01d6b6b6eed82667bbfff33b06b55f59536c20e99538`, event identities `dfd7fddd2a460ed1d3147bfca55c9e9034269c4ad0cecdc5de4751b1678cd3d6` and `70809b9c2febd343ec459ae79be667892d6aa170a3dcc4c5b7bf0c914665392a`, corridor boundary identities `24b330294cf15e014d25ae9eb0b883728e2e45dd999eb62b53f86c86a77aab07` and `a263206271e0126c0ec060b5ed7320ac76d848759980bce29dc8e1a37c350803`, and event identities `129635aab9379aa49593210c5dabdbe0ff2a206f665f54f5c9c8fe931fd9af54` and `9a49722f9ccdade242d9e72b6707544a7b293e5d9850c0ab8bacaa526d75292b`. The dev.8 head `ec71e2c9bfba2c2e00c9a11b5ed8d05370dda1d8` remains historical because EPS-ER9-0002 was not verified; its identities and the final reviewed dev.9 identities are recorded in `docs/GATE_0B_SLICE_4_BOUNDARY_EVENTS.md`. Exact head `24deb074d4ba0cf3e1044e466ecbbb6e0b2a4cc4` received `ENGINEERING_PASS` and renewed `SCIENTIFIC_PASS`; exact-head Ubuntu/OSMesa CI run `33047664727` succeeded.

Inspection output is encoded completely before publication. A same-directory temporary file is flushed and atomically linked to the requested path without replacement; existing files, symbolic or hard links, directories, incompatible parents, and create-time races fail closed. Temporary files are removed after either outcome.

## Determinism and rendering limits

The same seed/configuration produces byte-identical manifests and transition annotations in the declared environment fingerprint. The fixed-point analytic transport is separately tested for deterministic bytes and appearance/opaque-ID independence. The manifest records MuJoCo and NumPy versions, renderer name, selected GL backend, and operating system. The lock file pins the complete Python dependency environment.

Cross-platform RGB byte identity is not claimed. At the original reviewed schema, Windows/WGL and Ubuntu/OSMesa produced different dataset logical hashes while sharing episode-0 ecological hash `3de23fb70a1f68934b3dbb81a8929434ecc9b81c92c0fa5d55c5138e36963419`. Exact-head CI run `32917034763` at `f7faa8321f674d084e21808a4666c5f61fd5180a` confirmed the later Slice 1 transition-`0.1.0-dev.1` hash recorded above for the tested Windows/WGL and Ubuntu/OSMesa environments. Local Windows/WGL evidence and exact-head Ubuntu/OSMesa run `32929169048` at `446934899e1e814697e79f3d8b87506194176012` exposed and supplied the distinct `0.1.0-dev.2` corridor regression values above; that run failed at the then-unpinned values and skipped its dataset-command step. Run `32929628787` subsequently passed all checks and dataset commands on exact head `5451f21c6fd7df99c41758bdea647954b556c18a`. The later engineering request for changes applies to that successful head, and any correction head requires renewed review and CI. None of this establishes cross-platform RGB or corridor ecological identity, stability across arbitrary drivers, stability across future renderer or dependency versions, or stability across later scene families. Pixel/container identity, ecological-label identity, scene-content identity, and provenance remain distinct claims.

The corridor uses the same identity separation. Its solid-colour appearance rerender changes RGB while preserving scene-content and ecological-label identity for fixed geometry, camera, and action. The canonical single-occluder scene-content hash is `57e8b2f1df42d0a3d13e2fe3403d903a24bf7d8b5b7ce410291afcd091fdc351`; corridor episode-0 and episode-1 scene-content hashes are `16a1f5a214577c3d189be6a2c0e47eb4ee3dbe0dfd913b4739b3c6b8832b9525` and `1de8d8a877a29695958917251e76044bc267616bd4da5050ac49ea909e65bffd` for the canonical configuration. A seed-only change with fixed sampled content does not change these domains.

## PR #3 closeout evidence

The final reviewed implementation head was `4eb754e16dd8bb80eb0873be01060f39fddcbe44`; it received `ENGINEERING_PASS` and `SCIENTIFIC_PASS`. Exact-head Ubuntu CI run `32939498541` passed all 149 tests and both the single-occluder and corridor dataset workflows.

The final reviewed apparatus independently checks compiled positions, sizes, camera pose, and field of view. Resolved render dimensions are bound to every public and privileged raster artifact. Full manifest access is privileged through the typed API, and root-manifest aliases and escapes fail closed. The scene-content and ecological-identity hashes remain the values already recorded above.

This evidence does not establish full Gate 0B completion, authorise Gate 0C, or establish a scientific result.

## PR #5 closeout evidence

The final reviewed implementation head was `3086bc2eccc4b7492eb8a77660572e879a04ef0b`; it received `ENGINEERING_PASS` and `SCIENTIFIC_PASS`. Exact-head Ubuntu/OSMesa CI run `32971294963` passed locked installation, lint, formatting, typing, all 241 tests, and both two-episode generation, validation, and inspection workflows. Local Windows/WGL review evidence recorded 238 passed and 3 platform skips.

The accepted exact-edge policy is `inclusive_extent_plus_scaled_binary64_epsilon_v1`: each finite-plane local-axis limit is `extent + 16.0 * 2.220446049250313e-16 * max(1.0, extent)`. It is the declared binary64 inclusive comparison contract, not an ecological enlargement of the surface. Public analytic raster dimensions, FOV, camera position/rotation, and controlled geom identifiers fail closed before allocation or indexing. Inspection uses race-safe atomic no-replace publication.

At the PR #5 closeout, canonical evidence retained zero unexplained renderer-interior disagreement and correspondence-indexed inverse tests. Full oriented boundary ownership and ecological visibility events were then unavailable; Slice 4 subsequently made the bounded reviewed contract canonical. Arbitrary-subpixel dense-array inverse interpolation, cross-platform claims beyond locked Windows/WGL and Ubuntu/OSMesa, full Gate 0B, Gate 0C, and model work remain unavailable or unimplemented. No scientific result is claimed.

## PR #9 closeout evidence

The final reviewed implementation head was `24deb074d4ba0cf3e1044e466ecbbb6e0b2a4cc4`; it received `ENGINEERING_PASS` and renewed `SCIENTIFIC_PASS` after the preserved request-changes and correction history. Exact-head Ubuntu/OSMesa CI run `33047664727` passed locked installation, lint, formatting, typing, all 302 tests, and both two-episode generation, validation, and inspection workflows. Local Windows/WGL review evidence recorded 299 passed and 3 platform skips, coherent inspection composites, and matching final identities and canonical counts.

The final boundary/event identity pairs are single-occluder episode 0 `5a6a4792a3554d3b7670c73ac3e2c4d1ed7c438e12f2dabe8065f9dbc230a2c8` / `c41807b1245faa9fe1027584ca2dcc2e050956e6a4086bad783e74381abb611a`, single-occluder episode 1 `485aadab9af42e8ee5ca2194da02eff26626d2b887050675de4fcb192cad3aba` / `d9d2c91fe5fd10bf90fcff316dbf6c7956ef8e34baccb3b185971c0aa16192f9`, corridor episode 0 `ddbb23daf1f7e17c936e74fd68ad63075385dcfb7c3b60f810acc835e346b235` / `4ff67bac73f9c01abe6fecd8263ff8d2eda1636b9222dd9f6fc7916c7e414d33`, and corridor episode 1 `df0c356d09e3cd601a0c896c5b4b2df34fb52b21a9dccaaefea2d7a4f4570448` / `9c8a8e1a57d6d5c43b56690812ceafe8a25c662a1e2abfd3d1c2ef035ca0cbb3`.

The canonical counts are 186 attached, 504 silhouette, 24 junction-ambiguous, and 324 occluding boundary elements for each single-occluder episode, with 229 causal and 839 analytic-boundary-ambiguous pixels in each direction. Corridor episode 0 has 760 attached and 12 junction-ambiguous elements; episode 1 has 752 attached and 12 junction-ambiguous elements; both have zero occluding contours and zero causal events. Complete directional stable, frame entry/exit, boundary-ambiguous, and no-surface counts remain recorded in the Slice 4 contract.

The final numerical contract requires an actual witness inside the closed compiled contact cell. Contact-cell parameters have exact `[0, 1]` bounds; the `1e-12` feasibility slack applies only to inclusive image-association inequalities; and the strict in-front plane and cell bounds receive no slack. Segment, rectangle, and overlap-volume cells with no strictly in-front actual in-cell point are rejected, while point, segment, rectangle, and overlap-volume cases are accepted only when an actual supported in-cell witness exists. Local projected attachment, actual-boundary ambiguity, complete public graph equality, typed fail-closed permissions, and metric/identity leakage boundaries remain intact.

The review evidence is bounded to the two canonical scene families, static planes and axis-aligned boxes, the declared half-pixel association band, and the locked Windows/WGL and Ubuntu/OSMesa environments. Component topology, continuous boundary-component tracing, broader scenes and appearance assets, final evaluation seeds, Gate 0C, and models remain outside scope. The review establishes apparatus and construct validity, not a comparative scientific result.

## Remaining Gate 0B work

- Complete the remaining Gate 0B exit-criterion evidence and obtain separate owner authority before any full Gate 0B completion decision.
- Decide whether later authorised work should define component topology, continuous boundary-component attribution, or broader scene geometry. Slice 4 deliberately leaves these unavailable.
- Extend the reviewed cross-platform posture beyond the locked Windows/WGL and Ubuntu/OSMesa environments.
- Decide and freeze broader appearance assets and final evaluation seeds before comparative work.

Gate 0B completion is therefore not claimed. Gate 0C should not begin until these items and every Gate 0B exit criterion have owner-reviewed evidence.

## Why no model exists

The authorised task is infrastructure and data-contract hygiene. A model would advance prematurely into Gate 0D, obscure leakage and ontology errors behind training behaviour, and violate the explicit handoff. No neural model, LLM, or learned extractor is implemented.

## Supplied document integrity

The attached documents were copied without substantive or byte-level changes. Source and retained SHA-256 values match:

| Document | SHA-256 |
| --- | --- |
| `RESEARCH_CHARTER.md` | `4e15b97c6041e12a33bdb52fba5297672a2181df68bceb9a6df56395811b3b06` |
| `EPS_BENCH_V0.md` | `6c6cca701085a637e978d6e26113e7f90752e37d94773743e7be52b565219505` |
| `MILESTONE_0.md` | `8cff3844fabc69e7cb4edbc6bbe59a7d8b41b067521674df4071b8decbf4b157` |
| `CODEX_HANDOFF.md` | `829b96ee9938f65084ba2313d89b81c9e45ac0cacaf7befc8680657025d6e367` |

No substantive conflict was silently resolved. The milestone document describes full Gate 0B, while the prompt and handoff authorise only its smallest single-occluder slice; the narrower scope governs this change.
