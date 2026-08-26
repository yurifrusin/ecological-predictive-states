# Milestone 0 implementation notes

## Scope and architecture

This implementation completes the repository foundation in Gate 0A, preserves the reviewed single-occluder Gate 0B slice, and adds the separately authorised minimal corridor slice. It does not satisfy the full Gate 0B deliverable list in `MILESTONE_0.md` and makes no scientific claim.

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
- `ECOLOGICAL_ORACLE`: remapped surface regions, segmentation-derived boundary contacts, frame-normalised projected-image fractions, region correspondence, neutral raster mask changes, typed ecological-visibility-event availability, and typed occlusion availability/relations.
- `METRIC_BASELINE`: depth.
- `INSTRUMENTATION_ONLY`: camera world transform, raw MuJoCo geom IDs, raw simulator coordinates, sampled corridor geometry, and privileged generation records.
- `CONTROL_METADATA`: the compound transition and scene-family records that index explicitly typed records; neither is a learner input.

Every `DatasetLoader` is constructed with an explicit `ModalityPermissionSet`. It checks permissions before resolving or opening the requested artifact. The ecological view returns no `FrameRecord`, depth path, camera record, raw identifier, world-coordinate value, sampled corridor dimension, semantic apparatus name, scene-family field, or full manifest. Full manifest access is an explicit method requiring control-metadata, scene-family, and privileged-generation permissions. This is a typed cooperative API boundary, not a sandbox for hostile Python code; callers that deliberately bypass private attributes are outside its threat model.

Loading and validation share one fail-closed root-manifest resolver. `manifest.json` must be a single-link regular file whose resolved target remains inside the resolved dataset root; external or internal symbolic aliases, hard-link aliases, special files, and missing manifests are rejected before parsing. Every ordinary RGB, depth, and segmentation frame and every scene-specific counterfactual or raw-segmentation raster must also equal the resolved render width and height.

## Surface identifiers

MuJoCo geom IDs are read only during generation. A namespaced deterministic random generator produces a different random order, opaque `surface-<token>` identifier, and non-raw integer segmentation label for every episode. Ordinary segmentation arrays contain only zero (unlabelled renderer background) or those declared episode-local labels. The raw-ID-to-opaque-ID mapping exists only in `instrumentation.json`, whose modality is privileged generation data.

The same root seed and episode index reproduce the mapping exactly, but independently serialised episode indices use disjoint opaque identifiers even with identical geometry. Corridor episodes derive independent geometry-sampling, surface-remapping, and appearance namespaces from the episode seed. Whole-dataset validation enforces that surface-ID sets are disjoint across episodes. Numeric segmentation labels remain episode-local and may be reused by different episodes.

## Annotation derivation

Visible pixels and projected-image fractions are measured from remapped segmentation. `before_projected_image_fraction` and `after_projected_image_fraction` mean surface-labelled pixels divided by all image pixels; they are not fractions of a physical surface. Correspondence uses persistent opaque surface identity before and after and records exact before count, after count, and `same_image_coordinate_overlap_pixels`. That overlap is mask intersection at unchanged image coordinates, not optical-flow correspondence or an estimate of optical transformation.

`RegionMaskChange` reports only what aligned raster masks establish: gained image-coordinate pixels, lost image-coordinate pixels, region appearance, region disappearance, or an unchanged mask. A translation or projected expansion can therefore produce neutral gain/loss records without being called ecological accretion or deletion. Exact gain and loss counts are derived from same-coordinate mask differences. Actual ecological visibility events remain a separate typed annotation with `status: unavailable` and reason category `optical_transport_and_boundary_ownership_unavailable`. Dense optical transport and oriented boundary ownership are not implemented, so no accretion/deletion values are fabricated.

Boundary records count four-neighbour segmentation label transitions. Contacts between two declared surfaces are aggregated by opaque identifier.

The current occlusion oracle uses the appearance-invariant rule `counterfactual_occluder_exclusion_v1`. For each frame, generation renders ordinary raw geom segmentation and a privileged second segmentation with the candidate foreground occluder's MuJoCo render group disabled. A foreground-to-background relation is emitted for a frame only when at least one ordinary occluder pixel becomes the candidate background in the counterfactual segmentation. Generation stores the counterfactual array, revealed-pixel count, and logical reveal-mask hash only in privileged instrumentation; the ecological record contains only opaque surface IDs and derived frame membership. Validation requires exactly the three named apparatus surfaces and a bijection among semantic apparatus names, raw IDs, and opaque surface IDs. It reconstructs ordinary raw segmentation from the public labels and privileged bijection, requires the counterfactual to contain no occluder pixels, requires its changed-pixel mask to equal the ordinary occluder footprint exactly, and independently recomputes the background reveal evidence and ecological relation.

Dense optical flow is not fabricated. Its schema status is `unavailable` with a typed reason in every transition.

Occlusion uses a discriminated availability contract. Single-occluder transitions contain `status: available`, the `counterfactual_occluder_exclusion_v1` rule, and the independently checked relations. Corridor transitions contain `status: unavailable` and reason category `oriented_corridor_occlusion_oracle_unavailable` because no controlled oracle establishes oriented occlusion at corridor surface contacts. `status: available` with an empty relation list is reserved for a future oracle-supported known-empty result; it is not interchangeable with unavailable. Four-neighbour contacts are retained as boundary structure but are not promoted to occlusion or full boundary ownership.

## Hash domains

All hashes are SHA-256 and lowercase hexadecimal.

- Each image or array `logical_sha256` covers canonical `{dtype, shape}`, a zero separator, and contiguous semantic array bytes.
- Each JSON `logical_sha256` covers sorted, compact, UTF-8 canonical JSON without the file's final newline.
- Every `file_sha256` covers the complete container file bytes. It is separate from logical array or JSON identity.
- `ecological_label_sha256` covers schema version, executed action values, opaque surface references, logical segmentation hashes, visibility states, region correspondence, neutral region-mask changes, the ecological-visibility-event availability record, typed occlusion availability/relations, boundary structures, and the explicit dense-flow availability record. Known-empty and unavailable occlusion therefore have different identities.
- The ecological-label domain excludes RGB, depth, colour/appearance identity, camera transforms, raw simulator IDs, raw/world coordinates, paths, timestamps, and hostnames.
- `scene_content_sha256` covers non-appearance scene layout and camera/action trajectory. The single-occluder domain contains the scene family, fixed apparatus version, exact support/background/occluder shapes and poses, before/after camera positions and orientation rule, field of view, and action. The corridor domain contains the scene family, apparatus version and surface membership, sampled width and length, wall height and fixed thicknesses, before/after camera positions and orientation rule, camera height, field of view, and action. Both domains exclude episode seed, appearance, opaque IDs, segmentation labels, timestamps, and source/renderer provenance. Episode seed remains separately recorded in the manifest and privileged generation evidence.
- `dataset_logical_sha256` covers schema/generator versions, scene family, root seed, resolved-config logical hash, appearance variant, and each episode's stable identifier, derived seed, scene-content hash, ecological-label hash, and RGB logical hashes. It excludes renderer provenance, file-container hashes, timestamps, hostname, and absolute paths.
- `source_provenance_sha256` independently covers the typed inline source record: sanitized repository reference, exact commit and truthful dirty state, dirty-diff hash when applicable, lock-file identity, governing-document hashes, package version, and Python version. HTTP credentials and query/fragment data are removed; SSH user information is removed; file and local-path origins are replaced with a non-path redaction marker. Git unavailability is recorded as unavailable with a reason and never represented as clean.
- `renderer_execution_provenance_sha256` independently covers stable renderer/execution facts: MuJoCo and NumPy versions, renderer, backend, and operating system. These facts remain outside the scientific content domain.
- `content_provenance_binding_sha256` hashes the already-computed dataset logical hash together with the already-computed source-provenance and renderer/execution-provenance hashes. This binds content and both provenance domains without making any domain self-referential. The existing scientific content identity deliberately continues to exclude provenance.
- Volatile time, hostname, and Python build text live only in unreferenced `run.json`; they do not participate in manifest or scientific identity.

The dataset, resolved-configuration, privileged-instrumentation, and ecological-transition schema identifiers are coherently `0.1.0-dev.2`. Slice 1 used transition `0.1.0-dev.1`; its pinned episode-0 ecological hash was `8f7c7a7e8f70f9e84bf2f256ecf93f8d94f5327abe536c01a6b8f7e87aef3df0`. Slice 2 introduced forward actions and its original annotation posture changed the transition contract, while this correction replaces causal-sounding raster events and a bare occlusion tuple with explicit typed semantics. Keeping the old transition version would therefore be false.

The migration is not byte compatible: canonical transition JSON, ecological identity domains, artifact hashes, dataset hashes, and schema validation all change. Scientific meaning is preserved for unchanged Slice 1 observations: the same pixels, projected-image fractions, correspondence counts, boundary contacts, action, and counterfactual single-occluder evidence remain, but their claims are represented more precisely. Under `0.1.0-dev.2`, the expected single-occluder episode-0 ecological hash is `0210bdbce412c6cf199c1cad58b5e8831b97d8a112a0e82f285b6ea1c20df4a3` in both tested environments. Corridor raster boundaries differ between Windows/WGL and Ubuntu/OSMesa, so exact regression values are pinned to the recorded backend rather than falsely claimed as cross-platform: Windows/WGL episode 0/1 are `92e89e4e0eb23b2a50a39cb3803c490654899531a000a3c1ef139e875177f2f8` and `ea231fe400fabfeb1afea6f9ba58450700734d6b539cfd0a3d540b7ad3345980`; Ubuntu/OSMesa episode 0/1 are `8ca92d6161acc2029421f1182491a96837f01058486fb5ee0c0189a1c2ff9a22` and `65ed70d5ad141313070978217d84a73e8504c17be95be193568d569940e71189`. No public `0.1.0` scientific dataset is implied.

## Determinism and rendering limits

The same seed/configuration produces byte-identical manifests and transition annotations in the declared environment fingerprint. The manifest records MuJoCo and NumPy versions, renderer name, selected GL backend, and operating system. The lock file pins the complete Python dependency environment.

Cross-platform RGB byte identity is not claimed. At the original reviewed schema, Windows/WGL and Ubuntu/OSMesa produced different dataset logical hashes while sharing episode-0 ecological hash `3de23fb70a1f68934b3dbb81a8929434ecc9b81c92c0fa5d55c5138e36963419`. Exact-head CI run `32917034763` at `f7faa8321f674d084e21808a4666c5f61fd5180a` confirmed the later Slice 1 transition-`0.1.0-dev.1` hash recorded above for the tested Windows/WGL and Ubuntu/OSMesa environments. Local Windows/WGL evidence and exact-head Ubuntu/OSMesa run `32929169048` at `446934899e1e814697e79f3d8b87506194176012` exposed and supplied the distinct `0.1.0-dev.2` corridor regression values above; that run failed at the then-unpinned values and skipped its dataset-command step. Run `32929628787` subsequently passed all checks and dataset commands on exact head `5451f21c6fd7df99c41758bdea647954b556c18a`. The later engineering request for changes applies to that successful head, and any correction head requires renewed review and CI. None of this establishes cross-platform RGB or corridor ecological identity, stability across arbitrary drivers, stability across future renderer or dependency versions, or stability across later scene families. Pixel/container identity, ecological-label identity, scene-content identity, and provenance remain distinct claims.

The corridor uses the same identity separation. Its solid-colour appearance rerender changes RGB while preserving scene-content and ecological-label identity for fixed geometry, camera, and action. The canonical single-occluder scene-content hash is `57e8b2f1df42d0a3d13e2fe3403d903a24bf7d8b5b7ce410291afcd091fdc351`; corridor episode-0 and episode-1 scene-content hashes are `16a1f5a214577c3d189be6a2c0e47eb4ee3dbe0dfd913b4739b3c6b8832b9525` and `1de8d8a877a29695958917251e76044bc267616bd4da5050ac49ea909e65bffd` for the canonical configuration. A seed-only change with fixed sampled content does not change these domains.

## Remaining Gate 0B work

- Implement geometry-derived or analytic dense optical flow and its alignment tests.
- Define and serialise full oriented boundary ownership rather than only boundary contacts plus an occlusion edge.
- Extend cross-platform ecological-label expectations beyond the currently tested Windows/WGL and Ubuntu/OSMesa locked environments, and define any required tolerances for later scenes.
- Expand procedural appearances beyond solid-colour variants if texture-frequency testing is adopted.

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
