# Milestone 0 implementation notes

## Scope and architecture

This implementation completes the repository foundation in Gate 0A and only the prompt-authorised single-occluder vertical slice of Gate 0B. It does not satisfy the full Gate 0B deliverable list in `MILESTONE_0.md` and makes no scientific claim.

The path is intentionally short:

```text
strict YAML config
  -> generated MuJoCo XML and fixed kinematic camera transition
  -> RGB/depth/raw segmentation instrumentation
  -> privileged counterfactual occluder-exclusion segmentation
  -> episode-local opaque segmentation remap
  -> minimal segmentation-derived ecological annotations
  -> strict transition and manifest schemas
  -> permission-gated loader, validator, and inspection image
```

`src/epsbench/schema.py` is the public contract. Unknown schema and configuration fields are rejected. Dataset-relative POSIX paths are validated against traversal, Windows drive-qualified or drive-relative forms, and generation refuses to overwrite a non-empty directory. Inspection performs complete dataset validation before reading panels or creating an output image.

## Scene and action

`src/epsbench/sim/single_occluder.py` generates a small MuJoCo XML model using the official `mujoco` package. It contains a support plane, one opaque foreground panel, one opaque background surface, a directional light, and one fixed monocular camera. The executed action is a discrete lateral translation from `x=-0.35` to `x=0.35`; forward displacement and yaw are zero. MuJoCo world coordinates and camera matrices are retained only as privileged instrumentation.

The `base` and `alternate` appearance variants change procedurally specified surface colours while preserving geometry, camera trajectory, action, correspondence, occlusion order, and visibility events. Configuration validation rejects non-finite values, forward or yaw actions, zero lateral displacement, and action names with the wrong lateral sign. Whole-dataset validation separately checks the persisted before/after camera positions and proper, approximately orthonormal rotations against the configured action.

## Modality boundaries and loader permissions

- `SENSORY`: RGB and executed action.
- `ECOLOGICAL_ORACLE`: remapped surface regions, segmentation-derived boundary contacts, frame-normalised visibility fractions, region correspondence, occlusion relation, and visibility events.
- `METRIC_BASELINE`: depth.
- `INSTRUMENTATION_ONLY`: camera world transform, raw MuJoCo geom IDs, raw simulator coordinates, and privileged generation records.
- `CONTROL_METADATA`: the compound transition record that indexes explicitly typed records; it is not a learner input.

Every `DatasetLoader` is constructed with an explicit `ModalityPermissionSet`. It checks permissions before resolving or opening the requested artifact. The ecological view returns no `FrameRecord`, depth path, camera record, raw identifier, or world-coordinate value.

## Surface identifiers

MuJoCo geom IDs are read only during generation. A namespaced deterministic random generator produces a different random order, opaque `surface-<token>` identifier, and non-raw integer segmentation label for every episode. Ordinary segmentation arrays contain only zero (unlabelled renderer background) or those declared episode-local labels. The raw-ID-to-opaque-ID mapping exists only in `instrumentation.json`, whose modality is privileged generation data.

The same root seed and episode index reproduce the mapping exactly, but independently serialised episode indices use disjoint opaque identifiers even with identical geometry. Whole-dataset validation enforces that surface-ID sets are disjoint across episodes. Numeric segmentation labels remain episode-local and may be reused by different episodes.

## Annotation derivation

Visible pixels and projected-image fractions are measured from remapped segmentation. `before_projected_image_fraction` and `after_projected_image_fraction` mean surface-labelled pixels divided by all image pixels; they are not fractions of a physical surface. Correspondence uses persistent opaque surface identity before and after and records exact before count, after count, and `same_image_coordinate_overlap_pixels`. That overlap is mask intersection at unchanged image coordinates, not optical-flow correspondence or an estimate of optical transformation. Dense and region-level optical transformation remain unfinished Gate 0B work. Accretion counts pixels present only after the action; deletion counts pixels present only before. A shifted surface may correctly carry both events even when its total visible area is unchanged. Appearance and disappearance are used when visibility crosses zero.

Boundary records count four-neighbour segmentation label transitions. Contacts between two declared surfaces are aggregated by opaque identifier.

The current occlusion oracle uses the appearance-invariant rule `counterfactual_occluder_exclusion_v1`. For each frame, generation renders ordinary raw geom segmentation and a privileged second segmentation with the candidate foreground occluder's MuJoCo render group disabled. A foreground-to-background relation is emitted for a frame only when at least one ordinary occluder pixel becomes the candidate background in the counterfactual segmentation. Generation stores the counterfactual array, revealed-pixel count, and logical reveal-mask hash only in privileged instrumentation; the ecological record contains only opaque surface IDs and derived frame membership. Validation requires exactly the three named apparatus surfaces and a bijection among semantic apparatus names, raw IDs, and opaque surface IDs. It reconstructs ordinary raw segmentation from the public labels and privileged bijection, requires the counterfactual to contain no occluder pixels, requires its changed-pixel mask to equal the ordinary occluder footprint exactly, and independently recomputes the background reveal evidence and ecological relation.

Dense optical flow is not fabricated. Its schema status is `unavailable` with a typed reason in every transition.

## Hash domains

All hashes are SHA-256 and lowercase hexadecimal.

- Each image or array `logical_sha256` covers canonical `{dtype, shape}`, a zero separator, and contiguous semantic array bytes.
- Each JSON `logical_sha256` covers sorted, compact, UTF-8 canonical JSON without the file's final newline.
- Every `file_sha256` covers the complete container file bytes. It is separate from logical array or JSON identity.
- `ecological_label_sha256` covers schema version, executed action values, opaque surface references, logical segmentation hashes, visibility states, region correspondence, visibility events, occlusion relations, boundary structures, and the explicit dense-flow availability record.
- The ecological-label domain excludes RGB, depth, colour/appearance identity, camera transforms, raw simulator IDs, raw/world coordinates, paths, timestamps, and hostnames.
- `dataset_logical_sha256` covers schema/generator versions, root seed, resolved-config logical hash, appearance variant, and each episode's stable identifier, derived seed, ecological-label hash, and RGB logical hashes. It excludes renderer provenance, file-container hashes, timestamps, hostname, and absolute paths.
- `source_provenance_sha256` independently covers the typed inline source record: sanitized repository reference, exact commit and truthful dirty state, dirty-diff hash when applicable, lock-file identity, governing-document hashes, package version, and Python version. HTTP credentials and query/fragment data are removed; SSH user information is removed; file and local-path origins are replaced with a non-path redaction marker. Git unavailability is recorded as unavailable with a reason and never represented as clean.
- `renderer_execution_provenance_sha256` independently covers stable renderer/execution facts: MuJoCo and NumPy versions, renderer, backend, and operating system. These facts remain outside the scientific content domain.
- `content_provenance_binding_sha256` hashes the already-computed dataset logical hash together with the already-computed source-provenance and renderer/execution-provenance hashes. This binds content and both provenance domains without making any domain self-referential. The existing scientific content identity deliberately continues to exclude provenance.
- Volatile time, hostname, and Python build text live only in unreferenced `run.json`; they do not participate in manifest or scientific identity.

The unreleased schema identifier is `0.1.0-dev.1`. The terminology, transition modality, counterfactual evidence, and provenance corrections revise the development schema in place and do not imply that a prior public `0.1.0` scientific dataset existed.

## Determinism and rendering limits

The same seed/configuration produces byte-identical manifests and transition annotations in the declared environment fingerprint. The manifest records MuJoCo and NumPy versions, renderer name, selected GL backend, and operating system. The lock file pins the complete Python dependency environment.

Cross-platform RGB byte identity is not claimed. At the original reviewed schema, Windows/WGL and Ubuntu/OSMesa produced different dataset logical hashes while sharing episode-0 ecological hash `3de23fb70a1f68934b3dbb81a8929434ecc9b81c92c0fa5d55c5138e36963419`. The corrected development schema intentionally changes that ecological identity domain and the local Windows/WGL episode-0 regression value is now `8f7c7a7e8f70f9e84bf2f256ecf93f8d94f5327abe536c01a6b8f7e87aef3df0`. A renewed exact-head CI run must establish whether the corrected value also remains stable on locked Ubuntu/OSMesa; local success alone is not cross-platform evidence. Pixel/container identity, ecological-label identity, and source provenance remain distinct claims.

## Remaining Gate 0B work

- Add and validate the corridor scene family.
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
