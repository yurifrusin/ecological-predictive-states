# Gate 0B Slice 2 — corridor scene family

## Status and purpose

This slice tests whether the Gate 0B apparatus can represent a second, structurally different
environment without weakening the reviewed single-occluder contract or exposing privileged
metric state. It adds a minimal corridor before dense flow so that later flow and oriented
boundary-ownership definitions must work across both lateral occlusion disclosure and forward
corridor motion.

This is apparatus implementation pending independent review. It is not full Gate 0B completion,
Gate 0C authorisation, or a scientific result.

## Minimal corridor ontology

Every corridor episode contains exactly four privileged apparatus surfaces:

- `corridor_floor`;
- `corridor_left_surface`;
- `corridor_right_surface`;
- `corridor_end_surface`.

The corridor has no doorway, opening, junction, nested enclosure, moving surface, semantic
object, navigation target, collision task, or ceiling. It is open above, but configuration
validation requires the wall height to cover the complete camera optical field at the farthest
sampled end wall. As a result, ordinary frames contain only the four declared surface labels and
no uncontrolled renderer-background pixels. This is smaller and more controlled than adding a
fifth ceiling surface.

The four names above and `monocular_camera` are apparatus semantics. Names are necessary for
generation and validation, but they are privileged instrumentation, not surface features or
learner labels. Ordinary transition records contain only episode-local opaque surface IDs and
random non-raw segmentation labels.

## Geometry and action contract

`configs/corridor_v0.yaml` defines strict ranges for corridor width and length plus fixed wall
height, camera lateral position, camera starting-forward position, camera height, field of view,
and forward displacement. Width and length are sampled per episode. The generated MuJoCo XML is
parametric in those sampled values; it is not an immutable scene string.

The corridor action is pure forward translation:

```text
name == forward
delta_forward > 0
delta_lateral == 0
delta_yaw == 0
```

The camera starts beyond the entry plane, remains clear of both side walls, and ends strictly
before the end wall. Its height and rotation remain unchanged. Configuration validation rejects
an illegal path, and whole-dataset validation independently compares the persisted camera
artifacts, duplicated scene instrumentation, sampled geometry, and executed action. Lateral or
yaw contamination, zero or negative forward displacement, changed height or rotation, and travel
to or beyond the end boundary fail closed.

The canonical single-occluder scene retains its lateral-only camera configuration and exact
three-surface counterfactual occlusion apparatus.

## Deterministic episode variation

The root seed derives an episode seed using the existing `episode:<index>` namespace. Each
corridor episode then derives independent namespaces for:

```text
geometry-sampling
surface-remapping
appearance
```

Width and length use only the geometry seed. Opaque IDs and segmentation labels use only the
surface-remapping seed. The appearance seed controls a bounded colour variation inside each of
the two deterministic solid-colour variants. The same configuration and seed reproduce the same
logical records within the recorded renderer/execution fingerprint. Different episode seeds
produce controlled width or length differences as well as different episode-local opaque IDs.

## Scene-specific privileged instrumentation

The public schema uses a discriminator over `SingleOccluderInstrumentation` and
`CorridorInstrumentation`.

Single-occluder instrumentation retains its exact three-surface membership, raw-ID/opaque-ID
bijection, raw world positions, and `counterfactual_occluder_exclusion_v1` evidence.

Corridor instrumentation records:

- the exact four-surface apparatus set;
- distinct raw MuJoCo geom IDs and their exact opaque-ID bijection;
- raw geom world positions;
- privileged before/after raw renderer segmentation used to reconstruct the opaque arrays;
- sampled width, length, wall height, camera path, camera height, and field of view;
- before/after camera poses, duplicated for independent comparison with camera artifacts;
- episode, geometry, remapping, and appearance seeds;
- fixed geometry-sampling and appearance-generation rule identifiers.

All of these fields are privileged. Exact set equality, finite values, legal path bounds,
bijection, deterministic resampling, and agreement between sampled values and generated raw geom
positions are independently validated. The validator also reconstructs each public segmentation
from the preserved raw renderer array and raw-to-opaque bijection, preventing a consistently
rehashed but fabricated raw-ID mapping from passing.

## Occlusion posture and unavailable annotations

The closed optical field of a corridor contains surface contacts, but this slice has no controlled
counterfactual that establishes oriented occlusion relations at those contacts. Corridor
`occlusion_relations` is therefore explicitly empty. An empty set is valid evidence of restraint,
not a claim that all physical occlusion is absent. The single-occluder counterfactual rule is not
reused outside its assumptions and remains unchanged for that scene.

Dense optical flow remains a typed `unavailable` annotation with an explicit Gate 0B reason. The
four-neighbour boundary-contact counts remain minimal boundary structure and are not described as
full oriented boundary ownership. Neither unavailable modality is represented by fake zero data.

## Appearance invariance

The corridor supports `base` and `alternate` solid-colour variants. Appearance selection does not
enter geometry sampling or surface remapping. For identical seed, geometry, and action, the
alternate variant changes RGB logical identity while preserving opaque surfaces, segmentation,
correspondence, visibility events, empty occlusion relations, scene-content identity, and
ecological-label identity.

This is only a minimal invariance contract. Solid-colour replacement does not complete the final
texture-frequency, illumination-OOD, appearance-asset, or final-seed requirements.

## Modalities and permission boundary

The modality allocation remains:

| Class | Records in this slice |
| --- | --- |
| Sensory | RGB and executed action |
| Ecological oracle | Opaque surface regions, projected-image fractions, correspondence, visibility events, boundary contacts, and justified occlusion relations |
| Metric baseline | Depth |
| Instrumentation only | Camera transforms, raw IDs, raw world positions, sampled corridor geometry, seed namespaces, and scene-generation evidence |
| Control metadata | Compound transition and scene-family records |

`EcologicalTransitionView` contains neither the scene-family discriminator nor semantic names,
depth, camera pose, raw IDs, world coordinates, sampled dimensions, or generation evidence.
Loader permission checks run before the corresponding artifact is resolved or opened.

## Identity and provenance

The schema migration advances the dataset, resolved configuration, and privileged instrumentation
contracts from `0.1.0-dev.1` to `0.1.0-dev.2`. The unchanged ecological transition record remains
at `0.1.0-dev.1`; this cleanly preserves the reviewed single-occluder ecological identity rather
than adding control metadata to a learner-facing label domain. The historical and current
single-occluder episode-0 ecological hash is:

```text
8f7c7a7e8f70f9e84bf2f256ecf93f8d94f5327abe536c01a6b8f7e87aef3df0
```

`scene_family` is included in dataset scientific-content identity. Each episode also records a
`scene_content_sha256`; for corridor episodes its domain contains the scene family, episode seed,
and exact sampled geometry. Metric geometry therefore affects dataset content identity without
entering the ecological-label hash. Meaningful rendered layout or visibility differences still
change the ecological label through segmentation and derived ecological annotations.

Artifact/container hashes, ecological-label identity, dataset scientific-content identity,
source provenance, renderer/execution provenance, content/provenance binding, and volatile
`run.json` metadata remain separate domains. RGB byte equality across WGL and OSMesa is not
claimed. A corridor ecological regression hash will be pinned only after exact-head CI evidence.

## Remaining Gate 0B work

- Implement and validate analytic or geometry-derived dense optical flow across both scene
  families.
- Define and validate full oriented boundary ownership across both scene families.
- Decide the required cross-platform analytic/raster identity posture for later scenes.
- Expand appearance controls if texture-frequency and illumination-OOD assets are adopted.
- Freeze final appearance assets and evaluation seeds before comparative work.

Full Gate 0B completion is not claimed. Gate 0C and all model work remain unauthorised.
