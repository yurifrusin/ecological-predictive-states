# Gate 0B Slice 3: analytic optical transport

## Status and scope

This branch implements the analytic optical-transport slice of Gate 0B for the canonical single-occluder and corridor families. It is pending independent engineering and scientific review. It does not complete Gate 0B, establish oriented boundary ownership or ecological accretion/deletion, authorise Gate 0C, add a model, or establish a scientific result.

The annotation answers one bounded question: for a source pixel imaging a controlled static surface point, where does that point project after the prescribed observer movement, and is that exact point still visibly transportable? Controlled compiled geometry and camera transforms are privileged generation and validation instruments. RGB matching, learned flow, rendered motion vectors, depth, and renderer segmentation do not define the canonical result.

## Pixel, camera, and flow convention

Every ray samples a pixel centre. For zero-based column `c` and row `r`, the image coordinate is

```text
x_s = c + 0.5
y_s = r + 0.5
```

`x` increases right and `y` increases down. Let image width and height be `W` and `H`, and let MuJoCo's compiled vertical field of view be `fovy`. Define

```text
t_y = tan(fovy / 2)
t_x = (W / H) t_y
f_x = W / (2 t_x)
f_y = H / (2 t_y)
```

The unnormalised camera ray is

```text
q = ((x_s - W/2) / f_x,
     -(y_s - H/2) / f_y,
     -1)
```

and the world ray is the normalised `q` transformed by the compiled MuJoCo camera rotation. This uses MuJoCo's compiled camera position and orientation; it does not recreate a second scene-specific look-at convention.

For world point `P`, compiled camera position `C`, and compiled camera rotation `R`, `p = (P - C) R` is projected when `-p_z > 1e-12`:

```text
x_t = W/2 + f_x p_x / (-p_z)
y_t = H/2 - f_y p_y / (-p_z)
```

Directional transport is always `target pixel centre - source pixel centre`. Forward maps frame 0 to frame 1; backward maps frame 1 to frame 0. The units are image pixels.

## Analytic intersection and visibility

Centre-of-pixel rays are intersected explicitly with the compiled MuJoCo plane and oriented-box apparatus. Compiled geom positions, rotations, sizes, and raw identifiers exist only inside privileged computation. The nearest positive intersection establishes the controlled surface and exact world point. Unsupported controlled geom types fail closed.

The same point is projected into the target camera. It is transportable only if its source ray hit a controlled surface, it projects into the target image, the target ray's nearest controlled intersection is the same surface point within

```text
1e-7 * max(1, target point distance)
```

and neither source nor target sample falls in the analytic ambiguity band. The relative distance check catches both occlusion by another surface and self-occlusion of a farther point on the same geom.

## Boundary ambiguity

`four_neighbour_assignment_band_v1` compares the analytic nearest-surface assignment of horizontally and vertically adjacent rays, including controlled-to-uncontrolled changes. Both pixels adjoining a discontinuity are marked, yielding a deterministic one-pixel band on each sampled side of the analytic boundary. A directional sample is excluded if the source pixel is marked or if the containing target pixel is marked.

This rule is independent of RGB, depth, and rendered segmentation. It is a conservative exact-transport exclusion, not oriented boundary ownership and not a claim about figure/ground direction.

## Fixed-point representation and reason semantics

Forward and backward vectors are stored as `int32` arrays of shape `(H, W, 2)` with 1024 units per image pixel. Conversion is

```text
decoded_flow_pixels = vectors_fixed / 1024
```

Quantisation uses nearest rounding with ties to even. Each direction also has mandatory `uint8` validity and reason arrays of shape `(H, W)`. Invalid vectors are canonical zero; a valid static-camera vector can also be zero. The strict reason domain is:

| Code | Meaning |
| ---: | --- |
| 0 | valid transport |
| 1 | no controlled source surface |
| 2 | target out of frame |
| 3 | occluded at target |
| 4 | analytic boundary ambiguous |

The deterministic classification precedence is no controlled source, target out of frame, boundary ambiguity, target occlusion, then valid transport. Every valid pixel has code 0 and every invalid pixel has a nonzero code.

The maximum component-wise storage error is one half fixed-point unit, `1 / 2048` pixel. A forward/backward round-trip can therefore differ by at most one full fixed-point unit, `1 / 1024` pixel, before any discrete resampling consideration; the closed-form consistency test uses that declared tolerance.

## Schema, loader, and permissions

The wire-version matrix is deliberately non-uniform:

| Contract | Version | Reason |
| --- | --- | --- |
| Single-occluder config | `0.1.0-dev.2` | unchanged |
| Corridor config | `0.1.0-dev.2` | unchanged |
| Transition | `0.1.0-dev.3` | available transport bundle replaces the former unavailable field |
| Dataset manifest | `0.1.0-dev.3` | episode transport identity is added |
| Privileged instrumentation | `0.1.0-dev.3` | analytic/renderer diagnostics are added |

`AvailableDenseOpticalTransport` records the method, coordinate convention, quantisation, boundary rule, reason-code domain, complete forward/backward artifact records, and analytic identity. The unavailable branch remains discriminated and typed.

`Modality.ANALYTIC_OPTICAL_TRANSPORT` is an ecological-oracle modality. `read_analytic_optical_transport(episode_index)` checks that permission before reading the transition or any flow artifact and returns both directions together with validity, reasons, exact fixed-point arrays, decoded-flow conveniences, method metadata, and identity. Ecological-only views may contain these image-plane products but do not contain world points, rays, distances, raw geom identifiers, semantic apparatus names, camera transforms, depth, or metric scene geometry.

## Identity domains

Each episode has a separately inspectable `analytic_transport_sha256`. Its domain contains the method and reason-domain versions, coordinate convention, quantisation, boundary rule, ordered source/target frame indices, and the logical hashes of all six directional arrays. It excludes file paths and container hashes, episode seed, appearance, RGB, depth, segmentation labels, opaque surface identifiers, renderer backend, provenance, timestamps, and host metadata.

The ecological-label identity includes the public analytic annotation and its separate identity. The dataset scientific-content identity binds each episode's analytic identity. Raster artifact, scene-content, source-provenance, and renderer/execution identities remain distinct.

Local Windows/WGL evidence for the canonical configuration is:

| Family / episode | `analytic_transport_sha256` |
| --- | --- |
| Single occluder 0 and 1 | `e6d5fc664957c4e68b3c249ce5bd32113755b1c86d5fff8dbeaafc75b550c7cd` |
| Corridor 0 | `12ea54fea0b8d9716d12189fcba89397156f7c8111fcb7a488702ffb8240f3bf` |
| Corridor 1 | `9b41e6780bbf89654cda8c5f6d5f4d6326d64bac2594a1afb12db883f91395b6` |

Exact-head Ubuntu/OSMesa CI run `32950788201` at implementation head `04238d5a399ad89b3bc8195e2d53002e5facf346` reproduced all four local Windows/WGL analytic identities exactly. Those values are therefore pinned as one shared regression for the two locked environments. The same run passed all 195 Linux tests and generated, validated, and inspected both two-episode families. This is evidence only for the recorded dependency set and those WGL/OSMesa environments; it does not imply identity across arbitrary drivers, platforms, dependency versions, or future scene families. Raster-derived ecological and dataset hashes remain distinct and may be backend-specific.

## Independent validation and tests

Generation computes transport alongside the already compiled rendered scene. Whole-dataset validation independently recompiles the scene from the resolved configuration and privileged deterministic geometry record, recomputes both directions, and compares every fixed-point component, validity bit, and reason code. It also verifies artifact file and logical hashes, raster alignment, strict dtypes and shapes, reason-domain consistency, canonical zero invalid vectors, method and scale metadata, the analytic identity, and its episode/dataset bindings. Fully rehashed fabricated artifacts therefore still fail.

Closed-form tests cover lateral translation against a fronto-parallel plane, forward translation and radial expansion, valid static-camera zero transport, frame exit, target occlusion, analytic boundary ambiguity, and forward/backward consistency. Scene-family tests cover deterministic bytes, appearance invariance, opaque-ID independence, camera/action and geometry sensitivity, permission denial before access, ecological leakage boundaries, swapped/foreign/malformed artifacts, and preservation of the existing occlusion and unavailable-event posture.

## Renderer cross-checks and inspection

Rendered raw segmentation is a non-authoritative diagnostic only. Privileged instrumentation records analytic-versus-renderer assignment agreement away from the analytic boundary, excluded boundary counts, directional valid fractions, and reason-code frequencies. Validation requires at least 0.90 interior agreement so broad camera or geometry disagreement fails, while boundary rasterisation differences do not define transport.

The single-occluder plane is analytically infinite under MuJoCo's plane intersection semantics while its visual grid has a finite rendered extent, so local WGL interior agreement is approximately 0.933 rather than 1.0. The canonical corridor local agreement is 1.0. This discrepancy is preserved as a limitation and does not enter analytic identity.

Inspection validates the complete dataset before writing anything and shows before/after RGB, before/after opaque segmentation, forward/backward flow visualisations, and forward/backward validity/reason views. Its ordinary summary includes the method and analytic identity without metric or semantic apparatus values.

## Scientific limit and remaining work

Transport follows an already visible surface point. It does not say which oriented boundary owns a newly revealed or removed image region, so it cannot by itself establish accretion, deletion, or disocclusion. `ecological_visibility_events` therefore remains explicitly unavailable with `oriented_boundary_ownership_unavailable`; corridor occlusion also remains unavailable.

Remaining Gate 0B work includes full oriented boundary ownership, ecological visibility-event derivation, broader appearance-OOD controls, cross-platform posture beyond the locked evidence, and final benchmark freezing. Gate 0C is not authorised.
