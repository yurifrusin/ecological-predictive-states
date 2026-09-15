# Renderer readback investigation

Review profile: `DUAL_REVIEW`  
Evidence class: `PUBLIC_REPOSITORY_ONLY`  
Phase-gate effect: `NONE`

## Question and boundary

This finite follow-on records renderer state between the SDK's existing `mjr_render`
and `mjr_readPixels` calls and retains the raw float32 depth argument immediately
after `mjr_readPixels`, before the SDK's Python inverse projection. It does not add a
graphics context, render, readback, framebuffer change, shader, copy, extraction,
retry, fallback, replacement cell, model, label, tolerance, or gate decision.

The eight cells are pathology-selected diagnostic exemplars from the completed PR26
matrix. They are not prevalence samples and cannot identify a general driver. The
single-occluder episode changes appearance and opaque identifiers, not independent
geometry. Hybrid is omitted because its 32 pose endpoints exactly matched the
corresponding joint components in PR26.

## Fixed membership and budget

Root seed is 1729; resolution is 160 x 120. The only order is:

1. corridor episode 1: WGL joint4, OSMesa joint4, WGL joint0, OSMesa joint0;
2. single-occluder episode 0: WGL joint4, OSMesa joint4, WGL joint0, OSMesa joint0.

Each cell constructs one renderer, captures `before` then `after`, and calls RGB,
depth, then segmentation once at each pose. The total is eight contexts, 16 pose
endpoints, 48 SDK modality render calls, and 48 SDK readbacks. CPU preparation and
synthetic tests construct no graphics context. A reserved or failed cell permanently
stops the new namespace and preserves partial evidence.

## Immutable bindings

The fixed PR26 evidence ZIP has size 12,012,790 bytes, SHA-256
`0be040f4f37ea8d49ac76c0d2a35baf8cd49b1f1d2dde6216b8582af0685962f`,
and manifest SHA-256
`c906420ee9a01cc80c02d2ad9900f7bf3233253de9d3322c324a2e3501d76a80`.
Every manifest member is verified before use and again during each exact comparison.
All selected PR26 RGB, converted depth, encoded segmentation, decoded pairs, raw geom
IDs, and scene maps must match exactly in dtype, shape, and values. Any mismatch is a
preserved terminal result; it cannot be tuned or retried.

MuJoCo must be exactly 3.12.0. Its classic renderer source must hash to
`c193df6a8b8cc1659819abd0ded5c17dab0e3e64edb7af6a1e8d249bb4a4a548`,
and `readDepthMap` must equal `mjDEPTH_ZEROFAR`. Initialization binds the exact Git
HEAD/tree, clean worktree, dependency lock and config hashes. Each attempt holds an
exclusive one-attempt lock, reserves before preparation, rechecks those bindings,
and requires a completed two-way native Windows/WSL root and token handoff.

## Scoped SDK interception

For each existing `Renderer.render` call, Python temporarily wraps only module-level
`mujoco.mjr_render` and `mujoco.mjr_readPixels`. The wrappers require the exact
renderer viewport, scene, context, identity, role, and thread; each original must be
called exactly once. Originals are restored in `finally`, including exceptional
paths. Unexpected pre-existing wrappers, call counts, identity or restoration fail
closed. Raw depth is copied directly after the original readback, its bytes are
hashed before and after copying, and the retained copy must be bitwise identical;
the intercepted argument is not modified before SDK inverse projection.

After `mjr_render` returns and before readback, the probe records `MULTISAMPLE`,
`GL_SAMPLE_BUFFERS`, `GL_SAMPLES`, read/draw framebuffer bindings,
`GL_READ_BUFFER`, `GL_DRAW_BUFFER`, `readDepthMap`, clip depth mode and origin,
depth range/function, viewport, subpixel bits, projection/modelview matrices, scene
flags, frustum values, camera vectors, and GL errors. When the reported sample count
is four it also records four shading sample positions; these are not exhaustive
coverage proof.

Every such observation is explicitly `stage: post_render`. Framebuffer bindings and
other state may already have been restored by `mjr_render`; the queried values do not
establish draw-time multisample behavior. `GL_SAMPLE_BUFFERS` and `GL_SAMPLES` are
labelled as properties of the bound draw framebuffer at that post-render stage.
Static SDK source remains the only evidence about the draw path. The queried
projection matrix is reported state and does not prove internal arithmetic precision.

Matrices are saved twice as raw OpenGL column-major flat16 arrays, once from
`glGetFloatv` and once from `glGetDoublev`. A derived mathematical 4 x 4 row/column
array uses `reshape((4, 4), order="F")`. Both raw arrays must be finite and the
float query must equal the float32 cast of the double query. Arrays are published
before validation with nonfinite values intact; strict JSON uses tagged
representations for malformed scalar state.

## Interpretation

Raw depth records the float32 window-depth buffer delivered by the SDK readback.
The SDK forms reverse-Z projection coefficients in float32, performs the inverse in
float64, and casts the result to float32. Comparing raw depth, reported post-render
state, converted output, static SDK source, and the exact PR26 anchors can localize
the observed conversion boundary. It cannot prove an unobserved per-sample coverage,
resolve, rasterization, or driver mechanism, and it cannot establish generality from
these selected examples.

The implementation status is `IMPLEMENTED_PENDING_REVIEW`. No implementation agent
self-approves, and no graphics execution is allowed before independent exact-head
engineering and scientific review.
