# Prospective EPS capture diagnostic

Review profile: `DUAL_REVIEW`  
Evidence class: `PUBLIC_REPOSITORY_ONLY`

This is a bounded ordinary-development diagnostic only. It neither changes an
accepted capture contract nor alters the frozen benchmark, labels, topology,
configuration, or renderer.

The operator first runs `scripts/prospective_capture.py init` against the
shared fresh output root. The helper accepts only the complete fixed episode-0
packet: WGL and OSMesa preflight/generation receipts, corridor manifests,
resolved configs, instrumentation, and four raw decoded arrays. It checks the
reviewed packet's exact hashes, generation receipt-to-corridor-manifest link,
source head/tree, seed 1729 / episode seed 1703363364368450807, 160x120,
FOV 55, and `legacy_solid_base_v1`. No caller-selected evidence is trusted.

Run exactly one `next --backend wgl` or `next --backend osmesa` command per
host/cell in this order: WGL/4, OSMesa/4, WGL/0, OSMesa/0. Native Windows WGL
must have no `MUJOCO_GL` override. WSL OSMesa sets both `MUJOCO_GL=osmesa` and
`PYOPENGL_PLATFORM=osmesa` before importing MuJoCo/PyOpenGL. No attempt may be
made during implementation or review.

The durable ledger reserves a cell before capture. It rejects malformed,
expanded, reused, reserved/interrupted, and failed ledgers. A failed attempt,
input mismatch, baseline mismatch, unavailable provenance, or post-capture
validation failure permanently stops the matrix; partial arrays and receipts
remain in the fresh output root. Both 4-sample controls must byte-for-byte
match their own original before/after raw arrays before either 0-sample cell.

Before a renderer context exists, the callback uses the existing corridor and
appearance builders to compile and validate all recorded geometry (including
rotations) plus both world-camera poses. It sets only requested
`model.vis.quality.offsamples` before constructing the renderer. For each
frame it preserves historical RGB -> depth -> segmentation/update order,
original caller-owned segmentation RGB, returned `(objid,objtype)` pairs, and
the actual per-frame `segid -> (objid,objtype)` map. It independently applies
MuJoCo's `R + 256G + 65536B`, `segid + 1`, and vertical flip and rejects any
mismatch; raw geom IDs are derived from returned object/type pairs, never from
segids.

A successful cell records requested/actual samples, attachment facts, GL
identity and samples, source XML/compiled model/package/binary/renderer hashes,
actual host/backend, and both frames. FBO inspection queries renderbuffer
format/samples with `glGetRenderbufferParameteriv` and restores distinct read,
draw, and renderbuffer bindings even when inspection fails. The diagnostic
uses no fallback or guessed provenance.

Example arguments deliberately spell every bound original input. Substitute the
same absolute paths on both hosts and the one shared fresh output root:

```text
python scripts/prospective_capture.py init --output <fresh-root> \
  --preflight-wgl <...> --generation-wgl <...> --preflight-osmesa <...> --generation-osmesa <...> \
  --wgl-manifest <...> --osmesa-manifest <...> --wgl-config <...> --osmesa-config <...> \
  --wgl-instrumentation <...> --osmesa-instrumentation <...> \
  --wgl-before <...> --wgl-after <...> --osmesa-before <...> --osmesa-after <...>
```

This PR leaves execution for independent engineering/scientific review and
owner-directed operation. It makes no capture attempt itself.
