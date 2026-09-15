# Shared-raster capture study

Review profile: `DUAL_REVIEW`
Evidence class: `PUBLIC_REPOSITORY_ONLY`
Phase-gate effect: `NONE`

This finite diagnostic tests one narrow candidate for the Milestone 0 segmentation/depth alignment contract. During each ordinary depth render, it changes the existing native SDK call from `mjr_readPixels(None, depth, ...)` to `mjr_readPixels(id_rgb, depth, ...)`. It calls the original function once, after the existing SDK ID-color draw. It adds no render and no native readback. The candidate preserves the ID bytes and the **native SDK readback before metric conversion** before the SDK converts depth in place. That term does not mean unmodified hardware or raw OpenGL depth: MuJoCo's native `flipDepthIfRequired` path may normalize the values before Python receives them.

The ID and depth values are read from the same completed, depth-tested draw and the study requires a single-sample unresolved `offFBO`. The color and depth GL reads within `mjr_readPixels` are sequential, so this is not a hardware-atomic capture. The depth-tested interpretation is a source-bound inference from the exact MuJoCo 3.12.0 native draw path and retained scene flags; post-draw `GL_DEPTH_TEST` state is not treated as proof because native cleanup disables it.

## Fixed execution

The immutable order is single occluder then corridor; within each family, `legacy_solid_base_v1` then `legacy_solid_alternate_v1`; within each profile, repeat 0 then repeat 1. Every batch contains episodes 0–3 with root seed 1729, resolution 160×120, and component topology disabled. Across eight batches this is 32 contexts, 64 ordinary endpoints, 224 original `mjr_render` calls, and 224 original `mjr_readPixels` calls. Only the 64 ordinary depth calls receive the additional ID RGB destination.

No fallback is allowed. Before capture the study binds the clean source head/tree, lock, configs, registries, seed registry, imported package, canonical child provenance, MuJoCo Python source, native extension, shared library, and package source hashes. It requires OSMesa, zero actual samples, matching color/depth attachment dimensions, absent resolve FBO, `GL_RGB`, uint8 `H×W×3` packing, pack alignment 1, zero row length/skips, no pixel pack buffer, and `mjDEPTH_ZEROFAR`.

Each paired endpoint stores native ID bytes and native pre-metric float32 depth in their readback orientation, then applies one common vertical flip to derive the independently decoded object ID/type arrays and independently converted depth, and the ordinary canonical SDK segmentation/depth outputs used only for exact compatibility checks. The stable sidecar records the complete `(segid+1, objid, objtype)` map, logical episode/frame/context association, orientation, projection coefficients and precision order, hashes, and interpretation. Volatile renderer, scene, context, rectangle, thread, framebuffer, attachment, driver, and invocation facts remain in the outer receipt. Both ID and depth receive the same single vertical flip.

The paired loader is fail closed. Array access requires depth and privileged raw-ID authority before path access; metadata that exposes projection or instrumentation also requires instrumentation authority. Canonical ecological loaders are unchanged.

## Lifecycle and interpretation

A new `shared_raster_capture_ledger/v1` hash chain reserves a whole batch before dataset or context creation. A reservation, failure, or orphan is terminal and cannot be retried or replaced. Before the next reservation, validation rechecks the complete prefix, source binding, receipt hashes, canonical datasets, paired artifacts, permission probes, and context binding. Saved-data validation independently regenerates ID decoding and metric conversion from the preserved native arrays and compares them exactly with both stored candidate arrays and canonical SDK outputs. Repeat deterministic files must match exactly; only dataset-relative `run.json` is excluded by the canonical dataset comparison.

A successful finite execution has status `FINITE_SHARED_RASTER_CANDIDATE_EVIDENCE_ESTABLISHED_METRIC_AND_GEOMETRIC_ACCURACY_SEPARATE_UNQUALIFIED`. It establishes candidate shared-raster correspondence evidence only. Analytic geometry is descriptive; there is no fitted tolerance, full driver-cause claim, backend adoption, metric-accuracy qualification, geometric-ownership qualification, or Gate 0B decision.

Commands, after exact-head review and owner authorization to run, are:

```bash
python scripts/shared_raster_capture.py plan
python scripts/shared_raster_capture.py --source-root . init --output-root PATH
python scripts/shared_raster_capture.py --source-root . next --output-root PATH
python scripts/shared_raster_capture.py --source-root . validate --output-root PATH
```
