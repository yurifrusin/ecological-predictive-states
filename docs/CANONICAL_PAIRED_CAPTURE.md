# Experimental canonical paired ID/depth capture

Review profile: `DUAL_REVIEW`. Evidence class: `PUBLIC_REPOSITORY_ONLY`.
Phase-gate effect: `NONE`.

The shared-raster candidate study at PR29 head
`1a36c1605bd008fab9e13e270f9a20389b666204` retained joint ID/depth readback.
Its separately rendered canonical arrays were compatibility comparisons. This
work package makes a new canonical segmentation/depth pair derive from the
joint draw itself. It neither changes the provenance of old datasets nor
activates a production default.

## Producer and data contract

`generate_dataset(..., capture_mode="canonical_paired")` explicitly selects the
experimental producer. The ordinary API/CLI default remains `legacy`.

For each pose the simulator performs ordinary RGB, then one owned joint
ID-colour/depth draw/read. Single-occluder counterfactual segmentation remains
a separate draw. The owned API does not replace any process-wide SDK function.
It uses the existing renderer context and scene, makes that context current as
the SDK does, and restores scene flags and owned Mjr buffer selection in
`finally`. The same renderer context remains current on return.

The paired operation checks current context, unresolved offscreen framebuffer,
attachment identity/storage/sampling, dimensions, read/draw selection, packing
and PBO state, scene geometry/map, flags and cameras. The native draw installs
its projection/modelview and clip convention; those post-draw facts must remain
exactly unchanged through pre-read and post-read observations. No other scene,
camera or attachment input may change across the draw. There is one native
`mjr_render` and one `mjr_readPixels(id_rgb, depth, ...)` for a canonical pair.
The SDK performs sequential colour and depth GL reads, not hardware-atomic
readback. The depth-tested draw interpretation relies on the pinned SDK source
and its main ID-colour draw path; a post-draw GL depth-test flag is not evidence
that depth testing was active during the draw.

Both retained native arrays receive one common vertical flip. ID zero is
background. Every nonzero ID must occur in the exact `(segid + 1, objid,
objtype)` map, with unique segmentation IDs and geom object types. Existing
per-episode opaque remapping then produces the public two-dimensional `int32`
segmentation labels. Raw simulator identifiers never become ecological labels.

Retained depth is **native SDK readback before metric conversion**, not
unmodified hardware depth. Conversion follows MuJoCo 3.12.0: float32 near/far
and coefficients, the reverse-Z coefficient adjustment, float64 inverse, then
float32 output. This establishes derivation, not metric-depth accuracy.

New dataset schema `0.1.0-dev.11` requires both endpoint provenance artifacts
and binds their hashes into the dataset logical identity. Each versioned
privileged endpoint record binds episode/seed/configuration/scene, source and
renderer identities, retained arrays/state, exact raw-to-opaque map, orientation
and conversion, and the final canonical arrays' logical and file hashes. RGB
and counterfactual records explicitly identify their separate producers.

Stable native facts are scientific artifacts. Volatile context/FBO/object
handles remain in `run.json` operational observations and qualification receipts;
they are excluded from deterministic dataset identity and repeat comparisons.
The qualification receipt separately binds all saved files, including run
metadata. No ecological loader exposes that instrumentation. The paired reader
requires `DEPTH`, `MUJOCO_GEOM_IDS` and `PRIVILEGED_GENERATION_RECORDS` before
transition or path access, and uses the existing owned-file decoders.

Older dataset schemas 7–10 and their logical domains remain unchanged. Their
frame and episode records have no paired field. Equal values cannot create a
retrospective paired provenance claim.

## Initial scope and operation

Generation is restricted to Linux/WSL OSMesa, Python 3.11.15, MuJoCo 3.12.0,
NumPy 2.4.6, PyOpenGL 3.1.10, glfw 2.10.2, zero offscreen samples, 160×120,
the existing legacy base/alternate solid appearance profiles, and component
topology disabled. Revision-1 appearance generation is not included.

The inert qualification entrypoint is
`scripts/canonical_paired_qualification.py`. Its `plan` command creates no
context or study output. `init`, `next` and `validate` each take an explicit
`--output-root`; `--source-root` precedes the subcommand and must name the clean
imported checkout. Output must be outside that checkout. The operator must bind
independent engineering and scientific reviews and CPU verification to the
exact new PR head before executing a capture command.

The fixed schedule contains eight batches: two families, two legacy profiles,
two repeats, four episodes per batch, root seed 1729. Its budget is 32 contexts
and 160 native render/read pairs: 64 ordinary RGB, 64 canonical pairs and 32
single-occluder counterfactuals. `next` reserves exactly one batch before
creating its output/context. A failure or orphan reservation permanently stops
that namespace. A retained lock also prevents continuation. There is no retry,
reset, extension or replacement-namespace operation.

Qualification alone installs a scoped monitor around the pristine SDK
constructor and native render/read functions. It observes the actual calls made
by both the ordinary SDK modes and owned paired API, rejects excess/wrong-order
calls before invoking them, accounts for failed constructor attempts, and
restores the original functions and closes owned contexts in `finally`. This
monitor is not part of ordinary generation. Each completed prefix and its exact
source, runtime, artifact membership, native events and endpoint bindings are
checked before another batch can proceed.

Assessment validates complete datasets, replays derivation from retained arrays,
checks permission denial, saves inspection images, compares exact repeats
(excluding only `run.json`) and verifies appearance controls. Geometry diagnostics
remain descriptive. No appearance, geometric-boundary or metric acceptance
tolerance is introduced. Frozen Slice 6 remains `FAILED_CLOSED`; Gate 0B is
incomplete, and this package grants no Gate 0C/0D or model authority.

## CPU verification and remaining evidence

Focused tests cover native-call/flip/conversion behavior, input and readback
drift, restoration after injected failures, strict provenance and legacy schema
round trips, permission denial before access, finite ledger order, preserved
failures/orphans, constructor/call budgets and monitor restoration. Synthetic
CPU fixtures are not renderer qualification evidence. Qualification results and
independent exact-head reviews must remain external to this implementation PR.

No completed historical PR26–PR29 study may be initialized, resumed, retried,
extended or modified through this work package. Production activation, merge,
closeout and gate decisions remain separate owner decisions.
