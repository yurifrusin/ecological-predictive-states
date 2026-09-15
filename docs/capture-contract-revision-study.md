# Capture-contract revision study v1

Review profile: `DUAL_REVIEW`  
Evidence class: `PUBLIC_REPOSITORY_ONLY`  
Phase-gate effect: `NONE`

## Authority and scope

The owner approved the recommended bounded revision study, not adoption, merge,
closeout, benchmark revision, or gate advancement. This additive development
work package leaves accepted capture code, schemas, configuration, locks,
qualification seeds and all historical evidence unchanged. PR24 at
`bc7c65c145530f04a07a25a013af6d65f3f6ae65` and its completed four-cell ledger
remain immutable. New code and evidence have separate names and locations.

The question is whether single-sampled segmentation can be isolated from the
existing RGB/depth sampling policy, and what changes when all three modalities
are single-sampled. Neither backend is a ground-truth oracle. This is not an
EPS comparison or permission to implement models.

## Fixed development membership and finite budget

Use the existing unchanged `configs/corridor_v0.yaml` and
`configs/benchmark_v0.yaml`, root seed 1729, episode indices 0, 1, 2, 3 for
each family. Derive each episode seed with the existing `episode:<index>`
namespace and generate appearance/geometry using the existing generator's
exact seed namespaces. Persist integers exactly, never through binary64 JSON
number roundtrips. Use `legacy_solid_base_v1`, 160 x 120, vertical FOV 55
degrees, and the two camera poses already specified by each configuration.

Corridor has four sampled geometries. Single-occluder geometry is fixed;
its four seeds vary appearance and identifiers, not independent geometries.
These are development examples, not new held-out or qualification evidence.
No appearance-OOD, additional camera, counterfactual-exclusion, derived-label,
component-topology or downstream model qualification is included.

The exact nested order is family `corridor`, then `single_occluder`; within
each, episode 0, 1, 2, 3; within each episode the following six cells:

1. WGL / joint4
2. OSMesa / joint4
3. WGL / hybrid
4. OSMesa / hybrid
5. WGL / joint0
6. OSMesa / joint0

The initialized ledger enumerates all 48 cells explicitly. Each cell captures
before then after exactly once. Maximum: 96 frames, 64 renderer constructions
(one per joint cell, two per hybrid cell), and 288 modality render calls.
No auxiliary graphics probes, retry, replacement cell, adaptive seed, or
automatic extension is permitted. CPU compilation/analysis and synthetic
no-graphics tests do not consume the capture budget.

Run native WGL and WSL OSMesa on DESKTOP-TPUQMNG, with backend environment set
before MuJoCo/OpenGL import. Use one shared fresh host output root. Code must
have independent engineering/scientific exact-head reviews before any study
context is constructed. No study render is allowed during implementation.

## Three policies and isolation

`joint4`: one model/data/renderer, requested offsamples 4, RGB then depth then
segmentation per pose. `joint0`: same sequence, requested offsamples 0.
`hybrid`: two separately compiled model/data/renderer stacks from identical
source XML and assets. The primary stack is constructed at offsamples 4 and
captures RGB then depth. The secondary is constructed at offsamples 0 and
captures segmentation. Never share a mutable model between contexts.

Each pose sets the same compiled camera facts and static state in both hybrid
models. Explicitly make the owning context current before every update,
render, provenance inspection and context-sensitive cleanup. Do not rely on
the last-created context remaining current. Model offsamples are set before
context construction and never mutated afterwards.

Bind source/config/appearance hashes, complete compiled geometry/rotation/FOV
facts, model hashes and pose/state facts. Compile a normalized CPU-only model
hash with offsamples fixed to 4, so cross-policy equivalence is explicit and
only the prescribed sample setting may differ. Save the actual compiled-model
hash separately. Runtime checks must compare hybrid semantic/model/pose facts
before contexts are created; per-pose facts are checked before each capture.

Per backend/episode/pose, predeclare these isolation endpoints:

- hybrid RGB and depth exactly equal joint4 RGB and depth;
- hybrid encoded segmentation RGB, decoded object/type pairs, raw geom IDs
  and scene map exactly equal joint0 equivalents;
- same-policy cross-backend exact-array differences for all modalities;
- joint4 to joint0 per-backend exact changes and numeric depth differences.

Equality failures are retained empirical findings that prevent an isolation
claim; they are not permission to repair outputs or retry. Complete the
remaining fixed matrix when capture/provenance integrity remains valid.
Exact array equality requires matching shape, dtype and all values (with a
canonical array-byte hash also retained); non-finite depth is an integrity
failure. Scene maps compare sorted integer segid/object-id/object-type triples.

The two corridor episode-0 joint4 raw-ID controls must exactly reproduce their
own historical PR24 baseline arrays before any intervention. Bind the four
historical arrays and their known file hashes in the plan, and verify the
historical packet is unchanged before/after the new study. A failed anchor
blocks the matrix, since comparability to the preceding diagnostic is lost.

The immutable ordinary-development packet is
`prospective-capture-observations-bc7c65c.zip`, SHA-256
`072edf97d7cd44a7a95c6b189e358bc34ddfd50be4890167dacfd12ccf0d59a2`.
It is generated evidence retained with the completed PR24 report, not a
repository-tracked dataset or private holdout. Verify the archive and every
member listed in its `evidence-manifest.json`, with safe paths and no duplicate
members. The manifest and the four following `capture/` member hashes are
also fixed in the implementation plan (not operator-selected replacements).
Manifest SHA-256: `ba8018ef1a50d5ce3997d86aac36e7509096053866745fd3b73006b7752b305a`.
Under `capture/<backend>-offsamples-4/`, the files are:

| Backend | File | SHA-256 |
| --- | --- | --- |
| WGL | before_raw_geom_ids.npy | b5974ac705cd42a42e25abbbe8ed57cd3fec4218eb72b4d623c64fe81459b4d8 |
| WGL | after_raw_geom_ids.npy | 8308f8028b9b0b32ad294baf1ff4bcb6315cbd77ed5bbddc27bb94b35a58af9a |
| OSMesa | before_raw_geom_ids.npy | 6bcb47a40bd671cb8ace39753d2243d75727605dfafdf5d77172663bee956039 |
| OSMesa | after_raw_geom_ids.npy | 859ce504f08fc4bcb0ec493dedc13d81620c682201943866dee13a406caa4415 |

The reproduction source is PR24 exact head given above; original source
`d7b7ce8f04426e5869ef6e063f97c421fd10fffc` remains bound by that packet.

## Geometric alignment analysis

Use the existing declared centre-of-pixel camera convention and compiled
finite-plane/oriented-box intersection contract as privileged instruments,
independent of captured RGB, segmentation and depth. Preserve analytic raw
assignment, four-neighbour assignment boundary band, centre-ray hit distance,
and camera-axis depth for each pose. Camera-axis depth is
`-((hit_point - camera_position) @ camera_rotation)[..., 2]`, not Euclidean
ray length. Background/no controlled hit is a separate mask, not depth zero.

For all policies report exact raw-ID disagreement separately for analytic
interior, declared boundary band, and no-hit pixels. Save full maps and
counts; boundary disagreement is not automatically geometrically impossible.
Retain the original corridor opposite-wall sufficient exclusion separately.
The exclusion remains geometry-specific, not a general renderer oracle.

Report float64 depth residuals against analytic camera-axis centre depth over
finite controlled hits, separately inside/outside the analytic boundary band.
Provide count, signed mean, absolute mean, absolute median/p95/p99/maximum, and
counts above each fixed absolute diagnostic scale: 1e-6, 1e-5, 1e-4, 1e-3,
1e-2, 1e-1 model length units. NumPy quantiles use method `linear`.
These scales describe error; none is an acceptance tolerance or gate.

Also compute same-surface pixel-footprint depth ranges from the centre and
four corner rays using the same geometry. Mark a conservative sampled
footprint ambiguity when any corner assignment differs from the centre or
the existing boundary band is set. Retain all corner assignments/depths.
For non-ambiguous controlled hits, report depth outside the sampled range
with a fixed numeric guard of `1e-6 * max(1, abs(reference_depth))` model
length units, separately from the centre residual. This is a sampled
footprint diagnostic, not proof that all subpixel samples see one surface;
coplanar ties and unobserved subpixel occlusion remain limitations. No claim
that multisampled depth must equal centre depth is allowed.

RGB has no newly asserted geometric photometric oracle. Its exact comparisons,
shared projection/pose records, and changed-pixel locations relative to the
analytic boundary measure compatibility only. RGB/label boundary mixtures
remain explicitly unresolved rather than being described as pixel-perfect
alignment from array equality alone.

All statistical masks are fixed independently of rendered observations.
The boundary band is computed solely from analytic geometry assignments.
Report numeric and label failures without using them to change masks, sample
membership, thresholds, capture settings, or stopping rules. No aggregate may
hide an individual seed, pose, backend or policy failure.

## Evidence, failure handling and operational constraints

Reserve each cell durably before its first context construction. Hold an
exclusive atomic create lock across validation/reservation/capture/persistence
on the shared Windows/WSL filesystem; competing attempts fail without capture.
A leftover lock, reserved cell, failed cell, modified plan, wrong next backend,
or unsafe/reused output blocks further work. Do not auto-clear stale locks.
Plan, ledger and receipts use atomic publication and no artifact replacement.
This operational ledger is not review-state or approval automation.

The lock primitive is `os.open` with `O_CREAT | O_EXCL | O_WRONLY` on one
per-attempt lock file in the shared root, held until all result/receipt/state
publications and file flushes finish. An interrupted lock is never removed
automatically. The operator verifies the Windows path and its `wslpath -w`
translation agree and performs a two-way CPU-only token handoff on that same
root before the first capture; both runtimes must observe the other's token.
Reject symlinks, Windows reparse points and aliasing in every controlled path
component. WSL `/mnt/c` and the corresponding native C-drive path are the only
allowed host mapping for this study; no remote network filesystem.

Write each immutable artifact/receipt/state revision through an exclusively
created same-directory temporary file, flush/fsync, then `os.link` to publish
atomically with no replacement. Unsupported hard links fail closed. Remove
only that operation's temporary link after publication. Use append-only
hash-chained ledger revisions, rather than overwriting a final ledger. Validate
the full revision chain under the lock; any gap, unexpected temporary file,
invalid predecessor, reserved/failed tail or foreign entry blocks execution.
Release the lock only after all corresponding files are flushed and the
successful terminal revision exists: close its descriptor, verify ownership,
then unlink only this attempt's lock file and sync the parent where supported.
On failure or interruption retain the lock path and never auto-clear it.
Directory fsync is required where supported; any
platform limitation is recorded, not presented as a power-loss durability
guarantee. No automatic crash recovery or retry exists.

The concrete canonical model/state serialization and required-versus-empirical
classification table must be specified in the implementation document and
covered by CPU tests before exact-head review. Canonical array identities use
explicit dtype, shape, little-endian contiguous bytes and SHA-256; dictionaries
have sorted keys and finite values, with integer seeds preserved exactly.

Save immediately available RGB, depth, allocated segmentation out buffer,
returned pairs, derived IDs and actual scene map at stage granularity.
Get the map before segmentation readback, retain the allocated caller buffer
even if the SDK writes it then throws during decoding, and label incomplete
buffers unvalidated. Preserve already captured frames and modality-specific
provenance if a later context, pose, validation or cleanup fails. Never replace
the original exception with a cleanup exception or lose observations.

Record requested/actual context samples, actual observed backend, live main
and resolve FBO attachment format/type/sample facts and GL identity separately
for every renderer. Inspect under its own current context, preserving GL
bindings. Retain Python/MuJoCo/NumPy/PyOpenGL/glfw versions, package/binary/
renderer hashes, Git head/tree, dependency lock, actual host and invocation
records. Missing or inconsistent required provenance fails closed.

Independently decode retained RGB using R + 256G + 65536B, segid+1, the actual
scene map and vertical orientation, then compare every returned pair and raw
ID. An invalid readback/decode/schema/provenance/scene-equivalence result,
exception, interrupted attempt or anchor mismatch permanently stops this
study. Record the partial result; no automatic retry or fresh namespace.

## Deliverables and decision boundary

Additive study code/tests/design remain on a reviewed unmerged branch. All
generated evidence is untracked, hash-bound and independently assessable.
Report every cell, isolation endpoint, geometric diagnostic and limitation.
Obtain independent result assessments. Record installed tool versions and
changes if any; do not change system-wide tools or drivers.

The final recommendation may prefer further investigation of one candidate,
reject both, or remain inconclusive. It cannot adopt a method, redefine labels,
reinterpret frozen Slice 6, merge PR24/the new branch, or advance Gate 0B.
Any such action needs distinct owner authority. No Gate 0C/0D/model work.


## Concrete canonical identities and result schema

This implementation uses `revision_capture_cell/v1`. Each cell receipt contains:

- `identity`: exact ordinal, family, episode index, integer episode seed, backend, and policy;
- `source` and `config`: Git head/tree, dependency-lock digest, fixed config and
  appearance/seed-registry logical hashes, and the exact generator namespaces;
- `model`: normalized MJB SHA-256 compiled with offsamples 4, semantic-facts
  SHA-256, and a role-to-actual-MJB-SHA-256 mapping;
- `renderers`: one record for a joint policy and two for hybrid, each with role,
  requested and actual samples, observed backend, GL identity, live main and
  resolve framebuffer attachment facts, model extent and relative near/far
  settings, runtime versions, and package/binary/renderer source hashes;
- `poses`: ordered `before`, `after` records. Each binds `pose_facts`,
  `geometry_facts`, and `modalities`. Modalities reference RGB, depth, encoded
  segmentation RGB, decoded object/type pairs, raw geom IDs, and the sorted
  `[segid, object_id, object_type]` scene map;
- `stage_events`, terminal state, and failure details when applicable.

MuJoCo `cam_xmat` and geom `xmat` are serialized unchanged, reshaped as 3x3
row-major matrices. They are world-to-local under the analytic convention:
`local = world @ R`, `world_ray = local_ray @ R.T`, and camera-axis depth is
`-((hit - camera_position) @ R)[2]`. Finite planes and oriented boxes store
raw geom ID, centre, this rotation, and half extents. The state identity is
canonical JSON of sorted finite semantic facts and exact integer seeds.

Canonical arrays are C-contiguous, little-endian, non-object arrays identified
by dtype string, shape, and SHA-256 of their complete `.npy` bytes. JSON uses
sorted keys, compact separators, UTF-8, no NaN/infinity, and retains integers
as JSON integers. Every artifact reference also records completeness and
validation independently. Incomplete caller-owned segmentation buffers are
retained with `complete=false, validated=false`.

| Classification | Item | Failure effect |
| --- | --- | --- |
| Required integrity | fixed archive, manifest and every member hash | permanent stop |
| Required integrity | plan order, ledger chain, exclusive lock and paths | permanent stop |
| Required integrity | source/config/seed/appearance namespaces | permanent stop |
| Required integrity | normalized/actual MJB, semantic and pose equivalence | permanent stop |
| Required integrity | requested/actual samples, observed backend and GL provenance | permanent stop |
| Required integrity | finite depth, array schemas, independent decode and map | permanent stop |
| Required integrity | historical corridor-0 joint4 raw-ID anchors | block matrix |
| Empirical endpoint | hybrid RGB/depth equals joint4 | retain inequality; continue |
| Empirical endpoint | hybrid segmentation equals joint0 | retain inequality; continue |
| Empirical endpoint | same-policy cross-backend exact differences | retain; continue |
| Empirical endpoint | joint4-to-joint0 changes and depth residuals | retain; continue |
| Empirical endpoint | analytic alignment and footprint diagnostics | retain; continue |

An empirical inequality never changes masks, membership, settings, order, or
budget. It may prevent the corresponding isolation claim but is not an
integrity repair trigger. Required integrity failures preserve all available
observations, publish a failed terminal revision, retain the attempt lock, and
forbid retry.
