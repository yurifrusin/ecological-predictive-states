# A1 Docker execution candidate — prospective addendum

Declared before implementation: `DUAL_REVIEW`, `PUBLIC_REPOSITORY_ONLY`,
phase-gate effect `NONE`. Status: `IMPLEMENTED_PENDING_REVIEW`.
Source base: `aa59a5a1317c75f2c3e0edef56ce4eb298a3b872` (reviewed PR33 merge).
This is source authoring, not qualification, experiment launch, empirical evidence,
Gate 0D authorization, or independent approval.

## Prospective scope adjustment

The coordinator expressly selected a single initialized archive-file bind before
source review or any capture. The abandoned unreviewed pipe draft is preserved in
ignored coordination artifacts; no experiment occurred with that draft. Historical
PR33 descriptions remain historical. This addendum supersedes their absent-driver
limitations only for this prospective candidate; it does not retroactively prove
their broad whole-host resource wording or qualify any historical run.

The Docker cgroup bounds the native producer, loader, inspection, predictor controls,
evaluation and journal materialization: two assigned CPUs (0 and 1), two llvmpipe
threads, 8 GiB memory without swap, and 64 PIDs. The minimal Windows launcher,
Docker client/daemon and whole-host memory are outside the cgroup. There is no
hard aggregate whole-host RAM guarantee. No WSL marker is fabricated; the existing
WSL profile remains available and Docker is a separately admitted candidate.

Logical application-byte allocation is 255 MiB staging + 1 MiB shared memory +
767 MiB archive (including its 8 MiB terminal reserve) + 1 MiB fixed host receipts
= 1 GiB. The host receipt allocation holds a 64 KiB terminal reserve within itself.
Staging remains cumulative across fresh container tmpfs instances: deleting a
container, materialized file or output namespace never refunds reservations.
Every output artifact, inspection PNG, forecast map, exposure/result/failure record,
and restored dataset copy uses the archive's admission rules. Journal reads are
conservatively charged as materializations, even though their buffers are memory
only. Temporary internal array computations are under the cgroup memory limit;
this application-output allocation is not a count of every heap allocation.
Filesystem overhead/journals, daemon storage, source images and dependency preparation
are outside application-output bytes. Preparation artifacts are separately recorded.

The root filesystem is read-only, networking and daemon logs are disabled, capabilities
are dropped, privilege escalation is denied, and the only writable host mount is
the exact existing regular archive file at `/retained/history`. There is no host
directory, repository, Docker socket or receipt-log mount. This is cooperative
typed enforcement and chronology, not hostile-process confinement: a hostile process
could modify its writable archive file. External witnessed hashes detect history
changes and missing history denies continuation.

The host receipt format is canonical checksummed-chain version
`a1_host_receipts_v2`: each record binds its sequence and predecessor checksum.
Recovery checks canonical bytes, the full checksum chain and this driver's
finite legal chronology instead of trusting fields such as `COMPLETE.elapsed`.
Elapsed values must be finite, nonnegative numbers, excluding booleans; successful
cumulative active time cannot exceed 2700 seconds. Native completion requires the
fixed ordered attempts and their successful create/inspect/start/inspect/inspect/remove
command receipts. Unknown, premature, changed or uncertain histories do not release
another phase. Older unchained receipts remain preserved but cannot be silently migrated.

## Fixed identity, reservations and continuation

The binding fixes source HEAD/tree, the complete ordered configuration root,
immutable image ID and purpose domain. Dummy qualification has the separate
`a1_dummy_qualification_v1` domain. It cannot enter a native phase or become the
scientific archive. Native entrypoints reject dummy purpose; dummy entrypoints
reject native purpose. The initial source, seed 1729, every configuration/partition,
single geometry, actions and opaque surface-ID rules remain fixed.

The host initializes `STUDY` exactly once, fsyncs the archive and externally
witnesses its hash in a fixed fsynced host receipt log. Every container opens an
existing history using an externally supplied exact checkpoint. It never creates,
truncates, substitutes a current hash for a witness, or resets a study. The host
receipt anchor is exclusively locked against simultaneous cooperative launchers.
Missing, changed or torn receipts, missing archives, uncertain completion and
incomplete archive suffixes deny continuation. An attempted phase is never retried.
Terminal host receipts can survive a torn archive; sink failure may prevent further
durable evidence, which must be reported rather than converted into an ACK.

Each native cell has one sequential container, fixed `cells/00` … `cells/07`
prefixes, and durable context/render-read reservations before capture. The initial
phase captures ordinals 0/1 (2 contexts, 12 of 48 render-read pairs), then validates
and creates inspection images from retained development artifacts without rerendering.
The continuation phase captures ordinals 2…7 only after a separate explicit
coordinator decision on development evidence, retaining the same source/image,
archive and first two accepted captures. The authorization file binds phase,
the fixed archive/receipt anchor paths,
source/image, expected history, decision text and a prior dummy-qualification receipt
hash. This file records operational launch authorization; it is not automated
review disposition or gate-state evaluation. Supplying arbitrary text cannot itself
establish the scientific or engineering authority described by the review protocol.

Continuation additionally requires the exact externally authorized receipt-chain
root (`expected_receipt_history`), supplied alongside the archive checkpoint.
Before writing a host `PHASE` receipt or launching any container, the controller
commits an immutable `operations/phase-<phase>-consumed.json` archive marker binding
both prelaunch roots and the phase decision. The resulting archive checkpoint is
carried into the first launch. Restoring a valid older receipt prefix therefore
does not erase the consumed attempt: its archive checkpoint no longer matches,
and the marker path cannot be reused. A crash between the two durable stores is
uncertain and denies continuation, never granting a retry. This protects the
cooperative retained chronology, not hostile joint rollback of every independent
anchor or forgery of new owner authorization.

Dummy qualification retains the separate initialize → named attempt → bounded
command → matching checkpoint or terminal failure chronology without native
phases. Its finite task set is `dummy-complete`, `dummy-interrupted`,
`dummy-sink-failure`, `dummy-timeout` and `dummy-limits-overflow`; command verbs are
limited to create, inspect, start and remove. The last task names a separately
reviewed mechanical qualification component, not a new native entry task or
permission to substitute scientific capture. Intentional timeout/overflow stays a
terminal failed attempt whose evidence may be assessed by qualification.

The host controller enforces five minutes for each attached cell process and a
45-minute cumulative active-execution deadline across the two phases. Elapsed
active time is retained in the host ledger; the inactive review interval and
image preparation are separate. Process termination, readers, exact-owned cleanup
and filesystem operations have finite operational reserves/latency, so these
deadlines are not absolute kernel wall-time guarantees. No timeout resets time,
reservations or identity. Cleanup verifies the unique name, owner token and full
container ID, then removes only that exact owned container. A negative lookup
cannot exclude delayed daemon completion and is uncertain, never accepted cleanup.
The controller never restarts the Docker daemon or unrelated services.

## Complete lifecycle and provenance

The evaluator container re-materializes a fixed list of acknowledged artifacts;
restoration cannot chase its own appended copy records. A validated offset index
is rebuilt once after exact-history recovery and updated on commits, avoiding a
whole-archive scan per restored file. Restored bytes and journal reads are bounded
and charged before materialization. Journal JSON reads have a 4 MiB bound. Synthetic
complete-bundle checks stay within staging; actual fixed-study usage still requires
dummy/development qualification and must stop on admission failure.

Actual eight-capture metadata creates dataset-derived membership afterward. No
dataset identity is invented before capture. The existing typed before-only adapters
give predictors only own before/action snapshots and fixed development templates.
The entire four-member bundle (maps and diagnostics) commits through the same
archive before a durable whole-membership exposure record and any evaluator fate
decoding. Evaluation/failure records use immutable append paths in that archive.
Renewal after exposure remains forbidden. No held-out event map is decoded by
membership or forecast assembly; compound control-plane metadata retains the
previously documented inline-future-summary limitation.

The image recipe reuses the verified Python base digest whose interpreter is
3.11.16 only as preparation input. It installs CPython 3.11.15 explicitly with the
verified uv 0.11.26 digest, performs locked sync, and asserts mujoco 3.12.0,
numpy 2.4.6, PyOpenGL 3.1.10 and glfw 2.10.2. Its source is a real exact Git bundle
checkout; head/tree are checked and its canonical repository URL is restored without
fetching. The coordinator must verify that URL from the source repository before
building. The local image ID, installed packages and native OSMesa library versions
must be observed and recorded during preparation/qualification. No render runs
during image preparation. Capture's existing renderer provenance remains truthful.

## Evidence limits and remaining holds

Guarded source checks deny native/generation imports before bounded test collection.
They cover seed derivation, archive admission/index/copy persistence, external
checkpoint mismatch, failed phases without retry, complete archive-backed sealing,
exposure across restart and failed backing without target release. The old capture
entrypoint remains explicitly held. The new launcher requires its exact authorization
file and guarded tests do not invoke it with launch authority.

Source acceptance is not actual bind/fsync, termination, OOM, ENOSPC, malformed-log,
sink-failure or restart qualification. Independent exact-head source review precedes
any controller/image qualification with separate dummy-purpose archives. The
coordinator's staged native launch decision follows those observations. No daemon-crash, power-loss or
hostile-process durability guarantee is claimed. General pytest/conftest, historical
32-transition/192-cell/qualification/frozen studies, ML models, seed/partition
replacement, six-helper execution, licence changes and gate advancement are absent.
Failed and inconclusive outcomes remain retained and do not imply EPS superiority.
