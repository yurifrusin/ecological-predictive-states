# A1 retained output source contract

Review profile: `DUAL_REVIEW`. Evidence class: `PUBLIC_REPOSITORY_ONLY`.
Phase-gate effect: `NONE`. Status: `IMPLEMENTED_PENDING_REVIEW`.
Source base: `7c966dd21a615059ed46ff0548f102ee4f89fa2c`.

This bounded package adds optional typed synchronous publication to every write
owned by `generate_dataset`, including NumPy, RGB PNG, canonical JSON and volatile
`run.json`. An explicit writer is passed through generation helpers. Without an
injected publisher, serialized bytes and generation behavior are unchanged.
With one, each artifact receives a flushed start receipt before serialization;
successful completion returns only after archive chunks and the commit receipt
have been flushed and fsynced. That return is the artifact ACK. The caller must
wait for it before further publication or the next cell.

Retained generation requires an explicit canonical `publication_prefix`, selected
by the trusted caller from the fixed study's cell identity. Distinct dataset
calls use distinct stable cell prefixes in the SAME archive; changing an attempt
namespace does not create another budget, clear history or permit overwriting a
previously reserved path. This source package validates path syntax, not the
scientific membership-to-prefix mapping; that mapping belongs to the absent driver.

## Logical capacity and admission

The default application-byte allocation is 255 MiB staging + 1 MiB shared memory
+ 768 MiB archive = 1 GiB. The archive allocation includes every frame, start,
staging reservation, artifact duplicate, log, forecast journal, exposure record,
incomplete suffix and an 8 MiB terminal reserve. Reserve is inside the allocation,
not extra capacity. Tests use smaller allocations. No compaction, truncation,
deletion or new attempt namespace refunds capacity. Failed writes are charged
conservatively by their durable reservation, even if fewer bytes reach staging.

The writer reserves each staging write in the archive before writing it. A fresh
writer and exact-history resume inherit the archive's staging reservations.
Every archive frame is admitted against its exact encoded length plus framing
before writing. Ordinary publication cannot consume the terminal reserve;
bounded failure receipts can. Finite terminal capacity can itself exhaust;
failure of the sink or terminal admission denies continuation, never grants ACK.

This is a cooperative logical application-byte contract, not a host filesystem
quota or whole-host memory guarantee. Filesystem allocation overhead, filesystem
journals, Docker daemon storage, native allocator memory and unflushed bytes are
outside that claim. A future live controller must route ALL writable application
channels through this budget (including logs and lifecycle journals), enforce
the declared shared-memory ceiling, and reject other channels. That integration
is absent here; existing independent writers are not covered by inference.

## Exact archive format and recovery

One append-only file uses a four-byte unsigned big-endian payload length followed
by canonical UTF-8 JSON (sorted keys, compact separators, no NaN). Records have
strict consecutive integer `sequence` values starting at zero. The payload bound
is 100 KiB; binary chunks are at most 64 KiB before base64 encoding. Records are:

- `STUDY`: fixed study SHA-256 and exact allocation, only at sequence zero.
- `START`: canonical relative artifact path, reserved once across the history.
- `STAGE`: positive byte reservation for the current artifact, charged globally.
- `CHUNK`: base64 bytes for the current reserved artifact, bounded by its staging
  reservation; chunks preserve the artifact bytes without reserialization.
- `COMMIT`: exact total length and SHA-256 of all its chunks.
- `FAILURE`: bounded reason, preserving the currently incomplete artifact.

Unknown records, noncanonical or oversized frames, invalid sequence/path/length/
hash, unreserved bytes and duplicates fail closed. Absolute paths, traversal,
backslashes, empty components, symlinks and multiply linked files are denied.
Transfer checks file identity/stat stability around reads. These cooperative
checks do not constitute confinement against malicious concurrent filesystem
mutation. No native module is imported by output or retention helpers.

Recovery requires an externally retained expected whole-history SHA-256, the
same fixed study identity and the same allocation. Missing or changed history
denies resume. Creation is allowed only for the trusted study's initial history;
the live controller must retain that fixed archive location/identity across
attempts and must never invoke initial creation to replace missing history.
This source library cannot prevent an operator deleting history or supplying a
different directory; doing so cannot establish a prospective study.

Recovery validates complete frames and artifact commits, reports the committed
artifact prefix, incomplete paths and any torn trailing frame bytes, and leaves
all bytes intact. An interrupted object is `INCOMPLETE_UNCOMMITTED`, never a
successful or qualified artifact. Any incomplete artifact or torn suffix makes
recovery read-only; this package does not implement a restart after such a
failure. Already acknowledged artifacts and durable start/failure receipts
remain inspectable. No claim covers every unflushed native byte, intervals
between native read checkpoints, sink failure or host power loss. Native reads
currently finish before frame publication starts.

Generation propagates serialization/publication errors immediately; its caller
must stop that generation attempt. After a successfully flushed `FAILURE`, the
archive API permits the same process to append different paths, retaining all
failed-path bytes, staging charges and the failure receipt. It does not allow
reuse of the failed path, and it does not grant permission to retry or replace a
fixed scientific cell. A poisoned sink permits no continuation. Restart from an
archive containing any incomplete artifact is denied even if a failure receipt
was flushed; only read-only recovery remains available in this source package.

## Remaining integration and scientific holds

Capture entrypoints remain disabled. This package adds neither Docker transport
nor a server, experiment driver, lifecycle-journal integration, native capture,
development qualification, held-out release or empirical gate decision. The
existing Linux/Python/WSL runtime guard remains unchanged; a Docker candidate
requires separate explicit qualification and must not spoof that guard.
Any later qualification of development cells 0 and 1 must count as the first
two of the fixed eight contexts/48 render-read pairs, with the native source head
fixed beforehand and evidence retained, rather than additional captures.

Cases, seeds, partitions, appearance, action labels, scientific logical hashes,
opaque-ID rules and artifact formats are unchanged. Retention metadata is a
privileged control-plane artifact, not a learner input. Existing failed studies
and negative results remain historical. No model, six-helper programme, review
automation, licence change or historical qualification run is added.

Guarded synthetic checks cover byte equivalence, exact-cap admission and a
one-byte-over limit, persisted staging accounting, interruption, terminal reserve,
failed flush without ACK, prefix/suffix recovery, changed/missing study history
and malformed frames. Native smoke generation, validation and rendered
inspection remain held; source checks cannot establish runtime qualification.
