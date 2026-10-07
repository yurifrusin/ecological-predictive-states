# Fixed public event-cell mapping bridge

**DUAL_REVIEW / PUBLIC_REPOSITORY_ONLY**. **IMPLEMENTED_PENDING_REVIEW**.
Phase-gate effect **NONE**. This source package implements the coordinator's bounded
source decision of 2026-10-08, following corrected paper assessment SHA256
`084bfeba035f34c8ded30a25eb168d57ba1c9a881d20ff0d403f43d8ddcf0370`
and independent design review receipt
`f74a7a605f458e75e9fb05357041e2272f224acf7fb839bbe9d75024bc88299a`.
Base is PR76 merge `7aa329d7abf70d65f0648fbfb6727a1589841be7`.

This callable source bridge connects existing exact producer arguments, independent
face-reference qualification, observed-only Projection and PR76 probabilities for
one public cell. There is no CLI, operation admission, model, new renderer, controller,
cohort, partition or historical collector invocation. Real invocation requires a
separate exact-head admission and explicit operation decision. Source tests use
handwritten fake returns only and confer no physical qualification.

## Fixed prospective cell

Camera: untilted 32×32, 90-degree vertical field, origin `(ell,-3,1)`;
pixel ray `((2c-31)/32,1,(31-2r)/32)`. Exact two boxes:

- A: x `[27/10,29/10]`, y `[0,1/4]`, z `[1/5,7/5]`.
- B: x `[22/5,24/5]`, y `[3,13/4]`, z `[1/5,7/5]`.
- Floor: z=0, support half-extents `(1/10,1/10)`.

Prefix lateral positions are `-1/4,0`; executed action is `+1/4`.
Announced opposed lateral actions are `-1/2,+1/2`, forecasting their distinct
positions at episode index2. No forward/yaw motion. These are public evaluator
instruments, never learner fields. Fixed paper grid role labels are
0/background,1/floor,2/A,3/B:

|Position|A rows14–19: columns|B rows15–17: columns|Floor row21: columns|
|---|---|---|---|
|-1/4|31|28|17|
|0|29,30|27,28|15,16|
|-1/2|empty|29|18|
|+1/2|27,28|26|13|

All unlisted cells are background. Observed contacts change from empty to `{A,B}`;
floor never contacts either box. At future left the contact list is empty;
right retains `{A,B}`. These future contacts are not event targets. All three
regions are prefix-known. In appearance/disappearance/gain/loss order, A forecasts
have labels `0101` left and `0011` right; B/floor both `0011`. Expected Brier scores
are stable `(1/2,1/2)`, replay `(1/4,1/12)`, half `(1/4,1/4)`, saturation `(1/6,0)`.
Controls use observed Projection features only; no geometry-based action oracle exists.

## Access, chronology and retained evidence

`run_mapping` and retained-only `inspect` require an exact revalidated typed
`ModalityPermissionSet` containing only `PRIVILEGED_GENERATION_RECORDS`, before
filesystem/provider/identity/clock/callback access. This permission is distinct
from directory protection and later operation approval. Internal projections retain
the existing exact ecological permission set; neither metric fields nor raw roles,
tokens, nonce or provenance enter the structured/unstructured learner features.

The caller supplies exact source context (HEAD/tree, source-file hashes, runtime
versions, fixed-cell digest), independent source rechecking, external-byte accounting
and directory protection callbacks. An admission must bind these to the reviewed
head and actual environment; no-op callbacks in fake tests do not qualify operation.
Opaque role associations are randomized once per episode and retained reproducibly.

Every call has retained exact arguments before invocation. Raw and reference returns
are retained canonically before qualification or a post-return timeout check. Each
position has one raw plus one audit call, no retry/refill. Complete grid agreement,
no tie/sampled boundary and complete reference footprint evidence are prerequisites
for an observed identity. Both prefixes reconstruct the lifecycle and both endpoint
contacts; both action projections and all eight complete PR76 control candidates,
with lossless encodings, enter a before-only global hash seal before any future call.
Each future is acquired/qualified once and held immutably. Each control receives a
fresh evaluator-local callback permitting precisely one fetch of that retained target;
scoring and retained inspection make zero additional physical calls.

Inspector reconstruction checks attempted arguments, ordered call prefix, complete
before seal, exact role grid, mapping, source/binding, candidates and reports. It
uses retained returns and qualified/projection utilities, never producer/audit calls.
Final file hashes bind all retained originals; admission/retention must bind the
whole bundle externally. Volatile closure metadata stays outside scientific hashes.

## Bounds and interpretation

Four raw plus four audits maximum; 60 cooperative seconds from before output claim
through closure; 16 MiB inclusive accounting. Formula is
`2*(retained bytes + measured external bytes + 128 KiB logs + current reserve) + 4 MiB review`.
Doubling reserves the first archive and duplicates of external wrapper/admission/partials;
external callback must include all such bytes without omissions. A 1 MiB reserve funds
one at-most-512 KiB canonical fixed-adapter return plus terminal closure before each call.
Unexpected oversized/out-of-contract returns or external growth can prevent safe retention;
they cannot produce PASS and require an inconclusive admission receipt with all extant
partials preserved. This source does not promise a hard host memory/time/storage controller.
Returned physical objects are never silently discarded and retried as fresh observations.
No dependencies/installations/GPU are required locally.

Completion and qualification are separate from cell outcome. PASS requires all eight
calls, exact frozen-cell agreement and reconstructed reports within bounds; it qualifies
only this exact public cell. Qualified producer/reference agreement contradicting the
paper support/events is FAIL, even if collection stops at that contradiction.
Producer/reference disagreement has no independently qualified truth and is explicitly
INCONCLUSIVE, as are ties/boundaries, identity/provenance uncertainty, interrupted calls
and resource failure. FAIL does not refute EPS. Partial evidence remains exposed and
retained, with no exclusions/reruns/cap changes or efficacy inference.

The source guard patches physical producer/reference and historical collector call paths,
blocks native/model imports, disables pytest plugin autoload/conftest, and runs only
handwritten validation/inspection including denial-before-side-effects tests. Scoped
Linux/Windows CI excludes unrelated native qualification jobs on this source branch.

No graph advantage, prevalence, independent-group uncertainty, physical A2 or two-family
qualification follows. Original 0B/0C/0E remain incomplete; 0D is not completed and no
model/stub authority exists. Recovery18/21, Slice6 FAILED_CLOSED and held48 persist.
Closed motion/learning failures and exposed prefixes stay closed; no old outcome access.
The charter matched-budget appearance hypothesis remains **UNTESTED**.

PR77 bounded correction: EPS-ER77-0001 and EPS-ER77-0002 are
**IMPLEMENTED_PENDING_VERIFICATION**, with separate renewed reviews required.
The new source pin includes transitive `appearance.py` and `utils/seeding.py`;
the guarded checker rejects any imported local runtime module outside that pin,
and virtual byte-change regressions verify both added dependencies are checked.
Historical source lists remain unchanged. Terminal completion/failure/analysis
vocabulary and structure are validated before disposition acceptance. A recorded
operational failure requires UNRESOLVED/INCONCLUSIVE regardless of whether all
calls completed; internally contradictory PASS receipts reject. Qualified
Contradiction remains FAIL only with supporting reconstructed evidence. Complete
and incomplete operational INCONCLUSIVE packets remain representable.

Local source validation used the existing eps-mask-models environment: Python3.11.15,
uv0.11.26, ruff0.16.4 and mypy1.20.2. Locked/offline sync dry-run checked38 packages
and reported no changes. Scoped lint, formatting, fresh typing and17 guarded fake
checks passed, including handwritten smoke construction, validation and retained
inspection. The bounded correction passes36 guarded checks, scoped lint/format,
fresh typing and diff checking. No package was installed and no physical
producer/audit was executed.
