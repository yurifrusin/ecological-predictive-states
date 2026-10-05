# A1 prospective controls and source lifecycle

Review profile: `DUAL_REVIEW`. Evidence class: `PUBLIC_REPOSITORY_ONLY`.
Phase-gate effect: `NONE`. Status: `IMPLEMENTED_PENDING_REVIEW`.
This source protocol is not a scientific freeze, independent review disposition,
launch permission, merge approval or empirical result.

## Fixed scope and lineage

The source base is commit `83c505829779ba10a8b86c42352f9a0191ed2b61`,
tree `d2c6193fa4ef5a417fa45dc813031759c4d0ccbe`, imported from the
owner-authorized complete Git bundle. No source was reconstructed from selected
ZIP files. Implementation uses an isolated branch and preserves the original
source candidate, eight configurations, seed 1729, single-occluder geometry,
legacy appearance, height 1.25, vertical FOV 55 degrees, 160x120 raster, actions
-0.7/+0.7 and existing development/held-out order. No configurations, assets,
seeds, partitions or model contracts change.

The four held-out cases are a finite sanity comparison on one geometry.
No EPS superiority, statistical generalization, appearance-OOD robustness,
learned perception, memory or A2 inference follows. Model work and Gates 0C/0D
remain separately unauthorized. Prior held-out outcome exposure is unknown at
implementation time: any future prospective evaluation requires the owner's
confirmation of no earlier exposure and separate exact-head launch authority.

## Exactly one additional alignment control

Keep the original same-action Hamming-nearest-template selection and the original unaligned mask copier unchanged. For each held-out view, compare every fixed development template with the same `(action name, signed delta)` using the original owned-boundary descriptor and count unequal descriptor pixels. Equal distances retain immutable development order. Select once; the aligned control uses that exact selected template. It may not select a second template after alignment or inspect any held-out target.

Add one global integer horizontal translation, derived exclusively from selected-template BEFORE contours and the case's own BEFORE contours. This is a simple image-plane copying control, not a registration platform. It adds a challenge to the original comparison; it does not guarantee better comparator performance. Rank pairing can mispair contours, and one global translation cannot correct scale or deletion-band width.

1. Keep only frame-0 `HORIZONTAL` axis `OCCLUDING_CONTOUR` elements (the schema's horizontal axis means boundaries between horizontal neighbours). Use both defined owner sides. Opaque IDs and segmentation label numbers do not enter matching.
2. Group each view's edges by `(row, owner_side)`. Sort each group by column. Pair the first edge with the first, second with second, through the shorter group. Boundary input order must not matter. Duplicate `(row, column, owner_side)` entries are invalid input and fail closed rather than acquiring extra weight. Do not match across rows or owner sides, infer geometry, or optimize correspondence.
3. For every paired edge compute integer `heldout.column − template.column`. Sort this fixed list and use its lower median, at zero-based index `(number_of_pairs − 1) // 2`. This is always an integer; even-size ties deterministically choose the lower central value, with no averaging or rounding. There is no shift search, scale, vertical translation, per-row warping or parameter.
4. If there are zero pairs, use shift zero, set the lower-median index to null and report `NO_ALIGNMENT_SUPPORT`. Do not compute or invent a median index in this case. Unequal group cardinalities retain all rank-paired edges and report unmatched counts on each side. They do not drop or exclude the case. These deterministic pairing limitations are part of the control's interpretation.
5. Translate every selected-template deletion-mask pixel at `(r,c)` to `(r,c+shift)` without wrap or interpolation. Out-of-raster values are discarded; newly exposed columns are zero. No clipping correction, renormalization or target-derived filling. Missing support still produces an aligned-control forecast, numerically equal to the unaligned mask.

Seal diagnostics with the forecast: selected development ordinal, original Hamming distance, number of eligible edges and pairs, unmatched edges per view, support status, lower-median index, final integer shift, minimum/maximum paired displacement (null without support), number of paired displacements differing from that shift (zero without support), count of nonzero mask pixels before translation, retained nonzero pixels and clipped nonzero pixels. A zero shift, disagreement among paired displacements or severe clipping stays reportable. Diagnostics never determine inclusion or retrospective comparator choice. There is exactly one aligned comparator.

## Signed-action diagnostic and interpretation

For each held-out case compute the original boundary rule twice on the identical immutable own-before snapshot: once with its correct fixed action and once with the opposite action name/sign at the same magnitude. Change only those action fields; reject nonzero forward/yaw or mismatched name/sign. Neither forecast receives its opposite-action episode, paired endpoint, other held-out before view, target, correspondence, transport or provenance. The wrong-action control is the boundary rule only, not a second copying comparator.

Seal both maps before held-out target access. Report their map equality, number of unequal pixels, maximum absolute difference and, when interpretable, AP(correct), AP(wrong) and signed AP difference against the case's own target. A changed map establishes action sensitivity only. A positive AP difference supports finite direction-specific localization on that case; equality or a nonpositive difference provides no such support. Report every case and sign without choosing a numerical empirical threshold later. No pooled significance or aggregate success statistic.

Original primary assessment remains separately named: the original boundary rule must strictly exceed the original unaligned copier's AP in all four held-out cases, with every case interpretable. Add a separately named additional-alignment-control observation: strict improvement over the aligned copier in every interpretable case, requiring all four interpretable cases for an all-four statement. Add a separately named action-specificity observation: strictly positive correct-minus-wrong AP in all four interpretable cases. These do not redefine the original primary criterion. An original-primary positive alongside rule parity or inferiority to the aligned copier is expressly limited by that additional control. Failures or absent direction sensitivity do not by themselves falsify the charter hypothesis.

## Evaluation domain and complete coverage

Retain grouped-score-tie AP, evaluated only on canonical stable transport (0) versus occluding deletion (1). Retain all six counts 0…5, their sum equal to raster size, excluded-category counts for exit (2), ambiguity (3), uncontrolled (4) and unresolved occlusion (5), and coverage of all eight members/four held-out cases. Reject wrong dtype, wrong shape, values outside 0…5 and binding mismatches.

A held-out case is `INCONCLUSIVE` if it has no deletion positives, no stable negatives, or any unresolved occlusion. Preserve all reasons, even if multiple apply. AP is absent/None when either class is missing; if unresolved pixels coexist with both classes, AP may be reported as descriptive only and never used as a conclusive criterion. Never replace an inconclusive case or omit it from reports. A finite case failing a strict comparison remains a retained negative. Categories 2–4 are reported rather than relabelled as deletion or silently removed from the coverage account.

## Minimal membership, snapshot and sealing lifecycle

This is cooperative typed access and chronology enforcement, not confinement of malicious Python. No OS sandbox or review-state automation is proposed.

1. Fix source identity, algorithm version/addendum identity and immutable ordered eight-member membership before assembly. Control-plane membership binds the configuration/action/partition identity and typed canonical dataset logical identity plus canonical episode/transition artifact identifiers; predictors receive no configuration, paths, source identities or generation instruments. Reject duplicate ordinals/identities, unknown partitions/actions, changed ordering, missing members and dimensions inconsistent with the fixed study.
2. Loader checks required modality permissions before artifact access. Project an independently owned, immutable snapshot of action, before segmentation, exactly visible opaque references and frame-0 boundaries. Deeply snapshot nested action/boundary values, not merely the dataclass or array; mutation of source objects cannot change an accepted snapshot. Labels are equality tokens only. No after, events, transport, correspondences, depth, camera, raw IDs, geometry, hidden roots or full manifest reaches a predictor.
3. Establish exactly four development snapshots and their canonical binary deletion templates in fixed order. Bind and lock their membership, before views, action, target-derived mask and canonical provenance. Do not accept arbitrary caller-supplied templates, append held-out members or reorder ties. Development outcome access occurs only in this development assembly stage.
4. For each held-out member, predictor input consists solely of its own before/action snapshot and the already fixed development template tuple. Invoke isolated predictor calls without access to another held-out snapshot, file paths, target loaders or a shared held-out collection. The coordinator may validate membership but must not expose that collection as forecast input.
5. Atomically commit one immutable forecast bundle containing ALL FOUR held-out members before making ANY held-out target accessible to the evaluator. Each member contains original rule, unaligned copier, aligned copier, wrong-action rule and diagnostics. Validate completeness/unique membership first. Bind source head/tree or equivalent immutable source identity, ordered membership root, field-scope/version, own-view digest, correct and switched action digests, selected-template/development root, algorithm identity, array dtype/shape/content and diagnostic values. Hash canonical logical content; keep timestamps, host paths and volatile run metadata outside scientific hashes. A partial bundle, mutable backing data or failed commit does not release targets. Before any exposure, a changed dependency invalidates the seal and requires a fresh bundle. That renewal is forbidden after exposure under step 6.
6. A distinct protected evaluation component checks the complete committed bundle and durably persists an irreversible target-exposure record bound to the fixed membership and committed attempt BEFORE the first held-out target release. Failure to persist denies release. Mark the entire fixed held-out membership exposed conservatively even if release or evaluation subsequently fails or exposes only part of a target. Target envelopes contain observed immutable metadata separately derived from validated actual canonical dataset/transition records and compare it to the requested member, including dataset identity, episode/transition, own-before identity, actual action, canonical before-fate artifact and canonical provenance. Reject duplicate/unknown/missing target members, reused targets for a different case, shape/action/before/source/membership mismatches and modified forecasts. No target reference or outcome is returned to predictors. Dataset provenance may be checked by the evaluator/control plane; it is not an ecological predictor input.
7. After any partial or full exposure, changed forecasts, source, views, templates, membership, ordering or algorithm cannot regain prospectivity through a new seal, reset, retry or namespace. Preserve the exposed attempt and its results/failure; stop further prospective evaluation for that changed attempt. Evaluation of the original unchanged committed bundle may finish, retaining exposure history. No automatic new partition, replacement case or launch authority follows. A future study requires a distinct owner decision; the current eight members remain fixed. The exposure record is experiment chronology, not review-state automation or an OS confinement claim.

Inward cases are especially hazardous: one transition's AFTER endpoint can resemble or equal another case's BEFORE endpoint, and reversing endpoints could pool held-out information. Do not construct forecasts by swapping endpoints or permit cross-case held-out views as context. All forecasts must be sealed together before any held-out outcome read, even where public modality permissions would permit those outcome artifacts. Permission is not chronology.

The typed source lifecycle assembles four development templates, invokes each
held-out forecast with only its own before view and that fixed template tuple,
and commits one complete bundle before evaluator target release. Predictors do
not receive control-plane metadata or readers. The evaluator retains the full
maps, diagnostics, exposure record, per-case results and failure types in the
protected journal directory. The journal filename is fixed for this study and
must be retained across process restarts and attempts; changing an output
namespace cannot clear its chronology. This cooperative contract does not
prevent an operator deleting the protected history or supplying a different
protected directory. Such action cannot establish a prospective study.

Dataset identity uses the existing canonical dataset logical SHA-256; member
uniqueness uses dataset identity, canonical episode ID and canonical transition
artifact logical identity. Distinct datasets may all retain episode-000000.
Observed target metadata is separate from the requested Member and is derived
from the actual validated manifest, transition and own-before snapshot; every
dataset, episode, transition, before, action, fate-artifact and canonical provenance
binding is compared at evaluation. Predictors receive none of this metadata.

Canonical before-fate identity is the existing artifact logical SHA-256, obtained
from validated metadata without decoding held-out event pixels. Target decoding
must validate that identity at release. Dataset integration remains dependency
injected and cannot authorize native capture or real-data evaluation.

## Source checks and remaining holds

Guarded CPU-only synthetic tests cover original retrieval and scores, sorted
rank pairing, lower-median translation, unmatched and absent support, clipping,
wrong-action own-view forecasts, immutable snapshots, canonical categories,
complete bundle retention, missing or changed members, persistence failure and
irreversible exposure after partial evaluation. Checks deny native, OpenGL,
GLFW, generation and canonical renderer imports before selected test collection.
Both capture entrypoints fail before producer invocation. General test collection,
native smoke generation, validation, rendered inspection, real-data scoring,
qualification and experiment launch are unperformed and held.

Offline locked sync uses CPython 3.11.15 and uv 0.11.26; the lockfile is unchanged.
Source lint, formatting, typing and diff checks are recorded in the worker evidence.
These checks establish source behavior only and are not independent approval.
Frozen Slice 6 remains FAILED_CLOSED, Gate 0B incomplete, recovery 18/21 with
three wrapper gaps, dummy execution unreported, hard aggregate memory/output
caps unresolved and six helpers parked. No gate, merge, PR, push, capture, model,
private outcome access or scientific efficacy claim is authorized by this package.
