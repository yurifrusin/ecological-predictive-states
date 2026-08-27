# Research-software review protocol

## Purpose and authority

This tool-neutral protocol separates implementation, engineering review, scientific review,
ownership, empirical gate evaluation, and closeout. In particular, it distinguishes whether

```text
software behaves as designed
```

from whether

```text
the design tests what it is intended to test
```

CI supports review but does not substitute for engineering or scientific review. A PR-level
`SCIENTIFIC_PASS` is not an empirical result, and successful closeout does not imply gate
advancement. Exact-head owner authority remains mandatory before any closeout: implementation-merge
approval for the merge path or an explicit record-only-closeout decision for the terminal path.

The scientific authority order remains:

1. `docs/RESEARCH_CHARTER.md`;
2. `docs/EPS_BENCH_V0.md`;
3. `docs/MILESTONE_0.md`;
4. gate-specific frozen artifacts within their declared scopes;
5. this review protocol;
6. task-specific prompts.

This governance protocol does not override a higher scientific authority.

Every work package declares a review profile and an evidence class before implementation. Every
approval or review disposition applies only to the exact `(repository, pull request, head SHA)` it
names.

## Review profiles

### `DUAL_REVIEW`

`DUAL_REVIEW` requires independent engineering and scientific review. It is the default for
scientific code; simulator or apparatus contracts; schemas and labels; modality boundaries;
identity or provenance domains; benchmark definitions; evaluation or training logic; frozen
configuration; comparison fairness; gate logic; and future EPS/Unfrozen integration.

### `ENGINEERING_ONLY`

`ENGINEERING_ONLY` is permitted only for narrowly mechanical work that cannot alter scientific
interpretation. The PR must explain why scientific review is unnecessary. Examples include
non-semantic tooling, path or security hardening, mechanical CI maintenance, and mechanically
constrained implementation work. It still requires a separate engineering reviewer and conveys no
closeout authority. When the classification is uncertain, use `DUAL_REVIEW`.

### `SCIENTIFIC_ONLY`

`SCIENTIFIC_ONLY` is permitted only for a non-executable design memorandum or review that changes
no source, tests, dependencies, configurations, benchmark, or executable contract.

### `CLOSEOUT_ONLY`

`CLOSEOUT_ONLY` permits only the exact merge, record, tag, release, and synchronisation actions
named by owner approval. It governs a separately authorised closeout PR and `CLOSEOUT_AGENT`; it is
not a substitute for the review profile of an implementation or correction PR.

## Evidence classes

### `PUBLIC_REPOSITORY_ONLY`

All evidence needed for exact review is accessible through repository content, reproducible
generated artifacts, CI, and public or connected GitHub records.

### `PRIVATE_REVIEW_BUNDLE`

`PRIVATE_REVIEW_BUNDLE` is required when exact review depends on evidence that must not enter the
implementation PR or public repository. Examples include hidden benchmark episodes, private oracle
labels, secret partitions, licensed sensory data, unreleased renders, confidential human
validation, and private learned-extractor qualification data.

The minimum private-evidence receipt records:

- repository;
- PR number;
- exact reviewed SHA;
- review profile;
- evidence class;
- private review-manifest logical root;
- whole private-bundle file SHA-256;
- aggregate source, candidate, and report roots;
- reviewer access status;
- coverage examined;
- unresolved dispositions;
- retention or destruction posture.

A receipt does not make a private bundle trustworthy by itself. The reviewer must actually access
and assess the evidence on which the disposition depends.

## Roles

### `OWNER_PI`

The owner/PI determines scientific scope, resolves substantive design decisions, accepts or rejects
review findings, approves an exact pull-request number and head SHA, and authorises merge, tag,
release, record-only closeout, or gate advancement. The owner distinguishes approval of an exact
head for implementation merge from a terminal decision that prohibits implementation merge. No
implementation, coordination, or review agent may infer or manufacture owner authority. Passing
reviews do not oblige the owner to merge.

### `IMPLEMENTATION_AGENT`

The implementation agent implements a bounded authorised work package, creates tests and evidence,
records assumptions and limitations, and leaves the pull request open for review. Its terminal
status is only:

```text
IMPLEMENTED_PENDING_REVIEW
```

It may self-check but may not approve its own implementation.

### `ENGINEERING_REVIEWER`

The engineering reviewer operates independently from the implementation agent and makes no active
review commit. It assesses specification compliance, code correctness, validation independence,
failure behaviour, reproducibility, security and unsafe-path handling, portability, test
sufficiency, and scope discipline.

Allowed dispositions are:

```text
ENGINEERING_PASS
ENGINEERING_REQUEST_CHANGES
ENGINEERING_BLOCKED
```

Every disposition names the exact repository, PR, and head SHA reviewed.

### `SCIENTIFIC_REVIEWER`

The scientific reviewer remains independent of the implementation changes under review and makes
no active review commit. It assesses hypothesis validity, construct validity, privileged-information
leakage, label provenance, causal and representational confounds, fairness of future comparisons,
falsifiability, conceptual drift, and interpretation boundaries.

Prospective allowed dispositions are:

```text
SCIENTIFIC_PASS
SCIENTIFIC_REQUEST_CHANGES
SCIENTIFIC_INCONCLUSIVE
SCIENTIFIC_DESIGN_NO_GO
```

`SCIENTIFIC_DESIGN_NO_GO` means:

> The PR's construct, comparison, or apparatus cannot be made scientifically valid through a
> bounded correction within its authorised scope.

Every scientific disposition names the exact repository, PR, and head SHA reviewed and states:

```text
Phase-gate effect: NONE
```

### `CORRECTION_IMPLEMENTER`

The correction implementer may address owner-accepted findings. It reports exactly one worker
status for every stable finding ID:

```text
IMPLEMENTED_PENDING_VERIFICATION
NOT_IMPLEMENTED_WITH_REASON
OUT_OF_SCOPE_PENDING_OWNER
```

It may self-check but may not mark its own response verified, passed, resolved, or independently
approved. Corrections return to the original worker where practical.

### `ORCHESTRATOR_COORDINATOR`

The orchestrator/coordinator may track the canonical base, PR, exact head, CI, reviews, findings,
and closeout state; route work among worker and reviewer sessions; maintain stable finding IDs;
detect invalidated reviews after a head change; draft correction, reviewer, and closeout handoffs;
and verify that the correct evidence reached the correct reviewer.

It may not infer owner approval, convert CI into a review pass, inherit independent review authority
because it has broad context, convert PR-level review into an empirical gate decision, or count an
orchestration summary as an engineering or scientific review.

A conversation may coordinate and, at a different moment, enter an explicitly declared review
role. That review must still identify the repository, PR, exact SHA, role, evidence class, and
disposition.

### `EMPIRICAL_GATE_EVALUATOR`

The empirical gate evaluator is distinct from the scientific reviewer. It evaluates frozen
empirical evidence only at an authorised gate and returns the gate-specific decision vocabulary.
For later EPS empirical gates, use clearly separate terms such as:

```text
PROCEED
REVISE
STOP
```

unless an authoritative future gate defines other terms. A PR-level review can validate gate
machinery but cannot decide the empirical gate before the frozen evidence exists.

### `CLOSEOUT_AGENT`

The closeout agent operates in a separate session after either `MERGE_CONVERGENCE` and exact-head
owner merge approval or `TERMINAL_DISPOSITION_CONVERGENCE` and an exact owner
record-only-closeout decision. It revalidates the applicable exact SHA, clean branch state, CI
status, unresolved findings, authorised scope, and owner authority. It may merge, record, tag,
release, close, or synchronise only when explicitly authorised. A record-only closeout agent may
not repair or reinterpret the terminal implementation.

## Active and canonical evidence lifecycle

The governing rule is:

> Active review records must not modify the implementation PR they review. Final immutable records
> may be canonicalised only through a separately authorised closeout PR: after
> `MERGE_CONVERGENCE` and exact-head owner approval for an implementation that will merge, or after
> `TERMINAL_DISPOSITION_CONVERGENCE` and an exact owner record-only-closeout decision for an
> implementation that will not merge.

Active review evidence may be a read-only reviewer chat, GitHub review or comment, isolated
untracked review note, private evidence receipt, or the reviewer's final exact-head response.
Canonical final evidence may be immutable final review records added by the linked closeout PR, an
owner approval or terminal-decision record, a closeout record, and canonical Git and GitHub
history.

Implementation and correction PRs do not create or update their own active
`docs/reviews/pr-*.md` records. Correction evidence belongs in the worker response and PR body or
comment. Active reviewers do not commit. This active-record prohibition applies to both merge and
terminal paths. Final records are historical documents about the reviewed SHA and need not have
existed inside that SHA. They must not use self-referential placeholders. Prior historical records
remain immutable. A governance file such as `docs/reviews/README.md` is directory policy, not an
active review record.

## Exact-head rule and convergence

The review unit is:

```text
(repository, PR, head SHA)
```

Any head change invalidates earlier dispositions not renewed for the new head. A bounded delta
re-review is permitted, but it must still name and assess the new exact SHA.

### `MERGE_CONVERGENCE`

`MERGE_CONVERGENCE` requires every pass required by the selected review profile to name the same
exact implementation SHA and the owner to approve that PR and SHA for implementation merge. A pass
on SHA A plus a pass on SHA B is not merge convergence. Ordinary linked closeout may follow only
under the exact actions authorised by the owner.

### `TERMINAL_DISPOSITION_CONVERGENCE`

`TERMINAL_DISPOSITION_CONVERGENCE` requires every final review disposition required by the selected
profile—engineering and/or scientific—to name the same exact final implementation SHA. At least one
terminal reviewer disposition or owner decision prevents implementation merge, and the owner
explicitly authorises `RECORD_ONLY_CLOSEOUT` for that PR and SHA. No implementation commit from the
terminal PR is merged.

Terminal reviewer dispositions include `ENGINEERING_BLOCKED`, `SCIENTIFIC_INCONCLUSIVE`, and
`SCIENTIFIC_DESIGN_NO_GO`. A request for changes can also end on this path when the distinct owner
decision is `CORRECTION_NOT_AUTHORISED` or `CORRECTION_DECLINED_WITH_REASON`. Other owner terminal
decisions include `OWNER_REJECTED` and `OWNER_ABANDONED`. `SCIENTIFIC_INCONCLUSIVE` or another
non-pass disposition is terminal only when the exact owner decision ends the implementation path;
otherwise correction or further review may continue.

Reviewer disposition, owner decision, and implementation-merge authority are separate facts. A
terminal disposition is not a pass, and a record-only-closeout decision is not approval of the
implementation.

## Finding lifecycle

Review findings use stable identifiers. The recommended EPS formats are:

```text
EPS-ER<PR>-<NNNN>
EPS-SR<PR>-<NNNN>
```

Existing historical formats remain valid. Implementers use the worker statuses defined above.
Reviewers alone use:

```text
VERIFIED
NOT_VERIFIED
SUPERSEDED_WITH_EXPLANATION
```

Reviewers do not implement their own findings. Engineering review does not determine scientific
validity, and scientific review does not substitute for engineering review.

## Non-reconstructive public records

For `PRIVATE_REVIEW_BUNDLE`, a public receipt may disclose only these non-reconstructive bindings:
repository; PR number; exact reviewed SHA; review profile; evidence class; private review-manifest
logical root; whole private-bundle file SHA-256; aggregate source, candidate, and report roots;
aggregate counts and coverage; reviewer-access status; stable finding IDs and unresolved
dispositions; and retention or destruction posture. These values bind the exact private evidence
set without exposing its item-level contents.

Public records must not disclose hidden episode or item IDs, per-item hashes, private prompts or
answers, private oracle labels, hidden renders, private rationales that reveal expected outputs,
licensed source content, or metadata enabling reconstruction. Reviewers must not quote hidden
content in public GitHub comments.

## Closeout boundaries

### Work-package closeout

Gate 0B Slices 1, 2, and 3 and a governance amendment are examples of work packages. Typical
merge closeout follows `MERGE_CONVERGENCE`, merges the implementation PR and a limited documentation
closeout, and usually creates no tag or Release.

### Terminal record-only closeout

`RECORD_ONLY_CLOSEOUT` is the archival work-package path after
`TERMINAL_DISPOSITION_CONVERGENCE`. It is a governance and evidence-preservation operation, not
approval of the implementation. It must:

1. begin on a new documentation branch from current canonical `main`, not from the terminal
   implementation branch;
2. leave the implementation PR unmerged and close it or leave it open only as the owner directs;
3. add only immutable review, owner-decision, and terminal-closeout records through a separately
   authorised documentation PR;
4. record the repository, implementation PR, exact final head, canonical base, dispositions, stable
   finding IDs, owner decision, and reason no implementation merge occurred;
5. preserve the non-reconstructive public-record rules for private evidence;
6. have no gate, benchmark-freeze, tag, Release, or scientific-result effect unless each effect is
   separately authorised;
7. prohibit the closeout agent from repairing, completing, or reinterpreting the terminal design;
8. preserve the terminal head in GitHub history and bind any external exact-head review receipt.

No commit from the rejected, inconclusive, blocked, owner-rejected, or abandoned implementation PR
may be merged or cherry-picked as part of `RECORD_ONLY_CLOSEOUT`.

### Engineering milestone closeout

Full Gate 0B completion is an engineering milestone. Closeout verifies every deliverable and exit
criterion and creates an engineering-milestone tag and Release only if authorised.

### Benchmark or preregistration freeze

This boundary freezes assets, seeds, splits, identities, evaluation protocol, and approvals. It
creates a separate scientific-checkpoint tag and Release only if authorised.

### Empirical decision gate

This boundary evaluates frozen evidence, returns the gate's empirical decision, and creates a
separate gate checkpoint only if authorised.

These implications are prohibited:

```text
work-package merge
does not imply engineering milestone completion

engineering milestone completion
does not imply benchmark freeze

benchmark freeze
does not imply empirical success

record-only closeout
does not imply implementation acceptance

record-only closeout
does not merge the terminal implementation

record-only closeout
does not imply engineering milestone completion, benchmark freeze, empirical success,
or a scientific result
```

## Machine-readable automation deferral

No review-state JSON schema, approval bot, or automated review-convergence gate is authorised yet.
The human workflow has recently changed, private-evidence and supersession semantics are not frozen,
and premature automation can create false certainty.

Reconsideration may occur only:

```text
after at least one complete work package has used the canonical v2 protocol
from implementation through linked closeout without requiring governance
correction
```

Even then, automation requires a separate owner-authorised work package.

## Chat and worktree allocation

The recommended mapping is:

```text
original worker chat:
  IMPLEMENTATION_AGENT / CORRECTION_IMPLEMENTER

dedicated reviewer chat:
  ENGINEERING_REVIEWER for one PR and its correction rounds

scientific-review venue:
  explicit SCIENTIFIC_REVIEWER role for each exact head

coordination venue:
  ORCHESTRATOR_COORDINATOR unless explicitly entering a review role

fresh closeout chat:
  CLOSEOUT_AGENT

fresh next-slice chat:
  new IMPLEMENTATION_AGENT
```

The repository is authoritative, not chat memory.

## Permanent separation and anti-bureaucracy controls

1. Assign the review profile and evidence class before implementation.
2. Keep work packages small and consolidate correction waves where practical.
3. Report future hardening separately from merge-blocking defects.
4. Avoid status-only commits.
5. Use `ENGINEERING_ONLY` for genuinely mechanical work.
6. Do not require tags or Releases for every slice.
7. The implementation session cannot serve as its own independent engineering reviewer.
8. The correction implementer cannot verify its own findings.
9. Any head change requires renewal of every affected exact-head disposition and approval.
10. Gate advancement is separate from CI, PR review, and ordinary closeout.
11. Repository infrastructure success is not a scientific result.
12. The owner may decline to merge despite passing reviews.
13. Preserve terminal negative outcomes through owner-authorised `RECORD_ONLY_CLOSEOUT`; do not
    erase them merely because pass convergence is impossible.
14. Periodically assess whether the protocol is catching material problems.

## Historical compatibility

Older records remain valid historical evidence under the protocol that governed them. Older active
review files already committed in implementation PRs, including records that used the earlier
`SCIENTIFIC_NO_GO` terminology, are grandfathered and must not be renamed, modified, or
reinterpreted. The v2 active-record rule applies to this governance PR's own reviews and all future
work packages. Older rejected or abandoned PR evidence may be canonicalised prospectively only
through an owner-authorised `RECORD_ONLY_CLOSEOUT` from current canonical `main`; existing
historical records remain unchanged. This implementation PR must not create a review record for
itself.
