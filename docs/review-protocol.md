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
advancement. Exact-head owner approval remains mandatory before any authorised closeout.

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
constrained closeout. When the classification is uncertain, use `DUAL_REVIEW`.

### `SCIENTIFIC_ONLY`

`SCIENTIFIC_ONLY` is permitted only for a non-executable design memorandum or review that changes
no source, tests, dependencies, configurations, benchmark, or executable contract.

### `CLOSEOUT_ONLY`

`CLOSEOUT_ONLY` permits only the exact merge, record, tag, release, and synchronisation actions
named by owner approval.

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
- private bundle file SHA-256;
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
release, or gate advancement. No implementation, coordination, or review agent may infer or
manufacture owner approval. Passing reviews do not oblige the owner to merge.

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

The closeout agent operates in a separate session after review convergence and exact-head owner
approval. It revalidates the exact approved SHA, clean branch state, CI success, mergeability,
unresolved findings, authorised scope, and approval validity. It may merge, record, tag, release, or
synchronise only when explicitly authorised.

## Active and canonical evidence lifecycle

The governing rule is:

> Active review records must not modify the implementation PR they review. Final immutable review
> records may be canonicalised only in the separately authorised closeout PR after exact-head
> convergence and owner approval.

Active review evidence may be a read-only reviewer chat, GitHub review or comment, isolated
untracked review note, private evidence receipt, or the reviewer's final exact-head response.
Canonical final evidence may be immutable final review records added by the linked closeout PR, an
owner approval record, a closeout record, and canonical Git and GitHub history.

Implementation and correction PRs do not create or update their own active
`docs/reviews/pr-*.md` records. Correction evidence belongs in the worker response and PR body or
comment. Active reviewers do not commit. Final records are historical documents about the reviewed
SHA and need not have existed inside that SHA. They must not use self-referential placeholders.
Prior historical records remain immutable. A governance file such as `docs/reviews/README.md` is
directory policy, not an active review record.

## Exact-head rule and convergence

The review unit is:

```text
(repository, PR, head SHA)
```

Any head change invalidates earlier dispositions not renewed for the new head. A bounded delta
re-review is permitted, but it must still name and assess the new exact SHA. Review convergence
requires every pass required by the selected review profile on the same SHA. A pass on SHA A plus a
pass on SHA B is not convergence. Owner approval names the converged SHA.

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

For `PRIVATE_REVIEW_BUNDLE`, public records may disclose only aggregate roots, aggregate counts,
aggregate coverage, stable finding IDs, dispositions, and retention or destruction status. They
must not disclose hidden episode or item IDs, per-item hashes, private prompts or answers, private
oracle labels, hidden renders, private rationales that reveal expected outputs, licensed source
content, or metadata enabling reconstruction. Reviewers must not quote hidden content in public
GitHub comments.

## Closeout boundaries

### Work-package closeout

Gate 0B Slices 1, 2, and 3 and a governance amendment are examples of work packages. Typical
closeout merges the implementation PR and a limited documentation closeout. It usually creates no
tag or Release.

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
13. Periodically assess whether the protocol is catching material problems.

## Historical compatibility

Older records remain valid historical evidence under the protocol that governed them. Older active
review files already committed in implementation PRs, including records that used the earlier
`SCIENTIFIC_NO_GO` terminology, are grandfathered and must not be renamed, modified, or
reinterpreted. The v2 active-record rule applies to this governance PR's own reviews and all future
work packages. This implementation PR must not create a review record for itself.
