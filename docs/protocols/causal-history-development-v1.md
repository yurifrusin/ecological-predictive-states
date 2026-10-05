# Causal-history development v1 — prospective design

Review profile: `DUAL_REVIEW`. Evidence class: `PUBLIC_REPOSITORY_ONLY`.
Phase-gate effect: `NONE`. Status: `IMPLEMENTED_PENDING_REVIEW`.
Source base: `a1bbcaf2e634c0c24c27ea0e0ed4aad5bd180151`.
This memorandum specifies a development question; it implements no sequence,
fixture, predictor or experiment. Numeric fixtures are NOT FROZEN; launch is disabled.

## Authority, question and historical boundary

The owner has delegated bounded scientific development decisions to the EPS
coordinator. Under that standing authority the coordinator proposes this explicit
scope exception: design, and subsequently separately authorize implementation of,
a six-sequence single-occluder causal-history diagnostic while Gate 0B remains
incomplete. Independent engineering and scientific review must assess the exception
and design. Acceptance of this document does not start or complete Gate 0C, alter
milestone criteria, freeze a benchmark, authorize capture or advance any gate.
Implementation and launch each need a distinct bounded decision. No machine-learning
model is permitted before explicit Gate 0D authorization.

The question is whether this particular causal optical state discards distinctions
needed to predict an already observed surface's future visible mask. It is not an
appearance-OOD comparison or a test of the charter's primary hypothesis. Retaining
an opaque token through occlusion is not predicting where its surface will reappear.

Completed A1 runtime source `624f2e0` remains immutable and exposed; its finite
positive comparison is contextual evidence only, represented by the
[non-reconstructive aggregate receipt](https://github.com/yurifrusin/ecological-predictive-states/pull/34#issuecomment-6003710363).
No private items are needed for this design. Never rerun or reuse A1 held-out cases
as prospective data. Preserve old failures, 18/21 recovery and its gaps, frozen
appearance inputs, FAILED_CLOSED Slice 6 and the parked six-helper programme.
This package changes none of them. Development uses distinct new identities,
seeds and simple appearance assets, never final benchmark roots or appearances.

## Fixed conceptual membership and timing

There are exactly three pairs, at most six short sequences, with two members per
pair and one decision/target per member. All surfaces are static; only lateral
observer commands are admitted. Forward and yaw are zero. No dynamics, search,
additional scene family, parameter fitting or case replacement is allowed.

| Pair | Frames and decision | Intended discriminator |
| --- | --- | --- |
| 1: current-view aliasing | frames 0,1,2; decision t=1 after action u0; target frame 2 after announced u1 | Members have equal current optical views and equal u1, different completed histories and different future background masks. Test whether history separates what the current view aliases. |
| 2: complete occlusion | frames 0,1,2,3,4; decision t=3 after u0,u1,u2; target frame 4 after announced u3 | Background observed at 0 and 1, completely invisible at 2 and 3, revealed at 4. Match decision views and u3 across members; earlier optical history should carry the predictive distinction. |
| 3: shared ambiguity control | frames 0,1,2; decision t=1 after u0; target frame 2 after announced u1 | Equal full RGB/action histories AND equal oracle optical input histories, including opaque association modulo renaming, but different future masks because hidden static extent differs. |

Every member's designated background surface must have a nonempty observed mask
before the decision. The designation is evaluator-only; each rule forecasts ALL
previously observed tokens and receives no semantic target name. Pair 2 requires
at least one usable completed background flow before complete occlusion. Pair 3
must differ only in geometry outside all past optical evidence, including flow,
not through a changed appearance cue or different future command. Pair 1 and 2
must have distinguishable matched-window RGB/action histories; failure to achieve
this cannot be labeled E-specific information loss.

The numeric table is a later prerequisite, not supplied by this conceptual table.
Before any render it must bind all six members, exact geometry, lateral positions,
signed actions, raster, seeds, appearance, expected view/target relations, target
surface designation and source/renderer identity. Analytic fixture reasoning may
choose those constants prospectively; no rendered-output search is permitted.
The existing fixed wide background may not allow complete occlusion or Pair 3.
The smallest candidate extension is configurable static background/occluder box
width and lateral placement, keeping the existing scene family and fixed camera
orientation. A later design must demonstrate feasibility of all pair relations
and freeze the numeric table under independent review BEFORE capture. If exact
RGB/optical equalities are infeasible, stop this v1 design rather than relax them
or substitute cases after outcomes. This document is not that numeric freeze.

## Sequence identity and observation boundary

One compiled static scene, one episode and one randomized raw-to-opaque mapping
serve each sequence across every pose. The mapping is drawn once per episode,
independent of geometry/action/outcome, and remains privileged. Never concatenate
independent two-frame episodes to manufacture memory. Each indexed pose is captured
once, including equal positions reached at different indices; adjacent transitions
reference the same captured frame at their shared index, rather than recapture it.

A new sequence logical identity must canonically bind the version, ordered frame
indices and content identities, completed transition identities/actions, fixed
fixture/membership identity and renderer provenance. Existing two-frame episode
identities cannot silently stand for sequences. Source/fixture and raw mapping
bindings belong to the trusted control plane, not the predictor. Volatile paths,
host names and timestamps stay outside scientific hashes. Keep failed capture,
partial publication and exposure history; new directories cannot erase an attempt.

At decision t, the available window is frames 0..t and executed u0..u(t-1), plus
announced next command ut. The comparator observation window is exactly the same
RGB frames and commands. Flow for i->i+1 becomes available only after frame i+1
exists and only if i+1<=t. No t->t+1 flow, future correspondence, future mask,
counterfactual segmentation, hidden surface inventory, depth, pose, raw ID, world
coordinates or generation record reaches either rule. Oracle past segmentation,
boundaries, completed optical flow and past identity association are privileged
oracle ecological inputs: this is an association ceiling, not learned RGB extraction.
Shared RGB ambiguity is judged against RGB/action evidence, not a claim that RGB
alone could recover the supplied oracle associations.

Reuse typed fail-closed permissions and immutable owned projections. Existing
`ecological_only` permissions alone do not enforce time: a sequence projection
must reject future indices before artifact access and strip compound records of
future summaries and control metadata. Keep analytic assignment diagnostics
privileged; expose only completed vectors, validity and reason codes. The future
evaluator maps the designated previously observed token to its target using its
privileged mapping; it never feeds that mapping or target association back.

## Candidate causal state and primary collision audit

At t the proposed state S_t consists of current opaque segmentation and oriented
boundaries; observed-token set; per-token visible/remembered status, last nonempty
mask and its frame index j; latest usable completed motion summary and its source
transition index k and signed reference action u_k; and ordered past actions plus
ut. Persist no unobserved token. Missing association or inconsistent token continuity
is UNKNOWN_IDENTITY, never repaired from raw geometry. No eviction occurs in this
bounded window. This is a specified lossy history summary, not the complete past
frame/flow archive. Keep that archive evaluator-side for audit only.

A usable motion summary is the arithmetic mean of forward fixed-point vectors over
source pixels belonging to that token with validity=1 and reason VALID_TRANSPORT,
from the most recent completed transition with nonzero pure lateral action and at
least one such pixel. Retain the integer component sums, count, k, action, and counts
for every validity/reason combination, so the mean is exact rational arithmetic.
No samples yields NO_USABLE_FLOW; invalid flow is never treated as zero. Preserve
all original completed validity/reason fields in the evaluator audit archive.
A later transition without usable samples does not erase the stored summary,
including during full occlusion. Last nonempty mask j updates independently; motion
may be older than j. This stale-motion assumption and sample support are reported.

The current optical view means current segmentation and oriented boundaries only; it excludes stored masks, motion summaries and prior actions.
Primary audit compares S_t with the announced command included, not rule outputs.
Equality requires every non-ID field and array equal, under ONE bijective opaque-ID
renaming consistently applied across both entire input histories, retained state
and evaluator target designation. No per-frame remapping, fuzzy tolerance, renderer
cross-comparison or rounding of state is allowed. RGB/action histories compare exact
uint8 arrays and canonical signed action values over 0..t; histories have equal
lengths within each pair. Targets compare exact binary masks on the same raster
under the same token alignment. Separately record equality of the current optical
view, full oracle input history, candidate S_t, RGB/action history and target.

- S_t equal, target different, RGB/action history distinguishable: finite E-specific
  loss for THIS state and window; no deterministic function of S_t can answer both.
- S_t equal, target different, RGB/action history equal: shared observation ambiguity;
  Pair 3 must additionally meet its oracle-history equality requirement. Not evidence
  of E-specific loss, and not a defective forecast caused merely by token retention.
- S_t different: no collision witness, regardless of rule errors. Distinction retention
  is necessary evidence here, not proof of predictive sufficiency or RGB extraction.
- Invalid identity/alignment, missing artifacts or failed declared pair equalities:
  INCONCLUSIVE for the affected pair. Preserve the reason and both members.

## Exactly two illustrative rules

1. Current-mask persistence: for every observed token output its current binary mask,
   or the empty mask if remembered and currently invisible. No memory translation.
2. Fixed optical-history extrapolation: start from the token's last nonempty mask
   at j. Let v be its stored rational mean motion in pixels per reference u_k,
   dividing fixed-point vectors by the existing scale 1024. Let
   a=sum(u_i.lateral for i=j..t)/u_k.lateral. Translate every mask pixel by the
   integer vector round_ties_to_even(a*v), computed in exact rational arithmetic.
   Round once after summing all elapsed actions; never round each step. Drop pixels
   outside the raster without wrap or interpolation; union duplicates. Thus several
   hidden steps and the announced action all count. Reversal uses signed action:
   equal opposite commands cancel, rather than replaying one displacement per step.

The extrapolator supports only pure lateral commands and a nonzero reference
command. Zero elapsed displacement yields an unchanged mask even without motion
support; otherwise missing usable flow produces UNKNOWN_MOTION with no binary
forecast. Unsupported command or missing identity produces UNKNOWN, no instrument
imputation. Report j,k, support, signed cumulative action, rational displacement,
rounded shift, clipping and unknown reasons. A never-observed surface has no forecast.

No occluder completion, inferred unseen extent, depth scaling, new-pixel filling,
flow fitting or tuning is added. Warping an observed mask can fail because new area
is revealed, masks are clipped, occlusion changes or flow is not a constant translation.
That is a rule limitation, not proof all ecological states fail. Remembering a token
without a correct future visible mask cannot count as mask-prediction success.

## Evaluation and finite consequences

Seal both rules' forecasts/unknowns and collision-audit inputs for all six members
before any frame t+1 target release; generation may retain targets behind the typed
evaluator boundary. Reuse the reviewed complete-bundle/exposure and append-only
retention approach only after adapting its fixed membership/sequence bindings in
an independently reviewed implementation. No A1 journal is reset or repurposed.

Evaluator target is the actual visible binary mask of the designated already
observed background token at t+1, including an all-zero mask when fully hidden.
Report exact equality, symmetric-difference pixel count and IoU for every binary
forecast; two empty masks have IoU=1, exactly one empty mask IoU=0. UNKNOWN has
absent scores, never an empty-mask substitution. Report observed-token retention
separately. Pair relations and collision classification are primary; rule scores
are descriptive illustrative evidence. No fitted threshold, pooled significance,
benefit claim or generalization claim is available from six development sequences.

The prospectively designated four Pair 1/2 members support separate all-four exact
forecast adequacy statements for EACH rule: PASS only if all four fixtures are valid
and all four masks exact; FAIL if valid coverage is complete and any mask is wrong
or UNKNOWN; INCONCLUSIVE if any fixture is invalid, even if surviving cases pass.
Always publish every case and reason, including negatives. This finite criterion
does not establish state sufficiency: noncolliding states can still lack information.
Pair 3 is NEVER pooled into an all-six deterministic-perfect criterion. Its control
PASS requires equal stated histories/state, different targets and the expected shared
ambiguity; a deterministic rule cannot be correct on both. Failed equality/different-
target construction makes the control INCONCLUSIVE, not successful prediction.

A verified E-specific collision is a finite counterexample requiring revision of
this candidate state before any learning proposal. Shared ambiguity constrains the
observation/task specification; do not claim state revision alone cures it. Absence
of collisions plus all-four exact predictions supports only these fixed fixtures.
Rule failure without collision motivates a separately reviewed rule/state decision,
not automatic scale-up. STOP if the fixture construction, modality separation,
chronology, identity or resource limits cannot be established without departing from
this scope. Preserve incomplete/failed runs and report no empirical gate decision.

## Smallest subsequent package and holds

Current `single_occluder.render_transition` captures only two poses and uses fixed
box geometry. A bounded future package needs a true ordered-pose adapter reusing
one compilation and frame references; the minimal numeric geometry extension above
if independently justified; sequence identity and permissioned history/target
projections; this state, two rules and collision audit; retained all-six evidence.
Do not alter old schemas/identities silently or introduce an infrastructure programme.

Before launch, freeze the numeric table and exact source head; validate analytic
pair construction and target alignment independently; check deterministic regeneration,
one mapping per episode, shared-frame references, no future/metric leakage, signed
multi-step/zero/reversal arithmetic, absent flow, unknown identity, integer ties,
clipping and empty masks using synthetic checks. Renderer qualification, native
smoke/validation/inspection and real fixture access belong to that separately
authorized package and launch, not this design. Qualification captures count against
its fixed sequences; no discarded pilot captures or replacement members.

At most 22 indexed pose captures (6+10+6) are planned across six sequences, each
with one RGB and canonical paired depth/segmentation capture; depth remains privileged.
The later numeric table must fix raster and a single admitted renderer plus exact
render-read count. Before launch, demonstrate a conservative byte/memory bound
covering artifacts, audit arrays, forecasts, logs, journals, partial failures and
terminal reserve within existing retained-output capacity; no extra budget or
unbounded diagnostic channels follows from this document. Failure denies launch.
No installs, services, project execution, native capture or held-out access occurs
in this doc/CI package. Only static consistency and diff checks are appropriate.

Publication route is branch `codex/causal-history-design-20261006` to isolated target
`codex/a1-integration-base-20261005`; the exact CI exception retains source-only checks
and excludes historical quality/WGL/OSMesa jobs. It grants no runtime authority.
Independent reviews remain bound to the future PR and exact head; the implementing
agent cannot approve this design. No active review records or approval automation
are added.
