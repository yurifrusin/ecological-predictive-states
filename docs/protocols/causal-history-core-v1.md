# Native-free causal-history core v1

Review profile: `DUAL_REVIEW`. Evidence class: `PUBLIC_REPOSITORY_ONLY`.
Phase-gate effect: `NONE`. Implementation status: `IMPLEMENTED_PENDING_REVIEW`.
The causal-history development and fixture-design protocols govern this bounded core.
The numerical fixture table is unchanged and NOT FROZEN; native capture and launch
remain disabled. This is synthetic software evidence only, not an empirical result.

## Public API and ownership

`diagnostics.causal_history_core` provides `Command`, `OpticalFrame`,
`CompletedFlow`, `SequenceProvider`, `MemorySequence`, `TemporalProjection`,
`build_state`, `CausalState`, `persistence` and `extrapolate`. The two rules take
only an owned typed state and forecast every observed token. They have no provider,
path, sequence provenance, geometry, target designation or hidden inventory argument.
Frozen records own nested identity pairs and immutable bytes-backed raster arrays.
There is no filesystem producer, native adapter, model, fit, eviction or renderer.

Commands contain exact `Fraction` lateral/forward/yaw values. An adapter must use
canonical decimal strings or integer numerator/denominator pairs; converting a
binary float through an arbitrary decimal display is not the command contract.
The candidate lateral fractions are binary-exact. Unsupported elapsed commands
produce `UNKNOWN`; absent usable motion produces `UNKNOWN_MOTION`. A zero signed
elapsed sum preserves the last mask without motion. Invalid/missing visible identity
fails projection with `UNKNOWN_IDENTITY` rather than constructing a forecastable
state; it is not an empty-mask substitution. An unobserved token has no forecast.
`OpticalFrame.mask` is an internal visible-mask lookup; an absent current label is
empty only for a token already retained in a valid state.

## Explicit sequence-local payload semantics

The separate sequence-local v1 API uses nonnegative absolute `sequence_index` and
`source_index`; a completed transition ends at `source_index + 1`. It does not reuse
historical two-frame episode identities or widen their `Literal[0,1]` schemas.
Each boundary is a canonical immutable JSON payload of the existing
`OrientedBoundaryElement`, always with **local** `frame_index=0`, attached to the
absolute `OpticalFrame.sequence_index`. No compound two-endpoint boundary summary
is projected. Every lattice location, visible opaque side/owner reference and
segmentation neighbour mapping is validated explicitly; duplicates are rejected.
This prospective mapping validation is a future adapter qualification requirement,
not evidence of native ownership/classification or raster accuracy.

Segmentation is a nonnegative int32 raster: zero is uncontrolled, each positive
episode-local opaque label has exactly one visible `surface-<16 hex>` token.
Label/token encoding must remain bijective and consistent across all past frames.
The table contains visible labels only, never a raw simulator ID mapping or unseen
inventory. Real adapters must randomize the opaque mapping once per episode and
prove this boundary independently; these synthetic records cannot certify that.
Missing a visible label identity fails closed; an observed invisible token retains
its nonempty past mask and has an empty current mask.

`TemporalProjection` checks typed modalities and frame/end index against decision
time **before** any provider access. Completed vectors are int32 xy fixed-point
values at scale 1024 with uint8 validity/reasons (domain 0..4, validity=1 iff reason
`VALID_TRANSPORT=0`). No analytic assignment, future correspondence or instrumentation
is accepted by these records. RGB requires separate `RGB` permission. Provider
metadata is never copied into state. Sequence logical/provenance envelopes and
future evaluator targets remain trusted control-plane concerns for a later package.

## Literal memory and rules

State retains current segmentation/boundaries, the observed-token set as ordered
token records, visible/remembered status, last nonempty mask/index j, latest usable
motion sums/count/index k/reference command and source-token validity/reason counts,
ordered executed commands and the announced next command. Mask and usable motion
overwrite independently. Later invalid/empty support retains prior motion;
`motion=None` explicitly means `NO_USABLE_FLOW`. It stores no complete optical
history and deliberately permits the proposed overwrite collision.

Persistence uses the current mask (empty while remembered and invisible). History
extrapolation uses the exact rational mean, signed command sum j..t divided by the
nonzero reference lateral command, and rounds once with ties-to-even. It shifts
without wrap/interpolation and clips outside the raster, including arbitrarily
large rational shifts. Forecasts expose j, k/reference/support/reason counts,
cumulative command, rational displacement, integer shift and clipped-pixel count.
No occluder completion, newly revealed area, metric scaling or fitted motion is added.

## Trusted audit and limits

`diagnostics.causal_history_audit.archive` constructs a separately permissioned
matched RGB/action and full optical past archive; `audit_pair` rebuilds state to
validate history binding. One complete bijective token alignment is applied across
entire histories, retained state and both evaluator target designations. Exact
current view, full oracle history, state, RGB/action window and future binary target
relations are reported separately. Forecast differences are never collision evidence.

The audit distinguishes mathematical collision classification from declared
fixture validity. Pair timing is decision 2/3/1 for pairs 1/2/3; targets are t+1,
already observed before the decision, on the same raster. Pure lateral commands,
equal decision views/announced command and differing targets are required. Pair 1/2
require distinguishable RGB/action history. Pair 2 additionally requires background
visible at 0 and 1, hidden at 2 and 3, usable completed pre-occlusion flow, and a
nonempty revealed target. Pair 3 requires equal entire RGB/action and oracle
histories and state. Failed construction or alignment is `INCONCLUSIVE` with reason;
mathematical fields do not override fixture validity. Incomplete/invalid histories
remain inconclusive rather than repaired from instruments.

`score` reports exact equality, symmetric-difference count and rational IoU for a
binary forecast. Both empty means IoU=1; one empty means 0. UNKNOWN has absent
scores. Token retention is not mask prediction. This core does not aggregate an
all-four adequacy verdict, pool Pair 3, or claim efficacy. The later evaluator must
retain all cases, seal forecasts before target release, and apply the separately
specified all-four criterion. No native full-state/RGB equality, deterministic
capture, target provenance, future-release seal or 24-pose launch qualification is
established by the synthetic adapter. Gate 0B is incomplete; Slice 6 remains
FAILED_CLOSED, 18/21 recovery and the six-helper programme remain preserved.
