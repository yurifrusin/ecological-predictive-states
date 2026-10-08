# Bounded oracle disclosure source v1

Review profile: DUAL_REVIEW. Evidence class: PUBLIC_REPOSITORY_ONLY. Additive software
interfaces and public handwritten checks only. No study geometry, membership, seeds,
collection entrypoint, learned model, final comparison, original gate completion or
experiment-launch authority is supplied. The charter appearance hypothesis remains
untested. Historical failed/inconclusive studies and qualification gaps remain unchanged.

## Privileged instrumentation

`disclosure_producer.raster(access, lateral, boxes, extent, check)` admits exactly two
or three existing exact rational `Box` values and fixed untilted 32x32 calibration.
It reuses closed slab intersection and retains first-hit ties as label zero plus
explicit tie records. `disclosure_reference.audit` independently uses `Solid`, scalar
face intersections and projected convex hulls; it imports no producer. Both require
the exact frozen typed permission set `{PRIVILEGED_GENERATION_RECORDS}` before reading
geometry or invoking a callback. Mixed/denied access fails closed.

Coordinates, lateral position and positive floor extents are bounded by 8, exact
rational numerators/denominators by 64 bits; boxes lie ahead of the camera and strictly
above floor. Reference projected hull coordinates are additionally bounded by 8 before
enumerating support. Labels are 0=background, 1=floor, 2 onward=caller box order.
Callbacks support cooperative cancellation; no hard resource guarantee is implied.
Raw labels, geometry and these returns are privileged instruments, never consumer inputs.

`ancestry(access, boxes, extent)` hashes canonical physical descriptors after sorting
box descriptors, excluding trajectories, token spelling, episodes and volatile metadata.
This is a private grouping binding, not permission to expose reconstructive descriptors.
Binding does not authenticate a checkout or establish physical qualification by itself.

The reference returns all-unit sampled visible counts, target-only full/clipped support,
footprints, sampled hull-edge flags, continuous FOV-touch flags, nearest-hit boundaries,
ties and causes. `Audit.qualified` requires no sampled nearest-hit boundary or tie.
Full-grid producer/reference agreement is a separate adapter obligation. Continuous
FOV/footprint-edge flags affect cause certainty, not otherwise unambiguous sampled truth.
No tolerance is introduced. UNKNOWN units stay in all known-query denominators and their
sampled support remains unchanged.

`VISIBLE` means sampled presence, not a certificate of continuously unoccluded extent.
`FOV_EXIT` is strict whole-footprint axis separation from the image. `COMPLETE_OCCLUSION`
requires zero visible support, positive target-only support, a strictly in-frustum hull
and strict coverage at the front plane of a distinct earlier box. Label roles do not
select the target or covering box. Partial clipping, unsupported/subpixel cases, edge
uncertainty and union-only coverage remain `UNKNOWN`. Floor support/cause is unsupported
and UNKNOWN; sampled floor presence is still counted. Oracle association, complete image
and randomized episode-opaque mapping qualification remain the private adapter's duty.

## Equal lawful inputs and fixed controls

`disclosure_controls` consumes only existing `BoundedPrefixMemory`: three complete frames,
at most 1024 pixels, 16 observed-ever nodes, exact executed/announced commands and optional
four-neighbour contacts. Permission denial precedes source, feature, binding and saved
forecast reads. Rebuilding verifies owned causal-source/feature/alignment/binding fidelity.
Commands must be bounded exact nonzero lateral triples; duplicate prefix locations must
have identical masks under the static contract. Episode/source-head strings are bound
outside numerical features; they are syntactic identities, not authenticated Git state.

The support threshold is exactly 16 sampled pixels. For every known row, `predict` saves:

- `zero`, `one`, `half`: fixed probabilities 0, 1, 1/2; support is null.
- `current_support`: threshold the current mask count, including absent rows with zero.
- `exact_retrace`: threshold the first matching prefix location's count; no exact match
  yields probability 1/2, null support and an explicit fallback.
- `motion_only`: reuse A2's last-two-visible-centroid velocity divided by summed executed
  lateral displacement; transport the last observed mask horizontally using exact rational
  nearest-column rounding with half ties away from zero and crop at image bounds; threshold
  the resulting count. Insufficient history/zero denominator yields 1/2 and null support.
- `motion_overlap`: select other currently visible masks that intersect the query mask
  transported to the current position. Transport selected blockers using their own last-two
  sightings and subtract their union from the transported query. Threshold the residual
  count. Any selected blocker without valid velocity makes the whole overlap row fall back
  to 1/2/null, with its reason; blocker selection never consults role, depth or futures.

These are uncalibrated deterministic diagnostics, not learned probability estimates.
Ordinary-presence A2 controls are unchanged. Every row includes fallback/reason; overlap
also retains selected numeric row positions. Saved canonical bytes bind complete rows,
methods, threshold, feature and causal/action binding hashes. `validate` rederives the
fixed forecast and requires exact byte equality; it accepts no changed/manual prediction.

`matched(negative, positive)` requires equal-magnitude negative/positive queries and the
same complete numerical history, contacts, association, episode and source identity.
Principal prospective consumers must receive equivalent lawful information. Depth, pose,
raw IDs, hidden inventory, generation/provenance and cause labels remain excluded. No
generic or ecological consumer/model interface is implemented by this package.

`policy(p_minus,p_plus)` minimizes costs `(2,5-4*p_minus,5-4*p_plus)` with tie priority
abstain, negative, positive. Equal actions strictly below abstention therefore select
negative; any abstention tie abstains. `score` validates both matched saved forecasts
before reading trusted index-3 same-shape futures. Trust flags are external adapter
assertions, not inferred credentials. All known rows retain support, success, remembered
status, retrace flags, hindsight-oracle cost and each control's action/cost/regret. Constants
negative/positive and always-abstain policies are included. NEW counts/pixels are separate,
never retrospective known queries. Complete empty frames are valid. Saved forecasts and
future payload hashes bind scoring; software binding does not prove historical sealing
time. A later private operation must separately establish forecast-before-future chronology.

`optical_digest` sorts each token's complete mask history, jointly remaps contacts and
hashes shape/masks/presence/contact availability/pairs. It excludes executed and announced
commands, token spelling and provenance. Disjoint masks and a nonempty sighting make
histories distinguishable; no factorial permutation search is used.

## Development aggregation

`disclosure_report.Group` carries private adapter evidence assertions, ancestry, family,
complete/qualified/parity/contradiction flags, bounded canonical score bytes and an evaluator
cause for every known row. These records are not authenticated review dispositions or
qualification credentials. The adapter must bind them to actual retained instruments.
`summarize` preserves at most 24 records and reports slots versus distinct ancestries and
optical histories. Duplicate physical descriptors share a family/group and must share any
later split; they do not increase ancestry-based coverage. Each slot first averages its
before-eligible remembered queries; nonempty slot means average within ancestry, then
nonempty ancestry means average within family. Empty means stay null, with counts, never
zero; the two family means have equal weight. Method costs/regrets and cause-by-retrace/new
view counts are retained without treating rows/actions/pixels as independent evidence.

The developmental adequacy check requires all 24 records complete and qualified, at least
20 optical histories/10 per family, 16 eligible ancestries/8 per family, four complete
occlusion ancestries per family, four per family with new-view complete-occlusion task
success contrast between commands, and hindsight primary cost at most 7/4 per family.
Precedence is parity failure STOP; contradiction apparatus FAIL with predictive INCONCLUSIVE;
missing/unqualified evidence INCONCLUSIVE; fully qualified coverage/diversity/utility failure
FAIL, including all-empty populations. Only after adequacy, a fixed control with zero regret
throughout either family's new-view complete-occlusion stratum gives STOP; otherwise PASS
means developmental task adequacy only. Cause UNKNOWN never excludes a known query.
No review-state automation, empirical gate decision or efficacy claim follows.

## Public source checks

`python scripts/check_disclosure_source.py` admits no execution arguments and checks only
public hand-computed ray, Boolean-mask and artificial aggregate arithmetic fixtures.
Import guards deny native renderers, learned models, historical collectors and private
operations. Software smoke validation/inspection uses those fixtures, not a study table.
No private physical construction or historical output is a public test fixture. Scientific
source acceptance, a later private adapter/operation, and any explicit scoped Gate0D remain
separate authorities; no data launch, retry, model or comparison is authorized here.
