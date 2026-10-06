# Causal visible occupancy contract v1

Review profile: DUAL_REVIEW. Evidence class: PUBLIC_REPOSITORY_ONLY. Phase-gate effect: NONE.
Implementation status: IMPLEMENTED_PENDING_REVIEW. No independent approval is supplied here.

This source-only arbitrary-mask contract predicts next-view visible occupancy of previously
observed surface hypotheses. It is a restricted A4-like diagnostic, not A1 accretion/deletion,
A2 persistence, full ecological next-state prediction or an EPS sufficiency claim. No native
adapter, dataset, renderer, study execution, model or learned extractor is provided.

## Causal input and inventory

`CausalView` checks exactly SURFACE_REGIONS, REGION_CORRESPONDENCE and EXECUTED_ACTION typed
permissions and denies future requests before provider access. A complete consecutive prefix
0..t is required. `CausalInput` owns immutable Boolean token-mask bytes and fixed shape. Input and
forecast public mask arrays are fresh read-only views with detached metadata, so changing a view
shape/dtype cannot alter admitted bytes, Boolean interpretation or canonical identities. Opaque persistent
association is explicitly privileged oracle supervision. Local raster labels are adapter-only.
Masks carry image-plane occupancy; no hidden total surface area, raw identifier/map, invisible
catalogue, metric/pose/depth/generation records, seed/root, RGB, flow, ownership or boundary payload
enters the input. The exact three-component rational commands are forward/lateral/yaw commands,
not absolute pose. Executed commands end at t; one next command is announced.

Inventory is exactly the union of tokens visible in that prefix, fixed before the future is read.
One target mask per inventory token is constructed by the evaluator from t+1; absence gives an
empty target mask, not destroyed identity. New future tokens are unscored and never predictor
inputs. Report their count, occupied pixels and fraction of all nonzero future surface pixels;
zero denominator is undefined. Also report inventory future pixels. This restriction does not
claim whole-frame completeness. Token spelling/order and local label permutations cannot affect
control masks or scalar scores, apart from the corresponding channel permutation.

## Controls, limits and memory

Exactly two untrained controls ignore actions: current-mask persistence, and K-frame
agreement/abstention. Persistence receives the shared full-prefix-derived inventory header;
it is memoryless only with respect to raster history. Agreement predicts a mask when all K
selected recent masks agree, including empty masks for observed-but-now-absent tokens; otherwise
it returns whole-channel UNKNOWN. Reject fewer than K frames. Past agreement is not evidence of
future certainty. Last-seen summaries are not assumed sufficient. Revisit caches are deferred.

`Limits(max_frames,max_pixels,max_tokens)` are positive strict integers, supplied by the caller,
not fitted to outcomes. Complete frame and raster bounds and the union-token bound are checked
before token-mask expansion in provider/serialized materialization. Direct already-materialized
mask inputs are checked before contract snapshot copies; they cannot undo caller allocations.
Native target rasters are not admitted by this package; arbitrary target fixtures also obey the
pixel/token bounds. Missing history is rejected, never silently filled.

`storage` reports full materialized input Boolean buffers (one byte per pixel per visible token),
full input serialization bytes including commands/permissions/limits, inventory and rational-command
UTF-8 component bytes, forecast buffers/serialization, and the K-window's referenced input buffers.
Window references share the full resident input and add zero retained mask buffers; this is not a
K-window saving of the already materialized full prefix. Inventory/command component counts overlap
serialization and must not be added as separate copies. `snapshot_mask_bytes` includes owned input
and forecast buffers. These logical counts exclude Python object/allocator overhead and temporary
validation/provider/JSON/canonicalization buffers; they are not RSS, peak-memory or deadline claims.
Limits bound array expansion, not provider allocation or adversarial parser input-byte consumption.

## Saved forecast and evaluator

`forecast(input,rule,K)` yields immutable Boolean masks or whole-channel UNKNOWN for precisely the
causal inventory. Canonical forecast bytes bind the input SHA256, control/window, index and shape.
Save those bytes first. `evaluate(input,saved_bytes,target_provider)` validates their binding,
chronology, inventory and shape before fetching t+1; it never invokes a predictor. Invalid forecast
bytes cause zero target fetches. Serialized input requires the same external typed permissions
and caller limits; rejects extra fields, duplicate keys, non-Boolean mask cells and noncanonical
rational strings. UNKNOWN serializes as JSON null, not a label, probability, zero or missing channel.

This API/call-order separation establishes contract sequencing only. It does not prove historical
wall-clock timing or protect against a caller who already has evaluator privileges. Future actual
collection needs separately reviewed prospective chronology; no sealing/controller is added.

## Scores and honest uncertainty

N = inventory tokens times image pixels; U = UNKNOWN pixel slots; C = N-U; E = asserted-mask
symmetric-difference errors. Report integers, conditional E/C when C>0, pixel and channel coverage,
whole-episode assertion coverage and full-inventory ignorance interval [E/N,(E+U)/N] when N>0.
The interval is not a confidence interval. Complete abstention has zero coverage, undefined
conditional error and [0,1], never exact success. Empty inventory is NOT_APPLICABLE with undefined
coverage/error/exactness. Inventory-exactness requires all channels asserted and E=0; omitted new
coverage remains separate. Per-token target area/error/UNKNOWN and visible/absent strata preserve
empty and difficult cases; conditional accuracy alone cannot rank abstaining controls. No efficacy
PASS threshold or statistical generalization is supplied.

## Checks and interpretation

`uv run --locked python scripts/check_visible_forecast_source.py` installs native/generation import
denials before collecting only focused synthetic tests. It exercises serialization, validation and
inspection as source smoke. The exact branch/base routes to that selector and source/static checks;
old native/WGL/OSMesa/frozen jobs do not run. Locked sync, lint, formatting, typing and diff checks
remain required. Native generation/validation/inspection and historical full suites are inapplicable
and are not represented as passed.

Independent hand-written fixtures cover disconnected masks, holes, row/column degeneracies,
background-only input/target, reappearance, future-new occupancy, exact denominators, malformed
input/forecast, pre-fetch permissions/time denial, limits before expansion, immutability,
serialization, label/token renaming and visible positive changes. Symbolic ambiguous continuations
share the entire admitted input/actions but differ in future masks. A separate finite-window pair
has different full-prefix bytes/digests and older evidence, but identical selected K-state/control
masks: this concerns information loss in that selected window, not full-prefix insufficiency.
Neither example is a native geometry witness or a predictive win. Honest errors/abstentions and
explicit ambiguity are valid outcomes.

The independently reviewed fixed paired-appearance apparatus failure is preserved; this source
package admits no native forecast evidence or apparatus qualification. Whole Gate0B remains
INCOMPLETE, Slice6 FAILED_CLOSED, recovery18/21, historical studies immutable and six helpers parked.
No ML before explicit Gate0D authorization; no gates, benchmark freeze, new seeds/data, source
historical rewrites or texture/geometry/threshold search. Equal oracle inputs/targets force equal
scores within a deterministic control; appearance invariance is a pipeline property, not the
charter's matched-information architecture advantage. Independent exact-head engineering and
scientific reviews, owner scope decisions and any merge remain separate.
