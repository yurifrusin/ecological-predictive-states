# Restricted mask readers v2

This source package is `DUAL_REVIEW / PUBLIC_REPOSITORY_ONLY`, implemented under
explicit restricted Gate 0D source authorization. It makes no empirical claim.
The charter appearance hypothesis remains untested; original 0B/0C/0D/0E are not
completed. Closed studies and the stopped restricted model/controller remain closed.

The question is whether fixed observed-neighbour routing helps one-step oracle
mask prediction compared with learned all-node mixing given identical lawful
inputs, targets, parameters and fit opportunities. It is not a sensory extractor,
appearance-robustness test or proof of state sufficiency.

## Shared input and two readers

`restricted_mask_readers.Reader` consumes only unchanged PR71 `Projection`
features: current/previous masks, signed differences, visible/remembered status,
first/last-seen ages, six executed/announced command components, both endpoint
adjacency/availability/ages and both unions. Genuine full prefix, Action bytes,
episode/source/index/hash and opaque alignment remain outside model features.
Exact typed surface-region/correspondence/action access and revalidation precede
inference. Only completed frames 0/1, 32x32, at most three observed-ever nodes
are supported; inactive slots do not exist. Invalid context rejects. Valid
unavailable adjacency yields whole-channel UNKNOWN, including NEW; it never
silently means background. Missing trusted history cannot produce a Projection.

Each reader has 4531 CPU binary64 parameters, shared across nodes/pairs. Node
convolutions are 3→8 with 5x5, then 8→8 with 3x3; union convolutions are 2→8
with 5x5, then 8→8 with 3x3. Hidden layers use ReLU, same zero padding and bias.
Node pooled spatial features plus 10 metadata scalars use 18→16. The shared
ordered-pair function is 35→16→16 for both endpoints. E computes every pair
then multiplies by adjacency; L computes the same pairs and sums all messages,
with adjacency/availability/time still supplied. The known head is 70→8→1;
the NEW/background head is 74→8→2. No embeddings, dropout or pretrained encoder.
Parameter equality does not establish effective-capacity equality; empty edges
can leave E's message parameters inactive. The 7-pixel receptive field does not
guarantee support-plane, hidden-region or NEW prediction.

## Finite arithmetic and saved outputs

Pinned Torch 2.10.0+cpu, NumPy 2.4.6, one thread, deterministic algorithms and
disabled MKLDNN provide the shared implementation. All operands, parameters,
activations, logits, losses, gradients and Adam moments are CPU float64. Exact
rational commands must round-trip through binary32 unchanged; feature zero is
positive zero while signed-zero genuine Action metadata remains bound.
Nonfinite arithmetic rejects; no epsilon ties, NaN replacement or infinite padding.

Spatial means sequentially add row-major pixels from positive zero, then divide
by 1024. Node/pair/class-exponential sums sort each component numerically, with
negative zero before positive zero, and sequentially add from positive zero.
Pixel/batch loss means retain row-major/committed minibatch order. `_OrderedSum`
uses NumPy `add.accumulate` on an explicit positive-zero prefix. Its autograd
rule is the ordinary continuous sum derivative, not a finite-rounding derivative.
Affine/nonlinear operations are pinned-framework binary64; no cross-platform
last-bit or universally identical renamed optimizer-trajectory claim is made.

Initialization has no default random draws. For each declared tensor name and
flat weight index k, SHA256(K || UTF8('|'+name+'|'+decimal k)), first 8 bytes
big-endian shifted right 11, gives u=b/2^53. Set `(2*u-1)*sqrt(6/(fan_in+fan_out))`;
convolution fans include kernel area. Biases and moments start at positive zero.
No arm enters the stream. E/L start from identical retained bytes.

Prediction records retain little-endian binary64 active logits, context and weight
digest plus canonical saved Candidate. A unique maximum selects known/NEW/BG;
every exact maximum tie selects BG. Known/NEW masks are disjoint. `candidate`
revalidates and reconstructs those masks from retained logits; `evaluate_prediction`
checks the separately frozen selected-weight digest before one neutral target fetch.
Scoring uses saved Candidate bytes. Canonical checkpoints contain all weights,
Adam moments and integer step; no pickle or executable tensor loader is used.

## Trainer, fictional readiness and future operation boundary

Training supervision has a separate typed `Supervision` capability; it is an
external caller assertion, not proof of split membership. An admitted qualifier
must establish training rights and physical provenance. Complete bound targets
provide unweighted categorical cross entropy: max + log(sorted sequential sum
exp(logit-max)) - true logit. Adam uses binary64 lr 0.001, betas 0.9/0.999,
epsilon 1e-8, with no scheduler, clipping, decay, foreach or fused implementation.
Updates stage and validate all moments/weights before mutation.

`fit_comparative` is later-only: 400 shared committed batches of eight action
decisions, checkpoints 100/200/300/400, earliest exact-rational minimum development
neutral E/N. It has no final-target provider. The future driver must restrict the
development callback, preserve every fit/failure, freeze weights, seal ALL final
predictions before ANY final scoring, and apply original 1GiB inclusive reservation.
Paired group/seed contrasts use exact E/N before bootstrap conversion; aggregate
evaluation/bootstrap are outside this package. Three paired comparative keys and
112 memberships (64/16/32 independent geometry forks) remain unallocated.

Public fixtures are literal fictional software sets, not sampled geometry:
cases 1/2 have one 4x4 square at (8,8); 3/4 have two adjacent 4x4 squares at
(18,18)/(18,22), in both prefix and target. Left cases 1/3 add a four-pixel NEW
strip one column left; right cases 2/4 add it just right of the union. Executed
lateral is +0.75; announced lateral is ±0.5. Ages are 1/0, differences zero;
endpoint pairs are available-empty or the one known-known pair. Episode i is
32-digit hex, known token j is `surface-` plus hex16(16*i+j), evaluator-only NEW
is 16*i+n+1. All 1024 sites are independently checked. Exact episode/action/source
and literal target bind one external fetch; fitting reuses an immutable snapshot.

Public readiness uses SHA256('restricted-mask-model-v2-public-readiness'), identical
starting bytes, all four cases in order, lr 0.01, at most 200 updates/60 wall seconds,
checking saved-mask errors each update and stopping at the earliest zero error.
`public_readiness` is an exclusive-output callable, not launch authority. Exact
source review and one explicit admission per arm must precede its use. Each public
arm reserves 192MiB original retention including 64KiB terminal margin; two originals
plus same-size first archives plus 128MiB review use 896MiB, leaving 128MiB for archive
headers/wrappers. Actual inclusive accounting remains the admitted driver's duty.
Measured peak working set must remain ≤1GiB. Deadlines/resources are cooperative,
not a hard aggregate host guarantee. All setup, retention and binding failures are
INCONCLUSIVE; typed fitted nonfinite arithmetic or a complete intact 200-step
nonzero-error result is NO_GO. No rescue/tuning/retry is authorized.

The sampler defines prospective domain-separated unbiased 2^-32 lattice membership,
split grouping and closed-geometry collision rejection. Its local raw/audit receipt
validator does not prove execution or external Trust. There is no collection runner,
private fitting, membership generation or final study in this source package.
Ordinary CI denies every fit and membership entry and imports no native producer,
reference audit, renderer or stopped model. Source PASS is not scientific evidence.
