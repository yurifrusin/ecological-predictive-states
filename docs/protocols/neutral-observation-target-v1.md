# Neutral next-observation target v1

Review profile: DUAL_REVIEW. Evidence class: PUBLIC_REPOSITORY_ONLY.
Status: IMPLEMENTED_PENDING_REVIEW. Phase-gate effect: NONE.

This separate source-only adapter covers known prefix identities plus anonymous
first-observed occupancy. It preserves the existing visible-forecast APIs. It
provides no predictor, new state ledger, native adapter, model or study.

`Candidate(input, known_masks, new_mask)` accepts exactly the input inventory,
with a Boolean mask or whole-channel UNKNOWN (`None`) for every known token and
one reserved NEW channel. Asserted masks have the input shape and are disjoint.
They are owned immutable bytes; public views have detached metadata. The complete
existing `CausalInput` prefix, permissions, commands and limits are revalidated.
The NEW channel adds one mask beyond the bounded opaque prefix inventory; limits
still bound actual visible target tokens before expansion. No new oracle identity
is introduced by NEW, and no new numerical limit is fitted to outcomes.

Save candidate canonical bytes before calling `evaluate(input, saved, provider)`.
All binding, chronology, inventory, extra-field, strict Boolean, duplicate-key and
overlap validation happens before fetching exactly t+1. The evaluator never calls
a predictor. Its provider is explicitly evaluator-privileged; this call order
cannot establish wall-clock acquisition or protect against a privileged caller.

Known targets use exact opaque association equality. Previously observed but now
absent tokens retain known channels; their reappearance is never NEW. All target
pixels whose tokens were never observed form one anonymous NEW union, including
adjacent or disconnected surfaces. Known plus NEW masks partition all nonzero
future raster pixels. An empty prefix inventory still has a scored NEW channel.
The union loses new-surface count, partition and future persistent identity;
first-observed occupancy does not establish creation or causal disclosure.

`SemanticTarget.canonical_bytes()` and its digest contain only semantic version,
index/shape, known channel masks and NEW union. They exclude input/candidate and
provenance hashes, raw labels and unseen token spellings/count/partition. Different
candidates on the same input/target receive the same semantic target. Changing an
unseen partition without changing its union preserves semantic bytes/hash and a
fixed candidate's outcomes. Known-token renaming is equivariance, with potentially
changed channel-reference bytes/hash; local-label-only changes preserve bytes.

The separate immutable `Evaluation.receipt_bytes` binds input, saved candidate,
semantic target and the observed raster/association logical hash. It can differ
across candidates, unseen partitions and local labels. Raw associations never
enter the semantic target or report. Receipts and targets are evaluator-side,
never learner features. Item-level receipts are not authorized private-evidence
public disclosures; any future private route needs its own aggregate receipt.
No supplied hash independently certifies provenance or episode randomness.

The report owns an immutable mapping/tuple snapshot. Per-channel errors are exact
mask symmetric differences. N/U/C/E count channel-by-image-pixel assertion slots,
not unique occupied pixels: each image pixel contributes known-count plus NEW
slots. N is all slots; U is UNKNOWN slots; C=N-U; E is asserted erroneous slots.
Conditional error is E/C only if C>0. `channel_pixel_coverage` is C/N; whole-channel
`channel_coverage` counts asserted channels. Neither measures unique occupied-pixel
coverage or establishes complete-raster assertion from partial channels.
The ignorance interval is [E/N,(E+U)/N], not a confidence interval.
NEW ensures N>0 even for empty inventory. Complete assertion and exactness require
all channels asserted; abstention cannot become success. Target occupied/NEW pixel
counts and NEW fraction are separate; the fraction is undefined on background-only
targets. Scalar errors do not measure new-surface partition quality.

Neutral target/candidate labels for known channels call the existing unchanged
`derive_visibility`: image occupancy, aligned overlap, gained/lost pixels and
region zero crossings. UNKNOWN predictions have unknown neutral labels. NEW has
first-observed area only, never one fabricated surface lifecycle. Translation or
expansion changes masks without proving accretion/deletion. The annotation package
preserves its original exports/static imports, with native modules loaded lazily
so pure derivation can run without MuJoCo. Scientific algorithms are unchanged.

Optical causes are declaratively UNAVAILABLE_UNSUPPORTED because no qualified
witness/site-domain contract is admitted. There are no cause arrays, witness
fetches, per-site cause scores or combined neutral/causal score. Perfect raster
labels cannot satisfy original A1 optical-cause prediction. Analytic event maps
cannot become native-raster cause truth from matching shape alone.

`python scripts/check_neutral_target_source.py` runs only handwritten CPU checks,
with native/model/producer imports denied, pytest plugins/conftest/cache disabled.
It validates/inspects synthetic serialization, replay and immutable snapshots.
The dedicated exact branch/base CI route installs locked SDK-free tools and runs
bounded lint/format/fresh typing/checks; legacy quality/WGL/OSMesa jobs exclude this
branch. Local checks use the existing environment with explicit worktree paths;
no local installation, sync or environment rebinding occurs.

Finite checks establish target software contracts only. Full G conversion,
controlled-occlusion persistence, every visibility-label reconstruction, graph
visualization, majority/oracle baselines and original0C exits remain unmet. Native
provenance, timing, learned association and predictive sufficiency remain untested.
Any later efficacy comparison needs identical oracle segmentation/association
prefix, action/history, known/NEW targets, coverage and matched supervision/model/
data budgets, plus generic oracle-token controls and structural ablations. Oracle
segmentation versus RGB alone cannot isolate the charter's structural hypothesis.

Both aperture attempts remain closed inconclusive; recovery18/21, Slice6
FAILED_CLOSED, held48, parked helpers and all closed studies/failed evidence remain
unchanged. Original0B/0C/0E are incomplete. No Gate0D/models/GPU, data/seed/threshold,
freeze, renderer qualification or empirical result is authorized by this package.
