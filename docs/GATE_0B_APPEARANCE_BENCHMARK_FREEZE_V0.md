# Gate 0B Appearance Benchmark Input Freeze v0 correction candidate

## Authority and status

This PR #17 correction wave is bounded to `EPS-SR17-0001` and `EPS-ER17-0001` through
`EPS-ER17-0004`. Its review profile is `DUAL_REVIEW`, evidence class is
`PUBLIC_REPOSITORY_ONLY`, and closeout boundary is `BENCHMARK_OR_PREREGISTRATION_FREEZE`.
It starts from canonical base `08179fbdce909e9a0d6dbb2939c58f7ab5d0d8a7` and preserves the
reviewed request-changes head `8b34b78d5488af7103119697a286cfb8757cc125` as historical evidence.

This remains an implementation candidate. It does not freeze the benchmark, approve an exact
head, authorise merge or closeout, complete Gate 0B, advance Gate 0C or Gate 0D, authorise model
work, or establish a scientific result. A corrected head requires renewed independent engineering
and scientific review naming the same exact SHA.

## Lock supersession and preserved scientific content

The original prospective lock commit
`1a5929307dfcba1d726c650f5e1ce68771f66801` and its lock root
`a373a4742a6b5a3057b820b7d925c2e19fbddc0a83cc45384b8b81ea4e8b1464` remain immutable in
history. They are superseded for the corrected candidate because `EPS-ER17-0004` requires unique,
versioned domain envelopes. They are not amended or retrospectively represented as corrected.

The replacement lock uses `appearance_benchmark_freeze_definition_lock_v1`, explicitly binds that
supersession, and has root
`28acd2c340b3ef1e2b8b2b7b31ca69356cd9baa35b94ff8a598e020099883435`. The commit that most
recently changes the lock file is required by validation to have subject
`feat: replace appearance benchmark freeze v0 lock domains`, to descend from the historical lock,
and to contain the exact definition, seed registry, and replacement-lock snapshots. No corrected
qualification may begin until that commit is pushed and its remote readback is verified.

That replacement-lock commit is
`d4072f912cc58bbc1ca41ceb2652e41783dbf3e1`. The branch reference, fetched remote reference, and
GitHub commit API were read back at that exact SHA before corrected final-root execution began.

The correction does not change the five selected profile identities or roles, the two excluded
negative-evidence profiles, the sixteen episode roots or order, the two scene families, the primary
or secondary pairing relationships, appearance thresholds, supported renderer environments,
all-cells/both-renderers readiness rule, failure and root-retirement policy, or training/evaluation
exclusions. The selected profiles remain:

- `revision1_balanced_reference_v1` — `development_reference`
- `revision1_colour_shift_v1` — `held_out_colour_ood`
- `revision1_checker_low_v1` — `held_out_texture_presence_ood`
- `revision1_checker_high_v1` — `held_out_high_frequency_checker_ood`
- `revision1_combined_stress_v1` — `held_out_combined_appearance_ood`

`revision1_stripes_low_v1` and `revision1_illumination_shift_v1` remain excluded negative evidence.
Every primary OOD comparison uses the balanced reference at the same scene and episode root.
Checker-high versus checker-low remains the secondary frequency-only diagnostic.

## Public evaluation roots and exclusions

The registry retains root seed `314159`, `derive_seed_v1`, namespace
`gate0b-appearance-benchmark-v0-final-evaluation`, indices 0 through 15, and these canonical roots:

1. `988775886542458411`
2. `6042662804173454446`
3. `14022183079951728208`
4. `17670635520239758424`
5. `15123826810452275569`
6. `17286523330598615844`
7. `7896050055028401465`
8. `11538922899034961693`
9. `18335560522116882876`
10. `5187984745037597167`
11. `4446816192287790622`
12. `18081725484914221288`
13. `2878494741235076731`
14. `4279158373164189457`
15. `7775037187548969617`
16. `2012889273669341920`

The versioned registry root is
`6747d234aa5e843e1a09013b978c0e8ce55b4c71c0f3bb820cee46176b618652`.
Public does not mean development-eligible. These roots remain prohibited for training, validation,
tuning, preprocessing or architecture selection, hyperparameter or checkpoint selection, and
difficult-root replacement. Held-out OOD appearances remain prohibited for training or tuning.

## Renderer-selection dependency

Before any comparative model-result access, a future authorised protocol must prospectively:

- select exactly one supported renderer as the primary model-result renderer;
- classify the other renderer as `replication`, `robustness`, `sensitivity`, or `unsupported` for
  model-result purposes;
- make both choices without access to observed comparative results;
- prohibit renderer averaging or aggregation rules chosen from observed results; and
- preregister any renderer aggregation rule before comparative-result access.

The current dependency is `required_not_yet_satisfied`: both choices and any aggregation rule are
null, comparative model-result access is false, and no model protocol is frozen. Apparatus
qualification on both renderers does not satisfy this model-result dependency.

## Versioned logical domains

Every corrected freeze logical root hashes an explicit envelope containing
`epsbench_logical_domain_envelope_v1`, one unique immutable domain name, and the canonical payload.
The new definition identities include:

- benchmark definition: `d4c7d5026344f54dd5f1c5ef1ef60ffa8198595e7d643a5b986bf45a3cc14d07`
- profile roles: `5e62ecc0f985e99ea97180b35fc5bd9cf676bbb764c14c1f2bba83d72c139284`
- selected profile set: `b78edf9b099415a09243805de8d4f57c5a6e004a5c8b53a9d770df1645d7f174`
- excluded candidate set: `f7ba30e3c2ff53680cfef695dcaf9157c6d74c4b0f034900c8dca9860d0c498c`
- selected matrix membership: `b8f58f16eebed8c4454edfc7534a5d6c783487a3ef6acbb384cf6f1829f85e50`
- legacy-control membership: `fed7e0bd05f09877b5436e56a2cd0ddb8f72c2d0fdf6f80a81b72372f8ab93b1`
- training/evaluation policy: `16a4041fb22ee3fc49ee661bdebc121c3435f8d4d91fe0c37d62e22469ee1e27`
- renderer-selection dependency: `f9a36861cfd99a2993c926d34bbdd62526ca0bec2ba95d5129a5331d5976e4bd`
- admission thresholds: `b31238d377a04b7b360dae11d05d4082cb8b9ca9d4a3db974464663cd45310ed`

Prior v0 roots remain historical identities only and are not compared with corrected v1 roots.

## Independent source evidence and pairing

Every successful qualification cell retains its complete validated one-episode generator dataset,
an exact file manifest, and typed source identity record. Packet validation validates the retained
dataset and independently reconstructs the sampled/compiled geometry, camera trajectory, executed
action, raw-to-opaque surface remapping, scene content, analytic transport, oriented boundary
ownership, visibility events, and public occlusion relation. Each of those identities has a unique
v1 domain. Pairing checks and portable apparatus roots consume those independently reconstructed
values, not the top-level cell claims.

The retained source datasets are privileged apparatus evidence. They are not ecological-loader or
learner inputs, and their raw simulator identifiers, metric geometry, camera transforms, and
generation records remain subject to the existing typed modality boundary.

## Receipt and publication contract

`appearance_benchmark_renderer_qualification_receipt_v1` is strict, typed, and fail-closed. It
binds the exact renderer environment and fingerprint; clean source revision and Git tree;
replacement-lock commit and root; benchmark-definition and seed-registry roots; complete packet;
all 160 selected and 32 control outcome rows; profile identities, roles, counts, and readiness;
portable and renderer-local roots; versioned threshold margins; and a public packet-tree evidence
receipt excluding volatile `run.json` metadata.

Receipt creation first validates the complete packet, then validates the receipt against that
packet. A receipt used as counterpart evidence is insufficient by itself: validation must resolve
the separately published complete packet, its strict receipt, the exact CI publication record, and
live artifact/run/job metadata. The complete packet validator independently derives outcome maps,
profile readiness, roots, margins, counts, and source commit/tree provenance; readiness never
trusts receipt assertions or profile Boolean fields.

Publication is forbidden inside the packet, through traversal, a symlink, junction, reparse point,
hard-link, or other alias, and over any existing or racing target. Linux uses an unnamed file and
directory descriptor with handle-derived `linkat`; Windows uses native root-directory-relative
file creation, linking, reopening, and disposition. Staging, exclusive publication, reopening,
rollback, and cleanup remain bound to the same verified parent directory object. Parent swaps are
checked at both staging and publication boundaries, the target is atomic and no-replace, and the
complete source packet is revalidated on every publication exit.

## Qualification rule and current state

Each renderer must produce exactly 160 selected cells, 32 unique retained controls, and 192 total
cells. Every selected profile must pass all 32 cells in Windows/WGL and Ubuntu/OSMesa. Portable
apparatus roots must match, while renderer-local outcomes and raster evidence remain reported but
not equality-asserted. Any selected-cell or locked-renderer failure retires the complete root set
for a revised profile set.

The first post-readback WGL execution failed closed before packet publication because the dataset
artifact content identity had been incorrectly conflated with the new freeze-domain identity. It
produced no packet or receipt. Source-only commit
`f7b1e61b07a86ff021a73d6a4236fcfe0ea295d2` restored the dataset snapshot's canonical artifact
identity while retaining the separately domain-enveloped freeze root; no profile, role, episode
root, threshold, renderer, comparison, exclusion, failure rule, or replacement-lock byte changed.
The added regression generates and validates a retained final-seed dataset while asserting that the
two typed identities are distinct.

As historical evidence for exact head `62e09a920c10d50c62643543f6ebcbd26570b1d4`, the subsequent
complete Windows/WGL run at clean source commit `f7b1e61b07a86ff021a73d6a4236fcfe0ea295d2`
published packet root `630d23cf791983977bdd9e92ba33ccedb6a18d1faf819c5d2df139c83dac9310`.
All 160 selected cells were admitted; the 32 retained legacy controls were 16 admitted and 16
rejected. Independent packet validation and the complete adversarial corpus succeeded. The strict,
uniquely owned WGL receipt has root
`28036d50634b9521b62fa9ceb00d24fe32c3b94d7bb2182d984f2df49c8c76ac`, binds 6,550 packet
artifacts, the clean source Git tree `bbc29863fd4efef24c3d74cc8a99caff15ada504`, and these portable
apparatus roots:

- procedural assets: `e38f3132b1b8821b1df45ba0edae015081c8243eaac53e9a5c092c25e60de5a2`
- appearance assignment: `7caa2e43c07ad522e27d41a8d71a23abff22c57fcd820d8f607f0233ff49ff12`
- reconstructed source identity: `3218027ed2135815034a8162158d5940702ec25108cd726fe87a9410c1677948`
- within-renderer invariance outcomes: `20037a1b7103e38abb69ab62fa659d99ba15fbd6b4fae8be5e1e9d9bfc771ebe`

For that historical execution, the versioned threshold-margin root is
`eb23b2e2724f1603293e06a25d43344c87d5f954219c66ccdb8c5e5dd292c097`; every failure count and
near-threshold count is zero. The corrected status remains `qualification_incomplete` and the seed
disposition remains `pending_second_renderer_qualification` until exact-head Ubuntu/OSMesa CI and
cross-renderer reconstruction complete. `benchmark_frozen` remains false and `scientific_result`
remains null. No finding is self-verified by this implementation evidence.

The historical head later completed OSMesa execution but received `ENGINEERING_REQUEST_CHANGES`.
Its apparatus evidence and reviews do not carry forward. The owner-authorised second correction
requires complete exact-head WGL and OSMesa requalification after the new source commit. Those
artifact outcomes are necessarily post-commit external CI evidence and must be resolved through
their immutable Actions packet artifacts and publication records.

Exact head `b63121cdc7ef11340e0275498fda0f0b32213637` is retained failed evidence. Its WGL
packet completed, but publication-record creation rejected an incorrectly modeled artifact-upload
retention interval, PowerShell masked that first exit, and Linux independently rejected
cross-platform reconstruction before OSMesa generation. The corrected evidence contract binds
GitHub's exact run-created, artifact-created, and expiry timestamps and measures the configured
90-day interval from workflow-run creation. PR #17 freeze-layer audit copies use a twelve-decimal
portable serialization boundary and sign-normalize symmetric real-FFT row aliases; the shared
historical Slice 5 audit remains byte-compatible. Threshold comparisons continue to use unrounded values; profiles, evaluation episode roots,
thresholds, renderers, and locked scientific inputs are unchanged. Exact-head requalification must
start again after the source changes.

Second-correction exact head `dc17d8453bed2583c085aa7441255ca476d3e1f0` later completed both
renderer paths and received `SCIENTIFIC_PASS` plus `ENGINEERING_REQUEST_CHANGES`. The engineering
re-review verified `EPS-ER17-0002` through `EPS-ER17-0005`, left `EPS-ER17-0001` and
`EPS-ER17-0006` unverified, and raised `EPS-ER17-0007`. The owner-issued third correction requires
the production validator to consume only safely extracted, digest-verified raw packet and evidence
archives resolved from exact live GitHub metadata, and gives equivalent credential-free GitHub
HTTPS origins one canonical owner/repository identity. The verified findings and replacement lock
remain preservation constraints. All exact-head apparatus evidence must be regenerated.

## Current PR #17 status

| Boundary | Current source-record status |
| --- | --- |
| Apparatus status | `EXACT_HEAD_REQUALIFICATION_PENDING` |
| Post-commit exact-head CI evidence | `EXTERNAL_TO_SOURCE_COMMIT` |
| Implementation status | `THIRD_CORRECTION_SOURCE_CANDIDATE_PENDING_INDEPENDENT_REVIEW` |
| Review status | `RENEWED_EXACT_HEAD_DUAL_REVIEW_REQUIRED` |
| Owner status | `THIRD_CORRECTION_AUTHORISED; IMPLEMENTATION_MERGE_APPROVAL_NOT_GIVEN` |
| Benchmark-freeze status | `NOT_PERFORMED` |
| Model-protocol status | `NOT_FROZEN` |
| Gate status | `NOT_ADVANCED` |
| Scientific result | `NONE` |
