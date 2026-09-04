# Renewed Scientific Review — EPS PR #17 Fourth-Correction Exact Head

## A. Review identity

```text
Role: SCIENTIFIC_REVIEWER
Repository: yurifrusin/ecological-predictive-states
Reviewed PR: #17
Reviewed SHA: 4d47595ede0f1690f1d9d96075df9a3ff7ab60d9
Reviewed tree: ba112bcc4be4396648305bb1d1529cc002755ea1
Canonical base: 08179fbdce909e9a0d6dbb2939c58f7ab5d0d8a7
Historical lock commit: 1a5929307dfcba1d726c650f5e1ce68771f66801
Replacement-lock commit: d4072f912cc58bbc1ca41ceb2652e41783dbf3e1
Replacement-lock root: 28acd2c340b3ef1e2b8b2b7b31ca69356cd9baa35b94ff8a598e020099883435
Review profile: DUAL_REVIEW
Evidence class: PUBLIC_REPOSITORY_ONLY
Closeout boundary: BENCHMARK_OR_PREREGISTRATION_FREEZE
Independence: ESTABLISHED
```

This review was conducted under the supplied fourth-correction scientific-review authority. The prior third-correction scientific brief and scientific disposition were examined only as historical finding context; neither disposition nor verification was inherited for the new exact head.

The reviewer did not implement or materially direct the fourth correction and did not perform engineering review work. No tracked file, PR field, comment, review state, branch, merge state, release, tag, benchmark, gate, renderer selection, model protocol, or model result was modified.

## B. Evidence examined

### Exact-head preflight

PR #17 was freshly resolved as:

```text
state: open
merged: false
draft: false
base: main
base SHA: 08179fbdce909e9a0d6dbb2939c58f7ab5d0d8a7
head SHA: 4d47595ede0f1690f1d9d96075df9a3ff7ab60d9
head repository: yurifrusin/ecological-predictive-states
```

The exact commit independently resolves to tree:

```text
ba112bcc4be4396648305bb1d1529cc002755ea1
```

with subject:

```text
Correct GitHub evidence access posture
```

and sole parent:

```text
53b988372634e8a7d2d006021af5b5aadbf96d3c
```

The fourth correction is therefore exactly one direct-child commit from the previously reviewed head, with no intervening commit in the review unit.

Live repository metadata reports:

```text
repository: yurifrusin/ecological-predictive-states
visibility: private
archived: false
```

Recursive exact-tree inspection found the expected source, tests, documentation, and locked configuration files and no active `docs/reviews/pr-0017*.md` review record introduced into the implementation branch. The three governing freeze snapshots retain their established Git blob identities.

### Fourth-correction scope and delta

The correction is confined to evidence-access provenance, validation plumbing, tests, CI arguments, and lifecycle documentation. Its changed implementation surface comprises:

```text
.github/workflows/ci.yml
PROJECT_HISTORY.md
README.md
RESEARCH_LOG.md
docs/GATE_0B_APPEARANCE_BENCHMARK_FREEZE_V0.md
docs/IMPLEMENTATION_NOTES.md
docs/open-questions.md
scripts/check_freeze_counterpart_adversarial.py
scripts/check_freeze_documentation_consistency.py
scripts/fetch_github_actions_evidence.py
src/epsbench/cli/app.py
src/epsbench/freeze.py
src/epsbench/github.py
tests/unit/test_freeze.py
tests/unit/test_github_evidence_fetcher.py
```

The workflow now passes separately resolved repository metadata into publication-record creation for both renderer paths. The historical and current-status documents record the correction as limited to the inaccurate evidence-access posture and explicitly preserve the replacement lock, locked scientific inputs, earlier verified engineering work, and all non-advancement boundaries.

No profile definition, benchmark role, evaluation root, scene configuration, admission threshold, renderer definition, pairing rule, label rule, training/evaluation exclusion, failure policy, or model-result safeguard was changed.

### Replacement-lock preservation

The following exact-head files retain the same Git blob identities as at replacement-lock commit `d4072f912cc58bbc1ca41ceb2652e41783dbf3e1`:

```text
configs/appearance_benchmark_freeze_v0.yaml
configs/appearance_benchmark_freeze_v0_lock.json
configs/appearance_benchmark_v0_evaluation_episode_seeds.yaml
```

I also independently reconstructed the replacement-lock domain from the downloaded canonical snapshots. The recomputed value was exactly:

```text
28acd2c340b3ef1e2b8b2b7b31ca69356cd9baa35b94ff8a598e020099883435
```

The lock continues to bind:

- the same five selected profiles and roles;
- the same two excluded profiles;
- the same sixteen evaluation roots in the same order;
- the same two scene families;
- the same primary and secondary comparisons;
- the same admission thresholds;
- the same two renderer environments;
- the same training/evaluation exclusions;
- the same all-cells/both-renderers readiness requirement;
- the same failure and root-retirement policy;
- the same renderer-selection dependency.

### Governing evidence-class semantics

The governing protocol defines `PUBLIC_REPOSITORY_ONLY` by the location and verifiability of evidence, not by requiring every repository to have public visibility. It admits evidence accessible through repository content, reproducible generated artifacts, CI, and **public or connected GitHub records**. By contrast, evidence requiring undisclosed, off-platform private material belongs under `PRIVATE_REVIEW_BUNDLE`.

The fourth-correction documentation now states this distinction directly:

```text
repository visibility: private
artifact access: connected authenticated GitHub Actions
unpublished private off-platform evidence: not admitted
```

It also distinguishes visibility, artifact accessibility, reviewer access, expiry, and retention as separate facts.

The README consistently states that an independent reviewer requires connected repository and Actions-artifact access and retains all candidate-only and non-advancement language.

### Typed repository-visibility and artifact-access contract

The fourth correction introduces:

```text
appearance_benchmark_public_ci_packet_record_v2
```

with separate fields for:

```text
repository
repository_api_url
repository_html_url
repository_visibility
repository_private
repository_archived_at_record_creation
repository_disabled_at_record_creation
artifact_access_mechanism
```

The implementation accepts typed visibility values:

```text
private
public
internal
```

and typed artifact-access mechanisms:

```text
connected_authenticated_github_actions
public_github_actions
```

A private or internal repository requires the connected-authenticated mechanism; a public repository requires the public mechanism. A record whose visibility, private flag, and mechanism are internally inconsistent is rejected. The record is also compared against independently fetched live GitHub repository metadata rather than being treated as its own authority.

The dedicated tests cover:

- private repository → connected-authenticated access;
- public repository → public Actions access;
- visibility changes during production resolution;
- another owner or repository;
- lookalike hosts;
- credentials;
- query strings;
- fragments;
- extra path components;
- self-resealed visibility/access mismatches;
- private-record/public-live and public-record/private-live cross-mismatches.

From a scientific-provenance perspective, this is an improvement: it removes a materially inaccurate accessibility description without redefining any scientific evidence domain.

### Exact-head CI

Exact-head run:

```text
33706715326
```

completed successfully for:

```text
head: 4d47595ede0f1690f1d9d96075df9a3ff7ab60d9
tree: ba112bcc4be4396648305bb1d1529cc002755ea1
```

The three successful jobs were:

```text
quality — 100497296558
qualify-wgl — 100497296367
qualify-osmesa — 100509579302
```

The job records show exact-head checkout, locked installation, documentation/lock consistency, complete quality testing, all 192 WGL cells, all 192 OSMesa cells, packet validation, raw counterpart resolution, publication-record creation, receipt creation, and the complete artifact/access/archive adversarial paths.

The quality log reported:

```text
530 tests passed
lint passed
format check passed
type check passed
documentation/lock consistency passed
```

I treated CI as execution evidence, not as a substitute for scientific review.

### Live exact-head artifacts

All four exact-head artifacts were freshly resolved and were live and unexpired:

| ArtifactIDSizeRaw ZIP SHA-256 |              |            |                                                                    |
| ----------------------------- | ------------ | ---------- | ------------------------------------------------------------------ |
| Windows/WGL packet            | `9876515995` | `21803973` | `6618f9fe265912cb3bc67c99af1a56207a382fb84f20675e6d7dcad5b53a096e` |
| Windows/WGL evidence          | `9876815245` | `6527`     | `29f988417cc20fca3885b5122db01976296058151cf99e03ad7166e7b54ff473` |
| Ubuntu/OSMesa packet          | `9880381284` | `22346269` | `67ecd1a7e695e3b6fd36930258497ec7a92a89f76e22d0ee962891dcc50b6e18` |
| Ubuntu/OSMesa evidence        | `9881190409` | `6515`     | `4c756aefce7f350c4b50a4af77ead85cc7032ea7ba0e88dc37107237aeeda53c` |

All four are bound to run `33706715326`, repository ID `1346269440`, and exact head `4d47595...`. Their recorded expiry is:

```text
2026-12-02T02:12:06Z
```

I downloaded the raw ZIP bytes and independently reproduced all four recorded byte counts and SHA-256 digests before extraction.

Each evidence archive contained exactly:

```text
publication_record.json
renderer_receipt.json
```

The archives had no duplicate names, case- or normalisation-colliding names, unsafe traversal paths, absolute paths, links, special members, or file/directory prefix collisions.

### Publication records

Both independently inspected v2 publication records state:

```text
repository: yurifrusin/ecological-predictive-states
repository_visibility: private
repository_private: true
repository_archived_at_record_creation: false
repository_disabled_at_record_creation: false
artifact_access_mechanism: connected_authenticated_github_actions
```

Those values agree with the independently resolved live repository object. The records also bind the exact source SHA and tree, workflow run and attempt, job identity, artifact identity, size, digest, timestamps, expiry, packet identity, and complete-packet root.

I independently recomputed each publication-record hash from its canonical domain and reproduced:

```text
Windows/WGL:
f668dcda1679e397d6330a37938a1c37e36cd89bec0ac6468bba551267c47a4d

Ubuntu/OSMesa:
5bc44992ffeb058ea1a65320f872f92b2b8f0b79ce166047d1a847282a673c82
```

### Qualification identities and outcomes

Independent artifact-level validation reproduced:

| IdentityWindows/WGLUbuntu/OSMesa |                                                                    |                                                                    |
| -------------------------------- | ------------------------------------------------------------------ | ------------------------------------------------------------------ |
| Complete-packet root             | `8afe725076057bda3ff86a1925a1038fbbcd03141ade5f8acf9bde88afdb4481` | `43d5f6c88f24bf9439885db1de8bda947861f01dc2ce3ce3cd3022a05dee8079` |
| Packet-tree root                 | `5c210a1a2dc5cc7bf7d5fbb68700bb7cc3dfae2387d2f85a036456d23e0af41f` | `282398b62331fa09859dcd9e6a76864d0e6c2b6efdcfea9e0aae2a5b89193149` |
| Packet-file SHA-256              | `5d63e9e16515f9416b6f7fc1f51ba5fbf8bd886b83858d52fa2c45fff0afb81f` | `47c9ef071a24926192063fb1a6ef2836ec6511fd592ac58254b4c4dcba2e6916` |
| Renderer receipt                 | `75df56fcb5975b8db02b572af5ce3e9e76be1c86ddafc10178663b8cae97d78e` | `dfbd05ba51f9d1b656f4f1df9e054e245bf7a464835fd086f6c521585a4dae0c` |
| Publication record               | `f668dcda1679e397d6330a37938a1c37e36cd89bec0ac6468bba551267c47a4d` | `5bc44992ffeb058ea1a65320f872f92b2b8f0b79ce166047d1a847282a673c82` |
| Threshold-margin root            | `0ea13297585c1a2338d2ee12eb24cb74cf848c111f362be762d6ac1b42d9eaaa` | `7ec9d2f69b0faa6492db1483142ed630dafa95634f3a0a4527300a7a43119689` |

For each renderer:

```text
total cells: 192

selected:
160 admitted
0 rejected

controls:
16 admitted
16 rejected

each selected profile:
32 of 32 admitted
```

The matching portable apparatus roots are:

```text
procedural assets:
e38f3132b1b8821b1df45ba0edae015081c8243eaac53e9a5c092c25e60de5a2

appearance assignment:
7caa2e43c07ad522e27d41a8d71a23abff22c57fcd820d8f607f0233ff49ff12

source identity:
3218027ed2135815034a8162158d5940702ec25108cd726fe87a9410c1677948

within-renderer invariance:
20037a1b7103e38abb69ab62fa659d99ba15fbd6b4fae8be5e1e9d9bfc771ebe
```

All threshold failure counts and near-threshold counts were zero.

### Scientific-payload comparison with the previous head

I compared the fourth-correction packets with the previous exact-head packets.

The scientific payload was preserved:

- the file membership for each renderer remained the same;
- all representative contact-sheet PNGs were byte-identical;
- no `.png`, `.npy`, transition, instrumentation, metric array, ecological-label array, or other scientific binary payload changed;
- matrix differences were confined to source-commit and content-provenance bindings;
- top-level packet differences were confined to source commit, publication/receipt snapshots, execution-bound packet roots, and renderer-local retained-source-evidence roots;
- selected/control outcomes, threshold roots, portable roots, profile counts, scene counts, pairings, and readiness outcomes remained unchanged.

A representative retained-source manifest differed only in:

```text
source_provenance.git_commit
source_provenance_sha256
content_provenance_binding_sha256
```

This establishes that the fourth correction changed evidence provenance and access description rather than benchmark content or qualification semantics.

### Representative visual evidence

I inspected representative contact sheets covering:

```text
five selected profiles
× two scene families
× two renderer environments
```

The sheets showed coherent before/after RGB interventions, controlled segmentation, and admitted outcomes. No blank, degenerate, mislabeled, or scientifically changed render was observed. The fourth-head contact sheets were byte-identical to the corresponding third-head images.

### Reproduction classification

```text
Independently reconstructed or validated:
- exact PR/head/tree identity
- one-parent correction chronology
- replacement-lock root
- raw size and SHA-256 of all four archives
- evidence-archive exact member sets
- archive path/member safety
- publication-record hashes and typed access fields
- receipt hashes
- complete-packet roots
- public packet-tree roots
- selected/control counts
- per-profile and per-scene membership
- threshold summaries
- portable apparatus-root agreement
- non-advancement fields
- third-head versus fourth-head scientific-payload comparison
- representative contact-sheet evidence

Verified from exact-head CI:
- uv sync --locked
- ruff check
- ruff format --check
- mypy
- documentation/lock consistency
- complete 530-test suite
- all 192 Windows/WGL cells
- all 192 Ubuntu/OSMesa cells
- full packet, repository-access, archive, and counterpart adversarial paths

Not independently reproduced:
- fresh local rendering of all 192 WGL cells
- fresh local rendering of all 192 OSMesa cells
- complete repository command suite in a mounted exact-head checkout
- final local git status from a mounted reviewer worktree
```

No tracked repository checkout was mounted or modified during this review. Downloaded archives and extracted evidence were handled in isolated review directories.

## C. Stable scientific finding

```text
EPS-SR17-0001:
VERIFIED
```

The exact replacement lock still binds renderer-selection dependency root:

```text
f9a36861cfd99a2993c926d34bbdd62526ca0bec2ba95d5129a5331d5976e4bd
```

The following remain unchanged:

```text
primary_model_result_renderer: null
other_renderer_model_result_classification: null
aggregation_rule: null
comparative_model_result_access_authorised: false
dependency_status: required_not_yet_satisfied
```

The governing contract continues to require, before comparative model-result access:

- prospective selection of exactly one primary model-result renderer;
- prospective classification of the other renderer as `replication`, `robustness`, `sensitivity`, or `unsupported`;
- selection independent of observed comparative results;
- no result-dependent renderer averaging or aggregation;
- prospective preregistration of any aggregation rule.

Apparatus qualification on both renderers does not satisfy this model-result dependency. The fourth correction neither weakens nor satisfies it.

## D. Prior access-posture limitation

```text
Prior access-posture limitation:
RESOLVED
```

The previous scientific review recorded a nonblocking limitation because a private connected repository was described by the literal value:

```text
public_repository_authenticated_actions_artifact
```

The prior exact-head engineering review independently identified the same false literal posture as the unresolved part of `EPS-ER17-0005` and raised `EPS-ER17-0008`.

At the fourth-correction exact head:

- repository visibility is recorded as `private`;
- the live private flag is `true`;
- artifact access is recorded separately as `connected_authenticated_github_actions`;
- the values match live repository metadata;
- public and connected access are no longer conflated;
- an unconnected reviewer’s lack of access is explicitly stated;
- visibility or availability changes during artifact resolution are detected;
- publication records are compared against live repository metadata;
- off-platform undisclosed evidence remains inadmissible.

The specific inaccurate-public-posture limitation therefore no longer persists.

## E. Scientific determinations

1. **ACCEPTABLE** — The review is bound to the correct repository, PR, canonical base, exact head, and exact tree.
2. **ACCEPTABLE** — The fourth correction is exactly one direct-child commit of `53b988372634e8a7d2d006021af5b5aadbf96d3c` and is consistent with the documented owner-authorised correction boundary.
3. **ACCEPTABLE** — The prospective replacement lock is byte-for-byte preserved, independently reconstructs to the expected root, and remains authoritative for the candidate.
4. **ACCEPTABLE** — Every locked profile, role, profile membership, evaluation root and ordering, scene, comparison, threshold, renderer definition, exclusion, and failure rule is preserved.
5. **ACCEPTABLE** — No outcome-responsive benchmark tuning occurred. The change concerns access/provenance metadata, not observed qualification values or model results.
6. **ACCEPTABLE** — `EPS-SR17-0001` remains fully effective and is independently renewed as `VERIFIED`.
7. **ACCEPTABLE** — Repository visibility is now represented truthfully as `private`.
8. **ACCEPTABLE** — Artifact access is now represented truthfully as connected authenticated GitHub Actions access.
9. **ACCEPTABLE** — The v2 schema clearly separates repository visibility from artifact-access mechanism.
10. **ACCEPTABLE** — The prior inaccurate-public-posture limitation is resolved.
11. **ACCEPTABLE** — `PUBLIC_REPOSITORY_ONLY` retains its governing meaning: evidence must remain in repository content, reproducible artifacts, CI, or public/connected GitHub records and does not expand to unpublished off-platform evidence.
12. **ACCEPTABLE** — An appropriately connected independent reviewer can retrieve, authenticate, and validate all evidence required for this exact-head review.
13. **ACCEPTABLE\_WITH\_NONBLOCKING\_LIMITATION** — An unconnected reviewer cannot access this private repository or its Actions artifacts. That access limitation is now stated explicitly rather than concealed and does not make connected evidence scientifically invalid under the governing protocol.
14. **ACCEPTABLE** — Visibility changes, archival, disablement, expiry, and availability loss are procedurally visible and fail closed rather than leaving a stale accessibility claim silently authoritative.
15. **ACCEPTABLE** — The correction changes evidence provenance and access description only; it does not change the benchmark construct.
16. **ACCEPTABLE** — Portable scientific identities remain separated from renderer-local, packet, receipt, publication-record, source-provenance, and raw-archive identities.
17. **ACCEPTABLE** — WGL and OSMesa qualification continues to support apparatus readiness without implying model efficacy, EPS superiority, model generalisation, or cross-renderer model-result portability.
18. **ACCEPTABLE\_WITH\_NONBLOCKING\_LIMITATION** — Privileged packets remain prohibited from learner and model-development use. Governance cannot technically prevent deliberate external misuse of known final evidence, but the fourth correction does not enlarge that previously accepted risk.
19. **ACCEPTABLE** — Evaluation-only posture, public-root exposure, finite sixteen-root sampling, two-scene coverage, absent stripe-only coverage, absent illumination-only coverage, checker-frequency relativity, and combined-stress compound interpretation remain explicit.
20. **ACCEPTABLE\_WITH\_NONBLOCKING\_LIMITATION** — Current artifact retention is sufficient for this review. Durable preservation of final receipts, metadata, and preferably exact archives remains a later authorised closeout dependency before December 2, 2026.
21. **ACCEPTABLE** — All lifecycle and non-advancement claims remain scientifically truthful: no benchmark freeze, model-protocol freeze, comparative-result access, renderer selection, Gate 0B completion, Gate 0C/0D authorisation, owner approval, merge, closeout, or scientific result is asserted.
22. **ACCEPTABLE\_WITH\_NONBLOCKING\_LIMITATION** — Exact head `4d47595ede0f1690f1d9d96075df9a3ff7ab60d9` is scientifically acceptable as an Appearance Benchmark Input Freeze v0 candidate, subject to the explicit benchmark, connected-access, and retention limitations. This does not itself freeze or approve the candidate.
23. **ACCEPTABLE** — The reviewer performed no engineering adjudication, implementation, correction, owner action, merge, closeout, benchmark freeze, renderer selection, model work, tag, Release, gate action, or next work package.

## F. New findings

```text
New scientific findings: NONE
Blocking scientific findings: NONE
```

No `EPS-ER17-*` finding has been adjudicated or converted into a scientific finding.

## G. Limitations

### Benchmark limitations

The candidate remains bounded to:

```text
two scene families
sixteen evaluation roots
one development-reference appearance
four held-out Appearance-OOD appearances
```

It does not establish broad environmental, geometric, camera, dynamic, or ecological generalisation.

`revision1_stripes_low_v1` and `revision1_illumination_shift_v1` remain excluded negative evidence. No stripe-only or illumination-only robustness claim is authorised. Checker-high is a frequency diagnostic only relative to checker-low. Combined stress remains a compound intervention and cannot isolate one appearance axis.

Qualification establishes that the selected appearances can be instantiated while preserving the locked apparatus and satisfying prospectively specified admission criteria. It does not establish that a model is robust to those appearances.

### Private and connected evidence access

The evidence is independently reviewable by a reviewer with connected authenticated access to the repository and GitHub Actions artifacts. It is not anonymously or universally public.

Loss of connected access would prevent a new reviewer from retrieving the current artifacts. The v2 records now state this accurately, and live validation detects a changed visibility or availability posture. This is a real access constraint but not a defect under the governing evidence-class definition.

The label `PUBLIC_REPOSITORY_ONLY` is lexically broader than “public visibility,” but its authoritative repository definition is unambiguous: public **or connected** GitHub records are permitted; undisclosed private off-platform evidence is not.

### Evaluation-exposure limitation

The complete packets expose final roots and privileged apparatus evidence to authorised connected reviewers. The benchmark therefore depends on protocol discipline to prevent their use for:

```text
training
validation
preprocessing selection
architecture selection
hyperparameter selection
checkpoint selection
post-result role changes
difficult-root replacement
empirical debugging
```

The fourth correction neither enlarges nor removes that limitation.

### Artifact-retention limitation

The current exact-head artifacts expire on:

```text
2026-12-02T02:12:06Z
```

They are sufficient for the present review, and deterministic regeneration remains defined. They are not a permanent scientific archive.

A later separately authorised benchmark-freeze closeout should preserve at minimum:

- final renderer receipts;
- publication records;
- exact artifact metadata;
- preferably all four exact raw ZIP archives.

No archival, closeout, tag, or Release action was performed here.

### Reproduction limitations

The scientific payload was independently validated from downloaded exact-head artifacts, but I did not independently regenerate all 192 cells in either renderer.

The source quality suite, renderer executions, and full adversarial paths were verified from exact-head CI rather than rerun in a mounted local checkout. No claim is made that the repository command suite or final `git status --short` was executed locally.

### Deferred Gate 0D/0E requirements

Before any comparative model-result access, a separately authorised and prospectively frozen protocol must still:

```text
select one primary model-result renderer;
classify the other renderer;
define or explicitly reject an aggregation rule;
freeze training roots;
freeze validation roots;
freeze model-randomness roots;
freeze matched model budgets;
freeze task endpoints and evaluation metrics;
preserve modality permissions;
authorise comparative model-result access.
```

None of these requirements is satisfied by apparatus qualification or this review.

### Nonblocking future hardening

Longer-term archival retention and, eventually, a stable reviewer-access continuity plan would improve freeze reproducibility. Neither requires alteration of the present locked scientific construct.

## H. Disposition

The fourth correction truthfully distinguishes private repository visibility from connected authenticated artifact access, preserves the governing evidence-class boundary, preserves the complete locked scientific benchmark construct, retains `EPS-SR17-0001`, and introduces no new scientific-validity, leakage, provenance, comparison, adaptation, or interpretation defect.

```text
Role: SCIENTIFIC_REVIEWER
Disposition: SCIENTIFIC_PASS
Reviewed PR: #17
Reviewed SHA: 4d47595ede0f1690f1d9d96075df9a3ff7ab60d9
Reviewed tree: ba112bcc4be4396648305bb1d1529cc002755ea1
Canonical base: 08179fbdce909e9a0d6dbb2939c58f7ab5d0d8a7
Review profile: DUAL_REVIEW
Evidence class: PUBLIC_REPOSITORY_ONLY
Closeout boundary: BENCHMARK_OR_PREREGISTRATION_FREEZE
Repository visibility: PRIVATE
Artifact access mechanism: CONNECTED_AUTHENTICATED_GITHUB_ACTIONS
EPS-SR17-0001: VERIFIED
Prior access-posture limitation: RESOLVED
New scientific findings: NONE
Phase-gate effect: NONE
Scientific result: NONE
Historical lock: PRESERVED
Replacement lock: REVIEWED, NOT A BENCHMARK FREEZE
Exact-head CI: REVIEWED
Exact-head artifacts: REVIEWED
Freeze candidate: REVIEWED, NOT FROZEN
Benchmark freeze: NOT PERFORMED
Model protocol freeze: NOT PERFORMED
Comparative model-result access: NOT AUTHORISED
Primary model-result renderer: NOT SELECTED
Other renderer classification: NOT SELECTED
Renderer aggregation rule: NOT SELECTED
Full Gate 0B completion: NOT CLAIMED
Gate 0C authorisation: NOT GRANTED
Gate 0D authorisation: NOT GRANTED
Owner implementation approval: NOT GIVEN
Implementation merge approval: NOT GIVEN
Engineering re-review: SEPARATE AUTHORITY
Merge: NOT PERFORMED
Closeout: NOT PERFORMED
Tag: NOT CREATED
Release: NOT PUBLISHED
Empirical gate: NOT EVALUATED
Model work: NOT PERFORMED
Next work package: NOT BEGUN
Repository mutation by reviewer: NONE
```

## I. Routing statement

```text
Return this exact-head scientific disposition, the renewed status of
EPS-SR17-0001, the status of the prior access-posture limitation, and every new
EPS-SR17-* finding to the ORCHESTRATOR_COORDINATOR.

Do not perform or draft the engineering re-review.

The scientific disposition does not determine the engineering disposition.

After receiving this review, the coordinator is to freshly revalidate PR #17
and route the same exact head

4d47595ede0f1690f1d9d96075df9a3ff7ab60d9

and exact tree

ba112bcc4be4396648305bb1d1529cc002755ea1

to the independent ENGINEERING_REVIEWER unless another correction has changed
the review unit.

No owner implementation approval, merge, closeout, benchmark freeze, renderer
selection, model-protocol freeze, comparative-result access, tag, Release, gate
action, model work, scientific result, or next work package is authorised by
this review.
```
