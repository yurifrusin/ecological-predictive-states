## A. Review identity

```text
Role: ENGINEERING_REVIEWER
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
Repository visibility: PRIVATE
Artifact access mechanism: CONNECTED_AUTHENTICATED_GITHUB_ACTIONS
Independence: ESTABLISHED
```

The supplied fourth-correction brief defines this exact review unit, the eight stable engineering findings, the required evidence and determinations, and the read-only authority boundary. The preceding exact-head brief was used only as immutable finding history and not as inherited engineering authority.

I did not implement or materially direct the fourth correction. I performed no tracked repository write, correction, scientific review, owner action, PR mutation, merge, closeout, benchmark freeze, renderer selection, model work, tag, Release, gate action, or next-work-package action.

## B. Evidence examined

### Exact-head and scope preflight

Live GitHub records show that PR #17 remains open, unmerged, mergeable, and non-draft. Its base remains `main` at `08179fbdce909e9a0d6dbb2939c58f7ab5d0d8a7`, and its head remains `4d47595ede0f1690f1d9d96075df9a3ff7ab60d9`.

The reviewed commit has tree `ba112bcc4be4396648305bb1d1529cc002755ea1`, subject `Correct GitHub evidence access posture`, and exactly one parent, the prior engineering-reviewed head `53b988372634e8a7d2d006021af5b5aadbf96d3c`. The comparison is one commit ahead, zero commits behind, with the previous reviewed head as the merge base.

The fourth-correction delta contains these fifteen files and no additional changed file:

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

That surface is confined to repository/access metadata acquisition, publication-record typing and validation, tests, CI wiring, and lifecycle/access documentation. It does not alter the benchmark definition, seed registry, appearance registries, scene configurations, simulator or annotation code, thresholds, pairing policy, renderer policy, or training/evaluation policy.

The live repository object reports:

```text
full_name: yurifrusin/ecological-predictive-states
private: true
visibility: private
archived: false
disabled: false
```

No active `docs/reviews/pr-0017*.md` record was found in the reviewed branch.

### Governing and changed material

I examined the governing research and review documents, the current PR body, the fourth-correction owner-authority and implementation report, all files changed by the fourth correction, and the relevant preserved implementation paths in:

```text
src/epsbench/freeze.py
src/epsbench/schema.py
src/epsbench/data/provenance.py
src/epsbench/data/publication.py
scripts/check_freeze_adversarial_packets.py
tests/unit/test_provenance.py
```

The review protocol defines `PUBLIC_REPOSITORY_ONLY` as evidence available through repository content, reproducible generated artifacts, CI, and public **or connected** GitHub records. It does not require every underlying repository to be anonymously public.

The current documentation now consistently states that:

- this repository is private;
- the Actions evidence requires connected authenticated GitHub access;
- a public repository would instead use public GitHub Actions access;
- repository visibility, artifact access, expiry, and reviewer access are distinct facts;
- `PUBLIC_REPOSITORY_ONLY` excludes unpublished private off-platform evidence;
- source-committed implementation status is separate from post-commit CI evidence;
- engineering review, scientific review, owner approval, benchmark freeze, model-protocol freeze, gate state, and scientific-result state remain distinct.

This distinction appears in the README, project history, research log, freeze contract, implementation notes, and open-questions record.

The documentation-consistency checker tests multiple semantic requirements across the status-bearing documents, rejects current authority fabrication, checks multiple access-posture concepts rather than one fixed sentence, rejects retention of the obsolete combined access literal in production or test paths, and compares locked inputs byte-for-byte against the replacement-lock commit.

### Replacement-lock preservation and reconstruction

I compared the replacement-lock snapshots at `d4072f912cc58bbc1ca41ceb2652e41783dbf3e1` with the exact reviewed head.

The benchmark-definition blobs have the same Git blob SHA and exact contents at both revisions.

The replacement-lock blobs likewise have the same SHA and exact contents.

The final-evaluation seed-registry blobs also match exactly.

I independently removed the self-hash field, constructed the versioned `epsbench_logical_domain_envelope_v1` definition-lock envelope, serialized it canonically, and calculated:

```text
28acd2c340b3ef1e2b8b2b7b31ca69356cd9baa35b94ff8a598e020099883435
```

This matches the preserved replacement-lock root.

No access-posture field appears in the benchmark definition, replacement lock, selected or excluded profile identities, evaluation-root identity, thresholds, pairing policy, training/evaluation policy, renderer-selection dependency, or portable apparatus-root domains.

### Exact-head CI

The exact-head workflow is:

```text
Run: 33706715326
Attempt: 1
Head: 4d47595ede0f1690f1d9d96075df9a3ff7ab60d9
Tree: ba112bcc4be4396648305bb1d1529cc002755ea1
Status: completed
Conclusion: success
```

It is linked to PR #17 and its canonical base.

All three jobs succeeded at the same exact head:

```text
quality — 100497296558 — SUCCESS
qualify-wgl — 100497296367 — SUCCESS
qualify-osmesa — 100509579302 — SUCCESS
```

The job records show exact-head checkout and the expected production steps: locked installation, lint, formatting, typing, documentation/lock checks, full tests, matrix and qualification checks, stable-handle regressions, all 192 cells per renderer, raw counterpart acquisition, production publication-record creation, packet and counterpart adversarial validation, and evidence publication.

The quality job reports:

```text
530 passed
```

The WGL job reproduced:

```text
Complete-packet root:
8afe725076057bda3ff86a1925a1038fbbcd03141ade5f8acf9bde88afdb4481

Selected:
160 admitted

Controls:
16 admitted
16 rejected
```

It also reproduced the preserved replacement-lock and portable apparatus roots.

### Live artifact resolution and independent raw-archive checks

Live GitHub metadata reports exactly four current, unexpired artifacts, all tied to run `33706715326` and head `4d47595ede0f1690f1d9d96075df9a3ff7ab60d9`.

I independently downloaded all four raw ZIP archives and calculated their byte counts and SHA-256 values before extraction:

| ArtifactIDIndependently measured sizeIndependently calculated SHA-256 |              |            |                                                                    |
| --------------------------------------------------------------------- | ------------ | ---------- | ------------------------------------------------------------------ |
| WGL packet                                                            | `9876515995` | `21803973` | `6618f9fe265912cb3bc67c99af1a56207a382fb84f20675e6d7dcad5b53a096e` |
| WGL evidence                                                          | `9876815245` | `6527`     | `29f988417cc20fca3885b5122db01976296058151cf99e03ad7166e7b54ff473` |
| OSMesa packet                                                         | `9880381284` | `22346269` | `67ecd1a7e695e3b6fd36930258497ec7a92a89f76e22d0ee962891dcc50b6e18` |
| OSMesa evidence                                                       | `9881190409` | `6515`     | `4c756aefce7f350c4b50a4af77ead85cc7032ea7ba0e88dc37107237aeeda53c` |

Every independently measured value matched live GitHub metadata.

All four artifacts report:

```text
expires_at: 2026-12-02T02:12:06Z
expired: false
```

Before extraction, I checked canonical member names and types and rejected or would reject absolute paths, drive-qualified paths, `..` traversal, ambiguous backslash paths, duplicate names, normalized-name collisions, case-fold collisions, file/directory collisions, symlink or special-file entries, and extraction over existing files.

Observed packet contents were:

```text
WGL packet:
6743 regular members

OSMesa packet:
6745 regular members
```

Each evidence archive contained exactly:

```text
publication_record.json
renderer_receipt.json
```

No additional, missing, duplicate, linked, special, traversal, absolute, or colliding evidence member was present.

### Publication-record v2 verification

Both extracted records use:

```text
schema_version:
appearance_benchmark_public_ci_packet_record_v2
```

Both state:

```text
repository:
yurifrusin/ecological-predictive-states

repository_api_url:
https://api.github.com/repos/yurifrusin/ecological-predictive-states

repository_html_url:
https://github.com/yurifrusin/ecological-predictive-states

repository_visibility:
private

repository_private:
true

repository_archived_at_record_creation:
false

repository_disabled_at_record_creation:
false

artifact_access_mechanism:
connected_authenticated_github_actions
```

The exact-head parser recognizes only `private`, `public`, and `internal`; derives public access only for public visibility; derives connected authenticated access for private or internal visibility; requires exact canonical repository, API, and HTML identities; requires real Boolean values; rejects public/private contradictions; and rejects an archived or disabled live repository. It imports the same canonical repository-identity routine already used by source provenance.

The publication model is strict and version-exact. A v1 record, the obsolete `public_repository_authenticated_actions_artifact` literal, an omitted visibility/access field, a string in place of the private Boolean, an unknown field, or an unsupported enum value cannot satisfy the current v2 model. The record identity includes the normative repository URLs, visibility, private Boolean, archived/disabled posture, and access mechanism.

I independently recalculated each canonical publication-record identity:

```text
WGL:
f668dcda1679e397d6330a37938a1c37e36cd89bec0ac6468bba551267c47a4d

OSMesa:
5bc44992ffeb058ea1a65320f872f92b2b8f0b79ce166047d1a847282a673c82
```

Both match their record fields.

### Live-metadata flow and TOCTOU behavior

The production acquisition flow is:

```text
fetch live repository object
→ fetch artifact, workflow run and job metadata
→ download exact raw archive
→ refetch artifact metadata
→ fetch repository object again
→ compare normalized repository identity, visibility and availability
→ retain raw archive plus repository/artifact/run/job metadata
→ create publication record
→ validate record and receipt against the authenticated packet
→ derive counterpart readiness
```

The fetcher obtains repository metadata before artifact acquisition and again after acquisition. It also refetches artifact metadata after download and rejects disappearance, expiry, or identity change before writing the evidence bundle.

Both WGL and OSMesa production publication commands pass the independently fetched `live_repository.json`; they do not synthesize repository visibility from a caller-provided publication record.

Packet and evidence counterpart bundles retain distinct repository metadata files. Validation derives the trusted access object independently for each and rejects cross-bundle repository visibility or availability differences. Publication-record fields are compared against that live-derived object; recalculating `record_sha256` cannot convert a conflicting record into authority.

No credential or token is serialized into a publication record or retained as an access field.

### Visibility/access mismatch attacks

I reviewed and independently exercised the v2 comparison behavior for:

```text
private live repository + public record
private live repository + public access mechanism
private live repository + private=false
private live repository + wrong API URL
private live repository + wrong HTML URL
public live repository + private record
public live repository + connected-private mechanism
public live repository + private=true
internal visibility + wrong access mechanism
unsupported visibility
missing visibility
missing private Boolean
private represented as a string
archived repository
disabled repository
repository visibility changed after initial resolution
repository unavailable or incomplete after initial resolution
different owner
different repository
lookalike host
embedded credentials
query string
fragment
extra path component
self-resealed visibility mismatch
self-resealed access-mechanism mismatch
self-resealed archived/disabled mismatch
obsolete v1 schema
obsolete combined access literal
```

The production parser tests include private and public derivation, a repository visibility change during production resolution, and exact rejection of owner, repository, lookalike-host, credential, query, fragment, and extra-path substitutions.

The publication tests explicitly cover internally self-resealed but inconsistent access values, matching private/public postures, and a self-resealed public record compared with live private metadata.

The exact-head OSMesa target corpus reached production validation and rejected:

```text
private repository with self-resealed public posture
public live metadata with private-connected publication posture
evidence resolution with changed live repository visibility
publication record for a different packet
```

It ended with:

```text
all target-specific raw-artifact counterpart regressions passed
```

### Preserved raw-artifact and source-authority controls

The current adversarial corpus also reached and rejected:

- equal-length packet bytes with a different digest at `artifact_archive_digest_mismatch`;
- packet disappearance;
- packet expiry;
- packet/evidence archive cross-wiring;
- a detached extracted directory;
- an independently valid alternate packet whose receipt, publication record, packet tree, roots, margins, profiles, outcomes, and readiness dependencies were recomputed before substitution;
- evidence re-encoding;
- missing or extra evidence members;
- traversal and absolute members;
- duplicate, case-colliding, and file/directory-colliding members;
- symbolic links;
- receipt/publication-record mispairing.

The exact log states that the independently valid alternate packet and all resealed dependencies validated before substitution.

All nine authoritative source identities remain reconstructed from retained datasets:

```text
sampled_geometry
camera_trajectory
executed_action
surface_remapping
scene_content_identity
analytic_transport
oriented_boundary_ownership
visibility_events
public_occlusion_relation
```

The complete-packet adversarial corpus again rejected single-field substitutions, all-nine substitutions, complete matched-control dependency reseals, fully recomputed matrix/pair/portable/renderer/packet roots, wrong-root evidence, wrong-profile evidence, wrong-scene evidence, missing evidence, additional evidence, altered evidence, and extra packet-tree entries before reporting success.

The fourth-correction access metadata does not participate in these source identities or in scientific pairings.

### Stable-handle publication and canonical repository identity

The stable-handle receipt publication implementation was not bypassed. The fourth correction passes an additional repository-metadata input to publication-record creation; it does not create an alternate receipt-publication path.

The exact-head WGL job successfully ran the Windows stable-handle regression suite before packet generation. The preserved tests cover existing and racing targets, parent replacement, aliases, symlink/reparse paths, target substitution before reopen, no-replace semantics, rollback, cleanup, and packet revalidation.

The canonical repository identity remains:

```text
yurifrusin/ecological-predictive-states
```

Equivalent credential-free GitHub HTTPS forms with and without a terminal `.git` resolve to that identity. Other hosts, owners, repositories, extra path components, credentials, queries, fragments, unexpected ports, and ambiguous encodings remain rejected. The new live-repository parser imports rather than duplicates this canonicalization contract.

### Qualification identities and outcomes

Independent parsing and canonical hashing of the extracted packets and records reproduced:

| IdentityWindows/WGLUbuntu/OSMesa |                                                                    |                                                                    |
| -------------------------------- | ------------------------------------------------------------------ | ------------------------------------------------------------------ |
| Complete-packet root             | `8afe725076057bda3ff86a1925a1038fbbcd03141ade5f8acf9bde88afdb4481` | `43d5f6c88f24bf9439885db1de8bda947861f01dc2ce3ce3cd3022a05dee8079` |
| Packet-tree root                 | `5c210a1a2dc5cc7bf7d5fbb68700bb7cc3dfae2387d2f85a036456d23e0af41f` | `282398b62331fa09859dcd9e6a76864d0e6c2b6efdcfea9e0aae2a5b89193149` |
| Packet-file SHA-256              | `5d63e9e16515f9416b6f7fc1f51ba5fbf8bd886b83858d52fa2c45fff0afb81f` | `47c9ef071a24926192063fb1a6ef2836ec6511fd592ac58254b4c4dcba2e6916` |
| Renderer-receipt SHA-256         | `75df56fcb5975b8db02b572af5ce3e9e76be1c86ddafc10178663b8cae97d78e` | `dfbd05ba51f9d1b656f4f1df9e054e245bf7a464835fd086f6c521585a4dae0c` |
| Publication-record SHA-256       | `f668dcda1679e397d6330a37938a1c37e36cd89bec0ac6468bba551267c47a4d` | `5bc44992ffeb058ea1a65320f872f92b2b8f0b79ce166047d1a847282a673c82` |
| Threshold-margin root            | `0ea13297585c1a2338d2ee12eb24cb74cf848c111f362be762d6ac1b42d9eaaa` | `7ec9d2f69b0faa6492db1483142ed630dafa95634f3a0a4527300a7a43119689` |

Both packets contain:

```text
Total cells: 192
Selected: 160 admitted, 0 rejected
Controls: 16 admitted, 16 rejected
Each selected profile: 32 of 32 admitted
```

Both reconstruct the same portable apparatus roots:

```text
Procedural assets:
e38f3132b1b8821b1df45ba0edae015081c8243eaac53e9a5c092c25e60de5a2

Appearance assignment:
7caa2e43c07ad522e27d41a8d71a23abff22c57fcd820d8f607f0233ff49ff12

Source identity:
3218027ed2135815034a8162158d5940702ec25108cd726fe87a9410c1677948

Within-renderer invariance:
20037a1b7103e38abb69ab62fa659d99ba15fbd6b4fae8be5e1e9d9bfc771ebe
```

The OSMesa packet’s retained WGL counterpart publication record and receipt are byte-identical to the files from the authenticated WGL evidence archive.

The metadata correction changed exact source-bound packet, receipt, publication-record, and raw-archive identities, as expected. It did not change selected/control membership, profiles, ordered roots, thresholds, source reconstruction, pairings, portable apparatus roots, renderer environments, or authority flags.

Both receipts and the replacement lock retain:

```text
benchmark_frozen: false
model_protocol_frozen: false
comparative_model_result_access_authorised: false
full_gate_0b_complete: false
scientific_result: null
```

### Commands, reproduction, and final reviewer state

A direct local clone or worktree was not available through the connected private-repository interface. I therefore did not claim local execution of `uv sync`, Ruff, Mypy, Pytest, or renderer generation.

The exact-head CI did execute the repository-defined locked commands and complete renderer/adversarial paths. Separately, I:

- resolved live repository, PR, commit, workflow, job, and artifact metadata;
- downloaded all four raw archives;
- calculated all four raw sizes and SHA-256 values;
- preflighted and safely extracted all four archives;
- parsed both packets and both evidence archives;
- independently recalculated both publication-record identities;
- independently recalculated both receipt identities;
- independently recalculated both complete-packet roots;
- independently reconstructed the replacement-lock root;
- checked exact member sets, counts, portable roots, outcomes, and non-advancement fields;
- exercised representative v2 access/live-metadata mismatch and self-reseal cases;
- inspected the exact-head CI execution of the complete preserved adversarial suites.

All disposable reviewer archives, extraction directories, and mutation files were removed. No repository worktree was created, and no tracked or connected GitHub state was changed.

## C. Stable finding verification

```text
EPS-ER17-0001: VERIFIED
EPS-ER17-0002: VERIFIED
EPS-ER17-0003: VERIFIED
EPS-ER17-0004: VERIFIED
EPS-ER17-0005: VERIFIED
EPS-ER17-0006: VERIFIED
EPS-ER17-0007: VERIFIED
EPS-ER17-0008: VERIFIED
```

## D. Engineering determinations

| #DeterminationResult |                                                                                                                                                                                                  |                                         |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------- |
| 1                    | The review is bound to the correct repository, PR, canonical base, exact head, and exact tree.                                                                                                   | **PASS**                                |
| 2                    | The fourth correction is exactly one direct-child commit of the previous reviewed head and remains within the owner-authorised `EPS-ER17-0005`/`EPS-ER17-0008` scope.                            | **PASS**                                |
| 3                    | The historical original lock and prospective replacement-lock commit are preserved.                                                                                                              | **PASS**                                |
| 4                    | The replacement-lock snapshots are byte-identical and independently reconstruct to `28acd2c…`.                                                                                                   | **PASS**                                |
| 5                    | Every locked scientific input and policy is preserved.                                                                                                                                           | **PASS**                                |
| 6                    | All three exact-head CI jobs are bound to `4d47595e…` and succeeded.                                                                                                                             | **PASS**                                |
| 7                    | All four exact-head artifacts are live, unexpired, correctly named, and bound to the current run and head.                                                                                       | **PASS**                                |
| 8                    | The current publication-record schema is strictly versioned as `appearance_benchmark_public_ci_packet_record_v2`.                                                                                | **PASS**                                |
| 9                    | The schema separates repository visibility from artifact access mechanism.                                                                                                                       | **PASS**                                |
| 10                   | Both current records truthfully state `repository_visibility=private`.                                                                                                                           | **PASS**                                |
| 11                   | Both current records truthfully state `connected_authenticated_github_actions`.                                                                                                                  | **PASS**                                |
| 12                   | Independently fetched live GitHub repository metadata is authoritative for visibility.                                                                                                           | **PASS**                                |
| 13                   | The artifact access mechanism is derived from trusted visibility rather than accepted from the record.                                                                                           | **PASS**                                |
| 14                   | Record/live visibility mismatches are rejected.                                                                                                                                                  | **PASS**                                |
| 15                   | Record/live private-Boolean mismatches are rejected.                                                                                                                                             | **PASS**                                |
| 16                   | Archived or disabled record/live mismatches are rejected, and archived or disabled live repositories are not accepted.                                                                           | **PASS**                                |
| 17                   | A repository visibility or availability change during evidence acquisition is detected.                                                                                                          | **PASS**                                |
| 18                   | Recalculating a publication-record hash cannot conceal an access mismatch.                                                                                                                       | **PASS**                                |
| 19                   | Private/public cross-mismatches are tested through the production comparison path.                                                                                                               | **PASS**                                |
| 20                   | Unsupported, missing, mistyped, or malformed visibility values are rejected.                                                                                                                     | **PASS**                                |
| 21                   | Obsolete v1 records and the obsolete combined access literal cannot satisfy current exact-head evidence validation.                                                                              | **PASS**                                |
| 22                   | `EPS-ER17-0008` is independently verified.                                                                                                                                                       | **PASS**                                |
| 23                   | Lifecycle and evidence-access documentation is materially consistent.                                                                                                                            | **PASS**                                |
| 24                   | The access limitation for an unconnected reviewer is stated truthfully and is not confused with evidence nonexistence.                                                                           | **PASS**                                |
| 25                   | The authoritative `PUBLIC_REPOSITORY_ONLY` definition remains coherent and is consistently applied to public or connected GitHub evidence.                                                       | **PASS**                                |
| 26                   | `EPS-ER17-0005` is independently verified.                                                                                                                                                       | **PASS**                                |
| 27                   | `EPS-ER17-0001`, the raw-artifact-to-consumed-bytes binding, is independently reverified.                                                                                                        | **PASS**                                |
| 28                   | `EPS-ER17-0002`, single independently reconstructed source authority, is independently reverified.                                                                                               | **PASS**                                |
| 29                   | `EPS-ER17-0003`, stable-handle atomic publication, is independently reverified.                                                                                                                  | **PASS**                                |
| 30                   | `EPS-ER17-0004`, unique versioned domains and replacement-lock integrity, is independently reverified.                                                                                           | **PASS**                                |
| 31                   | `EPS-ER17-0006`, target-specific non-vacuous adversarial coverage, is independently reverified.                                                                                                  | **PASS**                                |
| 32                   | `EPS-ER17-0007`, canonical GitHub repository identity, is independently reverified.                                                                                                              | **PASS**                                |
| 33                   | Raw packet and evidence archives remain bound to live artifact metadata through independently calculated sizes and digests before extraction.                                                    | **PASS**                                |
| 34                   | No independently valid alternate packet plus complete receipt/publication/tree/readiness reseal can manufacture readiness.                                                                       | **PASS**                                |
| 35                   | All nine authoritative source identities remain independently reconstructed.                                                                                                                     | **PASS**                                |
| 36                   | Stable-handle publication remains atomic, exclusive, no-replace, non-aliasing, non-destructive, and fail-closed.                                                                                 | **PASS**                                |
| 37                   | All scientific, source, packet, receipt, publication, threshold, tree, and readiness roots retain unique versioned domains.                                                                      | **PASS**                                |
| 38                   | Canonical repository identity remains exact, shared, and reproducible across supported equivalent origins.                                                                                       | **PASS**                                |
| 39                   | Exact-head CI exercises the production repository-access and publication path for both renderers.                                                                                                | **PASS**                                |
| 40                   | Adversarial tests complete their setup, reach production validation, and assert the intended access or artifact boundary.                                                                        | **PASS**                                |
| 41                   | Both current renderer publication records validate against live private-repository metadata.                                                                                                     | **PASS**                                |
| 42                   | Both renderer packets are complete, exact-member-bound, independently hashable, and fail-closed.                                                                                                 | **PASS**                                |
| 43                   | Selected/control counts, profile outcomes, thresholds, pairings, evaluation roots, and portable roots are unchanged.                                                                             | **PASS**                                |
| 44                   | Portable scientific identities remain separate from publication/access, packet-file, and raw-archive identities.                                                                                 | **PASS**                                |
| 45                   | The current artifact-retention posture is adequate for this review; durable closeout preservation remains necessary before expiry.                                                               | **PASS\_WITH\_NONBLOCKING\_LIMITATION** |
| 46                   | All benchmark, model-protocol, comparative-access, owner, merge, closeout, gate, and scientific-result non-advancement flags are truthful.                                                       | **PASS**                                |
| 47                   | Exact head `4d47595ede0f1690f1d9d96075df9a3ff7ab60d9` is engineering-acceptable.                                                                                                                 | **PASS**                                |
| 48                   | The reviewer performed no implementation, correction, scientific review, owner action, merge, closeout, freeze, renderer selection, model work, tag, Release, gate action, or next work package. | **PASS**                                |

## E. New findings

```text
New engineering findings: NONE
Blocking engineering findings: NONE
```

## F. Limitations

### Engineering limitations

A direct local Git clone or detached worktree was not available through the connected private-repository interface. I therefore did not independently run the source tree’s `uv`, Ruff, Mypy, Pytest, or CLI commands in a local clone.

This did not prevent a reliable disposition because the exact source and diff were available read-only, all exact-head CI command paths and logs were available, all four current raw artifacts were accessible, and their raw bytes and contained identities were independently checked.

No local `git status --short` result is claimed. The equivalent reviewer end state is that no local repository worktree was created and no connected GitHub state was mutated.

### Local and renderer reproduction limitations

Renderer classification:

```text
Windows/WGL:
validated from independently downloaded exact-head evidence
verified from exact-head CI
not independently regenerated on reviewer-local WGL hardware

Ubuntu/OSMesa:
validated from independently downloaded exact-head evidence
verified from exact-head CI
not independently regenerated in a reviewer-local repository checkout
```

The current private repository was checked against live metadata. Public and internal repository behavior was assessed through the exact production parser/validator path using high-fidelity immutable fixtures rather than separate live public and enterprise-internal repositories.

The complete source-reseal, raw-artifact, archive-safety, and stable-handle suites were inspected in exact-head source and verified from exact-head CI. Representative access, self-reseal, raw-size/digest, extraction, and identity calculations were independently repeated; every platform-specific attack was not locally re-executed.

### Connected private-repository access limitations

An unconnected reviewer cannot access these private repository records or Actions artifacts merely because the evidence class is named `PUBLIC_REPOSITORY_ONLY`.

That limitation is now explicit and correctly modeled. It is not an unresolved defect: the authoritative evidence-class definition allows connected authenticated GitHub records, and current validation truthfully records the mechanism required.

Loss of connected access would be an access failure, not proof that the recorded artifact never existed. Current readiness validation still requires the artifact to remain live, unexpired, and accessible.

### Evidence retention and closeout requirements

All four artifacts are adequate for the present exact-head review and expire on:

```text
2026-12-02T02:12:06Z
```

Current validation rejects expired or unavailable evidence rather than silently treating it as ready.

A later separately authorised benchmark-freeze closeout should preserve durable copies of:

```text
renderer_receipt.json
publication_record.json
exact repository/artifact/run/job metadata
preferably all four exact raw ZIP archives
```

That is a nonblocking closeout dependency, not a present engineering defect. No archival or closeout action was performed.

### Nonblocking future hardening

Future tests could additionally run against dedicated live public and enterprise-internal fixture repositories rather than only high-fidelity parser fixtures. That would extend integration coverage but is not required to establish the current private-repository correction.

A durable evidence store beyond ordinary Actions retention would reduce later closeout risk. It requires separate authority and does not change this exact-head engineering disposition.

### Scientific matters outside engineering authority

I did not perform another scientific review and did not adjudicate `EPS-SR17-0001`.

The supplied coordination context records:

```text
SCIENTIFIC_PASS
EPS-SR17-0001: VERIFIED
Prior access-posture limitation: RESOLVED
New scientific findings: NONE
```

Those facts are separate authority. They do not themselves establish engineering correctness, owner approval, implementation-merge authority, benchmark freeze, model-protocol freeze, comparative-result access, gate advancement, or a scientific result.

## G. Disposition

```text
ENGINEERING_PASS
```

The fourth correction resolves the inaccurate access posture with a strict versioned record that truthfully describes the live private repository and its connected authenticated Actions access. Live repository metadata—not a caller-supplied or self-resealed record—is authoritative. Repository state is checked around raw artifact acquisition, the exact records are compared against that metadata, and private/public or visibility-change mismatches fail closed.

All previously verified raw-artifact, source-reconstruction, stable-publication, domain-separation, adversarial, and canonical-repository controls remain intact. Both renderer evidence chains independently validate, and no new engineering finding remains.

```text
Role: ENGINEERING_REVIEWER
Disposition: ENGINEERING_PASS
Reviewed PR: #17
Reviewed SHA: 4d47595ede0f1690f1d9d96075df9a3ff7ab60d9
Reviewed tree: ba112bcc4be4396648305bb1d1529cc002755ea1
Review profile: DUAL_REVIEW
Evidence class: PUBLIC_REPOSITORY_ONLY
Repository visibility: PRIVATE
Artifact access mechanism: CONNECTED_AUTHENTICATED_GITHUB_ACTIONS
Closeout boundary: BENCHMARK_OR_PREREGISTRATION_FREEZE
EPS-ER17-0001: VERIFIED
EPS-ER17-0002: VERIFIED
EPS-ER17-0003: VERIFIED
EPS-ER17-0004: VERIFIED
EPS-ER17-0005: VERIFIED
EPS-ER17-0006: VERIFIED
EPS-ER17-0007: VERIFIED
EPS-ER17-0008: VERIFIED
New engineering findings: NONE
Scientific review: SEPARATE AUTHORITY
Known exact-head scientific disposition: SCIENTIFIC_PASS
Known exact-head scientific finding status: EPS-SR17-0001 VERIFIED
Known access-posture limitation status: RESOLVED
Known new scientific findings: NONE
Scientific findings adjudicated by engineering reviewer: NO
Historical lock: PRESERVED
Replacement lock: REVIEWED, NOT A BENCHMARK FREEZE
Exact-head artifacts: REVIEWED
Publication-record schema: V2 REVIEWED
Repository-visibility binding: REVIEWED
Freeze candidate: REVIEWED BY ENGINEERING, NOT FROZEN
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
Merge: NOT PERFORMED
Closeout: NOT PERFORMED
Tag: NOT CREATED
Release: NOT PUBLISHED
Scientific result: NONE
Next work package: NOT BEGUN
Repository mutation by reviewer: NONE
```

## H. Routing statement

```text
Return this exact-head engineering disposition, renewed statuses for
EPS-ER17-0001 through EPS-ER17-0008, and every new EPS-ER17-* finding to the
ORCHESTRATOR_COORDINATOR.

Do not implement another correction.

Do not perform another scientific review.

The coordinator will consolidate this disposition with the existing exact-head
SCIENTIFIC_PASS, EPS-SR17-0001: VERIFIED, resolved scientific access-posture
limitation, and no new scientific findings.

No owner implementation approval, merge, closeout, benchmark freeze, renderer
selection, model-protocol freeze, comparative-result access, tag, Release, gate
action, model work, scientific result or next work package is authorised by
this review.
```
