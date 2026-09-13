# Independent engineering review — EPS PR #19

## Review identity and disposition

- **Role:** `ENGINEERING_REVIEWER`
- **Independence:** I did not implement or correct this work package. I made no source, test, fixture, lock, policy, review-record, GitHub, branch, tag, release, or merge mutation. Review scripts and extracted evidence were created only in an external temporary directory.
- **Repository:** `yurifrusin/ecological-predictive-states`
- **Pull request:** `#19`, `OPEN`, `DRAFT`, `UNMERGED`
- **Review profile:** `DUAL_REVIEW`
- **Evidence class:** `PUBLIC_REPOSITORY_ONLY` (connected access to the private repository was used)
- **Closeout boundary:** `WORK_PACKAGE`
- **Exact reviewed head:** `d7b7ce8f04426e5869ef6e063f97c421fd10fffc`
- **Exact reviewed tree:** `8ef8f5552750e322c01538fdf7be30a1855c0a63`
- **Canonical base:** `0e9e0593cc7a0f445718575152eca5fc6651edc4`
- **Canonical base tree:** `ec98e8bcc8ed62686ccc0f98a0fe8956690be698`
- **Disposition:** `ENGINEERING_PASS`

`FAILED_CLOSED` is the retained scientific qualification outcome. It is separate from this engineering disposition. This review gives no owner approval, merge authority, phase-gate effect, Gate 0C/0D authority, model authority, tag, release, or scientific-result claim.

## Exact identity and chronology verification

GitHub PR metadata, live branch refs, Git commit objects, and the local exact-head worktree agreed at both the start and end of review. The final readback showed PR head `d7b7ce8`, base `0e9e059`, merge state `CLEAN`, and an empty `git --no-optional-locks status --short` result.

The implementation range contains exactly two commits:

1. `cacdcb8ed0227c0d484d93178606b7fc01b3bf4c`, tree `b55c7c35d250128f7c74ea9fadddc3c3103571d5`, directly parented by the canonical base. It first adds the prospective lock.
2. `d7b7ce8f04426e5869ef6e063f97c421fd10fffc`, tree `8ef8f5552750e322c01538fdf7be30a1855c0a63`, directly parented by the lock commit. Its only delta is 43 added lines and zero deletions in `RESEARCH_LOG.md`.

Live run `34034710421` was created at the lock head at `2026-09-06T12:59:11Z`. The worker-reported first local final-root start is later, at `2026-09-06T13:00:10.756313+00:00`. I independently verified the remote lock-run identity and that no scientific source changed in the later commit; I did not independently recover that local timestamp from its original workstation directory.

`validate_topology_lock()` reconstructed the checked-in lock byte-for-byte with:

| Binding | Verified value |
|---|---|
| Lock commit | `cacdcb8ed0227c0d484d93178606b7fc01b3bf4c` |
| Lock tree | `b55c7c35d250128f7c74ea9fadddc3c3103571d5` |
| Lock root | `b15c431ac618709c63a5f31636d5024073967d0c3210d75e51ebd7ff925f98ee` |
| Definition root | `08839e3fd6198ebd649b5d677b0480c391cf2d7cc6fa68e75b0ec5a706d00952` |
| Scientific-source root (97 files) | `5c8c127aba9923944ae991dd90e0265f734be371b6c6ffc685a110bd1a5429be` |
| Qualification-membership root (192 cells) | `31b4673b0f40e4652146a4a89cbb3a14f610a6d24af9da8f748e468190ca746b` |

The Appearance Benchmark Input Freeze v0 lock at dependency commit `d4072f912cc58bbc1ca41ceb2652e41783dbf3e1` is an ancestor of the canonical base. Its lock bytes are identical to the reviewed working copy (file SHA-256 `39d7c8eeca551e8d339a64145060aac0613fe5556710a55a54c7b69b7d07e3de`). `git diff --check` passed. Tags and GitHub Releases were empty.

## Engineering assessment

### Component and event semantics

`src/epsbench/annotations/component_topology.py:102-525` implements the locked contract directly. Generation retains all nonempty, equal-label four-neighbour components, excludes zero, orders by opaque surface identity and row-major minimum, and derives frame-local component IDs from the declared domain. The independent path uses north/west union-find instead of generation flood fill.

Transport support at `component_topology.py:155-211` checks exact int32/uint8 shapes and canonical validity/reason/zero-vector rules. Generation uses vectorised int64 arithmetic; reconstruction uses Python integer `//`, which supplies floor toward negative infinity. Both implement the locked source-centre `+512`, scale-1024 target rule, exclude out-of-frame cells, reject cross-surface targets, retain every same-surface Cartesian pair, preserve separate directional counts, and create an edge when either count is positive.

Event construction at `component_topology.py:214-297` partitions every node by supported bipartite connectivity. Generation traversal and reconstruction set-equivalence closure are distinct. Cardinality produces continuation, split, merge, or complex. An isolated node becomes appearance/disappearance only when the surface is absent in the opposite frame or all component pixels carry public lifecycle code 1 or 2; otherwise it remains indeterminate. The capability status fails closed on any indeterminate event.

The typed records and strict version/domain literals in `src/epsbench/schema.py:575-715`, visibility-method binding at `schema.py:758-850`, and transition binding at `schema.py:999-1034` reject extra fields and inconsistent topology/transport/event/segmentation contracts. The portable graph deliberately excludes raster-local counts, masks, extents, and directional counts while retaining node, edge, event, and capability identity.

### Identity, permission, validation, and publication paths

The topology annotation is included in the visibility-event and ecological-label identity chain through `src/epsbench/data/identity.py:163-210`. Legacy generation remains opt-in compatible: default generation retains transition dev.9 and dataset dev.7/dev.8, while topology generation uses transition dev.10 and dataset dev.9/dev.10.

`src/epsbench/data/loader.py:164-295` checks the dataset/transition topology version relation. The dedicated topology loader checks `COMPONENT_TOPOLOGY` before opening the transition; topology-bearing ecological-transition and visibility-event projections do the same. The returned topology contains opaque surface IDs and public image-plane evidence, without appearance profile, role, root-registry, raw simulator identity, metric geometry, camera pose, or generation records.

`src/epsbench/data/validate.py:527-558` loads component maps through the existing owned-regular-file registry, then reconstructs every component, map, support count, edge, event, and root from retained public segmentation, transport, and event-code arrays using the independent algorithms. Dataset validation separately recomputes privileged apparatus evidence; those values are not inputs to the public topology derivation. Hard-link, traversal, alias, dtype, shape, logical-hash, and fully resealed false-claim tests are present.

Inspection validates the complete dataset before reading panels, requires output outside the dataset, refuses overwrite, and publishes through a no-clobber temporary-file/hard-link sequence at `src/epsbench/data/inspect.py:121-324`.

Qualification and packet validation at `src/epsbench/topology_qualification.py:185-563` reconstruct the prospective lock, enforce clean-head ancestry and remote readback before generation, bind exact source provenance and renderer fingerprints, retain failed cells, require exact 160 selected plus 32 control membership, compute within-renderer appearance invariance, and fail closed on any invalid/indeterminate/missing cell or cross-renderer difference. CI uploads complete packet directories, including failed outcomes, and reconstructs the published archives (`.github/workflows/ci.yml:447-489` and `620-680`).

No model, trainer, evaluator, checkpoint, Gate 0C ecological-state conversion, Gate 0D interface, dependency, licence, active review record, gate decision, tag, or release is in the 23-path implementation diff.

## Retained evidence independently examined

Authoritative CI run `34094328833` is a completed successful `pull_request` run at the exact reviewed head. Its three jobs are successful: quality `101654410171`, WGL `101654410416`, and OSMesa `101670353371`. Direct log readback confirmed 672 Linux tests passed, Ruff passed, 154 files were formatted, mypy found no issues in 72 source files, 52 Windows publication-safety tests passed, and the published packet roots were the values below.

I downloaded all four topology artifacts directly from GitHub and compared their bytes to live artifact metadata:

| Artifact | Bytes | Raw archive SHA-256 |
|---:|---:|---|
| WGL packet `10009780955` | 19,986,384 | `b5ccb60759a055582792ad1475f8ebbfec02d49ba80aee88477e49310dc867b6` |
| WGL publication `10009866027` | 6,523 | `904f12b5dcbfc32d402a10695ce81b814ee2c04bb54b64460446748f158b112e` |
| OSMesa packet `10019051886` | 19,461,675 | `ee68e403770eea22482840c62930dc7b392459cb43e0c2699959e16d4babddc2` |
| OSMesa publication `10019437688` | 8,803 | `fbd58fb4e21d54afaea5702bb2f6e35da0668797638cb43cd330c96813f2bf39` |

All were unexpired and bound by live metadata to run `34094328833`, the exact branch, and head `d7b7ce8`. Safe extraction yielded exactly 5,955 regular members in each packet. Full source reconstruction was run for each packet and again through the repository cross-renderer comparison; both repetitions passed.

| Result | WGL | OSMesa |
|---|---:|---:|
| Validated cells | 192 | 192 |
| Selected / controls | 160 / 32 | 160 / 32 |
| Single-occluder / corridor | 96 / 96 | 96 / 96 |
| Available / indeterminate | 96 / 96 | 192 / 0 |
| Appearance invariance | true | true |
| Packet root | `1798f2a8e7e13d427b8291694189d414d810c2e21704cf63bd12e9298e5d4116` | `04062f799fb216cf3e1cbfcf3dec766751babd9ba90555a5a31236b17d3066c4` |
| Local qualification | `failed_closed` | `qualified` |

The independent comparison covered 192 cells, found exactly 96 differences, all corridor cells across all six frozen profiles and 16 roots, and reproduced `portable_equality=false` and `qualification=failed_closed`. No actual split, merge, or complex event occurs in the frozen scene matrix. Closed-form tests exercise those branches; the absence in final-root evidence remains visible and is not relabelled as capability success.

The published receipt file SHA-256 values matched the PR body: WGL `11cf76415ffdb7a603a6475e733cc711c8eb785df9f310e35beec50315d8cd9f`, OSMesa `7c824b21e8473eca2ed71f4a46cec951e610720cb33939688a3074a148ea6d73`, cross-renderer `0cd9ebe6c37ca8758fa6c7443424ed2ed05c214e2c1340119b2ea2d2e563f984`.

The bounded local review run passed **102 tests in 8.19 seconds**: all new component-topology, qualification, and publication unit tests, plus complete single-occluder/corridor scene reconstruction and the three early permission-gate projections. I did not repeat the full renderer campaign.

## Findings

### Blocking findings

None.

### Optional hardening

**EPS-ER19-0001 — Add an explicit post-completion CI identity check.** `scripts/check_topology_publication.py:49-83` binds artifact name/digest/size/expiry, repository, run, head, and a unique renderer job by name/run/head. It does not bind the final run/job status or conclusion because the receipt is generated while that renderer job is still running. Current evidence is unaffected: I separately re-fetched the completed run and all three final successful job conclusions. A future, non-scientific hardening change could add a read-only post-run/closeout verifier that re-fetches the run and named jobs and requires `completed/success`. This does not require or justify changing the locked topology implementation, tests, fixtures, thresholds, or final-root evidence.

## Coverage limits

- Full packet replay used the repository's independent validator. I inspected the distinct flood/union-find, vectorised/Python-integer, and traversal/set-closure paths and ran their closed-form tests; I did not write a third topology implementation.
- I did not re-render final roots, repeat the original appearance-freeze campaign, or reproduce temporary Windows runner installation/registration.
- I verified the remote prospective-lock run and byte-identical scientific source after the lock. The first local final-root timestamp remains a worker-reported historical fact, rather than evidence I recovered from the original local directory.
- I reviewed the governing documents, full PR body, complete changed source/test surface, CI workflow, and retained topology publications. Unchanged historical implementation outside the dependency paths exercised by validation was not reviewed line-by-line.

## Evidence paths for closeout handoff

The byte-identical downloaded archives, live metadata, safe extractions, and temporary verification scripts are available on `DESKTOP-TPUQMNG` under:

`C:\Users\sarashera\AppData\Local\Temp\eps-pr19-eng-evidence`

Relevant entries are `artifact-10009780955.zip/.json`, `artifact-10009866027.zip/.json`, `artifact-10019051886.zip/.json`, `artifact-10019437688.zip/.json`, `wgl-packet`, `osmesa-packet`, `wgl-publication`, and `osmesa-publication`.

## Final disposition

For exact review unit `(yurifrusin/ecological-predictive-states, PR #19, d7b7ce8f04426e5869ef6e063f97c421fd10fffc)`: **`ENGINEERING_PASS`**.

Scientific qualification remains **`FAILED_CLOSED`**. Phase-gate effect: **NONE**.
