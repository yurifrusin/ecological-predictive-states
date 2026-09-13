# Independent scientific review — EPS PR #19

**Review request ID:** `eps-g0b-s6-pr19-scientific-renewal-20260913-d7b7ce8f`  
**Role:** `SCIENTIFIC_REVIEWER`  
**Review profile:** `DUAL_REVIEW`  
**Evidence class:** `PUBLIC_REPOSITORY_ONLY` (authenticated access to the connected private GitHub repository)  
**Closeout boundary:** `WORK_PACKAGE`  
**Review date:** 2026-09-13

## Independence and exact review unit

I acted only as the scientific reviewer. I did not implement or correct this work package, make an active review commit, alter source/tests/locks/fixtures/policy, rerun the final rendering campaign, mutate GitHub, choose a renderer, decide a gate, perform model work, or publish a scientific result. This is a new review based on evidence I examined in this review session.

I verified the live review unit as:

- repository: `yurifrusin/ecological-predictive-states`;
- PR: `#19`, open, draft, base `main`, head branch `codex/gate-0b-slice-6-component-topology`;
- reviewed head: `d7b7ce8f04426e5869ef6e063f97c421fd10fffc`;
- reviewed tree: `8ef8f5552750e322c01538fdf7be30a1855c0a63`;
- canonical base: `0e9e0593cc7a0f445718575152eca5fc6651edc4`;
- canonical base tree: `ec98e8bcc8ed62686ccc0f98a0fe8956690be698`;
- prospective first-add lock commit: `cacdcb8ed0227c0d484d93178606b7fc01b3bf4c`;
- prospective lock tree: `b55c7c35d250128f7c74ea9fadddc3c3103571d5`.

The working tree was clean. The reviewed head differs from the lock commit only by 43 added lines in `RESEARCH_LOG.md`; no scientific source, test, fixture, configuration, dependency lock, qualification rule, or renderer rule changed after the lock commit.

## Evidence examined

I read `AGENTS.md`, `docs/review-protocol.md`, `docs/RESEARCH_CHARTER.md`, `docs/EPS_BENCH_V0.md`, `docs/MILESTONE_0.md`, `docs/GATE_0B_SLICE_6_COMPONENT_TOPOLOGY.md`, the relevant portions of `docs/IMPLEMENTATION_NOTES.md`, the complete live PR body, and the full changed-path list and commit history.

I examined the public topology derivation, schema, identity chain, loader permission checks, dataset reconstruction, qualification and publication code, plus the closed-form, scene-integration, qualification, and publication tests. In particular, I checked `component_topology.py`, `generate.py`, `loader.py`, `validate.py`, `schema.py`, `topology_qualification.py`, and their relevant tests.

I independently recomputed the prospective lock and preflight. The results were:

- topology lock root `b15c431ac618709c63a5f31636d5024073967d0c3210d75e51ebd7ff925f98ee`;
- definition root `08839e3fd6198ebd649b5d677b0480c391cf2d7cc6fa68e75b0ec5a706d00952`;
- scientific-source root `5c8c127aba9923944ae991dd90e0265f734be371b6c6ffc685a110bd1a5429be`;
- qualification-membership root `31b4673b0f40e4652146a4a89cbb3a14f610a6d24af9da8f748e468190ca746b`;
- exact membership: 160 selected cells plus 32 legacy controls per renderer, spanning both frozen scene families, all sixteen roots, and the six frozen appearance/control rows.

Live GitHub metadata bound CI run `34094328833` to the reviewed head and PR. The run and its `quality`, `qualify-wgl`, and `qualify-osmesa` jobs all completed successfully. I read the quality evidence showing 672 passing Linux tests and the final WGL generation/publication lines. I also ran the bounded non-final topology and qualification unit suites locally: 83 tests passed.

I independently downloaded the four topology archives using the repository's authenticated evidence fetcher, recomputed their SHA-256 digests against live metadata, safely extracted them, and ran complete repository reconstruction of both 5,955-member packet archives plus the cross-renderer comparison:

| Artifact | Raw archive SHA-256 |
|---|---|
| WGL packet `10009780955` | `b5ccb60759a055582792ad1475f8ebbfec02d49ba80aee88477e49310dc867b6` |
| WGL publication `10009866027` | `904f12b5dcbfc32d402a10695ce81b814ee2c04bb54b64460446748f158b112e` |
| OSMesa packet `10019051886` | `ee68e403770eea22482840c62930dc7b392459cb43e0c2699959e16d4babddc2` |
| OSMesa publication `10019437688` | `fbd58fb4e21d54afaea5702bb2f6e35da0668797638cb43cd330c96813f2bf39` |

Both packets independently reconstructed every source cell. Their packet roots were `1798f2a8e7e13d427b8291694189d414d810c2e21704cf63bd12e9298e5d4116` (WGL) and `04062f799fb216cf3e1cbfcf3dec766751babd9ba90555a5a31236b17d3066c4` (OSMesa). The publication receipts and cross-renderer receipt matched their retained byte identities.

## Scientific assessment

### Construct and input provenance

The implementation measures an explicitly narrow construct: maximal four-connected regions of equal nonzero opaque segmentation label in the image plane, related across one transition by exact retained public analytic transport. It retains one-pixel components and uses no size threshold, appearance similarity, metric extent, or floating-point matching tolerance. Component IDs are frame-local surface/rank identities and are not represented as semantic or persistent object identities. This is consistent with the charter's surface-first oracle apparatus when interpreted at that narrow raster level.

The topology function accepts opaque segmentation, public analytic-transport arrays, public pixel-event codes, and opaque surface declarations. It does not accept RGB similarity, depth, camera/world pose, geometry, raw simulator identifiers, semantic names, appearance roles, generation records, or final-root metadata. Raw renderer IDs are remapped to per-episode opaque surface identities before the topology derivation, and the privileged raw-to-opaque map is not exposed by the dedicated ecological loader. Strict schemas reject added privileged fields. Typed `COMPONENT_TOPOLOGY` permission is required before opening the transition for the dedicated loader and for parent ecological projections.

The shared opaque surface identity across the two frames is oracle correspondence information. It is scientifically permitted for this oracle data-contract stage, but it does not demonstrate that correspondence can be inferred from RGB. Any later comparison that consumes this oracle information must preserve the charter's matched-interface and oracle-token ablations; this PR supplies no evidence about learned extraction or model efficacy.

### Prospective definition and non-adaptation

The first-add lock commit is directly parented by the authorised base, is the unique first addition of the lock, and remains an ancestor of the reviewed head. The current source, definition, and exact frozen membership independently rehash to the prospective lock. Current retained packet metadata records `scientific_adaptation_permitted=false` and begins after publication of the lock. Because the only post-lock commit changes `RESEARCH_LOG.md`, the observed final evidence could not have changed scientific source, thresholds, event rules, scenes, membership, or renderer posture at the reviewed head.

I did not need the worker's claimed earlier local execution timestamp to reach this conclusion; the current exact-head packet chronology, immutable lock bytes, and post-lock source history are sufficient for the evidence reviewed here.

### Qualification result and renderer divergence

The apparatus outcome is correctly `FAILED_CLOSED`:

| Renderer | Validated | Available | Indeterminate | Within-renderer appearance invariance | Local qualification |
|---|---:|---:|---:|---|---|
| Windows/WGL | 192 | 96 | 96 | true | `failed_closed` |
| Linux/OSMesa | 192 | 192 | 0 | true | `qualified` |

All 96 WGL corridor cells are indeterminate; all WGL single-occluder cells and all OSMesa cells are available. The cross-renderer comparison covers all 192 cells and finds exactly 96 differences, all in the corridor, so portable equality is false and overall qualification is `failed_closed`.

The divergence is material evidence about the construct rather than an ignorable execution detail. OSMesa corridor cells contain exactly four components in each frame. WGL corridor cells contain 17–28 components per frame, retain 34,962 zero-support same-surface pairs, 768 component disappearances, and 2,808 indeterminate events. These outcomes repeat across every frozen appearance/control row within each renderer. The evidence therefore supports renderer-sensitive raster fragmentation and transport withholding in this matrix; it does not establish which renderer is scientifically preferable, that one backend is erroneous, or that the topology is portable.

The precommitted rule required all cells, both renderers, no excluded failures, no accepted-backend substitution, and no relabelling of indeterminate components. The implementation retained the negative evidence and withheld qualification. A successful CI job here means that a complete failed-closed packet was generated and validated. It is not a successful topology qualification.

### Split/merge coverage and interpretation limits

The closed-form tests exercise one-to-many split, many-to-one merge, reversal duality, complex many-to-many structure, lifecycle isolates, exact target-cell boundaries, zero support, and one-direction support. Independent generation and reconstruction algorithms agree on those fixtures, and fully resealed false classifications are rejected.

The frozen renderer matrix contains zero split, zero merge, and zero complex event in both renderers. It therefore tests exact topology stability, appearance invariance, renderer portability, and uncertainty withholding on the selected scenes, but it does not demonstrate an integrated rendered split or merge transition. This is a disclosed scope limit, not a hidden positive claim. A future rendered split/merge exercise would be development coverage for a successor work package and could not replace or repair the present frozen result.

Four-connectivity, retention of one-pixel components, an edge from a single supported pixel in either direction, and exclusion of boundary/occlusion/exit samples make this exact raster oracle intentionally sensitive. The current frozen evidence contains no one-direction-only pair, so it validates that rule only in closed form. The result cannot be generalized to semantic objects, persistent identity, arbitrary geometry, additional scene families, different resolution, different renderers or dependency versions, continuous boundary components, dynamics, learned inference, or downstream task sufficiency.

### Fairness and preservation of negative evidence

The comparison uses the same prospectively frozen membership, profiles, roots, scene families, definition, and versioned renderer policies. No failed or difficult cell was removed. Within-renderer appearance invariance is separated from cross-renderer equality, preventing renderer-specific raster identity from being mislabeled as an appearance effect.

The negative outcome must remain attached to this exact method and matrix. Any future decision to accept one renderer, alter connectivity or support, filter small components, treat indeterminacy as a consumable state, or change portable comparison semantics requires a new prospective scientific contract. The already exposed final cells would then be historical or exploratory evidence for that revision, not untouched confirmation.

## Findings and disposition

No blocking scientific finding was identified, so no `EPS-SR19-0001` finding is assigned. The absence of rendered split/merge events and the renderer sensitivity are material limitations recorded above; they do not invalidate the honest fail-closed apparatus or the retained negative result.

**Disposition: `SCIENTIFIC_PASS`**

This pass means that the reviewed PR defines and applies a scientifically coherent, leakage-bounded topology apparatus and preserves its prospectively defined negative outcome. It does not convert `FAILED_CLOSED` into qualification and does not approve a renderer, Gate 0B completion, Gate 0C, Gate 0D, a model protocol, model work, comparative-result access, or any empirical claim.

**Phase-gate effect: NONE**

## Non-blocking continuation recommendation

A separate bounded development investigation may compare the two ordinary non-final episodes for each scene and locked renderer, identify the first divergence stage, and assess whether a controlled rendered split/reverse-merge case is expressible. It should remain disjoint from final membership, stop after the finite case list, retain failures, and leave this lock, implementation, and `FAILED_CLOSED` outcome unchanged.

The smallest substantive owner decision after that diagnosis is whether portable component-topology qualification is a prerequisite for entering Gate 0C, or whether a new successor state contract may explicitly represent withheld/indeterminate topology. The actual Milestone 0 Gate 0B exit criteria remain deterministic regeneration, appearance-preserving ecological labels, segmentation/depth alignment, analytic visibility-event unit validity, and permission-gated metric modalities. Additional renderer environments, broader scene geometry, and new appearance candidates listed in `IMPLEMENTATION_NOTES.md` are possible extensions; they are not themselves stated Milestone 0 exit criteria. This recommendation grants no advancement.

**Signed by role:** `SCIENTIFIC_REVIEWER` (role attestation only; no cryptographic signature)
