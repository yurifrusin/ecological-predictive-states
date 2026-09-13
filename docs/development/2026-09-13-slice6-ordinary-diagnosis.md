# Slice 6 ordinary-development diagnosis — 13 September 2026

The first observed cross-renderer topology divergence is already present in the raw corridor segmentation labels. The eight predeclared ordinary development cells all generated and passed the existing whole-dataset validator. Both single-occluder pairs have equal portable topology; both corridor pairs retain WGL indeterminacy and unequal portable topology.

This is a development record on the unchanged reviewed implementation. It does not revise the Slice 6 method, repair its frozen `FAILED_CLOSED` qualification, select a renderer as ground truth, or advance a scientific gate. The underlying graphics-level cause has not been established.

For repository preservation, this non-executable memorandum uses review profile `SCIENTIFIC_ONLY` and evidence class `PUBLIC_REPOSITORY_ONLY`: it changes no source, test, dependency, configuration, benchmark or executable contract, and its ordinary generated evidence is reproducible from the connected repository. It remains a proposal pending review of its own exact PR/head; the earlier PR #19 dispositions do not approve this memorandum.

## Execution identity and finite membership

- Repository: `yurifrusin/ecological-predictive-states`.
- Reviewed implementation: PR [#19](https://github.com/yurifrusin/ecological-predictive-states/pull/19), commit `d7b7ce8f04426e5869ef6e063f97c421fd10fffc`, tree `8ef8f5552750e322c01538fdf7be30a1855c0a63`.
- Implementation accepted by normal merge `f8a1b8584889fb955085517a7b51a4ad9a63ade9`; independent engineering and scientific reviews passed at the unchanged implementation head. The work-package closeout is separately recorded in PR [#20](https://github.com/yurifrusin/ecological-predictive-states/pull/20).
- Exact environment pins: Python 3.11.15, MuJoCo 3.12.0, NumPy 2.4.6. Windows/WGL and Ubuntu under WSL/OSMesa both used clean checkouts at the reviewed head, confirmed again after generation.
- Configurations: unchanged `configs/benchmark_v0.yaml` and `configs/corridor_v0.yaml`; seed 1729, 160 × 120, legacy appearance `legacy_solid_base_v1`.
- Two episodes per scene per renderer: eight cells, four paired transitions. Ordinary episode seeds, stored as decimal strings to preserve their full integer value: `1703363364368450807` and `2908570908161681051`.

The WGL (`preflight-wgl.json`) and OSMesa (`preflight-osmesa.json`) ledgers were written before either renderer generated results, at 01:07:13 and 01:07:45 UTC. They preserve the full resolved configurations, Git blob identities, file digests, and explicit case membership. The root and derived episode seeds were checked against the frozen candidate roots and their episode-0 derivations; there is no overlap. Config bytes and resolved configs match across both environments.

Generation ran from 01:08:21 to 01:08:59 UTC. The WGL (`generation-wgl.json`) and OSMesa (`generation-osmesa.json`) receipts bind the ledgers, manifest file digests and logical dataset identities. No final-evaluation cases were generated, no additional appearance sweep was performed, and no source/configuration/lock was edited.

The earlier local Phase A smoke is a separate historical result from a dirty pre-lock checkout; it is not substituted for these fresh records.

## Observed results

Each component count below includes every nonempty four-connected component, including one-pixel components. Counts are before → after. Pixel differences compare matching 160 × 120 frames.

| Scene / ordinary episode | WGL components | OSMesa components | Segmentation pixels differing, before / after | Portable graph | Capability, WGL / OSMesa |
| --- | --- | --- | --- | --- | --- |
| Single occluder / 0 | 3 → 3 | 3 → 3 | 0 / 0 | Equal | Available / available |
| Single occluder / 1 | 3 → 3 | 3 → 3 | 0 / 0 | Equal | Available / available |
| Corridor / 0 | 21 → 21 | 4 → 4 | 27 / 117 | Unequal | Indeterminate / available |
| Corridor / 1 | 22 → 21 | 4 → 4 | 27 / 205 | Unequal | Indeterminate / available |

This is a finite diagnostic sample, not an estimate of a renderer-wide failure rate. The two single-occluder episodes reuse the fixed scene geometry; their different episode/surface identities do not add geometric coverage.

Scene-content identities, camera records and opaque surface-identity maps match within every renderer pair. In the corridor pairs, raw geometry-label differences and public segmentation differences occupy exactly the same pixel locations. The remapping therefore exposes the upstream difference; these observations do not identify a downstream remapping defect.

The exact pixel comparison figure is retained as `corridor-development-comparison.png` in the development evidence directory named below. It shows ordinary corridor episode 0 before/after on both renderers and the complete difference masks.

The numeric geometry labels and role names in this figure are privileged simulator diagnostics. They are not introduced into EPS public topology inputs. Each source pixel is displayed as a 2 × 2 block with nearest-neighbour scaling. Neither image is labelled as ground truth.

## Trace from segmentation to graph outcome

The array and isolate trace (`development-trace.json`) establishes elementwise equality of forward/backward fixed transport vectors, validity arrays and reason arrays, plus the before-fate and after-origin event-code arrays, across every renderer pair. Thus the corridor graph difference persists with the same public transport and event inputs.

The additional WGL corridor components have no supported incident graph edge and no valid outgoing transport pixel:

| Episode / frame | Isolated components | Their sizes | Outgoing transport reasons |
| --- | ---: | --- | --- |
| 0 / before | 17 | 17 one-pixel components | 10 boundary-ambiguous pixels; 7 out of frame |
| 0 / after | 17 | 16 one-pixel components; one 92-pixel component | All 108 pixels boundary-ambiguous |
| 1 / before | 18 | 18 one-pixel components | 11 boundary-ambiguous pixels; 7 out of frame |
| 1 / after | 17 | 16 one-pixel components; one 90-pixel component | All 106 pixels boundary-ambiguous |

WGL corridor episode 0 has 27 indeterminate events, seven component disappearances and four continuations. Episode 1 has 28 indeterminate events, seven disappearances and four continuations. Their same-surface support tables contain 306 and 323 zero-support pairs respectively; each retains four supported edges. OSMesa has four continuations and four supported edges in each corridor episode, with no zero-support pair.

The validated annotation therefore follows the declared conservative rule: retain these segmentation components, retain absent support, and report indeterminacy where no permitted lifecycle proof exists. Filtering small components or relaxing support rules is not warranted by this diagnosis. Larger thin components also occur, so the observed limitation is not confined to a one-pixel size class.

Complete per-component sizes, bounds, locations, reason coverage, directional support counts, identities and event totals are retained in development-diagnostics.json (`development-diagnostics.json`), SHA-256 `dd57d8611e8906d769a8e5d27741736b9f56f5b92b2c6f4953bb95e3d65698d5`. A smaller summary (`development-summary.json`) retains the paired comparisons and frame totals.

## Split/merge coverage and bounded feasibility

| Evidence source | What it establishes | Remaining limitation |
| --- | --- | --- |
| Existing reviewed closed-form split/merge and reversal fixtures | Declared one-to-many/many-to-one graph semantics and reversal behavior | Does not establish that the simulator scenes produce those events |
| Frozen Slice 6 matrix | Recorded renderer behavior and honest failed qualification | No rendered split, merge or complex event |
| These eight ordinary development cells | Segmentation-first divergence and downstream support consequences | No rendered split, merge or complex event |

The current single-occluder configuration exposes lateral camera positions, camera height and field of view and accepts both lateral directions. Its occluder/background dimensions remain fixed in the scene builder. These existing fields provide a candidate route for a controlled occluder-crossing case and its reverse, but no such positive rendered pair was generated or established here. The corridor supports positive forward motion only; it is not an existing reverse-motion test vehicle.

The scientific follow-up identified a concrete, non-rendered candidate that existing model validation accepts: camera lateral position −1.0 → 0.0, forward position 0.0, height 1.25, vertical field of view 90°, and `lateral_right` with delta +1.0. Its exact reverse is 0.0 → −1.0 with `lateral_left` and delta −1.0; all other settings stay fixed. Analytic projected bounds suggest that the centred occluder spans the background vertically while leaving narrow lateral strips, whereas the offset occluder reaches the viewport edge. This establishes expressibility and geometric plausibility only, not a rendered split/merge or sufficient transport support.

Record that pair and the unchanged ordinary seed/profile before any rendering. Retain a failure to produce the intended supported event. Do not search seeds or geometries until an event appears or add the new case to the old final matrix. If existing fields fail, any scene extension requires its own separate proposal.

## Proposed next work and stopping point

Stop this execution after the recorded eight cells and the read-only coverage assessment. A diagnostic of the renderer segmentation production/readback path is the most direct next development step. Sample resolution and geometry-ID encoding/decoding are candidate mechanisms, not an exhaustive or established explanation. First audit the already generated differing coordinates against the scene seams and inspect the installed MuJoCo render path; this requires no new episode generation.

Before any intervention, write a separate finite protocol using ordinary corridor episode 0 only, both fixed backends, and an explicitly recorded baseline versus one capture/sample-mode intervention. Retain raw geometry labels and actual renderer settings. Inspect the installed MuJoCo render path to choose the intervention. Do not patch or postprocess saved segmentation masks, alter topology connectivity/support rules, introduce a new primary renderer, or relabel these intervention outputs as the original locked candidate.

Any changed renderer/method candidate needs a separately prospective definition and review before confirmatory execution. Previously exposed final cases remain historical or exploratory for that candidate. If the capture diagnosis does not support a well-defined correction, the material scientific options remain an explicitly indeterminate successor state, a separately justified future renderer-comparison contract, or stopping/deferring topology-dependent progress.

Gate 0B completion, Gate 0C/0D and model work remain separate decisions. No gate change follows from the implementation merge, this diagnosis, or a future development intervention.

## Preservation and reproducibility

Full generated datasets and the executed helper scripts remain outside the repository at `C:\Users\sarashera\EPS-development\slice6-20260913-ordinary8` on DESKTOP-TPUQMNG. This memorandum records compact findings and provenance; generated JSON summaries, data and figures remain untracked as required by repository guidance. The named evidence files share that absolute development-directory path. An `evidence-index.json` in the same directory preserves their byte counts and SHA-256 digests. The existing CLI can reproduce each renderer/scene cell set with `uv run --locked epsbench component-topology-audit --config configs/benchmark_v0.yaml --episodes 2 --output <new-empty-external-directory>` and the corresponding `configs/corridor_v0.yaml` command, using the pinned renderer environment. The execution used the reviewed `generate_dataset(..., component_topology=True)` followed by `validate_dataset` for two episodes of each unchanged config on each renderer. Generation receipts and manifests bind the result; the diagnostic code only reads those outputs and writes separate summaries/figures.

This document records observed development results and a non-operative proposal. It grants no approval to change a scientific contract or acceptance criterion.
