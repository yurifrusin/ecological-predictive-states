# Research log

This log records rationale, not authority; it does not replace the charter, benchmark specification, or milestone plan.

## 2026-08-26 — Ontology before scale

The first implementation fixes data boundaries and logical identities before adding scene breadth or learning. This makes it possible to falsify the representation proposal rather than letting an underspecified state, leakage, or training system absorb errors. A tiny deterministic slice is enough to test whether surfaces, boundaries, correspondence, and visibility events can be represented coherently.

## 2026-08-26 — Oracle annotations before learned extraction

Oracle annotations separate state sufficiency from perception. If an explicitly derived ecological state cannot support later controlled tasks, a learned RGB extractor cannot repair the ontology cleanly. Conversely, a later extractor should be compared with a measurable oracle ceiling rather than hiding all errors inside an opaque score.

## 2026-08-26 — Metric instrumentation versus perceptual state

MuJoCo coordinates, camera transforms, raw geom identifiers, and depth are useful measurement instruments. Their usefulness does not authorise them as ecological learner state. The schema and loader therefore keep metric baseline data and instrumentation available for generation/evaluation while denying them to ecological-only consumers.

## 2026-08-26 — Appearance invariance

The primary hypothesis concerns robustness under appearance change, so the data contract must establish the target invariant before model work. Re-rendering a fixed geometry and action under another appearance must change RGB identity while preserving the ecological-label identity. The current colour-variant test is a minimal contract check, not evidence for model robustness.

## 2026-08-26 — Deferring affordances and language

Affordances require body-scaled action constraints and are explicitly later work. Language would add semantic targets and confounds before the optical substrate is tested. Both are deferred so the first benchmark isolates surface persistence, occlusion, and visibility transformation without semantic object recognition, language supervision, or a robot assumption.

## 2026-08-26 — Minimal second scene family

The smallest adequate second scene family is a four-surface closed corridor with prescribed forward observer motion; openings and navigation remain deferred. The implementation uses an open top only because validated wall height keeps the full optical field on the four controlled surfaces. This adds a structurally different action/scene pair without inventing an occlusion oracle outside the single-occluder counterfactual contract.

## 2026-08-26 — Analytic image-plane transport

Gate 0B uses geometry-derived analytic image-plane transport rather than RGB matching, learned flow, or renderer-defined motion. Metric geometry is used only to generate and validate an ecological oracle expressed in image coordinates. The transport of an already visible static surface point remains distinct from oriented boundary ownership, accretion, deletion, and disocclusion.

## 2026-08-27 — Independent review workflow v2

Completed EPS review cycles showed that committing an active review record can change and therefore invalidate the exact implementation head the record describes. Broad coordination context is useful for tracking that convergence, but it must not collapse independent engineering, scientific, owner, closeout, or empirical-gate authority. Future hidden or licensed evidence also requires an exact review path whose public receipt cannot reconstruct private items or expected outputs.

EPS therefore separates active external review evidence from immutable records added during linked closeout, declares both a review profile and evidence class for each work package, and distinguishes work-package closeout, engineering-milestone closeout, benchmark or preregistration freeze, and empirical decision gates. Machine-readable review-state automation is deferred until the amended human process completes a full work package without governance correction. These refinements incorporate cross-project learning developed while Unfrozen Schemas adopted the EPS review model, but that memorandum is not authoritative for EPS; EPS adopts the changes prospectively under its own owner authority.

Moving active reviews outside implementation PRs solves exact-head circularity, but pass-only convergence would leave terminal negative outcomes without a canonical repository record. A rejected, inconclusive, blocked, owner-rejected, or abandoned implementation also needs preservation even though it cannot reach merge convergence. The record-only closeout path therefore archives exact-head review and owner-decision evidence through a separate documentation PR from canonical `main`, without merging or repairing the terminal implementation. This is governance and falsification-hygiene correction, not an empirical result.

## 2026-08-27 — Image-lattice boundary ownership

Oriented boundary ownership is defined on the image edge lattice and distinguishes occluding contours, attached junctions, controlled silhouettes, and multi-surface ambiguity. Metric geometry is privileged apparatus evidence; public ownership is expressed only through opaque surfaces and image-relative side.

The proposed Slice 4 representation is implemented pending independent review. This rationale does not make the work canonical or establish a scientific result.

## 2026-08-27 — Transport-causal visibility events

Accretion and deletion are derived only when analytic transport establishes occlusion/disocclusion and oriented boundary ownership identifies a supported owner/affected pair. Frame entry/exit and neutral mask change remain separate.

Component split/merge remains explicitly unavailable in this slice. Implementation and CI are infrastructure evidence only; Gate 0B completion and Gate 0C authorisation are not implied.

## 2026-08-28 — Procedural appearance candidates, not a freeze

Gate 0B appearance candidates use repository-generated procedural textures, palettes, and lighting rather than external assets while the licence question remains unresolved.

Freeze-eligible candidate profiles randomise style-slot assignment independently of geometry and opaque surface remapping so a semantic surface role is not permanently tied to one colour or texture slot.

Slice 5 produces a candidate registry, candidate seed pool, and reproducible admission packet. Final profile selection, split assignment, and evaluation-seed freeze remain separate owner-authorised benchmark-freeze work.

These implementation decisions do not make any candidate final, complete Gate 0B, authorise Gate 0C or Gate 0D, or establish a scientific result.

A pre-correction local Windows/WGL diagnostic retained 86 rejected cells and 74 admitted cells, caused only by the prospectively fixed controlled-surface exposure and textured-surface luminance-variation thresholds. Visual inspection then exposed that the renderer's built-in headlight had not been explicitly disabled, contrary to the declared lighting contract. That packet is superseded and is not qualifying local evidence. The exact profile definitions remain in `appearance_candidate_registry_v0`; no failed profile or seed was removed, relabelled, or tuned after viewing the outputs. The corrected renderer contract explicitly disables ambient, specular, and built-in headlight contributions and requires a fresh complete audit. This is negative apparatus and implementation evidence for later independent review, not a failed EPS hypothesis and not authority to freeze any profile.

Initial exact-head Ubuntu/OSMesa CI at `83a8f8d74ee21b8d13376e8c65e7c4db32b064b4` exposed that the nominally backend-independent ecological root improperly included raw compiled camera, geometry, and remapping hashes. The registry, seed-registry, and procedural roots matched Windows/WGL exactly; only that over-broad root differed. The correction narrows the cross-platform root to established public identities and each cell's structural-invariance result while retaining the raw hashes and their comparisons as private cell-level evidence. No candidate definition, seed, threshold, or result was removed or tuned.

## 2026-08-28 — Renderer-local raster identity versus portable appearance invariance

Scientific review of exact head `e6449cdfcf8a64e5ffd8339a325f21d48a893122` correctly
identified that the Slice 5 v0 root still conflated a complete renderer-derived ecological-label
identity with the portable outcome of an appearance intervention. Locked WGL and OSMesa preserve
each candidate's complete label relative to its matched control within that renderer, while the
canonical corridor raster labels differ slightly across renderers as already documented by Slice 4.
The red exact-head CI run `33130502434` is retained as valid negative evidence about the over-broad
v0 root domain, not as evidence that appearance altered ecological structure.

Audit v1 therefore reports raw complete ecological-label identities through a renderer-local root,
compares only established analytic/ecological identities across renderers, and gives the boolean
within-renderer invariance outcomes their own portable root. No expected raw label is dispatched by
renderer, and the canonical Slice 4 ecological-label definition is unchanged.

The same review found that ordinary validation depended on the ambient repository candidate-seed
registry even though style assignment used its index. Each dataset now snapshots that protected
registry and binds its identity and schedule source into the manifest, dataset logical identity, and
appearance-instance identity. Validation is self-contained. This does not promote candidate seeds to
final evaluation seeds: the unchanged result remains 44 admitted cells, 116 rejected cells, and no
profile admitted across every scene/seed cell. No freeze or scientific result follows.

## 2026-08-29 — Appearance Candidate Revision 1 prospective lock

The canonical Slice 5 profiles, seeds, thresholds, and 44/116 result remain immutable. Revision 1
adds new profile IDs rather than editing failed candidates in place.

Revision 1 uses the original eight seeds for diagnosis and a separately derived, prospectively
locked eight-seed qualification set. Neither set is a final evaluation-seed freeze.

Revision 1 profile definitions are committed before qualification rendering. Qualification
failures are retained and do not trigger within-work-package parameter tuning. The definition lock
chooses explicit ambient fill, moderate-luminance palettes, and one-cycle/four-cycle texture partners
from the baseline surface/normal and minification diagnosis. These are candidate interventions, not
a benchmark freeze or scientific result.

The immutable definition-lock commit `914550ce4e3a819dcbcd0bd5390e3c6034af5bf6` was pushed
before any Revision 1 candidate rendering. The complete locked Windows/WGL run subsequently
generated and validated 224 candidate cells and 32 controls. Design admitted 99/112 and untouched
qualification admitted 100/112. Balanced reference, colour shift, both checker frequencies, and
combined stress passed every local cell; stripe and illumination profiles retained their failures.
No definition, seed, threshold, or admission rule changed after qualification began. Exact-head
Ubuntu/OSMesa confirmation, dual review, and owner approval remain pending; no freeze or scientific
result follows.

## 2026-08-29 — Revision 1 portable profile disposition versus renderer-local cells

Scientific review of exact head `679d85fb40288715482f604d1585cdd6a045e0db` found that the
Revision 1 partition-outcome roots included per-cell RGB, exposure, and rendered-texture admission
outcomes but were asserted as portable. The red exact-head CI run `33247224228` is retained as
evidence against that root ontology. Windows/WGL admitted 99/112 design cells; Ubuntu/OSMesa
admitted 97/112. Both admitted 100/112 qualification cells, and both independently admitted the
same five profiles. The two additional OSMesa failures remain negative cell evidence in the
already-rejected stripe profile, not a portable membership disagreement.

Root-domain v1 therefore binds partition membership without renderer evidence, binds only
whole-profile dispositions for cross-renderer admission comparison, and explicitly labels the
unchanged per-cell outcome hashes as renderer-local. The portable profile root omits per-cell counts
and reasons; if a future renderer changes a whole-profile disposition, the disagreement is retained
and the cross-renderer admitted set becomes the intersection.

The same review found that the portable baseline-analysis root alone did not bind the complete
prospective diagnosis. Validation now reconstructs both analysis roots, checks the three child-root
receipts and every lock-facing baseline field, and requires the exact analysis snapshot from the
immutable definition-lock commit. Neither committed prospective record changes. This evidence
correction changes no profile, seed, control, threshold, renderer, admission rule, or experimental
outcome. The agreed admitted set supports only a later freeze proposal. Full Gate 0B, Gate 0C,
Gate 0D, models, benchmark freeze, and scientific results remain unauthorised.

## 2026-08-31 — Appearance Benchmark Input Freeze v0 candidate

The five cross-renderer admitted Revision 1 profiles are proposed as one development reference and
four held-out Appearance-OOD profiles. The rejected stripe-only and illumination-only candidates
remain excluded negative evidence and explicit coverage limitations; they are not silently removed
or evidence against the EPS hypothesis.

Sixteen new public evaluation-only episode roots were deterministically derived and prospectively
locked before any selected profile was rendered on them. They are distinct from apparatus-design,
candidate-qualification, training, validation, and model-randomness roots. If any selected profile
fails on either locked renderer, the entire sixteen-root registry is retired from future final
evaluation use for a revised profile set.

The pushed immutable lock commit is `1a5929307dfcba1d726c650f5e1ce68771f66801`. The subsequent
complete Windows/WGL audit admitted all 160 selected cells, with every selected profile passing
32/32 cells; all 32 legacy controls were retained. Exact-head Ubuntu/OSMesa CI independently
repeated the same 160/160 selected and 16/16 admitted/rejected control counts. Every profile passed
32/32 cells in both renderers, all portable apparatus roots matched, and the cross-renderer
readiness root is `da9aad546bf6223d1802cf57990779f952d410340e7bd37d1b07c83c081e65d5`.
The candidate is therefore `ready_for_dual_review`, with the seed registry
`eligible_for_owner_freeze_if_approved`.

Successful apparatus qualification creates only a freeze candidate. A benchmark freeze still
requires exact-head engineering and scientific passes, owner approval, and separately authorised
benchmark-freeze closeout. No model protocol, gate, or scientific result follows.

## 2026-09-01 — PR #17 bounded correction and replacement lock

The reviewed head `8b34b78d5488af7103119697a286cfb8757cc125` received immutable
`SCIENTIFIC_REQUEST_CHANGES` and `ENGINEERING_REQUEST_CHANGES`. The owner accepted
`EPS-SR17-0001` and `EPS-ER17-0001` through `EPS-ER17-0004` for one bounded correction wave and
specifically authorised a prospective replacement lock because the prior logical-root domains were
not unique and versioned.

The earlier lock commit `1a5929307dfcba1d726c650f5e1ce68771f66801` and root
`a373a4742a6b5a3057b820b7d925c2e19fbddc0a83cc45384b8b81ea4e8b1464` remain immutable
historical evidence. The corrected v1 replacement-lock root is
`28acd2c340b3ef1e2b8b2b7b31ca69356cd9baa35b94ff8a598e020099883435`. It retains the exact
profiles, roles, episode roots, scenes, comparisons, thresholds, renderers, exclusions, and failure
rules while adding the prospective model-result renderer-selection dependency, unique domain
envelopes, retained independently reconstructible cell-source evidence, and strict source-bound
renderer receipts with exclusive no-replace publication.

Corrected qualification remains prohibited until the replacement-lock commit has been pushed and
read back remotely. The obsolete v0 WGL receipt is removed from the corrected candidate; its
historical commit remains in Git. No benchmark freeze, model-result access, model work, gate
advancement, approval, merge, closeout, or scientific result follows from this correction stage.

## 2026-09-01 — Corrected WGL execution after replacement-lock readback

Replacement-lock commit `d4072f912cc58bbc1ca41ceb2652e41783dbf3e1` was pushed and read back
from the branch, fetched remote reference, and GitHub commit API before corrected final-root
execution. The first WGL execution then failed closed because the final seed registry's canonical
dataset-artifact identity had been conflated with its new freeze-domain identity. No packet or
receipt was published. This failed execution is retained as engineering evidence. Source-only
commit `f7b1e61b07a86ff021a73d6a4236fcfe0ea295d2` separated the two identities and added a real
retained-dataset regression; the replacement lock and all locked scientific content remained
unchanged.

The complete clean-source WGL rerun published packet root
`630d23cf791983977bdd9e92ba33ccedb6a18d1faf819c5d2df139c83dac9310`: all 160 selected cells
were admitted and all 32 legacy controls were retained as 16 admitted and 16 rejected. Independent
validation and all complete-packet adversarial regressions succeeded. The strict receipt root is
`28036d50634b9521b62fa9ceb00d24fe32c3b94d7bb2182d984f2df49c8c76ac`; it binds clean source
commit `f7b1e61b07a86ff021a73d6a4236fcfe0ea295d2`, Git tree
`bbc29863fd4efef24c3d74cc8a99caff15ada504`, 6,550 packet artifacts, complete outcome maps,
portable and renderer-local roots, and threshold-margin root
`eb23b2e2724f1603293e06a25d43344c87d5f954219c66ccdb8c5e5dd292c097`.

This is one-renderer apparatus evidence only. Status remains `qualification_incomplete` pending
exact-head Ubuntu/OSMesa and cross-renderer reconstruction. The benchmark is not frozen; no finding
is self-verified; and no review, owner, merge, closeout, gate, model, or scientific-result authority
follows.

## 2026-09-01 — PR #17 second correction authority and current source state

Historical exact head `62e09a920c10d50c62643543f6ebcbd26570b1d4` subsequently completed
two-renderer execution and received renewed scientific pass plus engineering request changes. Those
results, reviews, and owner dispositions remain exact-head historical evidence only. The owner
authorised a bounded second correction for `EPS-ER17-0001`, `EPS-ER17-0002`, `EPS-ER17-0003`,
`EPS-ER17-0005`, and `EPS-ER17-0006`; the verified domain-separation correction
`EPS-ER17-0004` and the complete replacement-lock boundary must remain unchanged.

The source correction requires a counterpart receipt to resolve a complete immutable packet plus a
live-verifiable CI publication record before any cross-renderer readiness can be constructed. It
reconstructs the nine compatibility hashes from retained source datasets, makes typed reconstructed
identities authoritative for admission/pairing/roots, and uses directory-handle-relative Windows
and Linux publication primitives. Fresh exact-head WGL and OSMesa packet artifacts are generated
only after the source commit exists. Their run, job, artifact ID, digest, source commit/tree,
availability, and 90-day retention record therefore cannot be embedded in the commit they evaluate;
they remain external exact-head CI evidence for independent review.

Exact head `2dec901f1465bff59f1a2507d3ed0a87235fca8b` completed the local WGL matrix and
corrected adversarial corpus, then Actions run `33522600616` failed before qualification and
published no packet or receipt. The hosted Windows runner had no usable WGL driver; Linux mypy also
identified platform-specific `ctypes` attributes that needed explicit portability handling. The
next source candidate uses a single-job ephemeral self-hosted Windows/WGL label for the locked WGL
apparatus and retains hosted Ubuntu/OSMesa CI. Because the source head changes, the successful local
execution at `2dec901f` is historical diagnostic evidence only and must be repeated.

Exact head `b63121cdc7ef11340e0275498fda0f0b32213637` then completed the full WGL matrix
and packet adversarial corpus in Actions run `33524945210`, publishing immutable packet artifact
`9809315485`. The first publication-record attempt rejected the live GitHub timestamps because the
validator incorrectly anchored the configured 90-day retention to the later artifact-upload time
rather than the workflow-run creation time. PowerShell continued to the receipt command and exposed
only that final exit status, so evidence artifact `9809732826` contained a receipt but no publication
record. Packet-bound OSMesa processing rejected the downloaded WGL evidence during independent
Linux recomputation before generating any OSMesa cell or artifact. This is retained failed evidence,
not qualification evidence. The source correction binds retention to the live run creation time,
keeps the artifact creation and expiry times exact, separates publication and receipt into distinct
fail-fast CI steps, and adds strict difference-path diagnostics for cross-platform reconstruction.
Changing the source head invalidates the successful WGL execution for qualification purposes.

Exact head `98d6b33e5f39e5ac1ef4a115b5fef364a7737937` subsequently produced a fresh local
192-cell WGL packet, passed standalone validation and every complete-packet adversarial case, and
reported packet root `ab840376b6d1af2b904c5c8ce26ecc6656b96e2f27f2b44dd4374826fe1cfabb`.
Exact-head Actions run `33536577509` nevertheless failed in its quality job before renderer
qualification because portable normalization had been applied to the shared historical Slice 5
audit and therefore no longer reproduced that immutable negative-evidence packet. The concurrent
WGL job was cancelled and no packet artifact was published. This failed run is retained. The
follow-up source correction restores the shared historical audit byte semantics and applies
portable float and signed-frequency normalization only to copied PR #17 freeze-layer evidence.
Changing the source head again invalidates the local WGL result for qualification purposes.

### Current PR #17 status

| Boundary | Current source-record status |
| --- | --- |
| Apparatus status | `EXACT_HEAD_REQUALIFICATION_PENDING` |
| Post-commit exact-head CI evidence | `EXTERNAL_TO_SOURCE_COMMIT` |
| Implementation status | `SECOND_CORRECTION_SOURCE_CANDIDATE_PENDING_INDEPENDENT_REVIEW` |
| Review status | `RENEWED_EXACT_HEAD_DUAL_REVIEW_REQUIRED` |
| Owner status | `SECOND_CORRECTION_AUTHORISED; IMPLEMENTATION_MERGE_APPROVAL_NOT_GIVEN` |
| Benchmark-freeze status | `NOT_PERFORMED` |
| Model-protocol status | `NOT_FROZEN` |
| Gate status | `NOT_ADVANCED` |
| Scientific result | `NONE` |
