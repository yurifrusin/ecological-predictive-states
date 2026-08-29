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
