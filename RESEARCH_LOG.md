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
