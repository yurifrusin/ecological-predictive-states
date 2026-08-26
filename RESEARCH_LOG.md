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
