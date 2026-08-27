# PR #9 scientific review record

This file preserves the initial scientific review of one exact revision. It is not a renewed review or a scientific pass.

## Review identity

- Repository: `yurifrusin/ecological-predictive-states`
- Pull request: `#9 — Gate 0B Slice 4: oriented boundaries and visibility events`
- Role: `SCIENTIFIC_REVIEWER`
- Reviewed SHA: `c82ac9d68463fea82ad132194007549ed955adf3`
- Disposition: `SCIENTIFIC_REQUEST_CHANGES`
- Review profile: `DUAL_REVIEW`
- Evidence class: `PUBLIC_REPOSITORY_ONLY`
- Phase-gate effect: `NONE`
- Scientific result: `NONE`

## Findings

### `EPS-SR9-0001`

Scene-level attachment pairs were being used as local image-boundary labels. Correction required a projected local contact-locus oracle so the same panel/support pair could have an attached base and an occluding side.

### `EPS-SR9-0002`

Pair-level causal uncertainty was being labelled `ANALYTIC_BOUNDARY_AMBIGUOUS`. Correction required reserving that code for the actual analytic transport boundary band and making supported interior occlusions causal.

### `EPS-SR9-0003`

The single-occluder public graph was not required to be complete. Correction required equality with all supported oriented owner/affected pairs and frame memberships, with the historical foreground/background counterfactual retained as a privileged cross-check.

## Review boundary

The scientific reviewer changed no repository state. Engineering review and owner approval were pending. Full Gate 0B completion and Gate 0C were not authorised.
