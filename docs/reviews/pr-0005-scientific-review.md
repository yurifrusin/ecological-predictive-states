# PR #5 scientific review record

This file preserves the scientific review of one exact revision. It is not a correction response or renewed review.

## Review identity

- Repository: `yurifrusin/ecological-predictive-states`
- Pull request: `#5 — Gate 0B Slice 3: add analytic optical transport`
- Reviewed head: `276ce4d7699a3482f7877cbc6677a2bd1a9d27ef`
- Reviewer role: `SCIENTIFIC_REVIEWER`
- Disposition: `SCIENTIFIC_REQUEST_CHANGES`
- Independent local execution by reviewer: none
- Repository mutation by reviewer: none
- Engineering review: pending
- Owner approval: pending
- Merge, tag, release, or gate advancement: absent
- Scientific result: none

## Findings

### `EPS-SR5-0001` — The analytic support plane is not the finite surface specified by the rendered optic array

The reviewed implementation analytically intersects the single-occluder support plane as infinite while MuJoCo renders its configured visual grid with a finite extent. It consequently tolerates approximately 6.7% non-boundary analytic/rendered assignment disagreement and can emit transport for a controlled support surface where the optic array shows uncontrolled renderer background.

The correction must make the controlled plane's compiled `geom_size[0:2]` an explicit finite optical extent, include that rule and all label-defining intersection/visibility constants in typed public method metadata and analytic identity, derive finite-edge ambiguity analytically, and require zero unexplained analytic/rendered interior disagreement. Renderer segmentation must remain a non-authoritative cross-check and must not define canonical transport or its identity. The old analytic hashes cannot be retained as current values.

### `EPS-SR5-0002` — The forward/backward consistency test does not evaluate backward transport at the transported correspondence

The reviewed test evaluates `F(p) + B(p)` on the same raster index and calls that a round trip. A true inverse correspondence evaluates the backward field at the transported destination, `F(p) + B(p + F(p))`.

The correction must add a closed-form exactly one-pixel lateral case that indexes backward transport at the corresponding destination and must add either a continuous-coordinate radial inverse or an explicitly interpolated correspondence-indexed radial inverse. Documentation must distinguish continuous analytic inversion, fixed-point quantisation, and discrete raster resampling.

## Interpretation boundary

This disposition applies only to reviewed head `276ce4d7699a3482f7877cbc6677a2bd1a9d27ef`. It does not verify a later correction, provide engineering review, record owner approval, authorise merge/tag/release, complete Gate 0B, authorise Gate 0C, or establish a scientific result.
