# PR #5 scientific correction response

This is the `CORRECTION_IMPLEMENTER` response to the scientific review of exact head `276ce4d7699a3482f7877cbc6677a2bd1a9d27ef`. It records implementation evidence, not independent verification. The original findings remain unchanged in `docs/reviews/pr-0005-scientific-review.md`.

## Response identity

- Repository: `yurifrusin/ecological-predictive-states`
- Pull request: `#5 — Gate 0B Slice 3: add analytic optical transport`
- Original reviewed head: `276ce4d7699a3482f7877cbc6677a2bd1a9d27ef`
- Correction role: `CORRECTION_IMPLEMENTER`
- Initial correction and cross-platform evidence head: `753c89dc54ad2c5f671f385237392d8794d94f37`
- Scientific re-review: pending
- Engineering review: pending
- Owner approval: pending
- Merge, tag, release, or gate advancement: not authorised
- Scientific result: none

## Finding responses

### `EPS-SR5-0001` — The analytic support plane is not the finite surface specified by the rendered optic array

- Accepted interpretation: the configured support patch is a finite optical surface even though MuJoCo plane collision semantics are infinite. Canonical analytic rays must accept a compiled plane hit only when its local x/y point lies inside `geom_size[0:2]`; renderer segmentation remains diagnostic only.
- Files changed: `src/epsbench/annotations/optical_transport.py`, `src/epsbench/annotations/__init__.py`, `src/epsbench/schema.py`, `src/epsbench/data/generate.py`, `src/epsbench/data/identity.py`, `src/epsbench/data/loader.py`, `src/epsbench/data/validate.py`, schema/identity/unit/integration tests, the Slice 3 contract, implementation notes, and this review cycle.
- Tests added: finite-plane rejection outside x/y extent; `NO_CONTROLLED_SOURCE_SURFACE` and canonical zero vectors outside the patch; finite-edge analytic ambiguity; exact analytic/rendered controlled assignment in both frames and families; no valid source transport on renderer background; finite-extent sensitivity; method/extent/identity corruption; preservation of appearance, opaque-ID, seed-history, action, geometry, permissions, and scene-family contracts.
- Identities changed: reviewed v1 single-occluder identity `e6d5fc664957c4e68b3c249ce5bd32113755b1c86d5fff8dbeaafc75b550c7cd` and corridor identities `12ea54fea0b8d9716d12189fcba89397156f7c8111fcb7a488702ffb8240f3bf`, `9b41e6780bbf89654cda8c5f6d5f4d6326d64bac2594a1afb12db883f91395b6` are retained only as rejected historical evidence. Corrected v2 values are `479d4540835dcc5d204e530766edfcc4bd74b971cc8efe9cbe391317d4ca6740` for both single-occluder episodes and `ddb4dff0fba18d89cd6c15eb988e672c93988d3d4eda2ae625ab317d36631015`, `a276190abe142bd6859cd0e29983964cbdd17c2cbbf291141ed6f32c6c3c5007` for corridor episodes 0/1. Exact-head Ubuntu/OSMesa evidence matches local Windows/WGL, so they are pinned as one shared regression for those locked environments.
- Local evidence: `uv sync --locked`, Ruff lint/format, mypy, all 205 tests with 2 Windows skips, both two-episode generation/validation/inspection workflows, and `git diff --check` pass. Single-occluder frames each have 18,270/18,270 agreeing non-boundary pixels, 930 analytically excluded boundary pixels, and zero unexplained disagreement. Corridor episodes also have exact agreement in every frame.
- CI evidence: exact correction head `753c89dc54ad2c5f671f385237392d8794d94f37`, Ubuntu/OSMesa run `32962076994`; 207 tests completed, both two-episode generation/validation/inspection workflows completed, and all four analytic identities matched Windows/WGL exactly.
- Remaining limitations: exact agreement is established only for the canonical apparatus and locked renderer environments; finite-plane edge exclusion remains a validity band, not oriented boundary ownership. No ecological accretion/deletion is inferred.
- Implementation status: `IMPLEMENTED_PENDING_VERIFICATION`.

### `EPS-SR5-0002` — The forward/backward consistency test does not evaluate backward transport at the transported correspondence

- Accepted interpretation: inverse consistency must evaluate `B` at `p + F(p)`, not at the original array index. Continuous inversion, fixed-point error, and discrete sampling are separate claims.
- Files changed: `tests/unit/test_analytic_optical_transport.py`, `docs/GATE_0B_SLICE_3_ANALYTIC_FLOW.md`, `docs/IMPLEMENTATION_NOTES.md`, and this response.
- Tests added: a lateral fronto-parallel plane translation chosen for exactly negative one-pixel forward transport; every valid source indexes the corresponding destination and satisfies `F(p) + B(p + F(p)) == 0` exactly in fixed-point units. A continuous-coordinate radial expansion test evaluates the inverse at the expanded correspondence without implying a same-index raster round trip.
- Identities changed: none independently of `EPS-SR5-0001`; test semantics do not enter the data identity domain.
- Local evidence: the correspondence-indexed fixed-point residual is exactly zero and the continuous radial inverse residual is within eight float64 epsilons. The closed-form lateral, radial, static-zero, frame-exit, target-occlusion, and boundary tests all pass within the complete 205-test local suite.
- CI evidence: the correspondence and closed-form tests completed in the 207-test Ubuntu/OSMesa run `32962076994` on exact correction head `753c89dc54ad2c5f671f385237392d8794d94f37`.
- Remaining limitations: arbitrary subpixel raster correspondences require an explicitly selected interpolation/resampling rule before a dense-array inverse statistic can be claimed. This correction does not add such a rule to the benchmark.
- Implementation status: `IMPLEMENTED_PENDING_VERIFICATION`.

Until independent re-review, both findings remain `IMPLEMENTED_PENDING_VERIFICATION`.
