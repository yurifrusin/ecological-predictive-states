# PR #5 scientific re-review record

Repository: yurifrusin/ecological-predictive-states

Pull request: #5

Role: SCIENTIFIC_REVIEWER

Reviewed SHA: 3086bc2eccc4b7492eb8a77660572e879a04ef0b

Disposition: SCIENTIFIC_PASS

Review profile: DUAL_REVIEW

Evidence class: PUBLIC_REPOSITORY_ONLY

Phase-gate effect: NONE

Scientific result: NONE

## Review history

- Initial head `276ce4d7699a3482f7877cbc6677a2bd1a9d27ef` received `SCIENTIFIC_REQUEST_CHANGES`.
- Correction head `eee0eaa4b1de50b4dea1e391dba6f32b588e763c` received an intermediate `SCIENTIFIC_PASS`.
- Engineering review of `eee0eaa4b1de50b4dea1e391dba6f32b588e763c` requested `EPS-ER5-0001` through `EPS-ER5-0003`.
- Engineering corrections created `3086bc2eccc4b7492eb8a77660572e879a04ef0b` and required renewed exact-head scientific review.

## Scientific conclusion

- `inclusive_extent_plus_scaled_binary64_epsilon_v1` uses:

  ```text
  limit =
  extent
  + 16.0
    * 2.220446049250313e-16
    * max(1.0, extent)
  ```

  This is an explicit numerical equality policy for the inclusive finite optical edge, not a meaningful ecological enlargement of the surface.
- The exact edge is accepted and values beyond the declared tolerance are rejected.
- The comparison rule and constants are typed and included in `analytic_transport_sha256`.
- Method, intersection, and extent advance to `analytic_static_scene_transport_v3`, `compiled_plane_and_oriented_box_nearest_hit_v3`, and `finite_plane_visual_extent_v2`.
- Transition and privileged-instrumentation contracts advance to `0.1.0-dev.5`.
- Dataset-manifest wire shape remains `0.1.0-dev.3`.
- Canonical label arrays are byte-identical to v2, while identity changes deliberately because the formal numerical contract changed.
- Fail-closed public-input validation strengthens apparatus integrity without altering valid canonical labels.
- Atomic no-replace inspection changes neither dataset nor ecological identity.
- Finite optical surfaces, zero unexplained renderer-interior disagreement, correspondence-indexed inverse evidence, metric-leakage boundaries, neutral mask-change semantics, unavailable boundary ownership, and unavailable ecological visibility events remain intact.
- The pass concerns apparatus and construct validity only.
- Evidence remains limited to the canonical apparatus, bounded scene scale, and locked Windows/WGL and Ubuntu/OSMesa environments.
- Arbitrary-subpixel dense-array inverse interpolation remains undeclared.
- Full oriented boundary ownership, ecological accretion/deletion, full Gate 0B, Gate 0C, and any model remain outside scope.
- The scientific reviewer performed no repository mutation or independent local execution.
