# Open questions

These questions require owner review or later gate work:

1. Which repository, documentation, code, and future dataset licence or licences should apply? No licence has been selected.
2. Will independent review accept `analytic_static_scene_transport_v1` as the Gate 0B dense image-plane transport method, including its fixed-point precision, visibility tolerance, and analytic boundary-ambiguity posture? The implementation exists but is not settled before review.
3. What per-pixel representation should encode oriented boundary ownership without leaking depth or world coordinates?
4. Does the separately hashed analytic transport remain identical beyond the tested Windows/WGL and exact-head Ubuntu/OSMesa locked environments, and what posture should later platforms or dependency versions use? Complete raster-derived ecological hashes may remain backend-specific.
5. Which appearance assets and final evaluation seeds should be frozen before comparative work?
