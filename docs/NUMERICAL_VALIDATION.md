# Numerical validation policy

Exact identities, hashes, seeds, permissions, discrete membership and intentionally exact
rational constructions remain exact. Compare computed continuous quantities using an explicit
quantity-specific budget chosen before outcome inspection: units, scale, precision,
conditioning, absolute allowance near zero and justified relative allowance where needed.
State the smallest task-relevant difference and why the budget is smaller. Library defaults
or universal epsilon values are not a policy.

Keep finite/type/shape/range and geometric consistency checks. Retain actual operands and
residuals; never round evidence into agreement. Discontinuous boundary/tie/clipping decisions
need exact logic or preset ambiguity handling; numerical smallness alone does not establish
identical labels. Numerical allowance is separate from statistical uncertainty, effect size
and empirical success. Do not expand allowances after outcomes or rescore historical results.

## Aperture camera v2

Work package: `DUAL_REVIEW` / `PUBLIC_REPOSITORY_ONLY`; implementation and independent
review are separate. Source acceptance grants no operation or empirical authority.

The one live development contract is `corridor-aperture-box-capability-v2`. Historical v1 at
PR #64, head `8280bdbee97f71750de46fcfe1bb7bf9fc698457`, and its exact-contract tests/results remain
authoritative and unchanged. Runtime purpose `corridor_aperture_native_v2` and new config root
are required; v1 admission cannot authorize v2.

Compiled component allowance `b64 = 64 * 2^-52`; scene/draw `b32 = 4 * 2^-23 + b64`; position/translation
allowance scales by `max(1, q)`, with `q` in `{2, 4, 6}`. These are prospective engineering allowances, not
proved SDK error bounds. Direct nominal caps, rigidity, compiled/scene/draw consistency and
bounded affine inversion accompany them. Tiny scale/shear residuals within every budget are
admitted; material/reflected/wrong-axis/degenerate changes are rejected. At-bound components
are admitted; exceeding a bound rejects. Exact affine structure and snapshot binding remain.

For target world corners, `g = 4/10^6 + 1/10^12` bounds existing coordinate allowance, `|V|_1 <= 14.2 + 3g`.
Thus direct modelview-coordinate discrepancy is bounded by `(20.2 + 3g) b32 < 25 b32 = E`. With `d = 79/20 - g`,
`x = 61/20 + g`, `z = 1/10 + g` and `P = 1/500000`, target horizontal discrepancy is bounded by
`64 E (d + x)/(d (d - E)) + 64 P (x + E)/(d - E)`, and vertical by
`64 E (d + z)/(d (d - E)) + 48 P (z + E)/(d - E)`.
These are below 1/1024 pixel (approximately 0.00044113/0.00020047 pixel). This motivates admission,
not identical labels, rasterization accuracy, or non-target near-plane invariance.

Realised rays use rational inverse of retained OpenGL column-major affine modelview `A, t`:
`C = -A^-1 t`; `direction = A^-1 (x/p0, y/p5, -1)`. The unnormalized parameter is actual camera depth.
Whole-target frame/clipping checks use transformed corners; slab intersections use actual
origin and positive-denominator corner bounds. Six-face hits, ties, edges and clipping stay
exact. Strict whole-volume clearances and unknown dispositions are not relaxed. Camera facts
and compact residual reports remain privileged; existing capture records retain raw operands.

Pinned camera convention: [MuJoCo 3.12 setView](https://github.com/google-deepmind/mujoco/blob/3.12.0/src/render/classic/render_gl3.c#L718-L777)
and [lookAt](https://github.com/google-deepmind/mujoco/blob/3.12.0/src/render/classic/render_util.c#L126-L159).
This source contract has no launch, retrospective-pass, empirical-gate or EPS-result effect.
