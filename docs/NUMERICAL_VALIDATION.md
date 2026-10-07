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

The camera numerical policy introduced by `corridor-aperture-box-capability-v2` is retained by v3. Historical v1 at
PR #64, head `8280bdbee97f71750de46fcfe1bb7bf9fc698457`, and its exact-contract tests/results remain
authoritative and unchanged. Historical v2 is PR #65, head `0282812622185b1b6d3ad57264f60366337821e0`.

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

## Aperture declarations v3

The prospective live contract is `corridor-aperture-box-capability-v3`, purpose
`corridor_aperture_native_v3`, with a distinct configuration root. Old purposes and roots
cannot authorize it. `DUAL_REVIEW` / `PUBLIC_REPOSITORY_ONLY` remains required.

MuJoCo 3.12 [visual map declarations](https://github.com/google-deepmind/mujoco/blob/3.12.0/include/mujoco/mjmodel.h#L142-L156)
store near/far as binary32. This is a pinned API storage assumption, not a dynamically
measured C-field precision; recorded array dtypes are separate measured facts. Validate their exact widened realizations: near
`5368709/536870912` and far `30`, after finite floating type and positive ordered range
checks. Incoming values are never rounded; adjacent binary32 values and unsupported
declared precision reject. This is a representation identity, with zero residual budget,
not a larger coordinate tolerance.

The existing privileged numerical record retains direct declaration values and hexadecimal
operands, precision, observed/expected values, residual, budget and status before domain
validation can fail. Missing compiled box coordinates, FOV and array dtypes share that
record's existing 32 KiB cap. Camera admission and overall unvalidated domain status are
distinct; raw draw operands remain in the existing paired capture record. Sink or cap
failure is terminal. Camera, lateral projection, exact structural and geometric budgets are unchanged.

Conditional public arithmetic models of final-only versus intermediate binary32 projection
rounding do not establish an actual GL error bound or a future apparatus pass. No historical
result is rescored and no operation or gate authority follows from this source correction.

The conditional depth policy `conditional-binary32-depth-v3` covers normal round-to-nearest
binary32 sum/difference/product/division for ordinary projection coefficients, exact halving
and Sterbenz reverse-Z subtraction. Identical eye near/far operands must be exactly widened
binary32 in `[2^-30, 2^30]`, with `f > 3n` and the rounded ordinary coefficient interval within
`[1,2]`. This conservative exponent window keeps sums/differences, `2fn` and divisions normal
and finite. Require the reverse-depth lower endpoints to remain at least `2^-126`, excluding
underflow/flush-to-zero and degenerate propagation. These are declared engineering assumptions,
not measured or proved driver internals.

With `A=n/(f-n)`, `B=fn/(f-n)`, `C=(f+n)/(f-n)`, `u=2^-24` and
`eta=(1+u)^2/(1-u)-1`, use absolute p10 allowance `epsA=eta*C/2`, with `epsA<A`.
Propagate `epsB=max(eta*B, (1/500000)*abs(B))`, covering the unchanged p14 check.
`deltaNear=(epsB+n*epsA)/(1+A-epsA)` and `deltaFar=(epsB+f*epsA)/(A-epsA)`.
Only p10 consistency and projected-far consistency use amended budgets. Keep existing p0, p5,
p14, projected-near and scene/model checks. Actual whole-target near margin must strictly
exceed `max(deltaNear,(1/500000)*n)`; far margin must strictly exceed `deltaFar`.
Compact amended residuals and these strict clearance checks share the privileged prevalidation
record. Actual projection planes and rays, exact first-hit/tie/boundary/clipping decisions and
other whole-target clearances remain. No label invariance, actual driver guarantee, historical
recertification or operation authority is claimed.
