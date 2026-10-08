# Opt-in oracle region-unit point samples

Review profile: DUAL_REVIEW. Evidence class: PUBLIC_REPOSITORY_ONLY.

`point_sample_qualification.VERSION` is `oracle-region-unit-point-sample-v1`.
Its privileged `qualify(access, policy, lateral, boxes, extent, raw, independent)`
predicate accepts separately retained producer Raster and independent Audit records.
It performs no observation. A future driver must authenticate both records against
these exact inputs, source/runtime identities and attempted-call records before use;
the returned dataclass is evidence, not an approval credential. The driver also
authenticates complete footprint, cause and unit-status diagnostics; this predicate
validates grids and sampled tie/boundary containers, not those continuous certificates. The input hash binds
ordered boxes, floor, lateral calibration and policy. Invalid/missing records raise,
rather than becoming background. Permission denial precedes caller input reads.

The domain remains the fixed exterior camera `(lateral,-3,1)`, the existing nonzero
32 by 32 rational grid rays, two or three bounded positive-front boxes strictly
above the bounded floor patch. Inside/on-surface origins and coplanar floor rays
are outside this contract. Existing rational and coordinate caps apply.

For each closed box or floor patch take its first positive hit parameter, if any.
Exterior origins and closed bounded solids ensure that a nonempty hit has a minimum.
No hit is background. A unique nearest **region unit** has a determinate label,
including edge, corner and singleton tangency hits. Multiple faces of one solid
are one unit; coincident face distances are deduplicated. Distinct units tied at
the nearest parameter are unresolved; no order, identity priority or epsilon breaks
a tie. Farther ties cannot displace a unique nearer unit. Floor edges/corners follow
the same closed-set rule. The box label denotes the whole solid, not a face normal.

Qualification requires complete producer/reference label agreement and no sampled
nearest distinct-unit tie in either record. Boundary flags remain available and
are not qualification failures under this policy. The exact slab producer and
independent face-enumeration audit remain unchanged. Legacy `Audit.qualified`,
continuous footprint/FOV diagnostics, conservative UNKNOWN causes, lawful loaders,
controls, 16-pixel support threshold and costs 1/5/2 remain unchanged. Unknown causes
never become background or exclude queries. Existing reports do not opt in.

Point-ray mathematical definiteness, implementation agreement, and perturbation or
finite-pixel robustness are separate properties. Closed-set tangencies can be
fragile under arbitrarily small perturbations; this predicate certifies no robustness,
area-average pixel semantics or native numerical agreement. Public hand-derived
fixtures check each exact algorithm before cross-comparison: unique face/edge/corner,
singleton tangency, floor interior/edge/corner, no hit, distinct-unit face/edge ties,
hidden farther ties and close unequal rational depths. They also check permutation
covariance, permission denial and invalid records. Existing threshold fixtures are
retained. These are software checks, not scientific efficacy evidence.

Any eventual adoption must require this exact policy in its definition, qualification,
report and operation provenance, together with source/runtime/input bindings. A source
hash alone cannot select it. This prospectively changes admissibility relative to the
legacy boundary exclusion while preserving a coherent closed-solid ideal target.
No old result is requalified: closed studies remain inconclusive or failed as recorded.
Exposed construction tables cannot become fresh qualification or held-out evidence.
A later prospective plan must report sampled boundaries/ties separately from causes
and support, and specify sensitivity reporting before data. It must not move geometry
in response to observed boundaries. No cohort, operation, model or gate is authorized.
