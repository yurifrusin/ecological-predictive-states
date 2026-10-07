# Known-region events v1: model-free source contract

Review profile **DUAL_REVIEW**; evidence class **PUBLIC_REPOSITORY_ONLY**.
Status **IMPLEMENTED_PENDING_REVIEW**. Phase-gate effect **NONE**.

This additive adapter implements the accepted Stage A proposal SHA256
`39d65ccd5b66b7359fc0ed44f20566c217714979eba94199422f721e47e79d6b`,
independently reviewed proposal receipt SHA256
`745ff2cefae0e1421b92cce08116c6a3b125daa5b1b6c8fe65e8fb398c9d665e`.
It changes no historical Projection, lifecycle, neutral mask/NEW target or score contract.

## Input and information equivalence

Only an exact revalidated Projection with the exact typed permissions, complete episode-start
observations 0/1, nonempty observed-ever inventory and both endpoint relation availability flags
true is admitted. Immutable prefix and lifecycle reconstruction remain the existing authority.
Unavailable relations reject this primary domain; an empty pair list is a valid observation.
Adapter assertions require independent domain qualification before any empirical use; this
package establishes source behavior on handwritten rasters only.

`structured` returns the unchanged numeric feature payload with explicit node and endpoint
fields. `unstructured` is a versioned positional JSON serialization of every field, including
both endpoint ages, availability and pair lists. It is a codec, not a model or a fixed padded
network tensor. `reconstruct` inverts that codec; it is not an input qualification API. Each
encoder accepts an external complete row permutation and remaps both relation lists to the
new row indices. Row-to-opaque-token binding stays with the caller's Projection alignment and
permutation; token spellings, episode/source IDs, hashes, provenance, metric instruments and
unseen future inventory are absent from learner features. Prefix mask shapes/counts remain
lawful observed information. Round trips and joint relabel/permutation checks cover relations.

`omit_relations=True` sets supplied pair lists to null and explicitly marks omission while
retaining endpoint availability and all other fields. This removes the explicit field, not
contact information recoverable from masks; it is a separate ablation from E/U organisation.

## Candidates, targets and exact loss

`save` seals all four probabilities for every prefix-known token, bound to the entire existing
Projection binding (input/state/features, action, chronology, inventory, shape and provenance).
Caller rows may be unordered; sealed rows use canonical token order. Booleans, nonfinite numbers,
out-of-range values, missing/extra/duplicate rows or events, unknown fields and noncanonical
serialized candidates reject before any future operation. The four events are marginals and
need not sum to one. Probabilities accept exact int/Fraction or finite Python float. Floats
convert to their exact IEEE rational value, not rounded decimal; serialized values are reduced
Fraction strings. Validation reconstructs identical canonical bytes. Scores use Fraction
arithmetic throughout, retaining zeros and ties.

Only after complete candidate validation does `evaluate` request one evaluator-only
`Provider.observation(2)`. Exact TrustedObservation assertions must establish complete image,
complete stable association, no unresolved counts and a present correctly indexed/shaped raster.
Unknown association rejects; a missing known token means empty support only under those
assertions. These assertions do not establish real renderer/adapter qualification. Neutral
`_target` supplies unchanged complete known/NEW masks; unseen future tokens do not become named
input nodes or event targets. No retry or per-node future fetch occurs.

For current support B and future support A the event order is appearance (B empty/A nonempty),
disappearance (B nonempty/A empty), gained support (A minus B nonempty), lost support
(B minus A nonempty). Gain and loss may coexist at constant area; remembered regions may
reappear. These are observational predicates, not physical accretion/deletion or occlusion cause.
All false is stable. Episode Brier loss is sum of squared probability errors divided by 4n.
Reports include full N=C=4n, U=0 coverage, event Brier/frequencies and visible/remembered strata;
empty strata have null loss. Immutable evaluator probabilities, labels and per-event losses
permit later calibration summaries without another future fetch. Invalid predictions produce
no score, never a favorable coverage-adjusted score. The receipt binds candidate and neutral
semantic target; its qualification is explicitly external assertions only.

Four untrained controls share this scorer: stable zeros, preceding-transition replay,
constant 1/2 and current-visibility saturation (visible: 0,0,1,1; remembered: 0,0,0,0).
TRAIN-group-only status frequencies with frozen unseen-status fallback 1/2 are a future
comparison requirement; no fitting implementation is supplied.

## Scope and unresolved readiness

The source guard blocks real raster/reference/audit, native renderer, models, closed collectors
and study drivers before import, with plugin autoload/conftest disabled. Checks construct only
public handwritten rasters and fake providers; smoke validation/inspection is the same bounded
handwritten fixture. Scoped CI uses that guard on Linux/Windows and excludes unrelated native
qualification jobs for this branch. No dependencies or licenses change.

Future comparisons must average branches within episode, episodes within physical/generative
ancestry group, and independent groups equally; group-level splitting and interval resampling
must preserve related histories/actions/appearance rerenders. This package supplies per-episode
arithmetic only; it has no grouping/splitting/bootstrap/cohort/launch machinery. Margins, prevalence,
saturation/status-frequency sensitivity, power, matched model/input/preprocessing/compute costs,
qualified family coverage and calibration policies still need prospective acceptance. No learned
model, stub, trainer or fit is implemented or authorized. A2 remains deferred and two qualified
families are not established. Original 0B/0C/0E remain incomplete, 0D not completed, recovery
18-of-21, Slice6 FAILED_CLOSED and held48 persist. All closed fits and the 120-second motion
timeout remain closed. Original matched-budget appearance hypothesis remains **UNTESTED**.
