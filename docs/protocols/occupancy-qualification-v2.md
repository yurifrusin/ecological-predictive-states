# Fixed analytic occupancy qualification source package

Review profile: DUAL_REVIEW. Evidence class: PUBLIC_REPOSITORY_ONLY.
Implementation status: IMPLEMENTED_PENDING_REVIEW. Phase-gate effect: NONE.

This package implements the prospective development-only analytic proposal v2,
SHA256 `753f461c597ea5e58c29c63cfec0bf92403d6a39ac53a4c1905537fc67571494`,
on baseline `bf9d2f80caa92184232f70c101d835263433b323`, tree
`cf3711631fdb498d467465ab4fceb478b0f84f07`. It tests annotation and converter
readiness, not learned efficacy, compactness, biological perception or appearance robustness.

`occupancy_producer` is the isolated actual collection entrypoint. It pins existing
Camera/Box/Scene.frame source by logical LF SHA256 and uses raw segmentation only.
The fixed 32-square camera, support and two boxes and R/H/F/P rational table are
literal source, with no candidate selection, fitted tolerance or alternate backend.
R/H share prefix and identity; F/P remain distinct. Caller-supplied private 32-byte
nonces must be distinct and produce nine unique opaque associations across the three
units; no allocator or launch CLI is provided. Inputs see only actually observed tokens.
Coordinates, source identities, roles, nonce/mappings, raw labels, rational reference
and status records stay in private retained evidence, outside the causal input.

Actual collection requires a separate exact-source launch decision and a clean
checkout matching the caller's expected reviewed head. Source implementation or CI
success cannot authorize launch. No actual nonce, prefix, target or study has been
allocated or collected by this source package's tests. Actual evidence would require
PRIVATE_REVIEW_BUNDLE and separately authorized packaging. Never publish the output
directory, item-level hashes, inventories or the inspection result for actual data.

The independent scalar reference enumerates six faces per box and the support plane
with exact rational arithmetic; it calls no producer projection/ray/slab code. Unique
raw labels must agree at every centre. Ties, boundary flags, ideal/float disagreements
and missing/unsupported truth fail conservatively. Independent eight-vertex convex
hulls establish whole unclipped footprints and positive-depth ranges. Full raster
lattice support includes centres outside the image and differs from clipped support.
Continuous strict containment on the nearer front face certifies full background
occlusion; discrete zero visibility alone cannot. Strict separating axes support
unoccluded certification. Support whole extent is UNKNOWN_DOMAIN, not a pure cause;
finite native near/far limits are NOT_APPLICABLE. Every observed inventory channel
remains in scoring, regardless of status, forecast error or UNKNOWN.

The pure collection module gathers all six prefix observations before constructing
four inputs. It reuses visible_forecast_contract CausalView, immutable CausalInput,
Forecast, fixed persistence/K2 controls, evaluate and storage without changes.
Wrapper summaries add current-visible/current-absent strata and token reappearance
from admitted source and target rows, preserving all UNKNOWN inventory channels.
An explicit row-major Boolean buffer with frame/token/shape/action metadata is
losslessly reconstructed and the same rules must agree; this is fidelity, not an
independent learned baseline or advantage claim. All eight complete saved forecasts,
inputs/actions/inventories, flat buffers and source plan are globally sealed before
any target access. Both controller and actual adapter check the retained seal;
mutable completion flags alone cannot open target access. Repeated evaluator reads
use immutable retained targets. No predictor runs during evaluation.

One fresh private directory retains append-only hash-linked events, complete pending
membership, raw/reference provenance, seals, masks and reports. Files use exclusive
creation and flush/fsync. Byte space is reserved before writing; all retained files,
including identity records, manifests, seals, runtime metadata and failure receipts,
share 16MiB total with 256KiB reserved for failure. No extra archive/review copy is
allowed. Limits are two prefix frames, 1024 pixels, three tokens, ten actual producer
calls and 60 seconds cooperative including geometry and writes. Rational hull lattice
extent is bounded by normalized magnitude eight; centre intersections are at most
10x1024x13 face/plane candidates. These limits do not claim RSS or host preemption.
A failure preserves pending membership, initiating bounded private error context, stage and
closed failure kind; no retry/resume/replacement. Scoring completion is distinct from
required physical relation qualification. Runtime versions/elapsed and bounded private error context are retained but
excluded from logical inspection hashes.

Source verification command:

```text
uv sync --locked
uv run --locked ruff check src tests scripts/check_occupancy_source.py
uv run --locked ruff format --check src tests scripts/check_occupancy_source.py
uv run --locked mypy src tests scripts/check_occupancy_source.py
uv run --locked python scripts/check_occupancy_source.py
git diff --check
```

The dedicated checker denies actual producer, causal-history fixture, native/render,
data generators and old controllers BEFORE pytest collection; plugins/conftest and
ambient selectors are disabled. It uses handwritten noncandidate rational geometry,
fake masks/providers, clocks and faults only. CI selects this checker only for head
`codex/occupancy-qualification-20261007` and base `codex/a1-integration-base-20261005`;
the same exact route excludes broad quality/WGL/OSMesa jobs and the historical
check_a1_source default. That default must never be used for this package.
Synthetic generation, validation and inspection exercise the retention lifecycle;
actual table smoke is deferred because it would consume prospective qualification
evidence before independent source review and separate launch authority.

Preserve historical 18/21 recovery, Slice6 FAILED_CLOSED, incomplete original 0B/0C,
no Gate0D, held48 sequence, all closed failures and parked privileged helpers. No
model, gate change, license, native/build change or active review record is included.
