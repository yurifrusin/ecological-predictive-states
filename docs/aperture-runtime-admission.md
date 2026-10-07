# Prospective aperture runtime admission

Review profile: `DUAL_REVIEW`. Evidence class: `PUBLIC_REPOSITORY_ONLY`.
This source correction addresses EPS-ER60-I-0003. The earlier operation closed
inconclusive before producing any frame; adding this route neither qualifies the
apparatus nor reopens that attempt. No new execution is authorized here.

The default `CanonicalPairedRenderer` runtime gate now has a distinct aperture
candidate. Presence of any `EPS_APERTURE_RUNTIME`, `EPS_APERTURE_BINDING`, or
`EPS_APERTURE_IMAGE` field selects this route exclusively, including empty fields.
A denied aperture candidate cannot fall back through WSL or older study markers.
When all three fields are absent, historical runtime routes retain their existing
predicates. The common pinned Linux, OSMesa, Python, MuJoCo, NumPy, PyOpenGL and
glfw checks still apply.

A future separately reviewed outer admission must verify the immutable image and
supply `EPS_APERTURE_IMAGE=sha256:<image digest>`,
`EPS_APERTURE_RUNTIME=docker_candidate_v1`, and canonical JSON in
`EPS_APERTURE_BINDING`. Its strict fields are `source_head`, `source_tree`,
`configuration_root`, `purpose` (`corridor_aperture_native_v1`), `image`, and
`manifest_sha256`. The image value must match the outer image field. These values
are supplied prospectively from verified inputs; no current commit is hardcoded.
The candidate does not inspect the Docker daemon or establish image authenticity.
Its image trust boundary is the separately verified outer admission; environment
agreement alone does not replace that verification or authorize execution.

The candidate hashes the prepared public `/invocation/manifest.json` bytes and
checks its source head, source tree and configuration root. The prepared manifest
need not contain its final image digest, avoiding circular image identity.
Observed clean Git head/tree must match the runtime binding and the current
apparatus configuration root. Missing, malformed, duplicate-key or noncanonical
runtime bindings, unavailable public facts, mixed historical runtime markers and
source/configuration/image/preparation mismatches reject. Public errors disclose
only rejection categories, without binding values or scientific inputs.

The observed cgroup limits must remain one CPU quota, 4 GiB memory, zero swap and
64 processes. CPU affinity and thread counts are not additional admission
requirements. The source check never imports a native SDK or accesses private
inputs. Runtime eligibility does not change typed privileged access, the default
disabled adapter, source cleanliness, consumed-view behavior, finite roster,
resource measurement, hashes, reference math, thresholds or result precedence.

The guarded public tests execute the repository's extracted default constructor
and runtime gate against controlled fake SDK/version/runtime facts. They exercise
missing admission, exclusive route selection, rejected bindings/resources and
historical-route noninterference. Extraction checks the actual source call chain
but does not exercise module import or a real SDK, graphics context, container,
renderer, image or full native adapter. Native smoke, locked environment sync and
scientific generation/validation/inspection remain deferred under this explicit
source-only authorization. Later build, image, admission and native execution
require separate exact review and authority.
