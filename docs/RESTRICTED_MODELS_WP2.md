# Restricted model WP2 source boundary

Review profile: DUAL_REVIEW / PUBLIC_REPOSITORY_ONLY. This package implements the
frozen restricted architectures after explicit restricted Gate0D source permission.
It does not complete original Gates0B/0C/0D/0E or authorize actual export, health,
training, forecast, scoring or efficacy interpretation. Collection remains bound to
accepted PR55 source1d719a20b800bb100449876947d176051b97ec90 and tree
f698d65f6944ea037211a3e2b7e290b89a851966. Model source/launch identities are separate.
The frozen proposal and architecture identities remain unchanged.

Optional environment: `uv sync --locked --group restricted-models`. Exact
PyTorch2.10.0+cpu uses the explicit [official CPU wheel index](https://download.pytorch.org/whl/cpu/torch/)
([official packaging instructions](https://pytorch.org/get-started/previous-versions/)).
No GPU/vision/audio dependency is installed. The author environment uses Python3.11.15,
NumPy2.4.6, one CPU/inter-op thread and deterministic tensor operations. The measured
project environment logical payload is662,146,229 bytes, including base dependencies.
Shared package-cache storage and filesystem metadata are separate from this payload
measurement; it is not a host disk guarantee. Ordinary dependencies remain unchanged.

`restricted_models.py` implements the exact reset-after GRUs, shared encoder,
six fixed streams, sequential reductions, deterministic frozen initialization,
99913/99984 used parameters and384 recurrent float32 values. Diagnostics retain
connected gradients even where information removal makes them zero. No old masks
or encoded features enter readout. Prefix activations retained by autograd count
as training storage, independently from recurrent state.

`restricted_model_export.py` is an evaluator-only capability. It authenticates the
accepted archive/SealA once, reads qualified train/development payloads only, checks
complete committed expanded schedules and paired initialization values, and releases
immutable lawful examples with sanitized integer order indexes. It never reads any
split/identity/schedule/bootstrap seed file or derives membership. Geometry/decision
identifiers and evaluation targets stay evaluator-side. Actual expanded-schedule
preparation and initialization release require later admission. Every subsequent
payload read is checked against the retained SealA binding. Isolated processes and
private ACLs remain necessary: Python objects cannot defeat hostile same-process code.

`restricted_model_training.py` supplies the common float32 Brier/Adam policy,
1000-update final checkpoint and explicit binary validation without pickle. No fit
runs on import. Source checks invoke only bounded public synthetic batches, never
`train`. Whole data/order/labels/init/source/checkpoint bindings remain in retained
structured reports. Both families use exactly the same sanitized schedules.

Resource accounting traces executed tensor forward/backward/loss/Adam operators.
Named nonlinear evaluations are unit operations under the frozen arithmetic
convention, not machine instructions. Structural operations and unclassified work
are retained; unknown required work/provenance fails matching. Known Python
predictor/Adam scalar arithmetic is required, validated and included
in forward/full-training matching alongside the tensor breakdown: candidate families
charge198 normalization/summary operations per three-observation example; dense has
none of these scalar summaries. Each Adam parameter tensor charges two power calls
(named proxy units) and two subtractions per update. One step-index increment is
charged per update.
Bookkeeping and initialization are excluded from arithmetic counts but included in
inclusive fit CPU/wall clocks starting before model and optimizer construction.
Whole-launch supervision must additionally cover module startup/export/forecasts/
scoring and retained failure/output overhead. Forward
and total forward+backward+loss+Adam each require <=10% matching; phase breakdowns
have no additional tolerance. Reports include buffers, saved-activation conservative
bounds, process-lifetime peak working set when honestly available, and cumulative
allocations explicitly labeled as non-peak. Missing memory observation remains
INCONCLUSIVE. No caller VERIFIED string or3F planning estimate establishes SUPPORT.

`restricted_model_forecasts.py` creates a disjoint new archive through the
authenticated exporter, binds the original qualification archive/SealA plus the
model source/launch, and requires all24 fully validated final checkpoints and all54
complete forecast groups. Cheap controls use the fixed train-only rules. SealB binds
all forecasts and measured runs before scoring access. Scoring has no predictor
callback, checks original retained target bytes, reconstructs fixed WP1 denominators
and endpoints with the committed bootstrap, and denies target access if matching is
missing. Failures and partial forecast work are retained; no recollection or mutation
of the closed qualification archive is permitted. Actual launch admission, operating
supervision and immutable reviewer receipts remain separate from these source APIs.

Run `uv run --locked --group restricted-models python scripts/check_restricted_models_source.py`.
The guard rejects simulator/producer/reference imports, blocks actual export/train/
forecast entrypoints and runs only explicitly SYNTHETIC_SOURCE_ONLY checks. Synthetic
seals cannot score or establish actual readiness. No native/oldA1 test is run.
