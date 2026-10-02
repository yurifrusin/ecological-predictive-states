# EPS machine continuation — 2 October 2026

Review profile: `ENGINEERING_ONLY`. Evidence class: `PUBLIC_REPOSITORY_ONLY`.
Phase-gate effect: `NONE`. This documentation-only package records checkout and transfer
instructions. It changes no source, test, dependency, scientific contract or execution authority.
Independent engineering review is required; scientific review is unnecessary for these
non-semantic transfer instructions. Active review records remain external.

## Select the current code

The repository is `https://github.com/yurifrusin/ecological-predictive-states`. The default
`main` branch is at `25cdf0e13774be2152a294a3f408000a1d45e48e`. Latest experimental code is
PR30 at `3c9ec6e196010959b558d5a6fc49fc82c660603d`, tree
`14017584f28a7b13bafe34cb579db2c2c15206f3`. The TPUQMNG Windows and WSL checkouts match this
source and are clean. No unpublished source changes were found in the inspected EPS worktrees.

| Draft PR | Branch | Exact head |
| --- | --- | --- |
| 26 | `feat/capture-contract-revision-study` | `6e614b7e0d8b4ce421851d667ec1f07d75ab67e3` |
| 27 | `feat/renderer-readback-investigation` | `be2628dd3edf36e95a3ef8f3af289297597370e4` |
| 28 | `feat/osmesa-joint0-qualification` | `b692a4968f4441a136fe7dadaec486ef9165a3d3` |
| 29 | `feat/shared-raster-capture-study` | `1a36c1605bd008fab9e13e270f9a20389b666204` |
| 30 | `feat/canonical-paired-capture` | `3c9ec6e196010959b558d5a6fc49fc82c660603d` |

Each row is based on the preceding row; PR26 is based on `main`. This handoff branch,
`docs/machine-continuation-20261002`, is based on PR30 and includes the entire code stack plus
these instructions. The separate draft PR24 (`feat/prospective-eps-capture-diagnostic`,
`bc7c65c145530f04a07a25a013af6d65f3f6ae65`) remains a separate historical diagnostic branch.
Do not combine or merge the drafts merely to copy their files.

Clone a full repository, rather than copying a Git worktree directory whose `.git` pointer
still refers to the original machine:

```sh
git clone --branch docs/machine-continuation-20261002 https://github.com/yurifrusin/ecological-predictive-states.git EPS
cd EPS
git rev-parse HEAD
git rev-parse HEAD^
git status --short
```

Use `Set-Location EPS` for the second command in PowerShell. The parent must be the PR30 head
above for the initial handoff commit. Record the cloned head, tree and clean status before work.
Create a new work branch for subsequent corrections; retain the exact historical PR heads.

## Restore a project environment

Git, `uv`, Python and system graphics libraries must be available on the destination. Recreate
the project environment there; copying a Windows or Linux `.venv` between machines is unsuitable.
The checked-in `.python-version` selects the 3.11 series, so request the previously used exact
Python version explicitly:

```sh
uv sync --locked --python 3.11.15
```

The unchanged `uv.lock` SHA-256 is
`d8fbbd09590dd2c937db822668d168ed73947d79e3772de7dce1e311b89ebafc`.
The recorded project runtime used Python 3.11.15, MuJoCo 3.12.0, NumPy 2.4.6, glfw 2.10.2,
mypy 1.20.2, Ruff 0.16.4, pytest 8.4.2 and Pydantic 2.13.4. PyOpenGL 3.1.10 was an additional
project-environment installation and is absent from this lock. To restore that dependency in
Linux/WSL after locked sync:

```sh
uv pip install --python .venv/bin/python PyOpenGL==3.1.10
```

On Windows use `.venv\Scripts\python.exe` for that interpreter argument. A later exact
`uv sync` can remove this extra installation; restore it afterward and record versions. This
handoff does not change the lock or add a new dependency declaration.

Canonical paired capture is currently restricted to Linux/WSL OSMesa, zero samples and
160 × 120. A Windows destination can use WSL for this path; its checkout should reside in the
WSL Linux filesystem. For an Ubuntu/WSL graphics environment, the existing CI installs
`libgl1-mesa-dri`, `libglfw3` and `libosmesa6`. Set `MUJOCO_GL=osmesa` before graphics imports
in that environment. These are environment prerequisites, not permission to capture or a
claim that the destination apparatus is qualified. See `CANONICAL_PAIRED_CAPTURE.md`.

Inert startup checks that create no renderer context or study output are:

```sh
.venv/bin/python -m epsbench.cli.app --help
.venv/bin/python scripts/canonical_paired_qualification.py plan
```

Use `.venv\Scripts\python.exe` in PowerShell. Inspect `AGENTS.md`, the research charter,
benchmark contract, milestone and review protocol before continuing. Apply the recorded
minimum-sufficient subagent allocation and project-specific tooling policies.

## Current engineering work

[PR30 CI run 34965963988](https://github.com/yurifrusin/ecological-predictive-states/actions/runs/34965963988)
is completed with `failure`, superseding the 15 September snapshot that called it queued.
The Ubuntu quality job passed installation, lint and formatting, then failed
`uv run mypy src tests`: 82 errors in nine test files. Subsequent quality-job tests and smoke
commands were skipped. The self-hosted WGL job had no assigned runner and was cancelled after
24 hours; the dependent OSMesa qualification job was skipped.

The older guarded local/WSL receipts report 217 selected CPU tests passed and a successful
bare `python -m mypy` check, whose configured package scope is `epsbench`. That result does
not establish that the broader `src tests` CI command passes. The first engineering task on
the destination is a separate bounded correction of the test typing failures, followed by
independent review; address runner availability separately. Preserve the original implementation
heads and all study evidence. A new source head does not inherit their exact-head reviews.

## Transfer evidence separately

Generated data, environments, operator records and active reviews are deliberately untracked.
A Git clone transfers the source and its history, not these materials. The accompanying
`EPS-machine-continuation-evidence-2026-10-02.zip` preserves the available ordinary diagnostic
and development evidence for PR24 and PR26–PR30, review/assessment bundles, CPU receipts and
the pending topology proposal. Its transfer index binds each original file's byte count and
SHA-256. Keep it outside the source checkout and verify hashes before use.

The key current archives are:

| File | SHA-256 |
| --- | --- |
| `EPS-TPUQMNG-canonical-paired-evidence-2026-09-15.zip` | `f1a2fda28498d53f5c74361018c74024642cfb81dc2606682975f7684b4d20a2` |
| `EPS-PR30-assessment-handoff-2026-09-15.zip` | `027d504259560686f749452ec402fa89f45b12928f83cd76f3d82067b85f775f` |
| `EPS-retained-public-topology-exploration-2026-09-15.zip` | `b5ed25bc8305699a73f25cc0eb9db68e9e06aeb10213a0ecae373e49af22b0cc` |

The evidence bundle is a transfer copy, not canonical review closeout or new authority. It is
not a complete replacement for every historical GitHub Actions artifact or frozen apparatus
archive; consult the committed freeze and closeout records for those older external identities.
Retain the original evidence on TPUQMNG. Never initialize, resume, retry, extend, overwrite or
replace the completed PR24/PR26–PR30 studies during migration.

The proposed eight-transition forward/reverse split/merge development study remains pending
owner authorization. Its resolved membership and proposal are preserved for consideration,
not execution. Legacy remains the default; frozen Slice 6 is `FAILED_CLOSED`, Gate 0B remains
incomplete, and no Gate 0C/0D, model, merge or closeout authority follows from this handoff.
