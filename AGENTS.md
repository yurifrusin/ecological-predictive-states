# Repository guidance

This repository tests a falsifiable engineering hypothesis; it does not assume an ecological predictive state is superior. Read `docs/RESEARCH_CHARTER.md`, `docs/EPS_BENCH_V0.md`, and `docs/MILESTONE_0.md` before changing scientific contracts.

- Treat depth, world coordinates, camera pose, raw simulator identifiers, and generation records as metric or privileged instrumentation. Never expose them through ecological-only loaders or learner inputs.
- Enforce modalities with typed, fail-closed permissions. Names and directory conventions are not access control.
- Preserve deterministic seeds, canonical logical identities, artifact hashes, strict validation, and renderer provenance. Keep volatile run metadata outside scientific hashes.
- Randomise opaque surface identifiers per episode and keep raw-to-opaque mappings privileged.
- Preserve failed experiments, negative results, and limitations. Repository success is not scientific evidence.
- Do not implement any machine-learning model before explicit Gate 0D authorisation. Do not advance gates silently.
- Do not purchase, require, or assume a physical robot.
- Work on reviewed branches, keep generated data and ordinary artifacts untracked, and run locked sync, lint, formatting, typing, tests, smoke generation, validation, inspection, and `git diff --check` before review.
- Do not add a repository, documentation, or data licence without owner approval.

## Review roles

Follow `docs/review-protocol.md`. An agent implementing or correcting a work package may self-check but may not independently approve that work package. Reviews and owner approval are bound to an exact pull request and head SHA; a changed head requires renewed review. CI success does not advance a gate or establish a scientific result.
