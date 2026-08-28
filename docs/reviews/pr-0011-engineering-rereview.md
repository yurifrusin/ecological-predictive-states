# PR #11 engineering re-review record

## Review identity

- Repository: `yurifrusin/ecological-predictive-states`
- Pull request: `#11 — Gate 0B Slice 5: add procedural appearance candidate audit`
- Role: `ENGINEERING_REVIEWER`
- Review profile: `DUAL_REVIEW`
- Evidence class: `PUBLIC_REPOSITORY_ONLY`

## Final finding ledger

- `EPS-ER11-0001` — truthful renderer filtering
- `EPS-ER11-0002` — dataset alias/path protection
- `EPS-ER11-0003` — audit evidence containment
- `EPS-ER11-0004` — contact-sheet integrity binding
- `EPS-ER11-0005` — packet counts and freeze posture
- `EPS-ER11-0006` — matched-control metadata
- `EPS-ER11-0007` — cell/evidence identity pairing
- `EPS-ER11-0008` — axis isolation
- `EPS-ER11-0009` — mixed-failure retention
- `EPS-ER11-0010` — strict audit-cell schemas
- `EPS-ER11-0011` — immutable snapshots and race resistance
- `EPS-ER11-0012` — contact-sheet manifest schema
- `EPS-ER11-0013` — shared logical decoding

## Exact-head re-review history

### Consolidated engineering-correction head

- Reviewed SHA: `93ebd95453b0a023a4c8b12fdd995990839aaf6d`
- Final status: not final; further correction was required for strict audit-cell schemas and owned-file race resistance.

This head did not receive a final engineering pass.

### Strict-schema and owned-file correction head

- Reviewed SHA: `febf122974fdca3b9e099744ad8d4f769e0e6ac2`
- `EPS-ER11-0010`: `VERIFIED`
- Final status: not final; further correction was required for immutable snapshots, strict contact-sheet manifest typing, and shared logical decoding.

This head did not receive a final engineering pass.

### Final re-review

- Role: `ENGINEERING_REVIEWER`
- Disposition: `ENGINEERING_PASS`
- Reviewed PR: `#11`
- Reviewed SHA: `2ea05867e7c8cb04ee4f9d0138394602b6c3e53a`
- Review profile: `DUAL_REVIEW`
- Evidence class: `PUBLIC_REPOSITORY_ONLY`
- `EPS-ER11-0001`: `VERIFIED`
- `EPS-ER11-0002`: `VERIFIED`
- `EPS-ER11-0003`: `VERIFIED`
- `EPS-ER11-0004`: `VERIFIED`
- `EPS-ER11-0005`: `VERIFIED`
- `EPS-ER11-0006`: `VERIFIED`
- `EPS-ER11-0007`: `VERIFIED`
- `EPS-ER11-0008`: `VERIFIED`
- `EPS-ER11-0009`: `VERIFIED`
- `EPS-ER11-0010`: `VERIFIED`
- `EPS-ER11-0011`: `VERIFIED`
- `EPS-ER11-0012`: `VERIFIED`
- `EPS-ER11-0013`: `VERIFIED`

## Final verification record

- Exact-head local Windows/WGL result: 398 passed with 9 platform skips.
- Exact-head Ubuntu/OSMesa CI run `33170997633`: 407 passed.
- Both ordinary two-episode workflows passed and were visually coherent.
- The complete 160-cell audit and independent packet validation passed.
- The final candidate result and all six portable roots matched exactly across locked environments.
- Renderer-local ecological-label and renderer-audit roots remained deliberately backend-specific.
- Ownership-to-hash, hash-to-decode, same-inode overwrite, hardlink, nonregular-file, noncanonical-path, containment, strict-schema, logical-hash, dtype, shape, media-type, contact-sheet, permission-before-open, and mixed-failure regressions were independently exercised.
- No new engineering finding remained.
- The reviewer changed no tracked repository state.
- Scientific validity and owner approval were outside engineering-review authority.
