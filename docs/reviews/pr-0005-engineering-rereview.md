# PR #5 engineering re-review record

## Review identity

- Repository: `yurifrusin/ecological-predictive-states`
- Pull request: `#5 — Gate 0B Slice 3: add analytic optical transport`
- Role: `ENGINEERING_REVIEWER`
- Reviewed SHA: `3086bc2eccc4b7492eb8a77660572e879a04ef0b`
- Disposition: `ENGINEERING_PASS`
- Review profile: `DUAL_REVIEW`
- Evidence class: `PUBLIC_REPOSITORY_ONLY`

## Verification record

- `EPS-ER5-0001`: `VERIFIED`
- `EPS-ER5-0002`: `VERIFIED`
- `EPS-ER5-0003`: `VERIFIED`
- No new engineering findings were identified.
- Local Windows/WGL result: 238 passed and 3 platform skips.
- Ubuntu/OSMesa CI run `32971294963`: 241 passed.
- Both two-episode dataset generation, validation, and inspection workflows passed.
- All 24 canonical v2/v3 vector, validity, and reason `.npy` files were byte-identical.

Final v3 analytic identities:

| Family / episode | `analytic_transport_sha256` |
| --- | --- |
| Single occluder 0 and 1 | `77821c734e4a5316851b9e57417a014e8caf292f568314da05e2493608960832` |
| Corridor 0 | `ffa9b31e91da7a4cdb68ce938beba901beecb3973684bc303d3e362fa01fa09a` |
| Corridor 1 | `64706a77347aa2a98e52d6da023ba80f7a04d1b7a94eaa809124c9b7fe9fca0a` |

The reviewer changed no tracked file or repository state. Scientific validity and owner approval were outside engineering authority.
