# PR #9 engineering re-review record

## Review identity

- Repository: `yurifrusin/ecological-predictive-states`
- Pull request: `#9 — Gate 0B Slice 4: oriented boundaries and visibility events`
- Role: `ENGINEERING_REVIEWER`
- Review profile: `DUAL_REVIEW`
- Evidence class: `PUBLIC_REPOSITORY_ONLY`

## Exact-head re-review history

### First re-review

- Reviewed SHA: `ec71e2c9bfba2c2e00c9a11b5ed8d05370dda1d8`
- Disposition: `ENGINEERING_REQUEST_CHANGES`
- `EPS-ER9-0001`: `VERIFIED`
- `EPS-ER9-0002`: `NOT_VERIFIED`
- `EPS-ER9-0003`: `VERIFIED`
- `EPS-ER9-0004`: `VERIFIED`

`EPS-ER9-0002` was not verified because feasibility slack could still produce a witness outside the true contact-cell parameter domain. This head was not approved, and its dev.8 identities remain historical evidence.

### Final re-review

- Reviewed SHA: `24deb074d4ba0cf3e1044e466ecbbb6e0b2a4cc4`
- Disposition: `ENGINEERING_PASS`
- `EPS-ER9-0001`: `VERIFIED`
- `EPS-ER9-0002`: `VERIFIED`
- `EPS-ER9-0003`: `VERIFIED`
- `EPS-ER9-0004`: `VERIFIED`
- Scientific escalation: `NO NEW ESCALATION; renewed exact-head scientific review required`

## Final verification record

- No new engineering findings were identified.
- Segment, rectangle, and overlap-volume cells with no strictly in-front actual in-cell point were rejected.
- Each segment, rectangle, and overlap-volume case was accepted when an actual in-cell point lay strictly in front.
- Point-manifold and camera-validation behavior remained correct.
- Local Windows/WGL result: 299 passed and 3 platform skips.
- Exact-head Ubuntu/OSMesa CI run `33047664727`: 302 passed.
- Both two-episode scene-family generation, validation, and inspection workflows passed.
- Inspection composites were coherent.
- Final identities and canonical counts matched across the two locked environments.
- The engineering reviewer changed no tracked repository state.
- Scientific validity and owner approval were outside engineering-review authority.
