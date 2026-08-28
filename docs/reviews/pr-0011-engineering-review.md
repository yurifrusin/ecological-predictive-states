# PR #11 engineering review record

This file preserves the initial engineering review of one exact revision. It is not a correction response or renewed review.

## Review identity

- Repository: `yurifrusin/ecological-predictive-states`
- Pull request: `#11 — Gate 0B Slice 5: add procedural appearance candidate audit`
- Role: `ENGINEERING_REVIEWER`
- Reviewed SHA: `5ee07912e8712ce3f511312a32c94e03fccbc3ca`
- Disposition: `ENGINEERING_REQUEST_CHANGES`
- Review profile: `DUAL_REVIEW`
- Evidence class: `PUBLIC_REPOSITORY_ONLY`

## Findings

### `EPS-ER11-0001`

Declared texture filtering did not match MuJoCo 3.12.0 classic-renderer behavior.

### `EPS-ER11-0002`

Dataset artifacts, including the seed-registry snapshot, required complete symlink, hardlink, regular-file, containment, and alias protection.

### `EPS-ER11-0003`

Audit retained-evidence paths could escape or alias outside the packet.

### `EPS-ER11-0004`

Representative contact sheets were not bound to or independently validated by the packet.

### `EPS-ER11-0005`

Packet counts, scene domain, final split, final seeds, and freeze posture required independent validation.

### `EPS-ER11-0006`

Stored matched-control metadata could contradict the canonical registry relation.

## Review boundary

Normal checks and CI had succeeded despite these defects. The engineering reviewer did not implement its own findings. Findings changing appearance or root identity required scientific escalation. Owner approval and closeout were pending.
