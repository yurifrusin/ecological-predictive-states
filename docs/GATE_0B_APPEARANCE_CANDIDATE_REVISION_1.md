# Gate 0B Appearance Candidate Revision 1

## Authority and status

This work package preserves the canonical Slice 5 apparatus and negative result while adding a
prospectively locked, versioned candidate revision. It uses review profile `DUAL_REVIEW`, evidence
class `PUBLIC_REPOSITORY_ONLY`, and closeout boundary `WORK_PACKAGE`. It is candidate evidence only:
no profile, split, or evaluation seed is frozen; full Gate 0B is incomplete; Gate 0C, Gate 0D, and
model work remain unauthorised; and no scientific result is claimed.

## Protected canonical baseline

The clean canonical base `9df0b7a37ab81b02855ef8016e60d79c4f671071` was independently
regenerated before Revision 1 definitions were materialised. All 160 cells generated and validated;
44 cells were admitted, 116 rejected cells were retained, all ten profiles were rejected at profile
level, and the admitted-profile set remained empty. The final split and final evaluation seeds are
null, and freeze status remains `candidate_packet_only_not_frozen`.

The protected portable roots are unchanged:

| Domain | SHA-256 |
| --- | --- |
| Appearance registry | `f90feb3cf9d798ab61c3adbb8d2b276c5d2cb131b95c5a9187df151f9b801d60` |
| Design-seed registry | `6c6816ae1f6657a710631f08a435cca4c623e81efc71ac4e6330a44338313482` |
| Procedural assets | `6eb582b0c636e79349a11edab50394c26b15f07d55327cdd081ef999131b5724` |
| Appearance assignment | `10a24a068c80cd2b85ff00a55b4d0c5a02ba64617f97ac808a9417197bcea90d` |
| Portable analytic identity | `2cb8b42739418faae9666dcb518de3e6655b9ee7f42c63c3acd337debcd224d5` |
| Appearance-invariance outcome | `2bc8edfe05b0401c48a8bc3d897e965c1925ec53dbdf4273d0ee60cc034b3cde` |

The original ten profiles, their profile hashes, the original eight ordered seeds, and all seven
hard admission thresholds remain exact. Revision 1 adds profile IDs; it does not edit failed
candidates in place.

## Deterministic failure diagnosis

`epsbench appearance-failure-analysis` first independently validates the complete canonical packet,
then derives profile, scene, seed, frame, semantic-surface, normal-class, style-slot, texture-source,
rendered-texture, lighting, and renderer breakdowns. It retains raw threshold margins separately
from pass/fail decisions. The portable analysis root is
`8ff7374fb4ea0144687a4bda3b25ae3936c7b8b7f0d1ace35ce071d64328dea2`; the local WGL
renderer-specific analysis root is
`74964814090c8526875552436cec5bfb5c433af98d2710716320fe2ce0b9a6ba`.

Across both frames, the local packet contains 294 lower-exposure surface failures and no
upper-exposure failures. Corridor left and right surfaces account for 128 and 144 respectively;
their near-zero luminance persists while balanced assignment moves every style slot across semantic
surfaces. This supports surface-normal/directional-light interaction as the dominant cause rather
than a fixed semantic colour. The dim illumination profile adds failures on camera-facing end,
background, and occluding surfaces, so light intensity and palette can compound the dominant cause.

There are 247 rendered texture-variation surface failures: 105 checker and 142 stripe occurrences.
Both low and high frequency fail on dark corridor side walls, while the 16-cycle checker also loses
variation on small projected single-occluder surfaces. In 216 surface-frame observations the source
texture standard deviation is at least 0.040 while the rendered surface still fails 0.025, separating
source variation from renderer minification/darkness. Seventy-nine texture margins and five lower
exposure margins lie within 0.005 of their thresholds and remain sensitivity warnings, not altered
decisions. Material-change failure occurs in 64 frame observations. Structural/ecological
invariance, determinism, and generation/validation have no baseline failures.

The reviewed WGL and OSMesa packets agree on 44/116 and profile rejection. The WGL-only diagnostic
packet cannot independently establish every item-level OSMesa margin sign; that causal comparison
remains explicitly uncertain. Renderer-specific matrices and exact values are therefore separate
from the portable analysis identity.

## Prospective definitions

The smallest evidence-supported response is an identity-bound ambient fill for opposing normals,
moderate-luminance palettes, and lower but distinct texture cycles. The canonical threshold remains
0.025; the profiles target, without admitting on, luminance `[0.10, 0.90]`, texture standard
deviation at least 0.040, and normalised RGB MAD at least 0.040.

| Profile | Matched control | Locked rationale |
| --- | --- | --- |
| `revision1_balanced_reference_v1` | `legacy_solid_base_v1` | Solid balanced reference with ambient 0.35, diffuse 0.65, and a moderate-luminance palette; declares colour and illumination. |
| `revision1_colour_shift_v1` | `revision1_balanced_reference_v1` | Changes only the moderate-luminance palette. |
| `revision1_checker_low_v1` | `revision1_balanced_reference_v1` | Changes only texture family and uses one source cycle. |
| `revision1_checker_high_v1` | `revision1_checker_low_v1` | Changes only frequency to four cycles, exactly the required 4:1 partner. |
| `revision1_stripes_low_v1` | `revision1_checker_low_v1` | Changes only family to one-cycle axis-aligned stripes. |
| `revision1_illumination_shift_v1` | `revision1_balanced_reference_v1` | Changes only bound illumination to ambient 0.40, diffuse 0.55, and a second direction. |
| `revision1_combined_stress_v1` | `revision1_balanced_reference_v1` | Prospectively combines the colour shift, four-cycle checker, and illumination shift. |

No profile uses semantic-surface-specific brightness, adaptive or seed-specific tuning, post-render
correction, renderer-specific values, specular/reflection/normal-map effects, shadows, camera
exposure changes, or geometry changes.

## Seed partitions and definition lock

The original eight ordered seeds are the diagnostic/design partition. The qualification registry is
derived once from root 271828 with `derive_seed_v1` and namespace
`gate0b-appearance-revision1-qualification`:

```text
13263716994628839049
10419656982453996572
17335017240715108969
6808163546807211275
10340178108013510301
2958807714473405968
1453940245817783884
12368970943374979214
```

Its registry hash is
`247db21c869703f5604e63678f0cf614f0b88040a05ce42c61629ddf712cefd5`. The list is
unique, recomputed from its declared derivation contract, disjoint from design seeds, and not final
evaluation evidence.

The Revision 1 registry hash is
`81da1bb9414e7c53e42bbf65b198aa61d8bb7ed3f81eb2bf5813a1edff245ac4`. It materialises
all original definitions without YAML inheritance and preserves every original profile hash. The
prospective definition-lock root is
`71d2ed7a9f45c55bf17ec518c08b5d0b827a7cf0ae2cc3c339dc09197c55f633`. It binds the
baseline, analysis, registry/profile hashes, exact seed lists, thresholds, design targets,
rationales, and schema/method versions while declaring `qualification_started: false`.

The validator reconstructs both the portable and renderer-specific analysis roots, binds the three
profile/surface/threshold child-report receipts, and requires exact canonical content equality with
`configs/appearance_revision1_baseline_failure_analysis.json` at the immutable definition-lock
commit. Thus the diagnosis, causal answers, renderer fingerprint, source-packet receipt, child
roots, baseline counts and roots, thresholds, method, and canonical base cannot be fully rehashed
into a different post-qualification account while retaining a valid lock relationship. The child
reports remain reproducible WGL evidence receipts; no cross-renderer equality is claimed for them.

Qualification may begin only after the exact commit adding this lock has been pushed. That commit is
never amended or rewritten. Once qualification begins, rendered definitions and admission semantics
cannot change within this work package.

## Version posture

| Contract | Baseline | Revision 1 |
| --- | --- | --- |
| Registry | `appearance_candidate_registry_v1` | `appearance_candidate_registry_v2` |
| Profile | `appearance_profile_v2` | `appearance_profile_v3` |
| Configuration | `0.1.0-dev.4` | `0.1.0-dev.5` |
| Dataset manifest | `0.1.0-dev.7` | `0.1.0-dev.8` |
| Privileged instrumentation | `0.1.0-dev.12` | `0.1.0-dev.13` |
| Appearance instance | `appearance_instance_v3` | `appearance_instance_v4` |
| Candidate audit | `appearance_candidate_audit_v2` | `appearance_candidate_revision_audit_v0` |
| Revision root domains | n/a | `appearance_candidate_revision_root_domains_v1` |

The profile schema advances because nonzero ambient is now accepted and explicitly rendered. The
config, manifest, instrumentation, and instance contracts advance only on Revision 1 records; the
canonical v1 serialisations remain unchanged. Partition seed snapshots remain protected
appearance-control instrumentation and are never exposed by ecological-only loaders.

## Qualification and packet posture

The locked audit generates 112 candidate cells in each partition: seven profiles by two scenes by
eight seeds, for exactly 224 Revision 1 candidate cells. Legacy reference controls may be generated
in addition. A profile is admitted only if all 16 design cells and all 16 qualification cells are
admitted. Every failure and control failure remains retained.

The immutable lock commit
`914550ce4e3a819dcbcd0bd5390e3c6034af5bf6` was pushed before qualification began. The
complete locked local Windows/WGL run then generated and validated all 224 candidate cells and all
32 controls. The design partition admitted 99/112 cells and rejected 13; the untouched
qualification partition admitted 100/112 and rejected 12.

Exact-head Ubuntu/OSMesa independently generated and validated the same membership: its design
partition admitted 97/112 and rejected 15, while qualification admitted 100/112 and rejected 12.
The two extra OSMesa failures occur in the already-rejected stripe profile. They are retained
renderer-local evidence, not a portable-identity failure.

| Profile | WGL design / qualification | OSMesa design / qualification | Cross-renderer profile outcome |
| --- | --- | --- | --- |
| `revision1_balanced_reference_v1` | 16/16 · 16/16 | 16/16 · 16/16 | admitted |
| `revision1_colour_shift_v1` | 16/16 · 16/16 | 16/16 · 16/16 | admitted |
| `revision1_checker_low_v1` | 16/16 · 16/16 | 16/16 · 16/16 | admitted |
| `revision1_checker_high_v1` | 16/16 · 16/16 | 16/16 · 16/16 | admitted |
| `revision1_stripes_low_v1` | 11/16 · 12/16 | 9/16 · 12/16 | rejected |
| `revision1_illumination_shift_v1` | 8/16 · 8/16 | 8/16 · 8/16 | rejected |
| `revision1_combined_stress_v1` | 16/16 · 16/16 | 16/16 · 16/16 | admitted |

All sixteen local illumination rejections were single-occluder material-change failures. The nine
stripe rejections comprise two single-occluder material-change failures and seven corridor
texture-variation failures. These failures are retained without threshold, seed, or profile tuning.

Root-domain v1 separates portable membership and whole-profile dispositions from per-cell
renderer-local admissions. Its exact cross-renderer comparison roots are:

| Domain | SHA-256 |
| --- | --- |
| Baseline failure-analysis | `8ff7374fb4ea0144687a4bda3b25ae3936c7b8b7f0d1ace35ce071d64328dea2` |
| Definition lock | `71d2ed7a9f45c55bf17ec518c08b5d0b827a7cf0ae2cc3c339dc09197c55f633` |
| Revision 1 registry | `81da1bb9414e7c53e42bbf65b198aa61d8bb7ed3f81eb2bf5813a1edff245ac4` |
| Qualification-seed registry | `247db21c869703f5604e63678f0cf614f0b88040a05ce42c61629ddf712cefd5` |
| Procedural assets | `bd563c543c1dffdd9038e86cb1aa6d97da566fa016d6f6683b11f4964d433ec0` |
| Appearance assignment | `9b7118e4faffb2a74980e9c1ca85b4758abafad6dd7e739b9b68af0224010523` |
| Portable analytic identity | `87459ac7f5f09b222ed029afaf3751881398917eb6184c86ce9305902f76ad4e` |
| Within-renderer invariance outcome | `6bad73a2b68310d2e7b3a19f91dc83d94f15213a91b0955dd541d3d22d5bb100` |
| Design-partition membership | `2493674443d31059d737f256415500a3951cd37ad50198fa3ce197cdf6c9fbaf` |
| Qualification-partition membership | `24eaee5fdb8f01965ac1091defe3d2bf9e62e66c3e862a86528f857e0041eaa3` |
| Profile-admission outcome | `8496521a261367e4fb1da340fc6a58b007eb9b2955fb8cafa44c2c8db0532d5b` |

Per-cell admission checks, statuses, reasons, and RGB/rendered-texture metrics are renderer-local.
The unchanged v0 cell-outcome domains retain these historical exact values under explicit names:

| Renderer-local domain | Windows/WGL | Ubuntu/OSMesa |
| --- | --- | --- |
| Design-partition outcome | `826545aa75737fd8d041b703fcf380e16791a70411c2bcb1c38f783a0fa9916c` | `96bb5661fdbff0491134ba52011bcea081f1cfb3de73e25956ac4d364e26764d` |
| Qualification-partition outcome | `fa976fb849d0037af7bcc8d63d1c554099ecc3f1061fd8ad98cd8d7cc8c3177c` | `fa976fb849d0037af7bcc8d63d1c554099ecc3f1061fd8ad98cd8d7cc8c3177c` |
| Ecological labels | `e1060ab78d6a2f2725b4c87e7ea485a74253eb51a792c247a52496a9b3f21ab6` | `7d916ca67af7f9bd4fed32e1d706b1236a3abe1ef2bfd81d8959aeb679dbc406` |
| Renderer-specific audit | `c63665c0e4f0c151b06a614ef8a8ef5d14eeb294be73702c0cb1144fd2828773` | `31d0a707acf7f7791b8a72908a9cd433f44878cb7b8f81d32176b36a8cef38d0` |

Matching qualification outcome hashes are incidental: the field domain remains renderer-local.
The initial exact-head CI failure at run `33247224228` and the Windows/WGL v0 packet root
`533624890830305bbd78958b4cfe2668aaca7838d36d48a52d0d0e0c56d0bc76` are preserved as
evidence for the rejected over-broad root ontology. No candidate result was removed or tuned.

The final packet status is `candidate_revision_packet_only_not_frozen`, with null final split and
evaluation seeds and `full_gate_0b_complete: false`. Portable membership/profile roots exclude RGB,
raw renderer-local labels, per-cell metrics, reasons, depth/segmentation hashes, provenance, paths,
clocks, hostnames, and contact sheets.
Renderer/source-specific roots retain metrics, raster identities, provenance, and reconstructible
contact sheets. Windows/WGL and Ubuntu/OSMesa each independently apply all criteria; if a later
profile disposition differs, the admitted set is the intersection and the disagreement is
preserved rather than forced to converge.

## Remaining work

Independent engineering and scientific re-review and owner approval remain pending. The agreed
five-profile admitted set only supports a later, separately authorised freeze proposal; it does not
freeze anything. Final evaluation seeds and a development/OOD split remain open. Full Gate 0B exit
criteria remain incomplete, and no action in this work package authorises Gate 0C, Gate 0D, a
model, a tag, a Release, or a scientific claim.
