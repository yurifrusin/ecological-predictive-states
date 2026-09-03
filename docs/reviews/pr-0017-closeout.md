# PR #17 Appearance Benchmark Input Freeze v0 closeout record

## Closeout identity

- Role: `CLOSEOUT_AGENT / CLOSEOUT_EXECUTOR`
- Repository: `yurifrusin/ecological-predictive-states`
- Review profile: `CLOSEOUT_ONLY`
- Evidence class: `PUBLIC_REPOSITORY_ONLY`
- Repository visibility: `PRIVATE`
- Artifact access mechanism: `CONNECTED_AUTHENTICATED_GITHUB_ACTIONS`
- Closeout boundary: `BENCHMARK_OR_PREREGISTRATION_FREEZE`
- Implementation PR: `#17 — Gate 0B: propose Appearance Benchmark Input Freeze v0`
- Approved implementation SHA: `4d47595ede0f1690f1d9d96075df9a3ff7ab60d9`
- Approved implementation tree: `ba112bcc4be4396648305bb1d1529cc002755ea1`
- Implementation merge commit: `4851bfc195a977f1bf9f29005b2579341dc75c38`
- Canonical `main` immediately after implementation merge:
  `4851bfc195a977f1bf9f29005b2579341dc75c38`
- Historical original lock commit: `1a5929307dfcba1d726c650f5e1ce68771f66801`
- Authoritative replacement-lock commit: `d4072f912cc58bbc1ca41ceb2652e41783dbf3e1`
- Authoritative replacement-lock root:
  `28acd2c340b3ef1e2b8b2b7b31ca69356cd9baa35b94ff8a598e020099883435`
- Scientific review: `SCIENTIFIC_PASS`
- Scientific finding: `EPS-SR17-0001 — VERIFIED` by the independent scientific reviewer
- Engineering review: `ENGINEERING_PASS`
- Engineering findings: `EPS-ER17-0001` through `EPS-ER17-0008 — VERIFIED` by the independent engineering reviewer
- Owner disposition: `EXACT_HEAD_APPROVED_FOR_MERGE_AND_BENCHMARK_FREEZE_CLOSEOUT`
- Owner authorisation status: `ISSUED AND EFFECTIVE`

The executor records the independent reviewer finding statuses and does not adjudicate them.

## Canonical freeze decision

```text
Appearance Benchmark Input Freeze v0: FROZEN
```

> This freeze status has effect only when this owner-authorised linked closeout commit is present on canonical `main` and final canonical-main readback has succeeded.

The immutable prospective replacement lock and both qualification receipts record
`benchmark_frozen=false`. That is the correct pre-freeze candidate-state fact and is not rewritten.
The owner-approved linked closeout event recorded here is the later canonical freeze decision.

The frozen benchmark-input scope contains exactly five selected appearance profiles:

1. `revision1_balanced_reference_v1` — `development_reference`
2. `revision1_colour_shift_v1` — `held_out_colour_ood`
3. `revision1_checker_low_v1` — `held_out_texture_presence_ood`
4. `revision1_checker_high_v1` — `held_out_high_frequency_checker_ood`
5. `revision1_combined_stress_v1` — `held_out_combined_appearance_ood`

The retained excluded profiles are `revision1_stripes_low_v1` and
`revision1_illumination_shift_v1`. The frozen sixteen final evaluation roots, in order, are:

1. `988775886542458411`
2. `6042662804173454446`
3. `14022183079951728208`
4. `17670635520239758424`
5. `15123826810452275569`
6. `17286523330598615844`
7. `7896050055028401465`
8. `11538922899034961693`
9. `18335560522116882876`
10. `5187984745037597167`
11. `4446816192287790622`
12. `18081725484914221288`
13. `2878494741235076731`
14. `4279158373164189457`
15. `7775037187548969617`
16. `2012889273669341920`

The replacement lock is unchanged and remains authoritative for these frozen inputs. The two
locked renderer apparatus qualifications are complete.

## Exact-head renderer evidence

Both renderer qualifications are bound to workflow `CI`, run `33706715326`, attempt `1`, exact
source SHA `4d47595ede0f1690f1d9d96075df9a3ff7ab60d9`, and exact source tree
`ba112bcc4be4396648305bb1d1529cc002755ea1`.

| Identity | Windows/WGL | Ubuntu/OSMesa |
| --- | --- | --- |
| Job ID | `100497296367` | `100509579302` |
| Environment | `windows_wgl_locked` | `ubuntu_osmesa_locked` |
| Complete-packet root | `8afe725076057bda3ff86a1925a1038fbbcd03141ade5f8acf9bde88afdb4481` | `43d5f6c88f24bf9439885db1de8bda947861f01dc2ce3ce3cd3022a05dee8079` |
| Packet-tree root | `5c210a1a2dc5cc7bf7d5fbb68700bb7cc3dfae2387d2f85a036456d23e0af41f` | `282398b62331fa09859dcd9e6a76864d0e6c2b6efdcfea9e0aae2a5b89193149` |
| Packet-file SHA-256 | `5d63e9e16515f9416b6f7fc1f51ba5fbf8bd886b83858d52fa2c45fff0afb81f` | `47c9ef071a24926192063fb1a6ef2836ec6511fd592ac58254b4c4dcba2e6916` |
| Renderer-receipt identity | `75df56fcb5975b8db02b572af5ce3e9e76be1c86ddafc10178663b8cae97d78e` | `dfbd05ba51f9d1b656f4f1df9e054e245bf7a464835fd086f6c521585a4dae0c` |
| Publication-record identity | `f668dcda1679e397d6330a37938a1c37e36cd89bec0ac6468bba551267c47a4d` | `5bc44992ffeb058ea1a65320f872f92b2b8f0b79ce166047d1a847282a673c82` |
| Threshold-margin root | `0ea13297585c1a2338d2ee12eb24cb74cf848c111f362be762d6ac1b42d9eaaa` | `7ec9d2f69b0faa6492db1483142ed630dafa95634f3a0a4527300a7a43119689` |
| Selected cells | `160 admitted; 0 rejected` | `160 admitted; 0 rejected` |
| Controls | `16 admitted; 16 rejected` | `16 admitted; 16 rejected` |
| Total cells | `192` | `192` |

The matching portable apparatus roots are:

- procedural assets: `e38f3132b1b8821b1df45ba0edae015081c8243eaac53e9a5c092c25e60de5a2`
- appearance assignment: `7caa2e43c07ad522e27d41a8d71a23abff22c57fcd820d8f607f0233ff49ff12`
- source identity: `3218027ed2135815034a8162158d5940702ec25108cd726fe87a9410c1677948`
- within-renderer invariance: `20037a1b7103e38abb69ab62fa659d99ba15fbd6b4fae8be5e1e9d9bfc771ebe`

Every threshold failure count and near-threshold count is zero.

## Actions artifact identities and retention

Repository ID `1346269440` is private, active, and not disabled. The four exact artifacts require
connected authenticated GitHub Actions access. They were independently downloaded as raw ZIP bytes
and their byte sizes and SHA-256 digests were recomputed before safe extraction.

| Artifact | ID | Raw ZIP bytes | Raw ZIP SHA-256 | Created | Expires |
| --- | --- | --- | --- | --- | --- |
| `appearance-freeze-wgl-packet-4d47595ede0f1690f1d9d96075df9a3ff7ab60d9` | `9876515995` | `21803973` | `6618f9fe265912cb3bc67c99af1a56207a382fb84f20675e6d7dcad5b53a096e` | `2026-09-03T03:01:42Z` | `2026-12-02T02:12:06Z` |
| `appearance-freeze-wgl-evidence-4d47595ede0f1690f1d9d96075df9a3ff7ab60d9` | `9876815245` | `6527` | `29f988417cc20fca3885b5122db01976296058151cf99e03ad7166e7b54ff473` | `2026-09-03T03:16:35Z` | `2026-12-02T02:12:06Z` |
| `appearance-freeze-osmesa-packet-4d47595ede0f1690f1d9d96075df9a3ff7ab60d9` | `9880381284` | `22346269` | `67ecd1a7e695e3b6fd36930258497ec7a92a89f76e22d0ee962891dcc50b6e18` | `2026-09-03T06:03:36Z` | `2026-12-02T02:12:06Z` |
| `appearance-freeze-osmesa-evidence-4d47595ede0f1690f1d9d96075df9a3ff7ab60d9` | `9881190409` | `6515` | `4c756aefce7f350c4b50a4af77ead85cc7032ea7ba0e88dc37107237aeeda53c` | `2026-09-03T06:34:11Z` | `2026-12-02T02:12:06Z` |

All four raw ZIPs were live and unexpired at preservation time. They remain Actions-retained and
are not committed. Their exact identities and current expiry are durable in the evidence manifest.
Byte-identical renderer receipts and publication records, plus credential-free repository, run,
job, and artifact metadata, are committed under `docs/reviews/`.

## Preserved limitations and non-authorisations

- Coverage is limited to two scene families and sixteen final roots.
- There is no admitted isolated stripe-only profile and no admitted isolated illumination-only
  profile.
- Checker-high is frequency-only only relative to checker-low.
- Combined stress is a compound intervention.
- Packet exposure requires future non-adaptive governance.
- Apparatus qualification is not model-efficacy evidence and does not establish EPS superiority,
  generalisation, or renderer-portable model results.
- The repository is private; current evidence access requires a connected authenticated reviewer.
- Actions artifacts are not indefinitely retained.
- The primary model-result renderer is not selected.
- The other-renderer model-result classification is not selected.
- A renderer aggregation rule is not selected.
- Comparative model-result access is not authorised.
- The model protocol is not frozen.
- Full Gate 0B completion is not claimed.
- Gate 0C and Gate 0D are not authorised.
- Model work was not performed.
- The empirical gate was not evaluated.
- Scientific result: `NONE`.
- Tag: not created.
- GitHub Release: not published.
- Next work package: not begun.

## Linked-closeout scope

This linked closeout is limited to the nineteen owner-authorised status, review, approval, closeout,
receipt, publication-record, GitHub-metadata, and evidence-manifest paths. It changes no code,
tests, configuration, schema, identity, workflow, dependency, lock, benchmark input, apparatus
outcome, or prospective replacement-lock byte.

The linked closeout PR number, its head, and its merge commit are deliberately reported in external
execution output and canonical GitHub history rather than self-referenced inside this commit.
