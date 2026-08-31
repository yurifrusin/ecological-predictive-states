# Gate 0B Appearance Benchmark Input Freeze v0 candidate

## Authority and status

This work package starts from canonical base
`08179fbdce909e9a0d6dbb2939c58f7ab5d0d8a7` under review profile `DUAL_REVIEW`,
evidence class `PUBLIC_REPOSITORY_ONLY`, and closeout boundary
`BENCHMARK_OR_PREREGISTRATION_FREEZE`. It creates an implementation candidate only. It does not
freeze the benchmark, select a primary model-result renderer, freeze a model protocol, complete
Gate 0B, authorise Gate 0C or Gate 0D, or establish a scientific result.

Benchmark freeze requires exact-head `ENGINEERING_PASS`, exact-head `SCIENTIFIC_PASS`, `OWNER_PI`
approval, and a separately authorised benchmark-or-preregistration-freeze closeout. No active
`docs/reviews/pr-*` record belongs in this implementation pull request.

## Protected canonical evidence

The unchanged Slice 5 apparatus reproduces 160 successful cells, 44 admitted cells, 116 rejected
cells, and rejection of all ten original profiles. Its protected registry and seed roots are
`f90feb3cf9d798ab61c3adbb8d2b276c5d2cb131b95c5a9187df151f9b801d60` and
`6c6816ae1f6657a710631f08a435cca4c623e81efc71ac4e6330a44338313482`.

The unchanged Revision 1 evidence is WGL design `99/13`, WGL qualification `100/12`, OSMesa
design `97/15`, and OSMesa qualification `100/12`. The protected Revision 1 registry,
definition-lock, qualification-seed, and cross-renderer profile-admission roots are respectively:

- `81da1bb9414e7c53e42bbf65b198aa61d8bb7ed3f81eb2bf5813a1edff245ac4`
- `71d2ed7a9f45c55bf17ec518c08b5d0b827a7cf0ae2cc3c339dc09197c55f633`
- `247db21c869703f5604e63678f0cf614f0b88040a05ce42c61629ddf712cefd5`
- `8496521a261367e4fb1da340fc6a58b007eb9b2955fb8cafa44c2c8db0532d5b`

Existing profiles, matched controls, design and qualification roots, rendering parameters,
admission thresholds, and retained negative evidence are unchanged.

## Proposed profile roles

The sole development reference and sole appearance eligible for later training or development on
separately authorised non-evaluation roots is:

- `revision1_balanced_reference_v1` — `development_reference` —
  `b2b64c03236ace288190fca68b702a6b09e6e156c0f43852dfa5db6ab24b9e52`

The four public but evaluation-only Appearance-OOD profiles are:

- `revision1_colour_shift_v1` — `held_out_colour_ood` —
  `3788b17e388c9e1d5805d0d518137f2ea8538629484cca0d8c5ed3e12693e658`
- `revision1_checker_low_v1` — `held_out_texture_presence_ood` —
  `82f051903058f26a72ae68f9a6e0f0833c8c36505c1789353daf078487c35173`
- `revision1_checker_high_v1` — `held_out_high_frequency_checker_ood` —
  `2b04eeb523d597778e89535cc0901209f0d2aaf1ef3c026e392e9a6b11595b71`
- `revision1_combined_stress_v1` — `held_out_combined_appearance_ood` —
  `e4cd4c7e268dd2cd354b5093e6895105843f1ea079d7c845a48b18aebde2606e`

Every primary OOD comparison uses the balanced reference at the same scene and evaluation episode
root. This benchmark relationship remains distinct from the candidate-admission matched control.
Checker-high retains checker-low as a secondary frequency-only diagnostic; it is not
frequency-only relative to the solid reference. Combined stress is a compound colour,
checker-frequency, and illumination shift, not an illumination-only condition.

The following rejected Revision 1 candidates remain excluded negative evidence:

- `revision1_stripes_low_v1` —
  `c2ed57abf5a4b411c5d93b08e5ce78cf188a623ec77b6bb5d3ade187641c2dcd`
- `revision1_illumination_shift_v1` —
  `6d0e18dffc77f2c48590ba42c7129b49c32601aaa150e4915fdc07c70fa76caa`

This leaves two explicit limitations: there is no admitted stripe-family-only condition and no
admitted illumination-only condition. Their rejection is not evidence against the EPS hypothesis.

## Public evaluation episode roots

The registry uses root `314159`, `derive_seed_v1`, namespace
`gate0b-appearance-benchmark-v0-final-evaluation`, and indices 0 through 15. In canonical order,
the roots are:

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

They are unsigned, unique, deterministically recomputed, and disjoint from both protected
eight-root sets. Registry root:
`eb3ca6083af203b325a16612590f0f2e56efdf4b99f1019148f01bd2caab94b2`.

Public does not mean development-eligible. These roots may not be used for training, validation,
tuning, preprocessing selection, architecture selection, hyperparameter selection, checkpoint
selection, or difficult-root replacement. The OOD profiles may not be used for model training or
tuning. Model-randomness, training, and validation roots remain null and deferred.

## Definition lock and qualification rule

The prospective definition identities are:

- benchmark definition: `a655a0e12cfe771a735145fb9d104119ce4160ba0039197f57bcb17c9dfa31fa`
- profile-role root: `0ebd33fd0a84b4bc980073709cc491622a852455aabf80685b12c0917f6abc4f`
- selected-profile-set root: `16a77b2617c44b9106f69afb15d8cc73d6611310548eae927ef67cff02d17460`
- excluded-candidate-set root: `cf1b36c408f766d72ba15acba66e83573803187a360d239fcbccd3d74d74e678`
- scene-family-membership root: `b1d22557512b5a764c0ed601d01f2e23d812926257cf4b353817e9ca19ab4922`
- selected-matrix-membership root: `1e114d28102a6ff284498bd53d537120d328342a06a0edd5662e863b0928f28d`
- legacy-control-membership root: `88887c69f9a9abaeb70b203a962dde27c94fad41f5bcea57c5f5ea293abc47b6`
- training/evaluation-policy root: `2a9efc513dab27bca20737e7538ba716b3ddf23e245b5be5c24f05b3ed369898`
- freeze definition lock: `a373a4742a6b5a3057b820b7d925c2e19fbddc0a83cc45384b8b81ea4e8b1464`

The unique additive lock commit is
`1a5929307dfcba1d726c650f5e1ce68771f66801`. It was validated and pushed before any selected
profile was rendered on any new root. Qualification must not change the role map, roots, scenes,
pairing, thresholds, renderer posture, policies, membership, or readiness rule.

Each renderer must contain exactly 160 selected cells (five profiles × two scenes × sixteen
roots), 32 unique legacy-base controls, and 192 total cells. Each selected profile must pass all 32
of its cells in both Windows/WGL and Ubuntu/OSMesa. Portable apparatus roots must match and every
cross-renderer profile readiness Boolean must be true. Renderer-local RGB, complete ecological
labels, metrics, outcomes, contact sheets, and packet roots are retained but not equality-asserted.

Any selected-cell failure on either locked renderer makes the candidate `not_ready` and retires
the entire sixteen-root registry from final-evaluation use for a revised profile set. A successful
qualification yields only `ready_for_dual_review` and
`eligible_for_owner_freeze_if_approved`; `benchmark_frozen` remains false.

## Frozen scope and deferred protocol

If later approved and specially closed out, this checkpoint freezes appearance roles and exact
profile identities, evaluation roots, scenes, pairing, appearance-admission criteria,
training/evaluation exclusions, supported apparatus environments, and portable/local identity
posture. It is not a complete experimental preregistration.

Gate 0D/0E must separately specify the primary model-result renderer, models, model seeds, training
and validation roots, data volume, optimiser, schedule, parameter and compute budgets, checkpoint
selection, statistical and bootstrap implementations, stopping rules, and empirical gate logic.

## Qualification outcome

After the immutable lock commit was pushed, Windows/WGL generated and independently validated the
complete 192-cell packet. All 160 selected cells were admitted; each of the five selected profiles
passed 32/32 cells. All 32 legacy controls were generated (16 admitted and 16 rejected under the
unchanged candidate-admission criteria). The complete WGL packet root is
`b2d593f3aea5b68fefff4130a1f466e9d24a320c99d344ff5fc634e5b3396a69`; its committed receipt root
is `003c7e28318f67febb754a91dfa814c1f7e6530ddd08035e4814d6757974ae0a`.

The WGL portable apparatus roots are:

- procedural assets: `e3bf5fb73d80ee537db8935e042c0ba7cae837f3f3955095053f69decf63602d`
- appearance assignment: `feaeb12c0c811c8ee9c60efd6c747b6d42f1490504b9a6c83bff22d33029d88f`
- analytic identity: `4cb7076a54bbaaa04a4045f35edb64b3d39440bda7c06fc8800ac575379f32a5`
- within-renderer invariance outcomes:
  `d6fe9e63549128877ec1d73997c0b4e0ae0c27cf5c19460f87856f6acab07617`

The WGL renderer-local selected, control, ecological-label, audit, and contact-sheet roots are,
respectively:

- `914f30d9b5304b1bd405b3c0fa699998fe53a181b6984db4758990a05044200a`
- `ea400adfd2176ae3d1b04be02782d437976c6b2f4abcebfeea6b2627ec08ab38`
- `aa2880b1db2d1c8aa2c8ce60e70d0987a4d3d60623371905251016b31a80d453`
- `8770115c3ea36b8d4a5e6671f2a74cea78da45fcdbaebe01ea6fbf41e2de41db`
- `3da26b945679c86a5c99a89a811f197aad5be103f0dc41bdeb3a587cb7f0e8d8`

No WGL threshold failure or near-threshold case was observed. Minimum passing margins were
`0.105866029445772` for changed controlled-pixel fraction, `0.04111574074074074` for normalized
controlled RGB MAD, `0.07243529411764704`/`0.3672700277464014` for lower/upper visible-surface
mean luminance, and `0.012713812139223166` for textured-surface luminance standard deviation.

Ubuntu/OSMesa exact-head CI also generated and independently validated the complete 192-cell
packet. All 160 selected cells were admitted; each selected profile passed 32/32 cells. All 32
legacy controls were generated (16 admitted and 16 rejected). At qualification source head
`8106c9683bcf921dc2055267da49a411a88b0a8c`, its source-bound complete packet root is
`868cdd14cf216d5c44cd398b6e05d3269f1be5b9d1b6b1d76517791ac83cc676`.

The four portable apparatus roots matched the WGL receipt exactly. The OSMesa renderer-local
selected, control, ecological-label, audit, and contact-sheet roots are, respectively:

- `914f30d9b5304b1bd405b3c0fa699998fe53a181b6984db4758990a05044200a`
- `ea400adfd2176ae3d1b04be02782d437976c6b2f4abcebfeea6b2627ec08ab38`
- `cd892af14b762e8c1eb6ac9aec0ac8953f09319ec0537d50eeb03bed0b064761`
- `30265e3d89fec4bea62ef951fd618c8dd2797917b8e6750e2f9be8559bd44c0d`
- `6ae74dee8a9a594aa6369a27c804ffc117feae1bb9e416cad2d58ff4e087197f`

These roots are reported locally and were not equality-asserted against WGL. No OSMesa threshold
failure or near-threshold case was observed. Minimum passing margins were `0.10594311261851536`
for changed controlled-pixel fraction, `0.04123835784313725` for normalized controlled RGB MAD,
`0.07274100815547455`/`0.36644622087556156` for lower/upper visible-surface mean luminance, and
`0.009543837563144336` for textured-surface luminance standard deviation.

Every selected profile is apparatus-qualified in both renderers. The portable cross-renderer
profile-readiness root is
`da9aad546bf6223d1802cf57990779f952d410340e7bd37d1b07c83c081e65d5`. Therefore the factual
candidate status is `ready_for_dual_review` and its seed-set disposition is
`eligible_for_owner_freeze_if_approved`. This is not review, approval, closeout, or freeze:
`benchmark_frozen` remains false and no scientific result exists.
