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
- selected membership root: `1e114d28102a6ff284498bd53d537120d328342a06a0edd5662e863b0928f28d`
- freeze definition lock: `a373a4742a6b5a3057b820b7d925c2e19fbddc0a83cc45384b8b81ea4e8b1464`

The additive lock commit is recorded after Stage A validation and pushed before any selected
profile is rendered on any new root. Qualification must not change the role map, roots, scenes,
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

Windows/WGL and Ubuntu/OSMesa final-root outcomes are recorded only after the immutable lock commit
has been pushed. Until both complete, the candidate is `qualification_incomplete`, not frozen, and
not a scientific result.
