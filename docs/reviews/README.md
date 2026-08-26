# Review-record directory policy

`docs/reviews/` contains immutable final evidence for completed EPS review cycles. It is not the
venue for active review of an implementation or correction PR.

## Active and final evidence

Active engineering and scientific evidence remains outside the implementation head in read-only
reviewer chats, GitHub reviews or comments, isolated untracked notes, external receipts, and final
exact-head reviewer responses. An implementation or correction PR must not create or update its own
`docs/reviews/pr-*.md` records.

After `MERGE_CONVERGENCE` and exact-head owner approval for implementation merge, a separately
authorised linked closeout PR may add final review, approval, and merge-closeout records here. After
`TERMINAL_DISPOSITION_CONVERGENCE` and an exact owner `RECORD_ONLY_CLOSEOUT` decision, a separately
authorised documentation PR from current canonical `main` may add final review, owner-decision, and
terminal-closeout records without merging the terminal implementation. Those records are
historical documents about the reviewed implementation SHA; they need not have existed inside that
SHA. They must not contain self-referential placeholder fields.

This `README.md` states directory policy and is not an active review record.

## Names and record types

Use a zero-padded PR number and a descriptive record type, for example:

```text
pr-0007-engineering-review.md
pr-0007-engineering-rereview.md
pr-0007-scientific-review.md
pr-0007-scientific-rereview.md
pr-0007-owner-approval.md
pr-0007-closeout.md
pr-0007-terminal-decision.md
pr-0007-record-only-closeout.md
```

Do not create a file for a review round or record type that did not occur.

## Immutability and compatibility

Historical records are immutable. Correction-response files committed inside older implementation
PRs are grandfathered evidence under the protocol then in force and must not be rewritten, renamed,
or reinterpreted. The prospective active-record prohibition applies to the governance-v2 PR's own
reviews and all later work packages. Evidence for an older rejected or abandoned PR may be
canonicalised prospectively only through an owner-authorised `RECORD_ONLY_CLOSEOUT` documentation
PR from current canonical `main`; records already present remain immutable.

For `PRIVATE_REVIEW_BUNDLE`, public records must remain non-reconstructive. They may report aggregate
roots, counts, coverage, stable finding IDs, dispositions, and retention or destruction status, but
must not expose hidden item IDs, per-item hashes, prompts, answers, oracle labels, renders, licensed
content, or metadata that permits reconstruction.
