# Research-software review protocol

This tool-neutral protocol separates implementation, engineering review, scientific review, ownership, and closeout. Every approval or review disposition applies only to the exact pull request and head commit it names.

## Roles

### `OWNER_PI`

The owner/PI determines scientific scope, resolves substantive design decisions, accepts or rejects review findings, approves an exact pull-request number and head SHA, and authorises merge, tag, release, or gate advancement. No implementation or review agent may manufacture owner approval.

### `IMPLEMENTATION_AGENT`

The implementation agent implements a bounded authorised work package, creates tests and evidence, records assumptions and limitations, and leaves the pull request open for review. Its terminal status is only:

```text
IMPLEMENTED_PENDING_REVIEW
```

It may self-check but may not approve its own implementation.

### `ENGINEERING_REVIEWER`

The engineering reviewer operates in a separate session from the implementation agent and initially makes no edits. It assesses specification compliance, code correctness, validation independence, failure behaviour, reproducibility, security and unsafe-path handling, portability, test sufficiency, and scope discipline.

Allowed dispositions are:

```text
ENGINEERING_PASS
ENGINEERING_REQUEST_CHANGES
ENGINEERING_BLOCKED
```

Every disposition names the exact pull request and head SHA reviewed.

### `SCIENTIFIC_REVIEWER`

The scientific reviewer remains independent of the implementation changes under review. It assesses hypothesis validity, construct validity, privileged-information leakage, label provenance, causal and representational confounds, fairness of future comparisons, falsifiability, conceptual drift, and interpretation boundaries.

Allowed dispositions are:

```text
SCIENTIFIC_PASS
SCIENTIFIC_REQUEST_CHANGES
SCIENTIFIC_INCONCLUSIVE
SCIENTIFIC_NO_GO
```

Every disposition names the exact pull request and head SHA reviewed.

### `CORRECTION_IMPLEMENTER`

The correction implementer may address owner-accepted findings. It reports exactly one of these statuses for every stable finding ID:

```text
IMPLEMENTED_PENDING_VERIFICATION
NOT_IMPLEMENTED_WITH_REASON
OUT_OF_SCOPE_PENDING_OWNER
```

It may self-check but may not mark its own response `VERIFIED`, passed, resolved, or independently approved.

### `CLOSEOUT_AGENT`

The closeout agent operates in a separate session after review and exact-head owner approval. It revalidates the exact approved SHA, clean branch state, CI success, mergeability, unresolved findings, authorised scope, and approval validity. It may merge, tag, or release only when explicitly authorised.

## Permanent separation rules

1. The implementation session cannot serve as the independent engineering reviewer.
2. The correction implementer cannot mark its own findings verified.
3. Scientific review does not substitute for engineering review.
4. Engineering review does not determine scientific validity.
5. Any change to the pull-request head invalidates earlier exact-head approval until the relevant reviews are renewed.
6. Review findings use stable identifiers.
7. Review dispositions identify the exact pull request and head SHA.
8. Owner approval identifies the exact pull request and head SHA.
9. Gate advancement is separate from successful CI.
10. Repository infrastructure success is not a scientific result.

Review records are append-only evidence for the revision they name. Correction responses live in separate files so the original findings and dispositions are not rewritten.
