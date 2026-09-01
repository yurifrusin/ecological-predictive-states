"""Check bounded PR #17 status language and the preserved replacement-lock boundary."""

from __future__ import annotations

import subprocess
from pathlib import Path

DOCUMENTS = (
    Path("README.md"),
    Path("PROJECT_HISTORY.md"),
    Path("RESEARCH_LOG.md"),
    Path("docs/GATE_0B_APPEARANCE_BENCHMARK_FREEZE_V0.md"),
    Path("docs/IMPLEMENTATION_NOTES.md"),
)

STATUS_MARKERS = (
    "`EXACT_HEAD_REQUALIFICATION_RESOLVED_FROM_EXTERNAL_CI_EVIDENCE`",
    "`EXTERNAL_TO_SOURCE_COMMIT`",
    "`SECOND_CORRECTION_SOURCE_CANDIDATE_PENDING_INDEPENDENT_REVIEW`",
    "`RENEWED_EXACT_HEAD_DUAL_REVIEW_REQUIRED`",
    "`SECOND_CORRECTION_AUTHORISED; IMPLEMENTATION_MERGE_APPROVAL_NOT_GIVEN`",
    "`NOT_PERFORMED`",
    "`NOT_FROZEN`",
    "`NOT_ADVANCED`",
    "`NONE`",
)

LOCKED_INPUTS = (
    Path("configs/appearance_benchmark_freeze_v0.yaml"),
    Path("configs/appearance_benchmark_v0_evaluation_episode_seeds.yaml"),
    Path("configs/appearance_benchmark_freeze_v0_lock.json"),
    Path("configs/appearance_candidates_revision1.yaml"),
    Path("configs/benchmark_v0.yaml"),
    Path("configs/corridor_v0.yaml"),
)

REPLACEMENT_LOCK_COMMIT = "d4072f912cc58bbc1ca41ceb2652e41783dbf3e1"
HISTORICAL_REVIEW_HEAD = "62e09a920c10d50c62643543f6ebcbd26570b1d4"


def main() -> None:
    for path in DOCUMENTS:
        text = path.read_text(encoding="utf-8")
        missing = [marker for marker in STATUS_MARKERS if marker not in text]
        if missing:
            raise AssertionError(f"{path} lacks current-status markers: {missing}")
        if HISTORICAL_REVIEW_HEAD not in text or "historical" not in text.lower():
            raise AssertionError(f"{path} does not distinguish historical exact-head evidence")
        status_start = text.rfind("| Apparatus status |")
        if status_start < 0:
            raise AssertionError(f"{path} lacks a bounded current-status table")
        status_end = text.find("\n\n", status_start)
        current = text[status_start : status_end if status_end >= 0 else None]
        for forbidden in (
            "ENGINEERING_PASS",
            "SCIENTIFIC_PASS",
            "MERGE_CONVERGENCE",
            "OWNER_APPROVED",
            "BENCHMARK_FROZEN",
            "MODEL_PROTOCOL_FROZEN",
            "SCIENTIFIC_RESULT",
        ):
            if forbidden in current:
                raise AssertionError(f"{path} fabricates current authority: {forbidden}")

    for path in LOCKED_INPUTS:
        committed = subprocess.run(
            ["git", "show", f"{REPLACEMENT_LOCK_COMMIT}:{path.as_posix()}"],
            capture_output=True,
            check=True,
        ).stdout
        if path.read_bytes() != committed:
            raise AssertionError(f"owner-preserved locked input changed: {path}")

    obsolete_receipt = Path("configs/appearance_benchmark_freeze_v0_wgl_qualification_receipt.json")
    if obsolete_receipt.exists():
        raise AssertionError("receipt-only counterpart evidence must not remain authoritative")
    print("freeze documentation and non-advancement status are consistent", flush=True)


if __name__ == "__main__":
    main()
