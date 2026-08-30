"""Exercise fail-closed validation against complete generated Revision 1 evidence."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from epsbench.audit import _hash_json
from epsbench.failure_analysis import _markdown, validate_failure_analysis
from epsbench.revision import validate_revision_audit
from epsbench.utils.canonical import canonical_json_bytes


def _expect_rejected(label: str, validator: Callable[[], Any]) -> None:
    try:
        validator()
    except ValueError:
        print(f"adversarial case rejected: {label}", flush=True)
        return
    raise AssertionError(f"adversarial case was accepted: {label}")


@contextmanager
def _replace_bytes(path: Path, replacement: bytes) -> Iterator[None]:
    original = path.read_bytes()
    path.write_bytes(replacement)
    try:
        yield
    finally:
        path.write_bytes(original)


@contextmanager
def _replace_with_hardlink(path: Path, external: Path) -> Iterator[None]:
    original = path.read_bytes()
    external.write_bytes(original)
    path.unlink()
    os.link(external, path)
    try:
        yield
    finally:
        path.unlink()
        path.write_bytes(original)
        external.unlink()


@contextmanager
def _replace_with_symlink(path: Path, external: Path) -> Iterator[bool]:
    original = path.read_bytes()
    external.write_bytes(original)
    path.unlink()
    try:
        path.symlink_to(external.resolve())
    except OSError:
        path.write_bytes(original)
        external.unlink()
        yield False
        return
    try:
        yield True
    finally:
        path.unlink()
        path.write_bytes(original)
        external.unlink()


def _rehash_revision_packet(packet: dict[str, Any]) -> None:
    logical = dict(packet)
    logical.pop("complete_packet_root_sha256")
    packet["complete_packet_root_sha256"] = _hash_json(logical)


def _rehash_failure_analysis(analysis: dict[str, Any]) -> None:
    renderer = dict(analysis)
    renderer.pop("renderer_specific_failure_analysis_sha256")
    analysis["renderer_specific_failure_analysis_sha256"] = _hash_json(renderer)


def _revision_provenance_regressions(revision_root: Path) -> None:
    packet_path = revision_root / "revision1_candidate_packet.json"
    original = json.loads(packet_path.read_text(encoding="utf-8"))

    altered_lock = json.loads(json.dumps(original))
    altered_lock["candidate_definition_lock_commit"] = altered_lock["source_provenance"][
        "git_commit"
    ]
    _rehash_revision_packet(altered_lock)
    with _replace_bytes(packet_path, canonical_json_bytes(altered_lock) + b"\n"):
        _expect_rejected(
            "alternate definition-lock ancestor", lambda: validate_revision_audit(revision_root)
        )

    forged_source = json.loads(json.dumps(original))
    forged_source["source_provenance"]["git_commit"] = forged_source[
        "candidate_definition_lock_commit"
    ]
    forged_source["source_provenance"]["git_dirty"] = False
    forged_source["source_provenance"]["dirty_diff_sha256"] = None
    _rehash_revision_packet(forged_source)
    with _replace_bytes(packet_path, canonical_json_bytes(forged_source) + b"\n"):
        _expect_rejected(
            "forged qualification source provenance", lambda: validate_revision_audit(revision_root)
        )


def _failure_content_regressions(analysis_root: Path, baseline_root: Path) -> None:
    analysis_path = analysis_root / "baseline_failure_analysis.json"
    markdown_path = analysis_root / "baseline_failure_analysis.md"
    original = json.loads(analysis_path.read_text(encoding="utf-8"))

    forged_renderer = json.loads(json.dumps(original))
    forged_renderer["renderer_fingerprints"] = ['{"renderer":"forged"}']
    _rehash_failure_analysis(forged_renderer)
    with _replace_bytes(analysis_path, canonical_json_bytes(forged_renderer) + b"\n"):
        _expect_rejected(
            "fully rehashed renderer fingerprint",
            lambda: validate_failure_analysis(analysis_root, source_packet=baseline_root),
        )

    forged_causal = json.loads(json.dumps(original))
    forged_causal["causal_diagnostic_answers"][0]["answer"] = "forged causal answer"
    _rehash_failure_analysis(forged_causal)
    with (
        _replace_bytes(analysis_path, canonical_json_bytes(forged_causal) + b"\n"),
        _replace_bytes(markdown_path, _markdown(forged_causal).encode("utf-8")),
    ):
        _expect_rejected(
            "fully rehashed causal answer",
            lambda: validate_failure_analysis(analysis_root, source_packet=baseline_root),
        )


def _failure_path_regressions(
    analysis_root: Path,
    baseline_root: Path,
    scratch: Path,
) -> None:
    def validator() -> None:
        validate_failure_analysis(analysis_root, source_packet=baseline_root)

    json_names = (
        "baseline_failure_analysis.json",
        "profile_failure_matrix.json",
        "surface_failure_matrix.json",
        "threshold_margin_summary.json",
    )
    for name in json_names:
        path = analysis_root / name
        payload = json.loads(path.read_text(encoding="utf-8"))
        noncanonical = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
        with _replace_bytes(path, noncanonical):
            _expect_rejected(f"noncanonical failure-analysis JSON: {name}", validator)

    for index, name in enumerate((*json_names, "baseline_failure_analysis.md")):
        path = analysis_root / name
        external = scratch / f"failure-hardlink-{index}"
        with _replace_with_hardlink(path, external):
            _expect_rejected(f"hard-linked failure-analysis artifact: {name}", validator)

    unexpected = analysis_root / "unexpected"
    unexpected.mkdir()
    try:
        _expect_rejected("additional failure-analysis directory", validator)
    finally:
        unexpected.rmdir()

    root_alias = scratch / "failure-analysis-root-link"
    try:
        root_alias.symlink_to(analysis_root.resolve(), target_is_directory=True)
    except OSError:
        print("adversarial case not exercised: root symlink unavailable", flush=True)
    else:
        try:
            _expect_rejected(
                "failure-analysis root link",
                lambda: validate_failure_analysis(root_alias, source_packet=baseline_root),
            )
        finally:
            root_alias.unlink()


def _revision_tree_regressions(revision_root: Path, scratch: Path) -> None:
    def validator() -> None:
        validate_revision_audit(revision_root)

    packet_path = revision_root / "revision1_candidate_packet.json"
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    noncanonical = (json.dumps(packet, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with _replace_bytes(packet_path, noncanonical):
        _expect_rejected("noncanonical revision packet JSON", validator)

    first_contact = revision_root / packet["contact_sheet_manifest"]["sheets"][0]["path"]
    with _replace_with_symlink(first_contact, scratch / "contact-target.png") as exercised:
        if exercised:
            _expect_rejected("linked Revision 1 contact sheet", validator)
        else:
            print("adversarial case not exercised: file symlink unavailable", flush=True)

    run_path = revision_root / "run.json"
    with _replace_with_hardlink(run_path, scratch / "run-hardlink.json"):
        _expect_rejected("hard-linked Revision 1 run metadata", validator)

    invalid_run = (json.dumps({"arbitrary": "volatile"}, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    with _replace_bytes(run_path, invalid_run):
        _expect_rejected("unstructured Revision 1 run metadata", validator)

    unexpected = revision_root / "unexpected"
    unexpected.mkdir()
    try:
        _expect_rejected("additional Revision 1 packet directory", validator)
    finally:
        unexpected.rmdir()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-packet", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--revision-packet", type=Path, required=True)
    args = parser.parse_args()
    validate_failure_analysis(args.analysis, source_packet=args.baseline_packet)
    validate_revision_audit(args.revision_packet)
    with tempfile.TemporaryDirectory(
        prefix=".epsbench-revision1-adversaries-",
        dir=args.revision_packet.parent,
    ) as directory:
        scratch = Path(directory)
        _revision_provenance_regressions(args.revision_packet)
        _failure_content_regressions(args.analysis, args.baseline_packet)
        _failure_path_regressions(args.analysis, args.baseline_packet, scratch)
        _revision_tree_regressions(args.revision_packet, scratch)
    validate_failure_analysis(args.analysis, source_packet=args.baseline_packet)
    validate_revision_audit(args.revision_packet)
    print("all complete-packet adversarial regressions passed", flush=True)


if __name__ == "__main__":
    main()
