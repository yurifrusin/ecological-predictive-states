"""Exercise fail-closed validation against complete freeze-candidate evidence."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from epsbench.freeze import validate_freeze_audit
from epsbench.utils.canonical import canonical_json_bytes, sha256_bytes


def _expect_rejected(label: str, validator: Callable[[], Any]) -> None:
    try:
        validator()
    except ValueError:
        print(f"adversarial case rejected: {label}", flush=True)
        return
    raise AssertionError(f"adversarial case was accepted: {label}")


@contextmanager
def _replace(path: Path, replacement: bytes) -> Iterator[None]:
    original = path.read_bytes()
    path.write_bytes(replacement)
    try:
        yield
    finally:
        path.write_bytes(original)


@contextmanager
def _hardlink(path: Path, external: Path) -> Iterator[None]:
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


def _rehash_packet(packet: dict[str, Any]) -> None:
    logical = dict(packet)
    logical.pop("complete_packet_root_sha256")
    packet["complete_packet_root_sha256"] = sha256_bytes(canonical_json_bytes(logical))


def _replace_report(
    packet: dict[str, Any],
    name: str,
    report: dict[str, Any],
) -> tuple[bytes, bytes]:
    report_bytes = canonical_json_bytes(report) + b"\n"
    changed_packet = json.loads(json.dumps(packet))
    changed_packet["report_file_sha256"][name] = sha256_bytes(report_bytes)
    _rehash_packet(changed_packet)
    return report_bytes, canonical_json_bytes(changed_packet) + b"\n"


def _replace_snapshot(
    packet: dict[str, Any],
    name: str,
    snapshot: dict[str, Any],
) -> tuple[bytes, bytes]:
    snapshot_bytes = canonical_json_bytes(snapshot) + b"\n"
    changed_packet = json.loads(json.dumps(packet))
    changed_packet["snapshot_file_sha256"][name] = sha256_bytes(snapshot_bytes)
    _rehash_packet(changed_packet)
    return snapshot_bytes, canonical_json_bytes(changed_packet) + b"\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("packet", type=Path)
    args = parser.parse_args()
    root = args.packet
    validate_freeze_audit(root)
    packet_path = root / "freeze_candidate_packet.json"
    packet = json.loads(packet_path.read_text(encoding="utf-8"))

    noncanonical = (json.dumps(packet, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with _replace(packet_path, noncanonical):
        _expect_rejected("noncanonical packet envelope", lambda: validate_freeze_audit(root))

    matrix_path = root / "selected_profile_matrix.json"
    original_matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    matrix = json.loads(json.dumps(original_matrix))
    matrix["cells"][0]["benchmark_role"] = "held_out_colour_ood"
    matrix_bytes, packet_bytes = _replace_report(packet, "selected_profile_matrix.json", matrix)
    with _replace(matrix_path, matrix_bytes), _replace(packet_path, packet_bytes):
        _expect_rejected("fully rehashed role corruption", lambda: validate_freeze_audit(root))

    matrix = json.loads(json.dumps(original_matrix))
    matrix["cells"][0]["candidate_seed"] = 1
    matrix_bytes, packet_bytes = _replace_report(packet, "selected_profile_matrix.json", matrix)
    with _replace(matrix_path, matrix_bytes), _replace(packet_path, packet_bytes):
        _expect_rejected("fully rehashed cell corruption", lambda: validate_freeze_audit(root))

    matrix = json.loads(json.dumps(original_matrix))
    matrix["cells"][0]["benchmark_pair_checks"]["same_scene_and_evaluation_root"] = False
    matrix_bytes, packet_bytes = _replace_report(packet, "selected_profile_matrix.json", matrix)
    with _replace(matrix_path, matrix_bytes), _replace(packet_path, packet_bytes):
        _expect_rejected("fully rehashed pair corruption", lambda: validate_freeze_audit(root))

    matrix = json.loads(json.dumps(original_matrix))
    matrix["cells"][0]["evidence"]["before"]["depth"]["path"] = "../escaped-depth.npy"
    matrix_bytes, packet_bytes = _replace_report(packet, "selected_profile_matrix.json", matrix)
    with _replace(matrix_path, matrix_bytes), _replace(packet_path, packet_bytes):
        _expect_rejected("fully rehashed evidence path escape", lambda: validate_freeze_audit(root))

    matrix = json.loads(json.dumps(original_matrix))
    matrix["cells"][1]["cell_id"] = matrix["cells"][0]["cell_id"]
    matrix_bytes, packet_bytes = _replace_report(packet, "selected_profile_matrix.json", matrix)
    with _replace(matrix_path, matrix_bytes), _replace(packet_path, packet_bytes):
        _expect_rejected(
            "fully rehashed duplicate cell identity", lambda: validate_freeze_audit(root)
        )

    readiness_path = root / "profile_readiness_summary.json"
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    readiness["profiles"][0]["cross_renderer_apparatus_qualified"] = False
    readiness_bytes, packet_bytes = _replace_report(
        packet, "profile_readiness_summary.json", readiness
    )
    with _replace(readiness_path, readiness_bytes), _replace(packet_path, packet_bytes):
        _expect_rejected("fully rehashed readiness corruption", lambda: validate_freeze_audit(root))

    seed_path = root / "evaluation_episode_seed_registry_snapshot.json"
    seed_snapshot = json.loads(seed_path.read_text(encoding="utf-8"))
    seed_snapshot["candidate_episode_seeds"][0] = 1
    seed_bytes = canonical_json_bytes(seed_snapshot) + b"\n"
    changed_packet = json.loads(json.dumps(packet))
    changed_packet["snapshot_file_sha256"][seed_path.name] = sha256_bytes(seed_bytes)
    _rehash_packet(changed_packet)
    with (
        _replace(seed_path, seed_bytes),
        _replace(packet_path, canonical_json_bytes(changed_packet) + b"\n"),
    ):
        _expect_rejected("fully rehashed seed corruption", lambda: validate_freeze_audit(root))

    lock_path = root / "freeze_definition_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["expected_total_cell_count"] = 191
    lock_bytes, packet_bytes = _replace_snapshot(packet, "freeze_definition_lock.json", lock)
    with _replace(lock_path, lock_bytes), _replace(packet_path, packet_bytes):
        _expect_rejected("fully rehashed lock corruption", lambda: validate_freeze_audit(root))

    definition_path = root / "benchmark_definition_snapshot.json"
    definition = json.loads(definition_path.read_text(encoding="utf-8"))
    definition["evaluation_use_policy"]["training_on_final_evaluation_roots_prohibited"] = False
    definition_bytes, packet_bytes = _replace_snapshot(
        packet, "benchmark_definition_snapshot.json", definition
    )
    with _replace(definition_path, definition_bytes), _replace(packet_path, packet_bytes):
        _expect_rejected("fully rehashed policy corruption", lambda: validate_freeze_audit(root))

    changed_packet = json.loads(json.dumps(packet))
    changed_packet["portable_definition_roots"]["benchmark_definition_sha256"] = "0" * 64
    _rehash_packet(changed_packet)
    with _replace(packet_path, canonical_json_bytes(changed_packet) + b"\n"):
        _expect_rejected("fully rehashed root corruption", lambda: validate_freeze_audit(root))

    invalid_run = (json.dumps({"arbitrary": "volatile"}, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    with _replace(root / "run.json", invalid_run):
        _expect_rejected("unstructured run metadata", lambda: validate_freeze_audit(root))

    with tempfile.TemporaryDirectory(prefix=".freeze-adversary-", dir=root.parent) as temporary:
        scratch = Path(temporary)
        with _hardlink(root / "run.json", scratch / "run-hardlink.json"):
            _expect_rejected("hard-linked run metadata", lambda: validate_freeze_audit(root))

        contact_manifest = json.loads(
            (root / "contact_sheet_manifest.json").read_text(encoding="utf-8")
        )
        contact_path = root / contact_manifest["sheets"][0]["path"]
        with _hardlink(contact_path, scratch / "contact-hardlink.png"):
            _expect_rejected("hard-linked contact sheet", lambda: validate_freeze_audit(root))

    unexpected = root / "unexpected"
    unexpected.mkdir()
    try:
        _expect_rejected("extra packet tree entry", lambda: validate_freeze_audit(root))
    finally:
        unexpected.rmdir()

    validate_freeze_audit(root)
    print("all freeze complete-packet adversarial regressions passed", flush=True)


if __name__ == "__main__":
    main()
