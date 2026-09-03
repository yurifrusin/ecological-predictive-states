"""Exercise fail-closed validation against complete freeze-candidate evidence."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

from epsbench.freeze import (
    SOURCE_IDENTITY_FIELDS,
    _domain_hash,
    _receipt_context,
    _root_domains,
    validate_freeze_audit,
)
from epsbench.utils.canonical import canonical_json_bytes, sha256_bytes


def _expect_rejected(
    label: str,
    validator: Callable[[], Any],
    expected_message: str | None = None,
) -> None:
    try:
        validator()
    except ValueError as error:
        if expected_message is not None and expected_message not in str(error):
            raise AssertionError(
                f"adversarial case {label!r} reached the wrong rejection: {error}"
            ) from error
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


@contextmanager
def _unavailable_directory(path: Path) -> Iterator[None]:
    moved = path.with_name(f"{path.name}.temporarily-unavailable")
    path.rename(moved)
    try:
        yield
    finally:
        moved.rename(path)


@contextmanager
def _swap_directories(first: Path, second: Path) -> Iterator[None]:
    temporary = first.with_name(f"{first.name}.swap-temporary")
    first.rename(temporary)
    second.rename(first)
    temporary.rename(second)
    try:
        yield
    finally:
        second.rename(temporary)
        first.rename(second)
        temporary.rename(first)


def _rehash_packet(packet: dict[str, Any]) -> None:
    logical = dict(packet)
    logical.pop("complete_packet_root_sha256")
    packet["complete_packet_root_sha256"] = _domain_hash("complete_packet", logical)


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
    parser.add_argument("--counterpart-evidence", type=Path)
    args = parser.parse_args()
    root = args.packet

    def validate() -> dict[str, Any]:
        return validate_freeze_audit(root, args.counterpart_evidence)

    validate()
    packet_path = root / "freeze_candidate_packet.json"
    packet = json.loads(packet_path.read_text(encoding="utf-8"))

    noncanonical = (json.dumps(packet, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with _replace(packet_path, noncanonical):
        _expect_rejected("noncanonical packet envelope", validate)

    matrix_path = root / "selected_profile_matrix.json"
    original_matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    control_path = root / "legacy_control_matrix.json"
    original_controls = json.loads(control_path.read_text(encoding="utf-8"))
    matrix = json.loads(json.dumps(original_matrix))
    controls = json.loads(json.dumps(original_controls))
    matrix["cells"][0]["benchmark_role"] = "held_out_colour_ood"
    matrix_bytes, packet_bytes = _replace_report(packet, "selected_profile_matrix.json", matrix)
    with _replace(matrix_path, matrix_bytes), _replace(packet_path, packet_bytes):
        _expect_rejected("fully rehashed role corruption", validate)

    matrix = json.loads(json.dumps(original_matrix))
    matrix["cells"][0]["candidate_seed"] = 1
    matrix_bytes, packet_bytes = _replace_report(packet, "selected_profile_matrix.json", matrix)
    with _replace(matrix_path, matrix_bytes), _replace(packet_path, packet_bytes):
        _expect_rejected("fully rehashed cell corruption", validate)

    matrix = json.loads(json.dumps(original_matrix))
    matrix["cells"][0]["benchmark_pair_checks"]["same_scene_and_evaluation_root"] = False
    matrix_bytes, packet_bytes = _replace_report(packet, "selected_profile_matrix.json", matrix)
    with _replace(matrix_path, matrix_bytes), _replace(packet_path, packet_bytes):
        _expect_rejected("fully rehashed pair corruption", validate)

    matrix = json.loads(json.dumps(original_matrix))
    matrix["cells"][0]["source_identity"]["sampled_geometry_identity_sha256"] = "0" * 64
    identity_roots = {
        field: matrix["cells"][0]["source_identity"][field] for field in SOURCE_IDENTITY_FIELDS
    }
    matrix["cells"][0]["source_identity"]["source_identity_root_sha256"] = _domain_hash(
        "source_identity_root", identity_roots
    )
    matrix_bytes, packet_bytes = _replace_report(packet, "selected_profile_matrix.json", matrix)
    with _replace(matrix_path, matrix_bytes), _replace(packet_path, packet_bytes):
        _expect_rejected(
            "fully rehashed mutually consistent source-identity claim",
            validate,
        )

    legacy_fields = (
        "geometry_sha256",
        "camera_trajectory_sha256",
        "action_sha256",
        "opaque_remapping_sha256",
        "scene_content_sha256",
        "analytic_transport_sha256",
        "oriented_boundary_sha256",
        "visibility_event_sha256",
        "occlusion_sha256",
    )
    matrix = json.loads(json.dumps(original_matrix))
    matrix["cells"][0][legacy_fields[0]] = "0" * 64
    matrix_bytes, packet_bytes = _replace_report(packet, "selected_profile_matrix.json", matrix)
    with _replace(matrix_path, matrix_bytes), _replace(packet_path, packet_bytes):
        _expect_rejected(
            "one fully resealed legacy source-identity substitution",
            validate,
            "legacy source claim differs from retained source evidence",
        )

    matrix = json.loads(json.dumps(original_matrix))
    for field in legacy_fields:
        matrix["cells"][0][field] = "0" * 64
    matrix_bytes, packet_bytes = _replace_report(packet, "selected_profile_matrix.json", matrix)
    with _replace(matrix_path, matrix_bytes), _replace(packet_path, packet_bytes):
        _expect_rejected(
            "all-nine fully resealed legacy source-identity substitutions",
            validate,
            "legacy source claim differs from retained source evidence",
        )

    matrix = json.loads(json.dumps(original_matrix))
    dependency = {(cell["scene_family"], cell["candidate_seed"]) for cell in matrix["cells"][:1]}
    assert len(dependency) == 1
    scene, evaluation_root = next(iter(dependency))
    group = [
        cell
        for cell in [*matrix["cells"], *controls["cells"]]
        if cell["scene_family"] == scene and cell["candidate_seed"] == evaluation_root
    ]
    assert len(group) == 6
    for cell in group:
        for field in legacy_fields:
            cell[field] = "1" * 64
    matrix_bytes = canonical_json_bytes(matrix) + b"\n"
    control_bytes = canonical_json_bytes(controls) + b"\n"
    changed_packet = json.loads(json.dumps(packet))
    changed_packet["report_file_sha256"]["selected_profile_matrix.json"] = sha256_bytes(
        matrix_bytes
    )
    changed_packet["report_file_sha256"]["legacy_control_matrix.json"] = sha256_bytes(control_bytes)
    _rehash_packet(changed_packet)
    with (
        _replace(matrix_path, matrix_bytes),
        _replace(control_path, control_bytes),
        _replace(packet_path, canonical_json_bytes(changed_packet) + b"\n"),
    ):
        _expect_rejected(
            "complete matched-control dependency-group legacy reseal",
            validate,
            "legacy source claim differs from retained source evidence",
        )

    matrix = json.loads(json.dumps(original_matrix))
    controls = json.loads(json.dumps(original_controls))
    complete_group = [
        cell
        for cell in [*matrix["cells"], *controls["cells"]]
        if cell["scene_family"] == scene and cell["candidate_seed"] == evaluation_root
    ]
    assert len(complete_group) == 6
    for cell in complete_group:
        for field in SOURCE_IDENTITY_FIELDS:
            cell["source_identity"][field] = "f" * 64
        identity_domain = {
            field: cell["source_identity"][field] for field in SOURCE_IDENTITY_FIELDS
        }
        cell["source_identity"]["source_identity_root_sha256"] = _domain_hash(
            "source_identity_root", identity_domain
        )
        for field in legacy_fields:
            cell[field] = "e" * 64
    definition_model, seed_model, lock_model = _receipt_context(root)
    contact_manifest = json.loads(
        (root / "contact_sheet_manifest.json").read_text(encoding="utf-8")
    )
    definition_roots, apparatus_roots, renderer_roots = _root_domains(
        definition_model,
        seed_model,
        lock_model,
        matrix["cells"],
        controls["cells"],
        contact_manifest,
    )
    matrix_bytes = canonical_json_bytes(matrix) + b"\n"
    control_bytes = canonical_json_bytes(controls) + b"\n"
    fully_resealed_packet = json.loads(json.dumps(packet))
    fully_resealed_packet["report_file_sha256"]["selected_profile_matrix.json"] = sha256_bytes(
        matrix_bytes
    )
    fully_resealed_packet["report_file_sha256"]["legacy_control_matrix.json"] = sha256_bytes(
        control_bytes
    )
    fully_resealed_packet["portable_definition_roots"] = definition_roots
    fully_resealed_packet["portable_apparatus_roots"] = apparatus_roots
    fully_resealed_packet["renderer_local_roots"] = renderer_roots
    _rehash_packet(fully_resealed_packet)
    with (
        _replace(matrix_path, matrix_bytes),
        _replace(control_path, control_bytes),
        _replace(
            packet_path,
            canonical_json_bytes(fully_resealed_packet) + b"\n",
        ),
    ):
        _expect_rejected(
            "fully recomputed matrix, pair, portable-root, renderer-root, and packet reseal",
            validate,
            "source identity differs from independent reconstruction",
        )

    matrix = json.loads(json.dumps(original_matrix))
    matrix["cells"][0]["evidence"]["before"]["depth"]["path"] = "../escaped-depth.npy"
    matrix_bytes, packet_bytes = _replace_report(packet, "selected_profile_matrix.json", matrix)
    with _replace(matrix_path, matrix_bytes), _replace(packet_path, packet_bytes):
        _expect_rejected("fully rehashed evidence path escape", validate)

    matrix = json.loads(json.dumps(original_matrix))
    matrix["cells"][1]["cell_id"] = matrix["cells"][0]["cell_id"]
    matrix_bytes, packet_bytes = _replace_report(packet, "selected_profile_matrix.json", matrix)
    with _replace(matrix_path, matrix_bytes), _replace(packet_path, packet_bytes):
        _expect_rejected("fully rehashed duplicate cell identity", validate)

    readiness_path = root / "profile_readiness_summary.json"
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    readiness["profiles"][0]["cross_renderer_apparatus_qualified"] = False
    readiness_bytes, packet_bytes = _replace_report(
        packet, "profile_readiness_summary.json", readiness
    )
    with _replace(readiness_path, readiness_bytes), _replace(packet_path, packet_bytes):
        _expect_rejected("fully rehashed readiness corruption", validate)

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
        _expect_rejected("fully rehashed seed corruption", validate)

    lock_path = root / "freeze_definition_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["expected_total_cell_count"] = 191
    lock_bytes, packet_bytes = _replace_snapshot(packet, "freeze_definition_lock.json", lock)
    with _replace(lock_path, lock_bytes), _replace(packet_path, packet_bytes):
        _expect_rejected("fully rehashed lock corruption", validate)

    definition_path = root / "benchmark_definition_snapshot.json"
    definition = json.loads(definition_path.read_text(encoding="utf-8"))
    definition["evaluation_use_policy"]["training_on_final_evaluation_roots_prohibited"] = False
    definition_bytes, packet_bytes = _replace_snapshot(
        packet, "benchmark_definition_snapshot.json", definition
    )
    with _replace(definition_path, definition_bytes), _replace(packet_path, packet_bytes):
        _expect_rejected("fully rehashed policy corruption", validate)

    changed_packet = json.loads(json.dumps(packet))
    changed_packet["portable_definition_roots"]["benchmark_definition_sha256"] = "0" * 64
    _rehash_packet(changed_packet)
    with _replace(packet_path, canonical_json_bytes(changed_packet) + b"\n"):
        _expect_rejected("fully rehashed root corruption", validate)

    for field, value in (
        ("benchmark_frozen", True),
        ("model_protocol_frozen", True),
        ("primary_model_result_renderer", "windows_wgl_locked"),
        ("full_gate_0b_complete", True),
        ("scientific_result", "fabricated_result"),
    ):
        changed_packet = json.loads(json.dumps(packet))
        changed_packet[field] = value
        _rehash_packet(changed_packet)
        with _replace(packet_path, canonical_json_bytes(changed_packet) + b"\n"):
            _expect_rejected(
                f"fabricated authority: {field}",
                validate,
                "freeze packet exceeds implementation authority",
            )

    changed_packet = json.loads(json.dumps(packet))
    changed_packet["schema_version"] = "appearance_benchmark_freeze_candidate_v0"
    _rehash_packet(changed_packet)
    with _replace(packet_path, canonical_json_bytes(changed_packet) + b"\n"):
        _expect_rejected(
            "historical-v0 packet identity substitution",
            validate,
            "freeze packet exceeds implementation authority",
        )

    changed_packet = json.loads(json.dumps(packet))
    changed_packet["root_schema_version"] = "unknown_root_domain"
    _rehash_packet(changed_packet)
    with _replace(packet_path, canonical_json_bytes(changed_packet) + b"\n"):
        _expect_rejected(
            "unknown packet root-domain substitution",
            validate,
            "freeze packet exceeds implementation authority",
        )

    invalid_run = (json.dumps({"arbitrary": "volatile"}, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    with _replace(root / "run.json", invalid_run):
        _expect_rejected("unstructured run metadata", validate)

    with tempfile.TemporaryDirectory(prefix=".freeze-adversary-", dir=root.parent) as temporary:
        scratch = Path(temporary)
        with _hardlink(root / "run.json", scratch / "run-hardlink.json"):
            _expect_rejected("hard-linked run metadata", validate)

        contact_manifest = json.loads(
            (root / "contact_sheet_manifest.json").read_text(encoding="utf-8")
        )
        contact_path = root / contact_manifest["sheets"][0]["path"]
        with _hardlink(contact_path, scratch / "contact-hardlink.png"):
            _expect_rejected("hard-linked contact sheet", validate)

    source_cells = original_matrix["cells"]
    baseline = source_cells[0]

    def source_root(cell: dict[str, Any]) -> Path:
        return root / cast(str, cell["source_evidence"]["dataset_path"])

    wrong_root = next(
        cell
        for cell in source_cells
        if cell["scene_family"] == baseline["scene_family"]
        and cell["profile_id"] == baseline["profile_id"]
        and cell["candidate_seed"] != baseline["candidate_seed"]
    )
    wrong_profile = next(
        cell
        for cell in source_cells
        if cell["scene_family"] == baseline["scene_family"]
        and cell["profile_id"] != baseline["profile_id"]
        and cell["candidate_seed"] == baseline["candidate_seed"]
    )
    wrong_scene = next(
        cell
        for cell in source_cells
        if cell["scene_family"] != baseline["scene_family"]
        and cell["profile_id"] == baseline["profile_id"]
        and cell["candidate_seed"] == baseline["candidate_seed"]
    )
    with _unavailable_directory(source_root(baseline)):
        _expect_rejected("missing retained source evidence", validate)
    additional = source_root(baseline) / "additional-source-evidence"
    additional.write_bytes(b"not declared")
    try:
        _expect_rejected("additional retained source evidence", validate)
    finally:
        additional.unlink()
    manifest_path = source_root(baseline) / "manifest.json"
    with _replace(manifest_path, b"{}\n"):
        _expect_rejected("altered retained source evidence", validate)
    for label, substitute in (
        ("wrong-root retained source evidence", wrong_root),
        ("wrong-profile retained source evidence", wrong_profile),
        ("wrong-scene retained source evidence", wrong_scene),
    ):
        with _swap_directories(source_root(baseline), source_root(substitute)):
            _expect_rejected(label, validate)

    unexpected = root / "unexpected"
    unexpected.mkdir()
    try:
        _expect_rejected("extra packet tree entry", validate)
    finally:
        unexpected.rmdir()

    validate()
    print("all freeze complete-packet adversarial regressions passed", flush=True)


if __name__ == "__main__":
    main()
