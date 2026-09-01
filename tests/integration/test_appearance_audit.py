from __future__ import annotations

import json
import os
import shutil
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

import epsbench.audit as audit
from epsbench.appearance import (
    appearance_profile_hash,
    load_appearance_registry,
    load_evaluation_seed_registry,
    resolve_appearance,
    seed_registry_hash,
)
from epsbench.data.paths import open_owned_regular_file, sha256_open_file
from epsbench.utils.canonical import (
    logical_array_hash,
    sha256_file,
    write_canonical_json,
)
from epsbench.utils.seeding import derive_seed

REPORT_NAMES = ("profile_summary.json", "seed_matrix.json", "negative_evidence.json")


def _write_rehashed_packet(
    packet_root: Path,
    packet: dict[str, Any] | None = None,
) -> None:
    packet_path = packet_root / "candidate_packet.json"
    if packet is None:
        packet = json.loads(packet_path.read_text(encoding="utf-8"))
    packet["report_file_sha256"] = {name: sha256_file(packet_root / name) for name in REPORT_NAMES}
    packet["packet_logical_root_sha256"] = audit._hash_json(audit._packet_logical_domain(packet))
    write_canonical_json(packet_path, packet)


def _failed_cell(
    dataset: Path,
    repeat: Path,
    packet_root: Path,
    config: Any,
    profile: Any,
    seed_index: int,
    candidate_seed: int,
    registry: Any,
    seeds: Any,
) -> dict[str, Any]:
    del dataset, repeat, packet_root, registry, seeds
    scene = config.scene_family.value
    return {
        "cell_id": f"{scene}--{profile.profile_id}--seed-{seed_index}",
        "scene_family": scene,
        "seed_index": seed_index,
        "candidate_seed": candidate_seed,
        "profile_id": profile.profile_id,
        "matched_control_profile_id": profile.matched_control_profile_id,
        "generation_status": "failed",
        "admission_status": "rejected",
        "rejection_reasons": ["generation_or_validation_failed"],
        "failure_type": "SyntheticFailure",
        "failure_message": "retained negative evidence",
    }


def _tiny_success_cell(
    dataset: Path,
    repeat: Path,
    packet_root: Path,
    config: Any,
    profile: Any,
    seed_index: int,
    candidate_seed: int,
    registry: Any,
    seeds: Any,
) -> dict[str, Any]:
    del dataset, repeat
    scene = config.scene_family.value
    cell_id = audit._cell_id(scene, profile.profile_id, seed_index)
    surface_names = (
        ("support_surface", "occluding_surface", "background_surface")
        if scene == "single_occluder"
        else (
            "corridor_floor",
            "corridor_left_surface",
            "corridor_right_surface",
            "corridor_end_surface",
        )
    )
    episode_seed = derive_seed(candidate_seed, "episode:0")
    appearance = resolve_appearance(
        registry,
        profile.profile_id,
        scene,
        surface_names,
        episode_seed,
        candidate_seed,
        seeds,
    ).record
    profile_index = next(
        index
        for index, item in enumerate(registry.profiles)
        if item.profile_id == profile.profile_id
    )
    rgb = np.full((4, 4, 3), 40 + 18 * profile_index, dtype=np.uint8)
    depth = np.zeros((4, 4), dtype=np.float32)
    segmentation = np.ones((4, 4), dtype=np.int32)
    evidence: dict[str, Any] = {}
    evidence_root = packet_root / "metric_evidence" / cell_id
    evidence_root.mkdir(parents=True)
    for frame_name in ("before", "after"):
        records: dict[str, Any] = {}
        for role, array, media_type, suffix in (
            ("rgb", rgb, "image/png", ".png"),
            ("depth", depth, "application/x-npy", ".npy"),
            ("segmentation", segmentation, "application/x-npy", ".npy"),
        ):
            path = evidence_root / f"{role}_{frame_name}{suffix}"
            if role == "rgb":
                Image.fromarray(array).save(path, format="PNG")
            else:
                np.save(path, array, allow_pickle=False)
            records[role] = {
                "path": path.relative_to(packet_root).as_posix(),
                "media_type": media_type,
                "dtype": str(array.dtype),
                "shape": list(array.shape),
                "file_sha256": sha256_file(path),
                "logical_sha256": logical_array_hash(array),
                "byte_count": path.stat().st_size,
            }
        evidence[frame_name] = records
    structural = audit._hash_json({"scene": scene, "seed_index": seed_index})
    return {
        "cell_id": cell_id,
        "scene_family": scene,
        "seed_index": seed_index,
        "candidate_seed": candidate_seed,
        "profile_id": profile.profile_id,
        "matched_control_profile_id": profile.matched_control_profile_id,
        "generation_status": "success",
        "episode_seed": episode_seed,
        "appearance_profile_sha256": appearance_profile_hash(profile),
        "appearance_instance_sha256": appearance.appearance_instance_sha256,
        "appearance_instance": appearance.model_dump(mode="json"),
        "evaluation_seed_registry_sha256": seed_registry_hash(seeds),
        "appearance_assignment_schedule_source": ("snapshotted_evaluation_seed_registry_v1"),
        "renderer_provenance": {
            "mujoco_version": "synthetic",
            "numpy_version": np.__version__,
            "renderer": "mujoco.Renderer",
            "backend": "synthetic-test-renderer",
            "operating_system": "synthetic",
        },
        "rgb_logical_sha256": [logical_array_hash(rgb), logical_array_hash(rgb)],
        "scene_content_sha256": structural,
        "analytic_transport_sha256": structural,
        "oriented_boundary_sha256": structural,
        "visibility_event_sha256": structural,
        "ecological_label_sha256": structural,
        "occlusion_sha256": structural,
        "action_sha256": structural,
        "camera_trajectory_sha256": structural,
        "geometry_sha256": structural,
        "opaque_remapping_sha256": structural,
        "depth_logical_sha256": [logical_array_hash(depth), logical_array_hash(depth)],
        "segmentation_logical_sha256": [
            logical_array_hash(segmentation),
            logical_array_hash(segmentation),
        ],
        "determinism_pass": True,
        "semantic_surface_labels": {
            surface_name: index + 1 for index, surface_name in enumerate(surface_names)
        },
        "source_texture_diagnostics": audit._source_texture_diagnostics(profile, scene, appearance),
        "evidence": evidence,
    }


@pytest.fixture()
def synthetic_packet(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(audit, "_dataset_cell", _failed_cell)
    output = tmp_path / "packet"
    audit.create_appearance_audit(
        Path("configs/appearance_candidates_v0.yaml"),
        Path("configs/evaluation_seed_candidates_v0.yaml"),
        Path("configs/benchmark_v0.yaml"),
        Path("configs/corridor_v0.yaml"),
        output,
    )
    return output


@pytest.fixture(scope="module")
def tiny_success_packet_base(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output = tmp_path_factory.mktemp("tiny-success-audit") / "packet"
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(audit, "_dataset_cell", _tiny_success_cell)
    try:
        audit.create_appearance_audit(
            Path("configs/appearance_candidates_v0.yaml"),
            Path("configs/evaluation_seed_candidates_v0.yaml"),
            Path("configs/benchmark_v0.yaml"),
            Path("configs/corridor_v0.yaml"),
            output,
        )
    finally:
        monkeypatch.undo()
    return output


@pytest.fixture()
def successful_packet(tiny_success_packet_base: Path, tmp_path: Path) -> Path:
    output = tmp_path / "packet"
    shutil.copytree(tiny_success_packet_base, output)
    return output


def test_audit_packet_is_complete_validated_and_retains_failures(
    synthetic_packet: Path,
) -> None:
    packet = audit.validate_appearance_audit(synthetic_packet)
    for required in (
        "candidate_packet.json",
        "appearance_registry_snapshot.json",
        "seed_registry_snapshot.json",
        "profile_summary.json",
        "seed_matrix.json",
        "negative_evidence.json",
        "representative_contact_sheets",
    ):
        assert (synthetic_packet / required).exists()
    assert packet["freeze_status"] == "candidate_packet_only_not_frozen"
    negatives = json.loads((synthetic_packet / "negative_evidence.json").read_text())
    assert len(negatives["rejected_cells"]) == 10 * 8 * 2


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("matrix_counts", {"rejected": 159, "admitted": 1}),
        ("profile_count", 9),
        ("candidate_seed_count", 7),
        ("scene_families", ["corridor", "single_occluder"]),
        ("final_split", {"development": ["balanced_solid_palette_v1"]}),
        ("final_evaluation_seeds", [1]),
        ("freeze_status", "benchmark_frozen"),
    ],
)
def test_fully_rehashed_packet_summary_and_freeze_mutations_are_rejected(
    synthetic_packet: Path,
    field: str,
    value: Any,
) -> None:
    # EPS-ER11-0005
    packet_path = synthetic_packet / "candidate_packet.json"
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    packet[field] = value
    _write_rehashed_packet(synthetic_packet, packet)
    with pytest.raises(audit.AppearanceAuditError):
        audit.validate_appearance_audit(synthetic_packet)


def test_packet_schema_rejects_an_unknown_field_even_with_a_rebuilt_root(
    synthetic_packet: Path,
) -> None:
    packet_path = synthetic_packet / "candidate_packet.json"
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    packet["unauthorised_extension"] = True
    _write_rehashed_packet(synthetic_packet, packet)
    with pytest.raises(audit.AppearanceAuditError, match="schema is not strict"):
        audit.validate_appearance_audit(synthetic_packet)


def test_fully_rehashed_matched_control_metadata_mismatch_is_rejected(
    synthetic_packet: Path,
) -> None:
    # EPS-ER11-0006
    matrix_path = synthetic_packet / "seed_matrix.json"
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    matrix["cells"][0]["matched_control_profile_id"] = "legacy_solid_base_v1"
    write_canonical_json(matrix_path, matrix)
    _write_rehashed_packet(synthetic_packet)
    with pytest.raises(audit.AppearanceAuditError, match="identity is not canonical"):
        audit.validate_appearance_audit(synthetic_packet)


def test_fully_rehashed_cell_id_swap_is_rejected(
    synthetic_packet: Path,
) -> None:
    matrix_path = synthetic_packet / "seed_matrix.json"
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    matrix["cells"][0]["cell_id"], matrix["cells"][1]["cell_id"] = (
        matrix["cells"][1]["cell_id"],
        matrix["cells"][0]["cell_id"],
    )
    write_canonical_json(matrix_path, matrix)
    _write_rehashed_packet(synthetic_packet)
    with pytest.raises(audit.AppearanceAuditError, match="identity is not canonical"):
        audit.validate_appearance_audit(synthetic_packet)


def test_audit_refuses_a_non_empty_output(
    synthetic_packet: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(audit, "_dataset_cell", _failed_cell)
    with pytest.raises(FileExistsError):
        audit.create_appearance_audit(
            Path("configs/appearance_candidates_v0.yaml"),
            Path("configs/evaluation_seed_candidates_v0.yaml"),
            Path("configs/benchmark_v0.yaml"),
            Path("configs/corridor_v0.yaml"),
            synthetic_packet,
        )


def test_atomic_directory_publication_does_not_replace_a_racing_target(
    tmp_path: Path,
) -> None:
    staging = tmp_path / "staging"
    destination = tmp_path / "destination"
    staging.mkdir()
    destination.mkdir()
    (staging / "packet.json").write_text("staged\n", encoding="utf-8")
    with pytest.raises(FileExistsError):
        audit._atomic_no_replace_directory(staging, destination)
    assert staging.is_dir()
    assert destination.is_dir()
    assert not (destination / "packet.json").exists()


def test_volatile_metadata_does_not_change_packet_root(synthetic_packet: Path) -> None:
    before = audit.validate_appearance_audit(synthetic_packet)["packet_logical_root_sha256"]
    (synthetic_packet / "run.json").write_text('{"arbitrary":"volatile"}\n', encoding="utf-8")
    after = audit.validate_appearance_audit(synthetic_packet)["packet_logical_root_sha256"]
    assert after == before


def _first_contact_sheet(packet_root: Path) -> tuple[dict[str, Any], Path]:
    packet_path = packet_root / "candidate_packet.json"
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    record = packet["contact_sheet_manifest"]["sheets"][0]
    return packet, packet_root / record["path"]


def test_missing_contact_sheet_is_rejected(synthetic_packet: Path) -> None:
    # EPS-ER11-0004
    _, sheet = _first_contact_sheet(synthetic_packet)
    sheet.unlink()
    with pytest.raises(audit.AppearanceAuditError, match="missing or additional"):
        audit.validate_appearance_audit(synthetic_packet)


def test_additional_contact_sheet_is_rejected(synthetic_packet: Path) -> None:
    directory = synthetic_packet / "representative_contact_sheets"
    (directory / "unexpected.png").write_bytes(b"not a sheet")
    with pytest.raises(audit.AppearanceAuditError, match="missing or additional"):
        audit.validate_appearance_audit(synthetic_packet)


def test_fully_rehashed_replacement_contact_sheet_is_independently_rejected(
    synthetic_packet: Path,
) -> None:
    packet, sheet = _first_contact_sheet(synthetic_packet)
    with Image.open(sheet) as original:
        replacement = np.zeros((original.height, original.width, 3), dtype=np.uint8)
    Image.fromarray(replacement).save(sheet, format="PNG")
    record = packet["contact_sheet_manifest"]["sheets"][0]
    record.update(
        {
            "logical_sha256": logical_array_hash(replacement),
            "file_sha256": sha256_file(sheet),
            "byte_count": sheet.stat().st_size,
        }
    )
    _write_rehashed_packet(synthetic_packet, packet)
    with pytest.raises(audit.AppearanceAuditError, match="independent reconstruction"):
        audit.validate_appearance_audit(synthetic_packet)


def test_contact_sheet_hardlink_alias_is_rejected(synthetic_packet: Path, tmp_path: Path) -> None:
    _, sheet = _first_contact_sheet(synthetic_packet)
    external = tmp_path / "external-contact.png"
    external.write_bytes(sheet.read_bytes())
    sheet.unlink()
    os.link(external, sheet)
    with pytest.raises(audit.AppearanceAuditError, match="hard-link alias"):
        audit.validate_appearance_audit(synthetic_packet)


def test_fully_rehashed_malformed_contact_manifest_is_rejected(
    synthetic_packet: Path,
) -> None:
    packet, _ = _first_contact_sheet(synthetic_packet)
    del packet["contact_sheet_manifest"]["sheets"][0]["dimensions"]
    _write_rehashed_packet(synthetic_packet, packet)
    with pytest.raises(audit.AppearanceAuditError, match="record is not strict"):
        audit.validate_appearance_audit(synthetic_packet)


@pytest.mark.parametrize(
    "mutation",
    ("boolean_seed", "float_seed", "float_dimension", "float_shape", "float_byte_count"),
)
def test_fully_resealed_contact_sheet_noncanonical_types_are_rejected(
    synthetic_packet: Path,
    mutation: str,
) -> None:
    # EPS-ER11-0012
    packet_path = synthetic_packet / "candidate_packet.json"
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    record = packet["contact_sheet_manifest"]["sheets"][0]
    if mutation == "boolean_seed":
        record["seed_index"] = False
    elif mutation == "float_seed":
        record["seed_index"] = 0.0
    elif mutation == "float_dimension":
        record["dimensions"][0] = float(record["dimensions"][0])
    elif mutation == "float_shape":
        record["shape"][0] = float(record["shape"][0])
    else:
        record["byte_count"] = float(record["byte_count"])
    _write_rehashed_packet(synthetic_packet, packet)
    with pytest.raises(audit.AppearanceAuditError, match="types are not canonical"):
        audit.validate_appearance_audit(synthetic_packet)


def test_corrupt_contact_sheet_is_rejected(synthetic_packet: Path) -> None:
    _, sheet = _first_contact_sheet(synthetic_packet)
    sheet.write_bytes(b"corrupt")
    with pytest.raises(audit.AppearanceAuditError, match=r"byte count mismatch|hash mismatch"):
        audit.validate_appearance_audit(synthetic_packet)


def _matrix_payload(packet_root: Path) -> tuple[Path, dict[str, Any]]:
    path = packet_root / "seed_matrix.json"
    return path, json.loads(path.read_text(encoding="utf-8"))


def _fully_reseal_matrix(packet_root: Path) -> None:
    matrix = json.loads((packet_root / "seed_matrix.json").read_text(encoding="utf-8"))
    cells = matrix["cells"]
    registry = load_appearance_registry(packet_root / "appearance_registry_snapshot.json")
    seeds = load_evaluation_seed_registry(packet_root / "seed_registry_snapshot.json")
    negative = {
        "schema_version": audit.AUDIT_SCHEMA_VERSION,
        "retention_rule": "all_rejected_candidates_and_failing_seeds_retained_v1",
        "rejected_cells": [
            {
                "cell_id": cell["cell_id"],
                "profile_id": cell["profile_id"],
                "scene_family": cell["scene_family"],
                "seed_index": cell["seed_index"],
                "rejection_reasons": cell["rejection_reasons"],
                "failure_type": cell.get("failure_type"),
                "failure_message": cell.get("failure_message"),
                "matched_control_failure_type": cell.get("matched_control_failure_type"),
                "matched_control_failure_message": cell.get("matched_control_failure_message"),
            }
            for cell in cells
            if cell["admission_status"] == "rejected"
        ],
    }
    write_canonical_json(packet_root / "negative_evidence.json", negative)
    packet_path = packet_root / "candidate_packet.json"
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    packet["roots"] = audit._roots(registry, seeds, cells)
    _write_rehashed_packet(packet_root, packet)


def test_fully_resealed_boolean_seed_index_is_rejected(
    synthetic_packet: Path,
) -> None:
    # EPS-ER11-0010
    matrix_path, matrix = _matrix_payload(synthetic_packet)
    matrix["cells"][0]["seed_index"] = False
    write_canonical_json(matrix_path, matrix)
    _fully_reseal_matrix(synthetic_packet)
    with pytest.raises(audit.AppearanceAuditError, match="noncanonical type: seed_index"):
        audit.validate_appearance_audit(synthetic_packet)


@pytest.mark.parametrize("nested", [False, True])
def test_fully_resealed_unknown_audit_cell_field_is_rejected(
    synthetic_packet: Path,
    nested: bool,
) -> None:
    # EPS-ER11-0010
    matrix_path, matrix = _matrix_payload(synthetic_packet)
    if nested:
        matrix["cells"][0]["admission_checks"]["benchmark_frozen"] = True
        expected = "admission-check schema is not strict"
    else:
        matrix["cells"][0]["unauthorised_extension"] = {"benchmark_frozen": True}
        expected = "cell schema is not strict"
    write_canonical_json(matrix_path, matrix)
    _fully_reseal_matrix(synthetic_packet)
    with pytest.raises(audit.AppearanceAuditError, match=expected):
        audit.validate_appearance_audit(synthetic_packet)


def _replace_with_external_hardlink(path: Path, external: Path) -> None:
    external.write_bytes(path.read_bytes())
    path.unlink()
    os.link(external, path)


def _overwrite_same_inode_same_size(path: Path) -> None:
    with path.open("r+b") as stream:
        payload = bytearray(stream.read())
        index = next(index for index, value in enumerate(payload) if value not in {0, 255})
        payload[index] ^= 1
        stream.seek(0)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


@pytest.mark.skipif(os.name == "nt", reason="Windows denies replacement of an open file")
def test_packet_replacement_between_ownership_check_and_hash_is_rejected(
    synthetic_packet: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # EPS-ER11-0011
    original_open = open_owned_regular_file
    replaced = False

    @contextmanager
    def replace_after_check(root: Path, relative_path: str) -> Any:
        nonlocal replaced
        with original_open(root, relative_path) as owned:
            if relative_path == "seed_matrix.json" and not replaced:
                replaced = True
                _replace_with_external_hardlink(
                    owned.path,
                    tmp_path / "external-check-to-hash.json",
                )
            yield owned

    monkeypatch.setattr(audit, "open_owned_regular_file", replace_after_check)
    with pytest.raises(audit.AppearanceAuditError, match="changed while being consumed"):
        audit.validate_appearance_audit(synthetic_packet)


@pytest.mark.skipif(os.name == "nt", reason="Windows denies replacement of an open file")
def test_packet_replacement_between_hash_and_decode_is_rejected(
    synthetic_packet: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # EPS-ER11-0011
    original_hash = sha256_open_file
    replaced = False

    def replace_after_hash(owned: Any) -> str:
        nonlocal replaced
        digest = original_hash(owned)
        if owned.path.name == "seed_matrix.json" and not replaced:
            replaced = True
            _replace_with_external_hardlink(
                owned.path,
                tmp_path / "external-hash-to-decode.json",
            )
        return digest

    monkeypatch.setattr(audit, "sha256_open_file", replace_after_hash)
    with pytest.raises(audit.AppearanceAuditError, match="changed while being consumed"):
        audit.validate_appearance_audit(synthetic_packet)


def test_packet_same_inode_overwrite_between_hash_and_decode_is_rejected(
    synthetic_packet: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # EPS-ER11-0011
    original_hash = sha256_open_file
    overwritten = False

    def overwrite_after_hash(owned: Any) -> str:
        nonlocal overwritten
        digest = original_hash(owned)
        if owned.path.name == "seed_matrix.json" and not overwritten:
            overwritten = True
            _overwrite_same_inode_same_size(owned.path)
        return digest

    monkeypatch.setattr(audit, "sha256_open_file", overwrite_after_hash)
    with pytest.raises(audit.AppearanceAuditError, match="bytes changed while being consumed"):
        audit.validate_appearance_audit(synthetic_packet)


def test_tiny_success_packet_validates_all_retained_evidence(
    successful_packet: Path,
) -> None:
    packet = audit.validate_appearance_audit(successful_packet)
    assert sum(packet["matrix_counts"].values()) == 160


def test_fully_rehashed_audit_evidence_path_escape_is_rejected(
    successful_packet: Path,
) -> None:
    # EPS-ER11-0003
    matrix_path, matrix = _matrix_payload(successful_packet)
    matrix["cells"][0]["evidence"]["before"]["rgb"]["path"] = "../outside.png"
    write_canonical_json(matrix_path, matrix)
    _write_rehashed_packet(successful_packet)
    with pytest.raises(audit.AppearanceAuditError, match="not canonical"):
        audit.validate_appearance_audit(successful_packet)


def test_fully_rehashed_audit_evidence_duplicate_alias_is_rejected(
    successful_packet: Path,
) -> None:
    matrix_path, matrix = _matrix_payload(successful_packet)
    cell = matrix["cells"][0]
    cell["evidence"]["after"]["rgb"] = deepcopy(cell["evidence"]["before"]["rgb"])
    write_canonical_json(matrix_path, matrix)
    _write_rehashed_packet(successful_packet)
    with pytest.raises(audit.AppearanceAuditError, match="duplicate packet artifact path"):
        audit.validate_appearance_audit(successful_packet)


def test_audit_evidence_hardlink_alias_is_rejected(
    successful_packet: Path,
    tmp_path: Path,
) -> None:
    _, matrix = _matrix_payload(successful_packet)
    record = matrix["cells"][0]["evidence"]["before"]["rgb"]
    path = successful_packet / record["path"]
    external = tmp_path / "external-evidence.png"
    external.write_bytes(path.read_bytes())
    path.unlink()
    os.link(external, path)
    with pytest.raises(audit.AppearanceAuditError, match="hard-link alias"):
        audit.validate_appearance_audit(successful_packet)


def test_audit_evidence_symlink_alias_is_rejected(
    successful_packet: Path,
    tmp_path: Path,
) -> None:
    _, matrix = _matrix_payload(successful_packet)
    record = matrix["cells"][0]["evidence"]["before"]["rgb"]
    path = successful_packet / record["path"]
    external = tmp_path / "external-evidence.png"
    external.write_bytes(path.read_bytes())
    path.unlink()
    try:
        path.symlink_to(external)
    except OSError as error:
        pytest.skip(f"symbolic links are unavailable: {error}")
    with pytest.raises(audit.AppearanceAuditError, match="symbolic-link alias"):
        audit.validate_appearance_audit(successful_packet)


def test_audit_evidence_special_file_is_rejected(successful_packet: Path) -> None:
    _, matrix = _matrix_payload(successful_packet)
    record = matrix["cells"][0]["evidence"]["before"]["rgb"]
    path = successful_packet / record["path"]
    path.unlink()
    path.mkdir()
    with pytest.raises(audit.AppearanceAuditError, match="regular file"):
        audit.validate_appearance_audit(successful_packet)


def test_fully_rehashed_swapped_evidence_is_rejected_by_parent_identity(
    successful_packet: Path,
) -> None:
    # EPS-ER11-0007
    matrix_path, matrix = _matrix_payload(successful_packet)
    first = matrix["cells"][0]
    second = matrix["cells"][8]
    first["evidence"]["before"]["rgb"], second["evidence"]["before"]["rgb"] = (
        second["evidence"]["before"]["rgb"],
        first["evidence"]["before"]["rgb"],
    )
    write_canonical_json(matrix_path, matrix)
    _write_rehashed_packet(successful_packet)
    with pytest.raises(audit.AppearanceAuditError, match="declared rgb identity"):
        audit.validate_appearance_audit(successful_packet)


def test_fully_rehashed_fabricated_declared_rgb_hash_is_rejected(
    successful_packet: Path,
) -> None:
    matrix_path, matrix = _matrix_payload(successful_packet)
    matrix["cells"][0]["rgb_logical_sha256"][0] = "0" * 64
    write_canonical_json(matrix_path, matrix)
    _write_rehashed_packet(successful_packet)
    with pytest.raises(audit.AppearanceAuditError, match="declared rgb identity"):
        audit.validate_appearance_audit(successful_packet)


def test_missing_matrix_cell_and_altered_admission_are_rejected(
    synthetic_packet: Path,
) -> None:
    matrix_path = synthetic_packet / "seed_matrix.json"
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    original = json.loads(json.dumps(matrix))
    matrix["cells"].pop()
    write_canonical_json(matrix_path, matrix)
    _write_rehashed_packet(synthetic_packet)
    with pytest.raises(audit.AppearanceAuditError, match="wrong cell count"):
        audit.validate_appearance_audit(synthetic_packet)
    write_canonical_json(matrix_path, original)
    _write_rehashed_packet(synthetic_packet)
    matrix = json.loads(json.dumps(original))
    matrix["cells"][0]["admission_status"] = "admitted"
    matrix["cells"][0]["rejection_reasons"] = []
    write_canonical_json(matrix_path, matrix)
    _write_rehashed_packet(synthetic_packet)
    with pytest.raises(audit.AppearanceAuditError, match="altered admission evidence"):
        audit.validate_appearance_audit(synthetic_packet)


def _successful_cell() -> dict[str, Any]:
    return {
        "cell_id": "single_occluder--legacy_solid_base_v1--seed-0",
        "scene_family": "single_occluder",
        "profile_id": "legacy_solid_base_v1",
        "matched_control_profile_id": "legacy_solid_base_v1",
        "seed_index": 0,
        "candidate_seed": 1,
        "generation_status": "success",
        "appearance_profile_sha256": "profile",
        "appearance_instance_sha256": "instance",
        "appearance_instance": {
            "seeds": {"candidate_schedule_index": 0},
            "assignment_schedule_posture": "candidate_registry_index",
            "style_assignment": {},
            "textures": [],
        },
        "evaluation_seed_registry_sha256": "seed-registry",
        "appearance_assignment_schedule_source": ("snapshotted_evaluation_seed_registry_v1"),
        "scene_content_sha256": "scene",
        "analytic_transport_sha256": "transport",
        "oriented_boundary_sha256": "boundary",
        "visibility_event_sha256": "event",
        "ecological_label_sha256": "label-windows",
        "occlusion_sha256": "occlusion",
        "action_sha256": "action",
        "camera_trajectory_sha256": "camera-windows",
        "geometry_sha256": "geometry-windows",
        "opaque_remapping_sha256": "remapping-windows",
        "admission_checks": {
            "structural_invariance": True,
            "portable_analytic_identity_equality": True,
            "ecological_label_equality": True,
            "depth_segmentation_invariance": True,
            "determinism": True,
        },
        "rgb_logical_sha256": ["rgb-before-windows", "rgb-after-windows"],
        "depth_logical_sha256": ["depth-before-windows", "depth-after-windows"],
        "segmentation_logical_sha256": [
            "segmentation-before-windows",
            "segmentation-after-windows",
        ],
        "evidence": {"before": {}, "after": {}},
        "renderer_provenance": {"backend": "wgl"},
        "frame_metrics": {},
        "determinism_pass": True,
        "admission_status": "admitted",
    }


def test_renderer_local_labels_can_differ_while_portable_outcome_matches() -> None:
    registry = load_appearance_registry(Path("configs/appearance_candidates_v0.yaml"))
    seeds = load_evaluation_seed_registry(Path("configs/evaluation_seed_candidates_v0.yaml"))
    cell = _successful_cell()
    windows = audit._roots(registry, seeds, [cell])
    ubuntu = deepcopy(cell)
    ubuntu.update(
        {
            "ecological_label_sha256": "label-ubuntu",
            "camera_trajectory_sha256": "camera-ubuntu",
            "geometry_sha256": "geometry-ubuntu",
            "opaque_remapping_sha256": "remapping-ubuntu",
            "rgb_logical_sha256": ["rgb-before-ubuntu", "rgb-after-ubuntu"],
            "renderer_provenance": {"backend": "osmesa"},
        }
    )
    assert audit._structural_domain(ubuntu) != audit._structural_domain(cell)
    assert ubuntu["admission_checks"]["structural_invariance"] is True
    linux = audit._roots(registry, seeds, [ubuntu])
    assert (
        linux["appearance_invariance_outcome_root_sha256"]
        == windows["appearance_invariance_outcome_root_sha256"]
    )
    assert (
        linux["portable_analytic_identity_root_sha256"]
        == windows["portable_analytic_identity_root_sha256"]
    )
    assert (
        linux["renderer_local_ecological_label_root_sha256"]
        != windows["renderer_local_ecological_label_root_sha256"]
    )


@pytest.mark.parametrize(
    "field",
    [
        "scene_content_sha256",
        "analytic_transport_sha256",
        "oriented_boundary_sha256",
        "visibility_event_sha256",
        "occlusion_sha256",
        "action_sha256",
    ],
)
def test_each_portable_identity_changes_both_portable_roots(field: str) -> None:
    registry = load_appearance_registry(Path("configs/appearance_candidates_v0.yaml"))
    seeds = load_evaluation_seed_registry(Path("configs/evaluation_seed_candidates_v0.yaml"))
    cell = _successful_cell()
    original = audit._roots(registry, seeds, [cell])
    changed = deepcopy(cell)
    changed[field] = f"changed-{field}"
    altered = audit._roots(registry, seeds, [changed])
    for root_name in (
        "portable_analytic_identity_root_sha256",
        "appearance_invariance_outcome_root_sha256",
    ):
        assert altered[root_name] != original[root_name]


def test_raw_label_changes_only_renderer_local_label_root_among_non_rgb_domains() -> None:
    registry = load_appearance_registry(Path("configs/appearance_candidates_v0.yaml"))
    seeds = load_evaluation_seed_registry(Path("configs/evaluation_seed_candidates_v0.yaml"))
    cell = _successful_cell()
    original = audit._roots(registry, seeds, [cell])
    changed = deepcopy(cell)
    changed["ecological_label_sha256"] = "renderer-local-alternate"
    altered = audit._roots(registry, seeds, [changed])
    assert (
        altered["renderer_local_ecological_label_root_sha256"]
        != original["renderer_local_ecological_label_root_sha256"]
    )
    for root_name in (
        "appearance_registry_sha256",
        "seed_registry_sha256",
        "procedural_asset_root_sha256",
        "appearance_assignment_root_sha256",
        "portable_analytic_identity_root_sha256",
        "appearance_invariance_outcome_root_sha256",
        "renderer_specific_audit_root_sha256",
    ):
        assert altered[root_name] == original[root_name]


def test_rgb_and_renderer_provenance_never_enter_portable_roots() -> None:
    registry = load_appearance_registry(Path("configs/appearance_candidates_v0.yaml"))
    seeds = load_evaluation_seed_registry(Path("configs/evaluation_seed_candidates_v0.yaml"))
    cell = _successful_cell()
    original = audit._roots(registry, seeds, [cell])
    changed = deepcopy(cell)
    changed["rgb_logical_sha256"] = ["different-before-rgb", "different-after-rgb"]
    changed["renderer_provenance"] = {"backend": "different-renderer"}
    altered = audit._roots(registry, seeds, [changed])
    assert (
        altered["portable_analytic_identity_root_sha256"]
        == original["portable_analytic_identity_root_sha256"]
    )
    assert (
        altered["appearance_invariance_outcome_root_sha256"]
        == original["appearance_invariance_outcome_root_sha256"]
    )


def test_ecological_label_inequality_changes_outcome_and_rejects_cell(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    registry = load_appearance_registry(Path("configs/appearance_candidates_v0.yaml"))
    seeds = load_evaluation_seed_registry(Path("configs/evaluation_seed_candidates_v0.yaml"))
    profile = next(
        item for item in registry.profiles if item.profile_id == "balanced_solid_palette_v1"
    )
    cell = _successful_cell()
    control = deepcopy(cell)
    cell["ecological_label_sha256"] = "candidate-label"
    control["ecological_label_sha256"] = "control-label"
    cell["semantic_surface_labels"] = {}
    cell["source_texture_diagnostics"] = []
    control["semantic_surface_labels"] = {}
    control["source_texture_diagnostics"] = []
    monkeypatch.setattr(
        audit,
        "_load_frame",
        lambda *_: (
            np.zeros((1, 1, 3), dtype=np.uint8),
            np.zeros((1, 1), dtype=np.float32),
            np.zeros((1, 1), dtype=np.int32),
        ),
    )
    monkeypatch.setattr(
        audit,
        "_frame_metrics",
        lambda *_: {
            "material_change_pass": True,
            "controlled_surface_exposure_pass": True,
            "textured_surface_variation_pass": True,
            "surface_diagnostics": [],
        },
    )
    original = audit._roots(registry, seeds, [deepcopy(control)])
    audit._evaluate_cell(tmp_path, cell, control, profile)
    assert cell["admission_checks"]["ecological_label_equality"] is False
    assert cell["admission_status"] == "rejected"
    assert "ecological_label_equality" in cell["rejection_reasons"]
    assert (
        audit._roots(registry, seeds, [cell])["appearance_invariance_outcome_root_sha256"]
        != original["appearance_invariance_outcome_root_sha256"]
    )


@pytest.mark.parametrize(
    "root_name",
    [
        "appearance_invariance_outcome_root_sha256",
        "renderer_local_ecological_label_root_sha256",
    ],
)
def test_fully_rehashed_root_corruption_is_rejected(
    synthetic_packet: Path,
    root_name: str,
) -> None:
    packet_path = synthetic_packet / "candidate_packet.json"
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    packet["roots"][root_name] = "0" * 64
    _write_rehashed_packet(synthetic_packet, packet)
    with pytest.raises(audit.AppearanceAuditError, match="root mismatch"):
        audit.validate_appearance_audit(synthetic_packet)


def test_packet_seed_registry_snapshot_is_independently_validated(
    synthetic_packet: Path,
) -> None:
    snapshot = synthetic_packet / "seed_registry_snapshot.json"
    payload = json.loads(snapshot.read_text(encoding="utf-8"))
    payload["candidate_episode_seeds"] = list(reversed(payload["candidate_episode_seeds"]))
    write_canonical_json(snapshot, payload)
    with pytest.raises(audit.AppearanceAuditError, match="registry snapshot"):
        audit.validate_appearance_audit(synthetic_packet)


def test_mixed_candidate_success_and_matched_control_failure_retains_complete_packet(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # EPS-ER11-0009
    def mixed_cell(*args: Any, **kwargs: Any) -> dict[str, Any]:
        config = args[3]
        profile = args[4]
        seed_index = args[5]
        if (
            config.scene_family.value == "single_occluder"
            and profile.profile_id == "legacy_solid_base_v1"
            and seed_index == 0
        ):
            return _failed_cell(*args, **kwargs)
        return _tiny_success_cell(*args, **kwargs)

    monkeypatch.setattr(audit, "_dataset_cell", mixed_cell)
    output = tmp_path / "mixed-failure-packet"
    audit.create_appearance_audit(
        Path("configs/appearance_candidates_v0.yaml"),
        Path("configs/evaluation_seed_candidates_v0.yaml"),
        Path("configs/benchmark_v0.yaml"),
        Path("configs/corridor_v0.yaml"),
        output,
    )
    packet = audit.validate_appearance_audit(output)
    matrix = json.loads((output / "seed_matrix.json").read_text(encoding="utf-8"))
    negatives = json.loads((output / "negative_evidence.json").read_text(encoding="utf-8"))
    assert len(matrix["cells"]) == 160
    assert sum(packet["matrix_counts"].values()) == 160
    affected = next(
        cell
        for cell in matrix["cells"]
        if cell["cell_id"] == "single_occluder--balanced_solid_palette_v1--seed-0"
    )
    assert affected["admission_status"] == "rejected"
    assert affected["rejection_reasons"] == ["matched_control_generation_or_validation_failed"]
    assert affected["matched_control_failure_type"] == "SyntheticFailure"
    retained = next(
        cell for cell in negatives["rejected_cells"] if cell["cell_id"] == affected["cell_id"]
    )
    assert retained["matched_control_failure_type"] == "SyntheticFailure"
    assert retained["matched_control_failure_message"] == "retained negative evidence"
