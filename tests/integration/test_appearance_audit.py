from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

import epsbench.audit as audit
from epsbench.appearance import load_appearance_registry, load_evaluation_seed_registry
from epsbench.utils.canonical import write_canonical_json


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


def test_missing_matrix_cell_and_altered_admission_are_rejected(
    synthetic_packet: Path,
) -> None:
    matrix_path = synthetic_packet / "seed_matrix.json"
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    original = json.loads(json.dumps(matrix))
    matrix["cells"].pop()
    write_canonical_json(matrix_path, matrix)
    with pytest.raises(audit.AppearanceAuditError, match="matrix is incomplete"):
        audit.validate_appearance_audit(synthetic_packet)
    write_canonical_json(matrix_path, original)
    matrix = json.loads(json.dumps(original))
    matrix["cells"][0]["admission_status"] = "admitted"
    matrix["cells"][0]["rejection_reasons"] = []
    write_canonical_json(matrix_path, matrix)
    with pytest.raises(audit.AppearanceAuditError, match="altered admission evidence"):
        audit.validate_appearance_audit(synthetic_packet)


def test_shared_ecological_root_excludes_private_compiled_metric_hashes() -> None:
    registry = load_appearance_registry(Path("configs/appearance_candidates_v0.yaml"))
    seeds = load_evaluation_seed_registry(Path("configs/evaluation_seed_candidates_v0.yaml"))
    cell: dict[str, Any] = {
        "cell_id": "single_occluder--legacy_solid_base_v1--seed-0",
        "generation_status": "success",
        "appearance_profile_sha256": "profile",
        "appearance_instance_sha256": "instance",
        "appearance_instance": {"style_assignment": {}, "textures": []},
        "scene_content_sha256": "scene",
        "analytic_transport_sha256": "transport",
        "oriented_boundary_sha256": "boundary",
        "visibility_event_sha256": "event",
        "ecological_label_sha256": "label",
        "occlusion_sha256": "occlusion",
        "action_sha256": "action",
        "camera_trajectory_sha256": "camera-windows",
        "geometry_sha256": "geometry-windows",
        "opaque_remapping_sha256": "remapping-windows",
        "admission_checks": {"structural_invariance": True},
        "rgb_logical_sha256": "rgb",
        "frame_metrics": {},
        "determinism_pass": True,
        "admission_status": "admitted",
    }
    windows_root = audit._roots(registry, seeds, [cell])["ecological_invariance_root_sha256"]
    ubuntu = deepcopy(cell)
    ubuntu.update(
        {
            "camera_trajectory_sha256": "camera-ubuntu",
            "geometry_sha256": "geometry-ubuntu",
            "opaque_remapping_sha256": "remapping-ubuntu",
        }
    )
    assert audit._structural_domain(ubuntu) != audit._structural_domain(cell)
    assert ubuntu["admission_checks"]["structural_invariance"] is True
    assert (
        audit._roots(registry, seeds, [ubuntu])["ecological_invariance_root_sha256"] == windows_root
    )

    changed_identity = deepcopy(cell)
    changed_identity["visibility_event_sha256"] = "changed-event"
    assert (
        audit._roots(registry, seeds, [changed_identity])["ecological_invariance_root_sha256"]
        != windows_root
    )
