from pathlib import Path

import pytest
from typer.testing import CliRunner

import epsbench.cli.app as cli_module
from epsbench.cli.app import app


def test_validate_and_inspect_cli(smoke_dataset: Path, tmp_path: Path) -> None:
    runner = CliRunner()
    validate_result = runner.invoke(app, ["validate", str(smoke_dataset)])
    assert validate_result.exit_code == 0, validate_result.output
    output = tmp_path / "inspection.png"
    inspect_result = runner.invoke(
        app,
        [
            "inspect",
            str(smoke_dataset),
            "--episode",
            "0",
            "--output",
            str(output),
        ],
    )
    assert inspect_result.exit_code == 0, inspect_result.output
    assert output.is_file()


def test_cli_exits_nonzero_for_invalid_dataset(tmp_path: Path) -> None:
    runner = CliRunner()
    invalid = tmp_path / "invalid"
    invalid.mkdir()
    result = runner.invoke(app, ["validate", str(invalid)])
    assert result.exit_code != 0


def test_cli_inspection_validates_first_and_writes_nothing_for_invalid_dataset(
    tmp_path: Path,
) -> None:
    runner = CliRunner()
    invalid = tmp_path / "invalid-inspection"
    invalid.mkdir()
    output = tmp_path / "must-not-exist.png"
    result = runner.invoke(
        app,
        ["inspect", str(invalid), "--episode", "0", "--output", str(output)],
    )
    assert result.exit_code != 0
    assert "Inspection failed" in result.output
    assert not output.exists()


def test_corridor_validate_and_inspect_report_scene_family(
    corridor_dataset: Path,
    tmp_path: Path,
) -> None:
    runner = CliRunner()
    validate_result = runner.invoke(app, ["validate", str(corridor_dataset)])
    assert validate_result.exit_code == 0, validate_result.output
    assert "corridor" in validate_result.output
    output = tmp_path / "corridor-inspection.png"
    inspect_result = runner.invoke(
        app,
        [
            "inspect",
            str(corridor_dataset),
            "--episode",
            "0",
            "--output",
            str(output),
        ],
    )
    assert inspect_result.exit_code == 0, inspect_result.output
    assert output.is_file()


def test_cli_inspection_refuses_existing_output_without_changing_bytes(
    smoke_dataset: Path,
    tmp_path: Path,
) -> None:
    runner = CliRunner()
    output = tmp_path / "existing-cli-output.png"
    original = b"preserve existing CLI output\n"
    output.write_bytes(original)
    result = runner.invoke(
        app,
        [
            "inspect",
            str(smoke_dataset),
            "--episode",
            "0",
            "--output",
            str(output),
        ],
    )
    assert result.exit_code != 0
    assert "already exists" in result.output
    assert output.read_bytes() == original


def test_generation_refuses_nonempty_destination(tmp_path: Path) -> None:
    runner = CliRunner()
    output = tmp_path / "nonempty"
    output.mkdir()
    (output / "existing.txt").write_text("preserve me\n", encoding="utf-8")
    result = runner.invoke(
        app,
        [
            "generate",
            "--config",
            "configs/corridor_v0.yaml",
            "--episodes",
            "1",
            "--output",
            str(output),
        ],
    )
    assert result.exit_code != 0
    assert "not empty" in result.output
    assert (output / "existing.txt").read_text(encoding="utf-8") == "preserve me\n"


def test_revision_audit_cli_reports_v1_portable_and_renderer_local_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    roots = {
        "portable_analytic_identity_root_sha256": "1" * 64,
        "within_renderer_invariance_outcome_root_sha256": "2" * 64,
        "design_partition_membership_root_sha256": "3" * 64,
        "qualification_partition_membership_root_sha256": "4" * 64,
        "profile_admission_outcome_root_sha256": "5" * 64,
        "renderer_local_design_partition_outcome_root_sha256": "6" * 64,
        "renderer_local_qualification_partition_outcome_root_sha256": "7" * 64,
        "renderer_local_ecological_label_root_sha256": "8" * 64,
        "renderer_specific_audit_root_sha256": "9" * 64,
    }
    packet = {
        "complete_packet_root_sha256": "a" * 64,
        "matrix_counts": {
            "design": {"admitted": 99, "rejected": 13},
            "qualification": {"admitted": 100, "rejected": 12},
        },
        "roots": roots,
    }
    monkeypatch.setattr(cli_module, "create_revision_audit", lambda *args: packet)
    monkeypatch.setattr(cli_module, "validate_revision_audit", lambda *args: packet)
    result = CliRunner().invoke(
        app,
        [
            "appearance-revision-audit",
            "--baseline-registry",
            "configs/appearance_candidates_v0.yaml",
            "--design-seeds",
            "configs/evaluation_seed_candidates_v0.yaml",
            "--revision-registry",
            "configs/appearance_candidates_revision1.yaml",
            "--qualification-seeds",
            "configs/appearance_revision1_qualification_seeds_v0.yaml",
            "--definition-lock",
            "configs/appearance_candidate_revision1_lock.json",
            "--single-config",
            "configs/benchmark_v0.yaml",
            "--corridor-config",
            "configs/corridor_v0.yaml",
            "--output",
            str(tmp_path / "packet"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "design-membership=" + "3" * 64 in result.output
    assert "profiles=" + "5" * 64 in result.output
    assert "design-outcomes=" + "6" * 64 in result.output
    assert "qualification-outcomes=" + "7" * 64 in result.output
