from pathlib import Path

from typer.testing import CliRunner

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
