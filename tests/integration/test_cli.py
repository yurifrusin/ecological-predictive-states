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
