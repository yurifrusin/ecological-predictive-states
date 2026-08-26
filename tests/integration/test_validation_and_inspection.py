import os
import shutil
from pathlib import Path

import pytest
from PIL import Image

from epsbench.data import DatasetValidationError, create_inspection_image, validate_dataset
from epsbench.schema import DatasetManifest


def test_missing_artifact_fails_dataset_validation(smoke_dataset: Path, tmp_path: Path) -> None:
    broken = tmp_path / "broken"
    shutil.copytree(smoke_dataset, broken)
    manifest = DatasetManifest.model_validate_json(
        (broken / "manifest.json").read_text(encoding="utf-8")
    )
    (broken / manifest.episodes[0].transition.path).unlink()
    with pytest.raises(DatasetValidationError, match="missing artifact"):
        validate_dataset(broken)


def test_inconsistent_artifact_fails_dataset_validation(
    smoke_dataset: Path, tmp_path: Path
) -> None:
    broken = tmp_path / "broken-hash"
    shutil.copytree(smoke_dataset, broken)
    manifest = DatasetManifest.model_validate_json(
        (broken / "manifest.json").read_text(encoding="utf-8")
    )
    rgb_path = broken / _first_rgb_path(broken, manifest)
    rgb_path.write_bytes(rgb_path.read_bytes() + b"corrupt")
    with pytest.raises(DatasetValidationError, match="byte count mismatch"):
        validate_dataset(broken)


def _first_rgb_path(root: Path, manifest: DatasetManifest) -> str:
    transition_path = root / manifest.episodes[0].transition.path
    from epsbench.schema import TransitionRecord

    transition = TransitionRecord.model_validate_json(transition_path.read_text(encoding="utf-8"))
    return transition.before.rgb.path


def test_inspection_is_readable_and_does_not_modify_identity(
    smoke_dataset: Path, tmp_path: Path
) -> None:
    manifest_before = (smoke_dataset / "manifest.json").read_bytes()
    output = tmp_path / "episode-0.png"
    create_inspection_image(smoke_dataset, 0, output)
    assert (smoke_dataset / "manifest.json").read_bytes() == manifest_before
    with Image.open(output) as image:
        image.verify()
    with Image.open(output) as image:
        assert image.width > 0 and image.height > 0


def test_inspection_refuses_to_write_inside_dataset(smoke_dataset: Path) -> None:
    with pytest.raises(ValueError, match="outside"):
        create_inspection_image(smoke_dataset, 0, smoke_dataset / "inspection.png")


def test_api_inspection_validates_first_and_writes_nothing_on_failure(
    smoke_dataset: Path,
    tmp_path: Path,
) -> None:
    broken = tmp_path / "invalid-inspection"
    shutil.copytree(smoke_dataset, broken)
    manifest = DatasetManifest.model_validate_json(
        (broken / "manifest.json").read_text(encoding="utf-8")
    )
    rgb_path = broken / _first_rgb_path(broken, manifest)
    rgb_path.write_bytes(rgb_path.read_bytes() + b"corrupt")
    output = tmp_path / "must-not-exist.png"

    with pytest.raises(DatasetValidationError, match="byte count mismatch"):
        create_inspection_image(broken, 0, output)
    assert not output.exists()


def test_api_inspection_refuses_existing_file_without_changing_bytes(
    smoke_dataset: Path,
    tmp_path: Path,
) -> None:
    output = tmp_path / "existing.png"
    original = b"existing inspection bytes\n"
    output.write_bytes(original)
    with pytest.raises(FileExistsError, match="already exists"):
        create_inspection_image(smoke_dataset, 0, output)
    assert output.read_bytes() == original


def test_api_inspection_refuses_existing_directory(smoke_dataset: Path, tmp_path: Path) -> None:
    output = tmp_path / "existing-directory.png"
    output.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        create_inspection_image(smoke_dataset, 0, output)
    assert output.is_dir()


def test_api_inspection_refuses_existing_symlink_without_changing_target(
    smoke_dataset: Path,
    tmp_path: Path,
) -> None:
    target = tmp_path / "symlink-target.png"
    original = b"linked target bytes\n"
    target.write_bytes(original)
    output = tmp_path / "inspection-link.png"
    try:
        output.symlink_to(target)
    except OSError as error:
        pytest.skip(f"symlink creation unavailable: {error}")
    with pytest.raises(FileExistsError, match="already exists"):
        create_inspection_image(smoke_dataset, 0, output)
    assert output.is_symlink()
    assert target.read_bytes() == original


def test_api_inspection_refuses_existing_hardlink_without_changing_target(
    smoke_dataset: Path,
    tmp_path: Path,
) -> None:
    target = tmp_path / "hardlink-target.png"
    original = b"hard-linked target bytes\n"
    target.write_bytes(original)
    output = tmp_path / "inspection-hardlink.png"
    os.link(target, output)
    with pytest.raises(FileExistsError, match="already exists"):
        create_inspection_image(smoke_dataset, 0, output)
    assert output.read_bytes() == original
    assert target.read_bytes() == original


def test_api_inspection_refuses_output_below_file_parent(
    smoke_dataset: Path,
    tmp_path: Path,
) -> None:
    parent = tmp_path / "not-a-directory"
    original = b"incompatible parent bytes\n"
    parent.write_bytes(original)
    with pytest.raises((FileExistsError, NotADirectoryError)):
        create_inspection_image(smoke_dataset, 0, parent / "inspection.png")
    assert parent.read_bytes() == original


def test_api_inspection_atomic_publish_loses_race_without_clobbering(
    smoke_dataset: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "raced.png"
    competing_bytes = b"competing writer won\n"
    real_link = os.link

    def competing_link(source: os.PathLike[str], destination: os.PathLike[str]) -> None:
        Path(destination).write_bytes(competing_bytes)
        real_link(source, destination)

    monkeypatch.setattr(os, "link", competing_link)
    with pytest.raises(FileExistsError, match="already exists"):
        create_inspection_image(smoke_dataset, 0, output)
    assert output.read_bytes() == competing_bytes
    assert not tuple(tmp_path.glob(".raced.png.*.tmp"))
