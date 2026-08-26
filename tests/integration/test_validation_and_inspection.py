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
