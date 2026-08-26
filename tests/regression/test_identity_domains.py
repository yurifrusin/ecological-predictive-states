from pathlib import Path

from epsbench.data.identity import ecological_label_domain
from epsbench.schema import DatasetManifest, TransitionRecord
from epsbench.utils.canonical import canonical_json_bytes

EXPECTED_EPISODE_0_ECOLOGICAL_HASH = (
    "8f7c7a7e8f70f9e84bf2f256ecf93f8d94f5327abe536c01a6b8f7e87aef3df0"
)


def test_ecological_label_hash_matches_locked_cross_platform_regression(
    smoke_dataset: Path,
) -> None:
    manifest = DatasetManifest.model_validate_json(
        (smoke_dataset / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest.episodes[0].ecological_label_sha256 == EXPECTED_EPISODE_0_ECOLOGICAL_HASH


def test_ecological_hash_domain_excludes_metric_appearance_and_privileged_data(
    smoke_dataset: Path,
) -> None:
    manifest = DatasetManifest.model_validate_json(
        (smoke_dataset / "manifest.json").read_text(encoding="utf-8")
    )
    transition = TransitionRecord.model_validate_json(
        (smoke_dataset / manifest.episodes[0].transition.path).read_text(encoding="utf-8")
    )
    encoded = canonical_json_bytes(ecological_label_domain(transition))
    forbidden = (
        b"depth_",
        b"camera_",
        b"rgb_",
        b"appearance",
        b"raw_geom",
        str(smoke_dataset.resolve()).encode(),
    )
    assert all(value not in encoded for value in forbidden)
    assert b"executed_action" not in encoded  # action values are included without a modality name
    assert b"delta_lateral" in encoded


def test_manifest_identity_excludes_volatile_run_metadata(smoke_dataset: Path) -> None:
    manifest_bytes = (smoke_dataset / "manifest.json").read_bytes()
    assert b"generated_at" not in manifest_bytes
    assert b"hostname" not in manifest_bytes
    assert str(smoke_dataset.resolve()).encode() not in manifest_bytes
