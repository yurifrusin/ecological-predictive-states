from pathlib import Path

from epsbench.data.identity import (
    analytic_transport_domain,
    compute_ecological_label_hash,
    ecological_label_domain,
)
from epsbench.schema import AvailableDenseOpticalTransport, DatasetManifest, TransitionRecord
from epsbench.utils.canonical import canonical_json_bytes

HISTORICAL_SLICE_1_TRANSITION_VERSION = "0.1.0-dev.1"
HISTORICAL_SLICE_1_EPISODE_0_ECOLOGICAL_HASH = (
    "8f7c7a7e8f70f9e84bf2f256ecf93f8d94f5327abe536c01a6b8f7e87aef3df0"
)
HISTORICAL_DEV_2_SINGLE_OCCLUDER_EPISODE_0_ECOLOGICAL_HASH = (
    "0210bdbce412c6cf199c1cad58b5e8831b97d8a112a0e82f285b6ea1c20df4a3"
)
HISTORICAL_DEV_2_CORRIDOR_ECOLOGICAL_HASHES_BY_BACKEND = {
    "wgl-default": (
        "92e89e4e0eb23b2a50a39cb3803c490654899531a000a3c1ef139e875177f2f8",
        "ea231fe400fabfeb1afea6f9ba58450700734d6b539cfd0a3d540b7ad3345980",
    ),
    "osmesa": (
        "8ca92d6161acc2029421f1182491a96837f01058486fb5ee0c0189a1c2ff9a22",
        "65ed70d5ad141313070978217d84a73e8504c17be95be193568d569940e71189",
    ),
}
EXPECTED_DEV_3_SINGLE_OCCLUDER_EPISODE_0_ECOLOGICAL_HASH_BY_BACKEND = {
    "wgl-default": "028ff38f7465ef4147635230db4427f7522e8771b32d5d77bba54859949ce70c",
}
EXPECTED_DEV_3_CORRIDOR_ECOLOGICAL_HASHES_BY_BACKEND = {
    "wgl-default": (
        "56f7a53536ed91a6ce97b1e20bdd9f2d1b4a796655faaec961335ad0879243d0",
        "cd057369c7d2750021ca9e9eba27671f4cc10ca80d54d8a347a8ae34da417d38",
    ),
}
EXPECTED_DEV_3_ANALYTIC_HASHES_BY_BACKEND = {
    "wgl-default": {
        "single_occluder": (
            "e6d5fc664957c4e68b3c249ce5bd32113755b1c86d5fff8dbeaafc75b550c7cd",
            "e6d5fc664957c4e68b3c249ce5bd32113755b1c86d5fff8dbeaafc75b550c7cd",
        ),
        "corridor": (
            "12ea54fea0b8d9716d12189fcba89397156f7c8111fcb7a488702ffb8240f3bf",
            "9b41e6780bbf89654cda8c5f6d5f4d6326d64bac2594a1afb12db883f91395b6",
        ),
    }
}


def test_ecological_label_hash_matches_locked_cross_platform_regression(
    smoke_dataset: Path,
) -> None:
    manifest = DatasetManifest.model_validate_json(
        (smoke_dataset / "manifest.json").read_text(encoding="utf-8")
    )
    expected = EXPECTED_DEV_3_SINGLE_OCCLUDER_EPISODE_0_ECOLOGICAL_HASH_BY_BACKEND.get(
        manifest.renderer_provenance.backend
    )
    if expected is not None:
        assert manifest.episodes[0].ecological_label_sha256 == expected


def test_corridor_ecological_hashes_match_locked_regressions(corridor_dataset: Path) -> None:
    manifest = DatasetManifest.model_validate_json(
        (corridor_dataset / "manifest.json").read_text(encoding="utf-8")
    )
    expected = EXPECTED_DEV_3_CORRIDOR_ECOLOGICAL_HASHES_BY_BACKEND.get(
        manifest.renderer_provenance.backend
    )
    if expected is not None:
        assert tuple(episode.ecological_label_sha256 for episode in manifest.episodes) == expected


def test_single_occluder_analytic_hash_matches_evidenced_backend_regression(
    smoke_dataset: Path,
) -> None:
    manifest = DatasetManifest.model_validate_json(
        (smoke_dataset / "manifest.json").read_text(encoding="utf-8")
    )
    expected_by_family = EXPECTED_DEV_3_ANALYTIC_HASHES_BY_BACKEND.get(
        manifest.renderer_provenance.backend
    )
    if expected_by_family is not None:
        assert (
            tuple(episode.analytic_transport_sha256 for episode in manifest.episodes)
            == expected_by_family["single_occluder"]
        )


def test_corridor_analytic_hashes_match_evidenced_backend_regression(
    corridor_dataset: Path,
) -> None:
    manifest = DatasetManifest.model_validate_json(
        (corridor_dataset / "manifest.json").read_text(encoding="utf-8")
    )
    expected_by_family = EXPECTED_DEV_3_ANALYTIC_HASHES_BY_BACKEND.get(
        manifest.renderer_provenance.backend
    )
    if expected_by_family is not None:
        assert (
            tuple(episode.analytic_transport_sha256 for episode in manifest.episodes)
            == expected_by_family["corridor"]
        )


def test_historical_slice_1_identity_is_recorded_as_migration_evidence() -> None:
    notes = Path("docs/IMPLEMENTATION_NOTES.md").read_text(encoding="utf-8")
    assert HISTORICAL_SLICE_1_TRANSITION_VERSION in notes
    assert HISTORICAL_SLICE_1_EPISODE_0_ECOLOGICAL_HASH in notes
    assert HISTORICAL_DEV_2_SINGLE_OCCLUDER_EPISODE_0_ECOLOGICAL_HASH in notes
    assert all(
        value in notes
        for values in HISTORICAL_DEV_2_CORRIDOR_ECOLOGICAL_HASHES_BY_BACKEND.values()
        for value in values
    )


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


def test_analytic_identity_domain_excludes_artifacts_ids_and_provenance(
    smoke_dataset: Path,
) -> None:
    manifest = DatasetManifest.model_validate_json(
        (smoke_dataset / "manifest.json").read_text(encoding="utf-8")
    )
    transition = TransitionRecord.model_validate_json(
        (smoke_dataset / manifest.episodes[0].transition.path).read_text(encoding="utf-8")
    )
    transport = transition.analytic_optical_transport
    assert isinstance(transport, AvailableDenseOpticalTransport)
    encoded = canonical_json_bytes(analytic_transport_domain(transport))
    forbidden = (
        b"episode_seed",
        b"appearance",
        b"surface-",
        b"segmentation",
        b"rgb",
        b"depth",
        b"renderer",
        b"provenance",
        b"path",
        b"file_sha256",
        b".npy",
    )
    assert all(value not in encoded for value in forbidden)


def test_manifest_identity_excludes_volatile_run_metadata(smoke_dataset: Path) -> None:
    manifest_bytes = (smoke_dataset / "manifest.json").read_bytes()
    assert b"generated_at" not in manifest_bytes
    assert b"hostname" not in manifest_bytes
    assert str(smoke_dataset.resolve()).encode() not in manifest_bytes


def test_unavailable_and_available_empty_occlusion_have_distinct_identities(
    corridor_dataset: Path,
) -> None:
    manifest = DatasetManifest.model_validate_json(
        (corridor_dataset / "manifest.json").read_text(encoding="utf-8")
    )
    transition = TransitionRecord.model_validate_json(
        (corridor_dataset / manifest.episodes[0].transition.path).read_text(encoding="utf-8")
    )
    available_empty = TransitionRecord.model_validate(
        {
            **transition.model_dump(mode="python"),
            "occlusion": {
                "status": "available",
                "oracle_rule": "counterfactual_occluder_exclusion_v1",
                "relations": (),
            },
            "ecological_label_sha256": "0" * 64,
        }
    )
    assert compute_ecological_label_hash(available_empty) != (
        compute_ecological_label_hash(transition)
    )
