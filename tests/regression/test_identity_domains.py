from pathlib import Path

from epsbench.data.identity import (
    analytic_transport_domain,
    compute_ecological_label_hash,
    ecological_label_domain,
    oriented_boundary_domain,
    visibility_event_domain,
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
REJECTED_PR5_SINGLE_OCCLUDER_EPISODE_0_ECOLOGICAL_HASH_BY_BACKEND = {
    "wgl-default": "028ff38f7465ef4147635230db4427f7522e8771b32d5d77bba54859949ce70c",
}
REJECTED_PR5_CORRIDOR_ECOLOGICAL_HASHES_BY_BACKEND = {
    "wgl-default": (
        "56f7a53536ed91a6ce97b1e20bdd9f2d1b4a796655faaec961335ad0879243d0",
        "cd057369c7d2750021ca9e9eba27671f4cc10ca80d54d8a347a8ae34da417d38",
    ),
}
REJECTED_PR5_CROSS_PLATFORM_ANALYTIC_HASHES = {
    "single_occluder": (
        "e6d5fc664957c4e68b3c249ce5bd32113755b1c86d5fff8dbeaafc75b550c7cd",
        "e6d5fc664957c4e68b3c249ce5bd32113755b1c86d5fff8dbeaafc75b550c7cd",
    ),
    "corridor": (
        "12ea54fea0b8d9716d12189fcba89397156f7c8111fcb7a488702ffb8240f3bf",
        "9b41e6780bbf89654cda8c5f6d5f4d6326d64bac2594a1afb12db883f91395b6",
    ),
}
REVIEWED_ER5_DEV_4_CROSS_PLATFORM_ANALYTIC_HASHES = {
    "single_occluder": (
        "479d4540835dcc5d204e530766edfcc4bd74b971cc8efe9cbe391317d4ca6740",
        "479d4540835dcc5d204e530766edfcc4bd74b971cc8efe9cbe391317d4ca6740",
    ),
    "corridor": (
        "ddb4dff0fba18d89cd6c15eb988e672c93988d3d4eda2ae625ab317d36631015",
        "a276190abe142bd6859cd0e29983964cbdd17c2cbbf291141ed6f32c6c3c5007",
    ),
}
EXPECTED_DEV_5_CROSS_PLATFORM_ANALYTIC_HASHES = {
    "single_occluder": (
        "77821c734e4a5316851b9e57417a014e8caf292f568314da05e2493608960832",
        "77821c734e4a5316851b9e57417a014e8caf292f568314da05e2493608960832",
    ),
    "corridor": (
        "ffa9b31e91da7a4cdb68ce938beba901beecb3973684bc303d3e362fa01fa09a",
        "64706a77347aa2a98e52d6da023ba80f7a04d1b7a94eaa809124c9b7fe9fca0a",
    ),
}
EXPECTED_DEV_6_CROSS_PLATFORM_BOUNDARY_HASHES = {
    "single_occluder": (
        "ea99e38626142a7c3b1845c234e5b577493fbf8ff55e40169a96bffaafe84f66",
        "c690400c1c1496da23b4d8ee91af6b52d7e163a415fb7929ffa4494061becbef",
    ),
    "corridor": (
        "77e4aba8aa763827ce3c3a1e20ebc03f134c0c44edb127235f071b32c769f503",
        "fd5c4360c0cc03d6be11bd9917de4909aa6f26129f95d69364d793c42e8e9ab3",
    ),
}
EXPECTED_DEV_6_CROSS_PLATFORM_VISIBILITY_EVENT_HASHES = {
    "single_occluder": (
        "c35fe87f68dde0b282f8069a6700895d135e956a3b5ba2190f3b606f259b8ebb",
        "c9032eb0250598e95902b3c4d5001ea724a48abf6cd66158fbd5374b2d312f2b",
    ),
    "corridor": (
        "2db60ab41fe0f93f1fd574c0d1218d374727199b51eb3bf15625f6264cefbf0a",
        "480e283e5eecc7fc21e3a21f56e114f44c69460681e167b9292c45fa8e7b5a15",
    ),
}


def test_corrected_ecological_label_does_not_reuse_rejected_pr5_identity(
    smoke_dataset: Path,
) -> None:
    manifest = DatasetManifest.model_validate_json(
        (smoke_dataset / "manifest.json").read_text(encoding="utf-8")
    )
    rejected = REJECTED_PR5_SINGLE_OCCLUDER_EPISODE_0_ECOLOGICAL_HASH_BY_BACKEND.get(
        manifest.renderer_provenance.backend
    )
    if rejected is not None:
        assert manifest.episodes[0].ecological_label_sha256 != rejected


def test_corrected_corridor_labels_do_not_reuse_rejected_pr5_identities(
    corridor_dataset: Path,
) -> None:
    manifest = DatasetManifest.model_validate_json(
        (corridor_dataset / "manifest.json").read_text(encoding="utf-8")
    )
    rejected = REJECTED_PR5_CORRIDOR_ECOLOGICAL_HASHES_BY_BACKEND.get(
        manifest.renderer_provenance.backend
    )
    if rejected is not None:
        assert tuple(episode.ecological_label_sha256 for episode in manifest.episodes) != rejected


def test_single_occluder_corrected_analytic_hash_matches_locked_cross_platform_regression(
    smoke_dataset: Path,
) -> None:
    manifest = DatasetManifest.model_validate_json(
        (smoke_dataset / "manifest.json").read_text(encoding="utf-8")
    )
    assert (
        tuple(episode.analytic_transport_sha256 for episode in manifest.episodes)
        == EXPECTED_DEV_5_CROSS_PLATFORM_ANALYTIC_HASHES["single_occluder"]
    )


def test_corridor_corrected_analytic_hashes_match_locked_cross_platform_regression(
    corridor_dataset: Path,
) -> None:
    manifest = DatasetManifest.model_validate_json(
        (corridor_dataset / "manifest.json").read_text(encoding="utf-8")
    )
    assert (
        tuple(episode.analytic_transport_sha256 for episode in manifest.episodes)
        == EXPECTED_DEV_5_CROSS_PLATFORM_ANALYTIC_HASHES["corridor"]
    )


def test_single_occluder_boundary_and_event_hashes_match_locked_cross_platform_regression(
    smoke_dataset: Path,
) -> None:
    manifest = DatasetManifest.model_validate_json(
        (smoke_dataset / "manifest.json").read_text(encoding="utf-8")
    )
    assert (
        tuple(episode.oriented_boundary_sha256 for episode in manifest.episodes)
        == (EXPECTED_DEV_6_CROSS_PLATFORM_BOUNDARY_HASHES["single_occluder"])
    )
    assert (
        tuple(episode.visibility_event_sha256 for episode in manifest.episodes)
        == (EXPECTED_DEV_6_CROSS_PLATFORM_VISIBILITY_EVENT_HASHES["single_occluder"])
    )


def test_corridor_boundary_and_event_hashes_match_locked_cross_platform_regression(
    corridor_dataset: Path,
) -> None:
    manifest = DatasetManifest.model_validate_json(
        (corridor_dataset / "manifest.json").read_text(encoding="utf-8")
    )
    assert (
        tuple(episode.oriented_boundary_sha256 for episode in manifest.episodes)
        == (EXPECTED_DEV_6_CROSS_PLATFORM_BOUNDARY_HASHES["corridor"])
    )
    assert (
        tuple(episode.visibility_event_sha256 for episode in manifest.episodes)
        == (EXPECTED_DEV_6_CROSS_PLATFORM_VISIBILITY_EVENT_HASHES["corridor"])
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
    assert all(
        value in notes
        for values in REJECTED_PR5_CROSS_PLATFORM_ANALYTIC_HASHES.values()
        for value in values
    )
    assert all(
        value in notes
        for values in REVIEWED_ER5_DEV_4_CROSS_PLATFORM_ANALYTIC_HASHES.values()
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


def test_boundary_and_event_domains_exclude_privileged_appearance_and_container_state(
    smoke_dataset: Path,
) -> None:
    manifest = DatasetManifest.model_validate_json(
        (smoke_dataset / "manifest.json").read_text(encoding="utf-8")
    )
    transition = TransitionRecord.model_validate_json(
        (smoke_dataset / manifest.episodes[0].transition.path).read_text(encoding="utf-8")
    )
    boundary = canonical_json_bytes(
        oriented_boundary_domain(transition.oriented_boundary_ownership)
    )
    events = canonical_json_bytes(visibility_event_domain(transition.ecological_visibility_events))
    for encoded in (boundary, events):
        assert all(
            value not in encoded
            for value in (
                b"appearance",
                b"raw_geom",
                b"world_",
                b"semantic",
                b"renderer",
                b"file_sha256",
                b".npy",
                str(smoke_dataset.resolve()).encode(),
            )
        )


def test_manifest_identity_excludes_volatile_run_metadata(smoke_dataset: Path) -> None:
    manifest_bytes = (smoke_dataset / "manifest.json").read_bytes()
    assert b"generated_at" not in manifest_bytes
    assert b"hostname" not in manifest_bytes
    assert str(smoke_dataset.resolve()).encode() not in manifest_bytes


def test_available_empty_and_unavailable_occlusion_have_distinct_identities(
    corridor_dataset: Path,
) -> None:
    manifest = DatasetManifest.model_validate_json(
        (corridor_dataset / "manifest.json").read_text(encoding="utf-8")
    )
    transition = TransitionRecord.model_validate_json(
        (corridor_dataset / manifest.episodes[0].transition.path).read_text(encoding="utf-8")
    )
    unavailable = TransitionRecord.model_validate(
        {
            **transition.model_dump(mode="python"),
            "occlusion": {
                "status": "unavailable",
                "reason_category": "oriented_corridor_occlusion_oracle_unavailable",
                "reason": "historical unavailable posture",
            },
            "ecological_label_sha256": "0" * 64,
        }
    )
    assert compute_ecological_label_hash(unavailable) != (compute_ecological_label_hash(transition))
