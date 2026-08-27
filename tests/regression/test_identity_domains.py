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
REJECTED_PR9_DEV_6_CROSS_PLATFORM_BOUNDARY_HASHES = {
    "single_occluder": (
        "ea99e38626142a7c3b1845c234e5b577493fbf8ff55e40169a96bffaafe84f66",
        "c690400c1c1496da23b4d8ee91af6b52d7e163a415fb7929ffa4494061becbef",
    ),
    "corridor": (
        "77e4aba8aa763827ce3c3a1e20ebc03f134c0c44edb127235f071b32c769f503",
        "fd5c4360c0cc03d6be11bd9917de4909aa6f26129f95d69364d793c42e8e9ab3",
    ),
}
REJECTED_PR9_DEV_6_CROSS_PLATFORM_VISIBILITY_EVENT_HASHES = {
    "single_occluder": (
        "c35fe87f68dde0b282f8069a6700895d135e956a3b5ba2190f3b606f259b8ebb",
        "c9032eb0250598e95902b3c4d5001ea724a48abf6cd66158fbd5374b2d312f2b",
    ),
    "corridor": (
        "2db60ab41fe0f93f1fd574c0d1218d374727199b51eb3bf15625f6264cefbf0a",
        "480e283e5eecc7fc21e3a21f56e114f44c69460681e167b9292c45fa8e7b5a15",
    ),
}
HISTORICAL_ER9_DEV_7_CROSS_PLATFORM_BOUNDARY_HASHES = {
    "single_occluder": (
        "da74e273f5e6786c9093709c07147ea2666db58bb64d687665dbf563e5f9e717",
        "1918d7616cdb39ebdd5a01d6b6b6eed82667bbfff33b06b55f59536c20e99538",
    ),
    "corridor": (
        "24b330294cf15e014d25ae9eb0b883728e2e45dd999eb62b53f86c86a77aab07",
        "a263206271e0126c0ec060b5ed7320ac76d848759980bce29dc8e1a37c350803",
    ),
}
HISTORICAL_ER9_DEV_7_CROSS_PLATFORM_VISIBILITY_EVENT_HASHES = {
    "single_occluder": (
        "dfd7fddd2a460ed1d3147bfca55c9e9034269c4ad0cecdc5de4751b1678cd3d6",
        "70809b9c2febd343ec459ae79be667892d6aa170a3dcc4c5b7bf0c914665392a",
    ),
    "corridor": (
        "129635aab9379aa49593210c5dabdbe0ff2a206f665f54f5c9c8fe931fd9af54",
        "9a49722f9ccdade242d9e72b6707544a7b293e5d9850c0ab8bacaa526d75292b",
    ),
}
NOT_VERIFIED_ER9_DEV_8_CROSS_PLATFORM_BOUNDARY_HASHES = {
    "single_occluder": (
        "203c074e745c47180b2be1c385712b2d4046319e1ba0251c8cc37957c85b57a7",
        "db4a039507db60e6baf16e3f09a47ba0fb059b30836440fb7ed346c413be301b",
    ),
    "corridor": (
        "a24fb5965418145b8abb9ee9ec7f344fb514895a492a4902f6cc0b327077423b",
        "02cca96eb7aa47dfbbae87a8c90eebe5c99b7d9686b6420c5050978f6d31f328",
    ),
}
NOT_VERIFIED_ER9_DEV_8_CROSS_PLATFORM_VISIBILITY_EVENT_HASHES = {
    "single_occluder": (
        "b29978dbee23c77a5b05402f39b85650bdfa6149238a3220ef58dc91b47a62fc",
        "6fb12d8fe4355fccb29d45befdc138fb5e20bf686d787f0c6d241a094c18b12d",
    ),
    "corridor": (
        "ab89d8eb8329b634a35de470337a552baaff8b25ef2344fed28e57f44eb5fb4a",
        "a9a5271000a73c2298ccc933c448ef1cc3ebbbace2cd4d14710091ae43230e44",
    ),
}
EXPECTED_DEV_9_CROSS_PLATFORM_BOUNDARY_HASHES = {
    "single_occluder": (
        "5a6a4792a3554d3b7670c73ac3e2c4d1ed7c438e12f2dabe8065f9dbc230a2c8",
        "485aadab9af42e8ee5ca2194da02eff26626d2b887050675de4fcb192cad3aba",
    ),
    "corridor": (
        "ddbb23daf1f7e17c936e74fd68ad63075385dcfb7c3b60f810acc835e346b235",
        "df0c356d09e3cd601a0c896c5b4b2df34fb52b21a9dccaaefea2d7a4f4570448",
    ),
}
EXPECTED_DEV_9_CROSS_PLATFORM_VISIBILITY_EVENT_HASHES = {
    "single_occluder": (
        "c41807b1245faa9fe1027584ca2dcc2e050956e6a4086bad783e74381abb611a",
        "d9d2c91fe5fd10bf90fcff316dbf6c7956ef8e34baccb3b185971c0aa16192f9",
    ),
    "corridor": (
        "4ff67bac73f9c01abe6fecd8263ff8d2eda1636b9222dd9f6fc7916c7e414d33",
        "9c8a8e1a57d6d5c43b56690812ceafe8a25c662a1e2abfd3d1c2ef035ca0cbb3",
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
        == (EXPECTED_DEV_9_CROSS_PLATFORM_BOUNDARY_HASHES["single_occluder"])
    )
    assert (
        tuple(episode.visibility_event_sha256 for episode in manifest.episodes)
        == (EXPECTED_DEV_9_CROSS_PLATFORM_VISIBILITY_EVENT_HASHES["single_occluder"])
    )


def test_corridor_boundary_and_event_hashes_match_locked_cross_platform_regression(
    corridor_dataset: Path,
) -> None:
    manifest = DatasetManifest.model_validate_json(
        (corridor_dataset / "manifest.json").read_text(encoding="utf-8")
    )
    assert (
        tuple(episode.oriented_boundary_sha256 for episode in manifest.episodes)
        == (EXPECTED_DEV_9_CROSS_PLATFORM_BOUNDARY_HASHES["corridor"])
    )
    assert (
        tuple(episode.visibility_event_sha256 for episode in manifest.episodes)
        == (EXPECTED_DEV_9_CROSS_PLATFORM_VISIBILITY_EVENT_HASHES["corridor"])
    )


def test_historical_slice_1_identity_is_recorded_as_migration_evidence() -> None:
    notes = Path("docs/IMPLEMENTATION_NOTES.md").read_text(encoding="utf-8") + Path(
        "docs/GATE_0B_SLICE_4_BOUNDARY_EVENTS.md"
    ).read_text(encoding="utf-8")
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
    assert all(
        value in notes
        for identity_group in (
            REJECTED_PR9_DEV_6_CROSS_PLATFORM_BOUNDARY_HASHES,
            REJECTED_PR9_DEV_6_CROSS_PLATFORM_VISIBILITY_EVENT_HASHES,
            HISTORICAL_ER9_DEV_7_CROSS_PLATFORM_BOUNDARY_HASHES,
            HISTORICAL_ER9_DEV_7_CROSS_PLATFORM_VISIBILITY_EVENT_HASHES,
            NOT_VERIFIED_ER9_DEV_8_CROSS_PLATFORM_BOUNDARY_HASHES,
            NOT_VERIFIED_ER9_DEV_8_CROSS_PLATFORM_VISIBILITY_EVENT_HASHES,
        )
        for values in identity_group.values()
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
