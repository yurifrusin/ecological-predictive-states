"""Synthetic sequence source/codec checks under the native import guard."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar
from xml.etree.ElementTree import fromstring

import numpy as np
import pytest

from epsbench.diagnostics.causal_history_core import Array, build_state
from epsbench.diagnostics.causal_history_native import (
    Capture,
    ProducedSequence,
    admit_flow,
    build_sequence_xml,
    produce_sequence,
)
from epsbench.diagnostics.causal_history_sequence import (
    HEIGHT,
    LOGICAL_OUTPUT_BYTES,
    MAX_SEQUENCE_BYTES,
    MAX_SIX_BYTES,
    MEMBERS,
    WIDTH,
    ArtifactProvider,
    FrameEvidence,
    candidate,
    canonical,
    commands,
    encode_sequence,
    parse,
    retention_estimate,
    validate_manifest,
)
from epsbench.schema import Modality, ModalityPermissionSet

CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs/development/causal_history_fixture_design_v1.json"
)


class FakeBackend:
    raw_ids = (10, 20, 30)
    compiled: ClassVar[dict[str, Any]] = {
        "raw_geom_ids": {"support_surface": 10, "occluding_surface": 20, "background_surface": 30},
        "raw_geom_world_positions": {
            n: [0.0, 0.0, 0.0]
            for n in ("support_surface", "occluding_surface", "background_surface")
        },
        "raw_geom_compiled_sizes": {
            n: [1.0, 1.0, 1.0]
            for n in ("support_surface", "occluding_surface", "background_surface")
        },
        "raw_geom_types": {
            "support_surface": "plane",
            "occluding_surface": "box",
            "background_surface": "box",
        },
        "raw_geom_world_rotations_row_major": {
            n: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
            for n in ("support_surface", "occluding_surface", "background_surface")
        },
        "camera_field_of_view_degrees": 60.0,
        "camera_world_position": [0.0, -3.0, 0.5],
        "camera_world_rotation_row_major": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        "statistics": {
            "meanmass": 1.0,
            "meaninertia": 1.0,
            "meansize": 1.0,
            "extent": 8.0,
            "center": [0.0, 0.0, 0.0],
        },
        "visual": {"znear": 0.01, "zfar": 20.0, "offsamples": 0},
    }

    def __init__(self, xml: str):
        self.xml = xml
        self.positions: list[float] = []
        self.draws: list[str] = []
        self.closed = False
        self.raw = np.full((HEIGHT, WIDTH), 10, dtype=np.int32)
        self.raw[:, 80:] = 20

    def capture(self, position: float) -> Capture:
        self.positions.append(position)
        self.draws.extend(("rgb", "paired"))
        raw_boundaries = tuple(
            {
                "frame_index": 0,
                "axis": "horizontal",
                "row": r,
                "column": 79,
                "negative_raw_geom_id": 10,
                "positive_raw_geom_id": 20,
                "kind": "unresolved_boundary",
                "owner_side": "none",
                "owner_raw_geom_id": None,
                "negative_counterfactual_next_raw_geom_id": 30,
                "positive_counterfactual_next_raw_geom_id": 30,
                "on_projected_attachment_locus": False,
            }
            for r in range(HEIGHT)
        )
        evidence = FrameEvidence(
            camera={
                "world_position": [position, -3.0, 0.5],
                "world_rotation_row_major": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                "vertical_field_of_view_degrees": 60.0,
            },
            compiled=self.compiled,
            paired_stable={"stable": "synthetic"},
            scene_map=({"objid": 10}, {"objid": 20}),
            near=float(np.float32(0.01)) * 8,
            far=160.0,
            orientation="synthetic",
            raw_boundaries=raw_boundaries,
            attachment={"included": True},
        )
        arrays: tuple[Array, ...] = (
            self.raw.copy(),
            np.ones((HEIGHT, WIDTH), dtype=np.float32),
            np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8),
            np.ones((HEIGHT, WIDTH), dtype=np.float32),
            *(np.full((HEIGHT, WIDTH), 30, dtype=np.int32) for _ in range(3)),
        )
        return Capture(
            np.full((HEIGHT, WIDTH, 3), int(position * 8) % 256, dtype=np.uint8),
            self.raw.copy(),
            position,
            evidence,
            arrays,
            raw_boundaries,
            {"host": "synthetic-host", "handle": len(self.positions)},
        )

    def transport(self, before: Capture, after: Capture) -> tuple[Any, Any]:
        vectors = np.zeros((HEIGHT, WIDTH, 2), dtype=np.int32)
        validity = np.ones((HEIGHT, WIDTH), dtype=np.uint8)
        reasons = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
        direction = SimpleNamespace(
            vectors_fixed=vectors,
            validity=validity,
            reasons=reasons,
            target_hit_assignment=self.raw.copy(),
        )
        transport = SimpleNamespace(
            forward=direction,
            backward=direction,
            before_surface_assignment=self.raw.copy(),
            after_surface_assignment=self.raw.copy(),
            before_boundary_ambiguous=np.zeros((HEIGHT, WIDTH), dtype=np.bool_),
            after_boundary_ambiguous=np.zeros((HEIGHT, WIDTH), dtype=np.bool_),
        )
        row, col = np.indices((HEIGHT, WIDTH), dtype=np.int32)
        return transport, np.stack((col, row), axis=-1)

    def close(self) -> None:
        self.closed = True


def produced(member: str = "chf-v1-p1-a") -> ProducedSequence:
    return produce_sequence(CONFIG.read_bytes(), member, factory=FakeBackend)


def encode(
    result: ProducedSequence | None = None, member: str = "chf-v1-p1-a"
) -> tuple[bytes, dict[str, bytes]]:
    result = produced(member) if result is None else result
    config, _, _ = candidate(CONFIG.read_bytes(), member)
    return encode_sequence(
        config=config,
        member=member,
        source_head="1" * 40,
        source_tree="2" * 40,
        image_digest="sha256:" + "3" * 64,
        renderer_identity={"runtime": "synthetic-not-qualified"},
        sequence=result.sequence,
        frames=result.frames,
        flows=result.flows,
    )


def test_six_sequences_one_compile_mapping_and_24_48_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    import epsbench.diagnostics.causal_history_native as module

    factories: list[FakeBackend] = []
    mappings: list[int] = []
    original = module.opaque_mapping

    def mapping(seed: int, raw_ids: tuple[int, ...]) -> tuple[tuple[int, int, str], ...]:
        mappings.append(seed)
        return original(seed, raw_ids)

    def factory(xml: str) -> FakeBackend:
        backend = FakeBackend(xml)
        factories.append(backend)
        return backend

    monkeypatch.setattr(module, "opaque_mapping", mapping)
    for member in MEMBERS:
        result = produce_sequence(CONFIG.read_bytes(), member, factory=factory)
        manifest, artifacts = encode(result, member)
        env = validate_manifest(manifest, tuple(artifacts))
        for i, transition in enumerate(env.flows):
            assert transition.source_frame == env.frames[i].identity
            assert transition.target_frame == env.frames[i + 1].identity
        assert all("raw" not in parse(b) for f in result.frames for b in f.optical.boundaries)
        assert all(
            len(f.optical.identities) == 2 for f in result.frames
        )  # unseen third token absent
    assert len(factories) == len(mappings) == 6
    assert sum(len(b.positions) for b in factories) == 24
    assert sum(len(b.draws) for b in factories) == 48
    assert all(b.closed for b in factories)
    assert len(set(mappings)) == 6


def test_config_binding_xml_and_commands_before_backend() -> None:
    config, pair, _ = candidate(CONFIG.read_bytes(), "chf-v1-p2-a")
    xml = fromstring(build_sequence_xml(CONFIG.read_bytes(), "chf-v1-p2-a"))
    statistic = xml.find("statistic")
    assert statistic is not None and statistic.attrib == {
        "meanmass": "1",
        "meaninertia": "1",
        "meansize": "1",
        "extent": "8",
        "center": "0 0 0",
    }
    quality = xml.find("visual/quality")
    assert quality is not None and quality.get("offsamples") == "0"
    support = xml.find("worldbody/geom")
    assert support is not None and support.get("size") == "4.0 7.0 0.1"
    assert [c.lateral for c in commands(config, "chf-v1-p2-a")] == [-0.5, -2, 0.125, 2.125]
    assert pair["decision"] == 3
    for key in config:
        changed = json.loads(canonical(config))
        del changed[key]
        with pytest.raises(ValueError):
            produce_sequence(
                canonical(changed), "chf-v1-p2-a", factory=lambda _: pytest.fail("compile")
            )
    for group, key in (("camera", "up_y"), ("appearance", "ambient"), ("support", "z")):
        changed = json.loads(canonical(config))
        changed[group][key] = 999
        with pytest.raises(ValueError):
            build_sequence_xml(canonical(changed), "chf-v1-p2-a")
    changed = json.loads(canonical(config))
    changed["pairs"][1]["poses"].reverse()
    with pytest.raises(ValueError):
        build_sequence_xml(canonical(changed), "chf-v1-p2-a")
    with pytest.raises(ValueError):
        candidate(CONFIG.read_bytes(), "old-a1-episode")


def test_roundtrip_owned_provenance_and_volatile_separation() -> None:
    result = produced()
    manifest, artifacts = encode(result)
    volatile = replace(result, operational=({"host": "different", "handle": 999},))
    assert encode(volatile) == (manifest, artifacts)
    provider = ArtifactProvider(
        manifest,
        tuple(artifacts),
        artifacts.__getitem__,
        ModalityPermissionSet.ecological_only(),
        2,
    )
    state = build_state(provider.projection(), result.flows[2].optical.command)
    assert state.current.sequence_index == 2 and len(state.tokens) == 2
    assert np.array_equal(provider.frame(1).segmentation, result.frames[1].optical.segmentation)
    with pytest.raises(ValueError):
        provider.frame(1).segmentation.setflags(write=True)
    env = provider.envelope
    env.config.clear()
    assert provider.envelope.config
    changed = json.loads(manifest)
    changed["source_tree"] = "4" * 40
    with pytest.raises(ValueError):
        validate_manifest(canonical(changed), tuple(artifacts))


def test_before_access_future_metric_and_rgb_denials() -> None:
    manifest, artifacts = encode()
    reads: list[str] = []

    def read(path: str) -> bytes:
        reads.append(path)
        return artifacts[path]

    provider = ArtifactProvider(
        manifest, tuple(artifacts), read, ModalityPermissionSet.ecological_only(), 2
    )
    assert reads == []
    for operation in (
        lambda: provider.frame(3),
        lambda: provider.flow(2),
        lambda: provider.rgb(1),
        lambda: provider.instrumentation(0),
        lambda: provider.projection().frame(3),
    ):
        with pytest.raises(PermissionError):
            operation()
        assert reads == []
    assert provider.frame(0).sequence_index == 0
    assert all(path.startswith("frame-0/") and "instrument" not in path for path in reads)
    rgb = ArtifactProvider(
        manifest,
        tuple(artifacts),
        read,
        ModalityPermissionSet(allowed=frozenset({Modality.RGB})),
        1,
    )
    assert rgb.rgb(1).dtype == np.uint8
    with pytest.raises(PermissionError):
        rgb.frame(1)


def test_flow_mismatch_retained_and_denied_without_rewriting() -> None:
    result = produced()
    flow = result.flows[0]
    actual = flow.arrays[0].copy()
    target = actual.copy()
    target[0, 0] = 30
    source = actual.copy()
    source[0, 1] = 30
    evidence, sm, tm = admit_flow(
        flow.optical.validity, flow.optical.reasons, source, actual, target, flow.arrays[9]
    )
    assert not evidence.admitted and evidence.source_mismatch_count == 1
    assert evidence.target_mismatch_count == 2 and sm[0, 1] and tm[0, 0]
    assert evidence.reason_counts == (HEIGHT * WIDTH, 0, 0, 0, 0)
    bad = replace(flow, evidence=evidence, arrays=(*flow.arrays[:10], sm, tm))
    retained = replace(result, flows=(bad, *result.flows[1:]))
    manifest, artifacts = encode(retained)
    reads: list[str] = []

    def read_bad(p: str) -> bytes:
        reads.append(p)
        return artifacts[p]

    provider = ArtifactProvider(
        manifest,
        tuple(artifacts),
        read_bad,
        ModalityPermissionSet.ecological_only(),
        2,
    )
    with pytest.raises(ValueError, match="admission failed"):
        provider.flow(0)
    assert reads == []
    assert np.array_equal(flow.optical.validity, bad.optical.validity)
    assert np.array_equal(flow.optical.vectors, bad.optical.vectors)


@pytest.mark.parametrize(
    "mutation",
    [
        "extra",
        "missing",
        "duplicate",
        "traversal",
        "shape",
        "dtype",
        "hash",
        "join",
        "command",
        "index",
    ],
)
def test_strict_manifest_rejections(mutation: str) -> None:
    manifest, artifacts = encode()
    obj = json.loads(manifest)
    paths = tuple(artifacts)
    if mutation == "extra":
        obj["unexpected"] = True
    elif mutation == "missing":
        paths = paths[1:]
    elif mutation == "duplicate":
        paths = (*paths, paths[0])
    elif mutation == "traversal":
        obj["artifacts"][0]["path"] = "../evil.json"
    elif mutation == "shape":
        obj["artifacts"][2]["shape"] = [10000000, 10000000]
    elif mutation == "dtype":
        obj["artifacts"][2]["dtype"] = "|O"
    elif mutation == "hash":
        obj["artifacts"][0]["sha256"] = "0" * 64
    elif mutation == "join":
        obj["flows"][0]["target_frame"] = obj["frames"][0]["identity"]
    elif mutation == "command":
        obj["flows"][0]["command"]["lateral"] = [-2, 2]
    elif mutation == "index":
        obj["frames"][1]["index"] = 3
    with pytest.raises((ValueError, KeyError)):
        validate_manifest(canonical(obj), paths)


def test_byte_and_object_array_bounds_and_hash_reads() -> None:
    assert MAX_SIX_BYTES == 6 * MAX_SEQUENCE_BYTES < LOGICAL_OUTPUT_BYTES
    bound = retention_estimate()
    assert bound["staging"] <= 255 * 1024**2
    assert bound["archive_with_framing_terminal"] <= 768 * 1024**2
    assert bound["total"] <= LOGICAL_OUTPUT_BYTES
    result = produced()
    invalid = replace(result.frames[0], rgb=np.full((HEIGHT, WIDTH, 3), None, dtype=object))
    with pytest.raises(ValueError, match="nonobject"):
        encode(replace(result, frames=(invalid, *result.frames[1:])))
    manifest, artifacts = encode(result)
    spec = validate_manifest(manifest, tuple(artifacts)).frames[0]
    artifacts[spec.segmentation] = b"bad"
    provider = ArtifactProvider(
        manifest,
        tuple(artifacts),
        artifacts.__getitem__,
        ModalityPermissionSet.ecological_only(),
        2,
    )
    with pytest.raises(ValueError, match="size/hash"):
        provider.frame(0)
    with pytest.raises(ValueError, match="duplicate"):
        parse(b'{"x":1,"x":2}')
    with pytest.raises(ValueError):
        parse(b'{"x":NaN}')
    with pytest.raises(ValueError, match="string ceiling"):
        canonical({"oversized": "x" * 65537})


def test_mandatory_evidence_mapping_and_failure_cleanup() -> None:
    from epsbench.diagnostics.causal_history_sequence import SequenceEvidence

    result = produced()
    with pytest.raises(ValueError):
        FrameEvidence.model_validate_json(b"{}")
    with pytest.raises(ValueError):
        SequenceEvidence(
            mapping=((10, 1, "surface-0000000000000001"),) * 3,
            compiled=result.sequence.compiled,
            xml_sha256="1" * 64,
        )
    with pytest.raises(ValueError):
        data = result.frames[0].evidence.model_dump(mode="json")
        data["camera"] = {}
        FrameEvidence.model_validate_json(canonical(data))

    class FailedBackend(FakeBackend):
        def capture(self, position: float) -> Capture:
            raise ValueError("synthetic capture failure")

    backend = FailedBackend("synthetic")
    with pytest.raises(ValueError, match="synthetic capture failure"):
        produce_sequence(CONFIG.read_bytes(), "chf-v1-p1-a", factory=lambda _: backend)
    assert backend.closed
    manifest, artifacts = encode(result)
    reader = ArtifactProvider(
        manifest, tuple(artifacts), artifacts.__getitem__, ModalityPermissionSet.all_modalities(), 3
    )
    assert reader.sequence_instrumentation().mapping == result.sequence.mapping
    assert reader.completed_instrumentation(0)[0].admitted
    assert reader.instrumentation(0)[0].near == result.frames[0].evidence.near


def test_valid_transport_with_uncontrolled_source_fails_admission() -> None:
    flow = produced().flows[0]
    uncontrolled = np.full((HEIGHT, WIDTH), -1, dtype=np.int32)
    evidence, mismatch, _ = admit_flow(
        flow.optical.validity,
        flow.optical.reasons,
        uncontrolled,
        uncontrolled,
        uncontrolled,
        flow.arrays[9],
    )
    assert not evidence.admitted and mismatch.all()
