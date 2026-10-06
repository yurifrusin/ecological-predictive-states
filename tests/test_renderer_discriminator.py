"""Guarded synthetic discriminator contracts. No native SDK/provider access."""

from __future__ import annotations

import copy
import io
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from epsbench.diagnostics import paired_appearance as p
from epsbench.diagnostics import renderer_discriminator as d
from epsbench.diagnostics.appearance_study import CONTRAST_V1, PAIRED_V1
from epsbench.utils.canonical import canonical_json_bytes, sha256_bytes
from tests.test_paired_appearance import synthetic_corridor

REPOSITORY = Path(__file__).resolve().parents[1]


def binding(purpose: str = d.DUMMY) -> d.Binding:
    return d.Binding.model_validate(
        {
            "preparation": d.preparation(REPOSITORY, "a" * 40, "b" * 40).model_dump(),
            "purpose": purpose,
            "image": "sha256:" + "c" * 64,
            "output_id": "e" * 32,
            "token": "f" * 32,
        }
    )


def decision(b: d.Binding) -> d.Decision:
    return d.Decision(
        binding_root=b.root,
        purpose=b.purpose,
        approved=True,
        authorization="Synthetic source control only, no native authority",
    )


def sampler_state(setting: str = "default") -> dict[str, Any]:
    return {
        "active_unit": 33984,
        "active_binding": 2004,
        "unit0_binding": 2004,
        "sampler_binding": 0,
        "texture_env_mode": 8448,
        "texture_matrix": np.eye(4, dtype=np.float32).reshape(-1).tolist(),
        "textures": [
            {
                "index": i,
                "target": 3553,
                "object": 2001 + i,
                "width": 128,
                "height": 128,
                "internal_format": 32849,
                "colorspace": "linear",
                "wrap_s": 10497,
                "wrap_t": 10497,
                "pixels": CONTRAST_V1.asset_pixel_hashes[0],
                "min_lod": -1000.0,
                "max_lod": 1000.0,
                "lod_bias": 0.0,
                "compare_mode": 0,
                "anisotropy": 1.0,
                **dict(zip(d.PARAMS, d.FIXED["sampling"][setting], strict=True)),
            }
            for i in range(4)
        ],
    }


class Sampler:
    def __init__(self, fault: tuple[str, int] | None = None):
        self.state = sampler_state()
        self.fault, self.calls = fault, {"query": 0, "set": 0, "restore": 0}

    def trip(self, name: str) -> None:
        self.calls[name] += 1
        if self.fault == (name, self.calls[name]):
            raise RuntimeError("synthetic one-shot fault")

    def bindings(self) -> tuple[int, int, int]:
        return self.state["active_unit"], self.state["active_binding"], self.state["unit0_binding"]

    def restore_bindings(self, saved: tuple[int, int, int]) -> None:
        self.trip("restore")
        self.state.update(
            zip(("active_unit", "active_binding", "unit0_binding"), saved, strict=True)
        )

    def query(self) -> dict[str, Any]:
        self.trip("query")
        return copy.deepcopy(self.state)

    def set_parameters(self, index: int, values: tuple[int, int, int, int]) -> None:
        self.trip("set")
        self.state["textures"][index].update(zip(d.PARAMS, values, strict=True))


def endpoint(ordinal: int, b: d.Binding, *, control_fault: bool = False) -> d.Endpoint:
    identity = d.MEMBERS[ordinal]
    old = synthetic_corridor(CONTRAST_V1.appearances[1], 1, CONTRAST_V1)
    e = copy.deepcopy(old.evidence)
    e.pop("appearance_record")
    e.pop("action")
    e.update(
        schema=d.SCHEMA + ":endpoint",
        source_head=b.preparation.source_head,
        source_tree=b.preparation.source_tree,
        binding_root=b.root,
        config_sha256=b.preparation.config_file,
        scene_xml_sha256=b.preparation.scene_root,
    )
    e["compiled"]["camera_world_position"] = [0.0, 1.2, 1.25]
    e["mapping"] = [list(v) for v in d.remapping((0, 1, 2, 3))]
    e["rgb_material"] = d.expected_material(identity, e["compiled"])
    e["paired_material"] = copy.deepcopy(e["rgb_material"])
    e["rgb_provenance"].pop("index")
    restoration = {
        key: d.expected_material(d.MEMBERS[2], e["compiled"])["model"][key]
        for key in ("geom_matid", "geom_rgba", "light_ambient", "light_diffuse")
    }
    e["model_restoration"] = {"before": restoration, "after": copy.deepcopy(restoration)}
    setting = "nearest" if identity.arm.endswith("nearest") else "default"
    e["sampler"] = {
        key: sampler_state("default" if key in ("original", "restored") else setting)
        for key in (
            "original",
            "selected",
            "rgb_before",
            "rgb_after",
            "pair_before",
            "pair_after",
            "restored",
        )
    }
    e["sampler"]["restoration"] = "RESTORED"
    arrays = {name: getattr(old, name).copy() for name in d.SPECS}
    opaque = np.zeros((120, 160), dtype=np.int32)
    for raw, label, _ in d.remapping((0, 1, 2, 3)):
        opaque[arrays["raw"] == raw] = label
    arrays.update(
        opaque=opaque,
        controlled=opaque > 0,
        horizontal=opaque[:, :-1] != opaque[:, 1:],
        vertical=opaque[:-1] != opaque[1:],
    )
    arrays["rgb"][:] = 224 if identity.arm.startswith("unit") else 22
    if identity.appearance == d.APPEARANCES[1] and identity.arm == "unit_nearest":
        rows, columns = np.indices((120, 160))
        arrays["rgb"][:] = np.where(((rows // 4 + columns // 4) % 2)[..., None], 224, 32)
    if (
        control_fault
        and identity.arm == "original_nearest"
        and identity.appearance == d.APPEARANCES[0]
    ):
        arrays["rgb"][:] = 23
    return d.Endpoint(identity, arrays, e)


def emit(sink: d.Sink, frame: d.Endpoint, ordinal: int) -> None:
    for stage in d.STAGES:
        value: Any = None
        if stage == "rgb_read_complete":
            value = frame.arrays["rgb"]
        elif stage == "rgb_state_complete":
            value = canonical_json_bytes(
                {"stable": frame.evidence["rgb_stable"], "material": frame.evidence["rgb_material"]}
            )
        elif stage in (
            "paired_draw_input",
            "paired_draw_output",
            "paired_read_input",
            "paired_read_output",
        ):
            value = canonical_json_bytes(frame.evidence["paired_stable"])
        elif stage == "paired_read_complete":
            value = tuple(
                np.flipud(frame.arrays[n]).tobytes() for n in ("native_id", "native_depth")
            )
        elif stage == "sampler_complete":
            value = canonical_json_bytes(frame.evidence["sampler"])
        elif stage == "endpoint_complete":
            value = frame
        sink.progress(stage, ordinal, value)


class Capture:
    def __init__(self, sink: d.Sink, fail: int | None = None, control_fault: bool = False):
        self.sink, self.fail, self.control_fault = sink, fail, control_fault
        self.visited: list[int] = []
        self.closed = False

    def capture(self, ordinal: int) -> d.Endpoint:
        self.visited.append(ordinal)
        if ordinal == self.fail:
            raise ValueError("synthetic invalid evidence")
        frame = endpoint(ordinal, self.sink.binding, control_fault=self.control_fault)
        emit(self.sink, frame, ordinal)
        return frame

    def close(self) -> None:
        self.closed = True


def test_exact_source_membership_protections_and_reuse() -> None:
    assert len(d.MEMBERS) == 16
    assert tuple(m.arm for m in d.MEMBERS[::4]) == d.ARMS
    assert [(m.appearance, m.repeat) for m in d.MEMBERS[:4]] == [
        (a, r) for a in d.APPEARANCES for r in (0, 1)
    ]
    assert d.config((REPOSITORY / d.CONFIG_PATH).read_bytes())["root"] == 2026100605
    proof = d.protection(REPOSITORY)
    assert "18/21" in proof["historical_limit"]
    assert proof["reuse_exception"]["file"] == CONTRAST_V1.asset_hashes[0]
    assert not set(d.seeds().values()) & {
        v
        for s in (PAIRED_V1, CONTRAST_V1)
        for f in p.FAMILIES
        for v in p.seed_domain(f, s).values()
    }
    xml, assets = d.scene_xml()
    assert len(assets) == 4 and len(set(assets.values())) == 1
    assert xml.count("<light ") == 1 and xml.count("<geom ") == 4
    assert xml.count("<material ") == 8 and 'pos="0 1.2 1.25"' in xml
    assert sha256_bytes(next(iter(assets.values()))) == CONTRAST_V1.asset_hashes[0]
    with pytest.raises(ValueError):
        d.config(d.CONFIG_BYTES.replace(b"2026100605", b"2026100606"))


def test_closed_binding_denies_cross_purpose_config_before_provider() -> None:
    b = binding()
    for key in ("config_file", "asset_root", "membership_root", "scene_root"):
        value = b.preparation.model_dump()
        value[key] = "0" * 64
        with pytest.raises(ValidationError):
            d.Preparation.model_validate(value)
    with pytest.raises(ValidationError):
        d.Binding.model_validate(
            {**b.model_dump(), "purpose": "appearance_contrast_calibration_native_v1"}
        )
    with pytest.raises(PermissionError, match="before SDK access"):
        d.require_native(REPOSITORY, b)
    with pytest.raises(PermissionError, match="before SDK access"):
        d.NativeCapture(REPOSITORY, b, None)  # type: ignore[arg-type]
    assert not d.candidate_runtime()
    with pytest.raises(ValidationError):
        d.Decision.model_validate({**decision(b).model_dump(), "approved": 1})


@pytest.mark.parametrize("fault", [None, ("query", 1), ("query", 2), ("set", 1), ("set", 4)])
def test_sampler_restoration_all_partial_override_failures(fault: tuple[str, int] | None) -> None:
    sampler = Sampler(fault)
    original = sampler.bindings()
    record: dict[str, Any] = {}
    try:
        with d.sampler_override(sampler, "nearest", record):
            d.validate_sampler(sampler.query(), "nearest")
            raise RuntimeError("synthetic draw/read/callback fault")
    except (ValueError, RuntimeError):
        pass
    d.validate_sampler(sampler.query(), "default")
    assert sampler.bindings() == original
    assert record["restoration"] == "RESTORED"


def test_sampler_restoration_failure_and_wrong_inventory_fail_closed() -> None:
    sampler = Sampler(("restore", 1))
    record: dict[str, Any] = {}
    with pytest.raises(ValueError, match="restoration failure"):
        with d.sampler_override(sampler, "nearest", record):
            pass
    assert record["restoration"] == "FAILED_CLOSED"
    for key, value in (("mag", 9728), ("width", 64), ("object", 0), ("pixels", "0" * 64)):
        state = sampler_state()
        state["textures"][0][key] = value
        with pytest.raises(ValueError):
            d.validate_sampler(state, "default")


def test_source_derived_uv_lighting_and_mathematical_ties() -> None:
    assert d.box_uv((0.5, -0.25, 1.0), 2) == (0.75, 0.625)
    assert d.box_uv((1.0, -0.5, 0.25), 0) == (0.25, 0.375)
    assert d.box_uv((-0.5, -1.0, 0.25), 1) == (0.25, 0.375)
    assert d.nearest_texel((0.5, 0.25)) is None
    assert d.nearest_texel((0.5 + 0.5 / 128, 0.25 + 0.5 / 128)) == (32, 64)
    assert d.ideal_gain((1, 0, 0), False) == 0.1
    assert d.ideal_gain((0, 0, 1), False) == 0.1 + 0.7 / math.sqrt(1.0625)
    assert d.ideal_gain((0, -1, 0), False) == 0.1 + 0.175 / math.sqrt(1.0625)
    assert d.ideal_gain((1, 0, 0), True) == 1.0
    rays = d.ideal_rays()
    assert sum(int((rays["category"] == i).sum()) for i in (0, 1, 2)) == 19200
    assert all(np.any((rays["owner"] == i) & (rays["category"] == 1)) for i in range(4))


def test_complete_factorial_preserves_reference_fail_replay_inspection_and_capacity(
    tmp_path: Path,
) -> None:
    b = binding()
    sink = d.Sink(tmp_path / "attempt", b, decision(b))
    capture = Capture(sink)
    result = d._drive(capture, sink)
    assert capture.visited == list(range(16)) and capture.closed
    assert result["status"] == "DIAGNOSTIC_COMPLETE", result
    assert result["arms"]["original_default"]["status"] == "FAIL"
    assert result["arms"]["unit_nearest"]["status"] == "PASS"
    assert "joint dependence" in result["interpretation"]
    assert result["absolute_native_uv_transfer_qualification"] == "UNRESOLVED"
    assert all(v["qualification"] == "UNRESOLVED" for v in result["ideal_diagnostics"])
    frames = d.replay(sink.root, b)
    assert len(frames) == 16
    assert d.assess(frames, b) == result
    from PIL import Image

    with Image.open(io.BytesIO(d.inspect(frames[0], b))) as image:
        assert image.size == (160, 120)
    bound = d.retention_bound()
    assert bound["endpoint_reserved_bytes"] <= p.MIB
    assert max(sink.used) <= bound["endpoint_reserved_bytes"]
    assert sink.shared <= 2 * p.MIB
    assert len(canonical_json_bytes(result)) <= p.MIB
    assert bound["total_with_two_exports_and_host_metadata"] == 55 * p.MIB
    # Altered raw read cannot be hidden behind recomputed derived arrays.
    raw = sink.root / "endpoints/e00/read_id.bin"
    raw.write_bytes(b"0" * raw.stat().st_size)
    with pytest.raises(ValueError, match="corruption"):
        d.replay(sink.root, b)


def test_unsupported_direct_control_retains_valid_matrix(tmp_path: Path) -> None:
    b = binding()
    sink = d.Sink(tmp_path / "control", b, decision(b))
    result = d._drive(Capture(sink, control_fault=True), sink)
    assert result["status"] == "DIAGNOSTIC_COMPLETE"
    assert result["solid_sampling_controls"][0]["status"] == "CONTROL_NOT_SUPPORTED"
    assert result["sampling_only_explanation"] == "UNQUALIFIED_CONTROL_NOT_SUPPORTED"
    assert len(d.replay(sink.root, b)) == 16


def test_invalid_evidence_stops_once_preserves_prefix_and_no_resume(tmp_path: Path) -> None:
    b = binding()
    sink = d.Sink(tmp_path / "failure", b, decision(b))
    capture = Capture(sink, fail=5)
    result = d._drive(capture, sink)
    assert result["status"] == "INCONCLUSIVE" and result["endpoints"] == 5
    assert capture.visited == list(range(6)) and capture.closed
    assert len(d.replay(sink.root, b)) == 5
    with pytest.raises(ValueError, match="collision"):
        d.Sink(sink.root, b, decision(b))
    with pytest.raises(ValueError, match="endpoint cap"):
        sink.put("endpoints/e05/oversize.bin", b"0" * (p.MIB + 1))
    assert sink.failed


def test_repeat_and_invariance_and_restoration_tampering_are_inconclusive() -> None:
    b = binding()
    frames = [endpoint(i, b) for i in range(16)]
    frame = frames[1]
    arrays = {name: a.copy() for name, a in frame.arrays.items()}
    arrays["rgb"][0, 0] = 1
    frames[1] = replace(frame, arrays=arrays)
    assert d.assess(frames, b)["status"] == "INCONCLUSIVE"
    e = copy.deepcopy(frame.evidence)
    e["model_restoration"]["after"]["light_diffuse"] = [[0, 0, 0]]
    with pytest.raises(ValueError, match="restoration"):
        d.validate_endpoint(replace(frame, evidence=e), b)
    e = copy.deepcopy(frame.evidence)
    e["sampler"]["pair_after"]["textures"][0]["object"] = 2222
    with pytest.raises(ValueError, match="identity drift"):
        d.validate_endpoint(replace(frame, evidence=e), b)
