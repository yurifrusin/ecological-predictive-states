from __future__ import annotations

import ast
import copy
import io
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from PIL import Image

from epsbench.diagnostics import paired_appearance as p
from epsbench.schema import ModalityPermissionSet
from epsbench.utils.canonical import canonical_json_bytes, sha256_bytes

ROOT = Path(__file__).resolve().parents[1]
NEAR = float(np.float32(0.01))
FAR = 20.0


def synthetic(appearance: str = p.APPEARANCES[0], index: int = 0) -> p.Frame:
    # Arbitrary three vertical label regions; not an evaluation of either candidate scene.
    raw = np.zeros((p.HEIGHT, p.WIDTH), dtype=np.int32)
    raw[:, 53:106] = 1
    raw[:, 106:] = 2
    native = np.zeros((*raw.shape, 3), dtype=np.uint8)
    native[..., 0] = raw + 1
    native_depth = np.full(raw.shape, 0.5, dtype=np.float32)
    scene_map = ((1, 0, 5), (2, 1, 5), (3, 2, 5))
    _, depth = p.derive_pair(native, native_depth, scene_map, NEAR, FAR)
    remap = p.mapping("single_occluder", (0, 1, 2))
    opaque = np.zeros_like(raw)
    for r, label, _ in remap:
        opaque[raw == r] = label
    rgb = np.full((*raw.shape, 3), 200, dtype=np.uint8)
    if appearance == p.APPEARANCES[1]:
        rows, cols = np.indices(raw.shape)
        rgb[((rows + cols) % 2) == 0] = 60
    geometry = p.expected_geometry("single_occluder")
    x = np.array([1.0, 0.0, 0.0])
    y = np.array([0.0, 0.16, 1.0])
    y /= np.linalg.norm(y)
    rotation = np.column_stack((x, y, np.cross(x, y))).reshape(-1).tolist()
    compiled = {
        "raw_geom_ids": dict(zip(p.SURFACES["single_occluder"], (0, 1, 2), strict=True)),
        "raw_geom_world_positions": {n: v[0] for n, v in geometry.items()},
        "raw_geom_compiled_sizes": {n: v[1] for n, v in geometry.items()},
        "raw_geom_types": {n: v[2] for n, v in geometry.items()},
        "raw_geom_world_rotations_row_major": {
            n: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0] for n in geometry
        },
        "camera_field_of_view_degrees": 55.0,
        "camera_world_position": [-0.35, -3.0, 1.25],
        "camera_world_rotation_row_major": rotation,
    }
    stable: dict[str, Any] = {key: 0 for key in p.REGISTRATION_KEYS}
    stable.update(
        {
            "rect": [0, 0, p.WIDTH, p.HEIGHT],
            "offSamples": 0,
            "offWidth": p.WIDTH,
            "offHeight": p.HEIGHT,
            "rnd_depth": False,
            "projection_matrix_float32": np.eye(4).reshape(-1).tolist(),
            "modelview_matrix_float32": np.eye(4).reshape(-1).tolist(),
            "scene_flags": [0] * 10,
            "scene_map": [{"segid_plus_one": a, "objid": b, "objtype": c} for a, b, c in scene_map],
            "near": NEAR,
            "far": FAR,
            "segment_enabled": True,
            "idcolor_enabled": True,
            "context_runtime": {"synthetic": "no_native_evidence"},
        }
    )
    stable.update(
        {
            "readPixelFormat": 6407,
            "readDepthMap": 1,
            "read_buffer": 36064,
            "draw_buffer": 36064,
            "pack_alignment": 1,
            "clip_origin": 36001,
            "clip_depth_mode": 37727,
            "framewidth": 0.0,
            "stereo": 0,
            "mjr_currentBuffer": 1,
            "query_bindings_restored": True,
            "extent": 1.0,
        }
    )
    stable["scene_flags"] = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0]
    stable["offscreen_attachments"] = {
        "offFBO": {
            "present": True,
            "draw_framebuffer_samples": 0,
            "color0": {
                "object_type": 36161,
                "component_type": 35863,
                "internal_format": 32856,
                "samples": 0,
            },
            "depth": {
                "object_type": 36161,
                "component_type": 5126,
                "internal_format": 36013,
                "samples": 0,
            },
        },
        "offFBO_r": {"present": False},
    }
    stable["ngeom"] = 3
    stable["context_runtime"] = {
        "actual_backend": "osmesa",
        "context_module": "mujoco.osmesa",
        "requested_offsamples": 0,
        "actual_offsamples": 0,
        "model_offsamples": 0,
        "width": 160,
        "height": 120,
        "sample_buffers": 0,
        "samples": 0,
        "mjr_off_width": 160,
        "mjr_off_height": 120,
        "color_storage_dimensions": [160, 120],
        "depth_storage_dimensions": [160, 120],
        "attachment_format": 32856,
        "attachment_component_type": 35863,
        "gl_samples": 0,
        "gl_vendor": "SYNTHETIC",
        "gl_renderer": "SYNTHETIC_NO_NATIVE",
        "gl_version": "SYNTHETIC",
    }
    stable["scene_geometry"] = p.expected_draw_geometry(compiled)
    camera = {
        "world_position": [(-0.35, 0.35)[index], -3.0, 1.25],
        "rotation_row_major": rotation,
        "fovy": 55.0,
    }
    stable["scene_cameras"] = p.expected_scene_cameras(camera, NEAR, FAR)
    rgb_state = copy.deepcopy(stable)
    rgb_state.update({"segment_enabled": False, "idcolor_enabled": False})
    rgb_state["scene_flags"][8:10] = [0, 0]
    material = p.expected_material("single_occluder", appearance, compiled)
    evidence = {
        "schema": p.VERSION + ":endpoint",
        "source_head": "a" * 40,
        "source_tree": "b" * 40,
        "config_sha256": sha256_bytes(p.config_bytes()),
        "appearance_record": p.visual_plan("single_occluder", appearance).record,
        "scene_xml_sha256": "c" * 64,
        "compiled": compiled,
        "camera": {
            "world_position": [(-0.35, 0.35)[index], -3.0, 1.25],
            "rotation_row_major": rotation,
            "fovy": 55.0,
        },
        "action": [0.0, 0.7, 0.0],
        "runtime": stable["context_runtime"],
        "mapping": [list(v) for v in remap],
        "scene_map": [list(v) for v in scene_map],
        "near": NEAR,
        "far": FAR,
        "orientation": p.ORIENTATION,
        "paired_stable": stable,
        "rgb_stable": rgb_state,
        "rgb_material": material,
        "paired_material": material,
        "rgb_provenance": {
            "producer": "mujoco.Renderer.render",
            "sdk_renderer_sha256": p.SDK_RENDERER_SHA256,
            "operation": "separate_ordinary_rgb_draw_before_owned_id_depth",
            "same_draw_as_pair": False,
            "index": index,
        },
    }
    return p.Frame(
        "single_occluder",
        appearance,
        index,
        rgb,
        native,
        native_depth,
        raw,
        depth,
        opaque,
        opaque > 0,
        opaque[:, :-1] != opaque[:, 1:],
        opaque[:-1] != opaque[1:],
        evidence,
    )


def test_fixed_config_assets_and_protection() -> None:
    assert (
        p.fixed_config((ROOT / "configs/development/paired_appearance_v1.json").read_bytes())
        == p.FIXED
    )
    bad = copy.deepcopy(p.FIXED)
    bad["roots"][0] += 1
    with pytest.raises(ValueError):
        p.fixed_config(canonical_json_bytes(bad))
    receipt = p.protection_check(ROOT)
    assert len(receipt["inputs"]) == 7
    assert "18/21" in receipt["historical_limit"]
    assert len(p.contexts()) == 8
    assert p.retention_bound()["total_with_two_exports_and_host_metadata"] <= 55 * p.MIB
    a = p.visual_plan("single_occluder", p.APPEARANCES[0])
    b = p.visual_plan("single_occluder", p.APPEARANCES[1])
    assert a.record["slots"] == b.record["slots"]
    assert a.record["seeds"] == b.record["seeds"]
    assert not a.asset_bytes and len(b.asset_bytes) == 3
    for i, slot in enumerate(p.slots("single_occluder")):
        pixels = np.asarray(Image.open(io.BytesIO(b.asset_bytes[f"dev-brick-{i}.png"])))
        assert np.array_equal(pixels, p.brick(slot))
        assert set(map(tuple, pixels.reshape(-1, 3))) == set(p.PALETTE[slot])


def test_protection_collision_is_terminal(tmp_path: Path) -> None:
    for filename in (*p.PROTECTED_SEEDS, *p.PROTECTED_APPEARANCES):
        dest = tmp_path / filename
        dest.parent.mkdir(exist_ok=True, parents=True)
        dest.write_bytes((ROOT / filename).read_bytes())
    path = tmp_path / p.PROTECTED_SEEDS[0]
    path.write_text("root_seed: 2026100601\n")
    with pytest.raises(ValueError, match="seed collision"):
        p.protection_check(tmp_path)
    path.write_bytes((ROOT / p.PROTECTED_SEEDS[0]).read_bytes())
    path = tmp_path / p.PROTECTED_APPEARANCES[0]
    import yaml

    data = yaml.safe_load(path.read_bytes())
    data["profiles"][0]["palette"]["single_occluder_slots"][0]["foreground_rgb"] = list(
        p.PALETTE[0][0]
    )
    data["profiles"][0]["palette"]["single_occluder_slots"][0]["background_rgb"] = list(
        p.PALETTE[0][1]
    )
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="alias"):
        p.protection_check(tmp_path)


def test_raw_zero_background_and_owned_serialization_inspection() -> None:
    frame = synthetic()
    p.validate_frame(frame)
    assert np.all(frame.opaque[:, :53] > 0)  # geom0 remains a controlled surface.
    native = frame.native_id.copy()
    native[0, 0] = 0
    raw, depth = p.derive_pair(
        native, frame.native_depth, ((1, 0, 5), (2, 1, 5), (3, 2, 5)), NEAR, FAR
    )
    assert raw[0, 0] == -1 and raw[0, 1] == 0 and depth.dtype == np.float32
    encoded = p.encode(frame)
    assert len(encoded) < p.ENDPOINT_CAP
    other = p.decode(encoded)
    assert p.repeat_equal(frame, other)
    assert Image.open(io.BytesIO(p.inspect(other))).size == (p.WIDTH, p.HEIGHT)
    with pytest.raises(ValueError):
        frame.rgb[0, 0] = 0
    frame.evidence["orientation"] = "bad"
    with pytest.raises(ValueError):
        p.validate_frame(frame)


@pytest.mark.parametrize(
    "field", ["raw", "depth", "opaque", "controlled", "horizontal", "vertical"]
)
def test_independent_array_corruption(field: str) -> None:
    frame = synthetic()
    array = getattr(frame, field).copy()
    array.flat[0] = not array.flat[0] if array.dtype == np.bool_ else array.flat[0] + 1
    with pytest.raises(ValueError):
        p.validate_frame(replace(frame, **{field: array}))


@pytest.mark.parametrize(
    "field",
    [
        "camera",
        "runtime",
        "rgb_stable",
        "rgb_material",
        "mapping",
        "config_sha256",
        "appearance_record",
    ],
)
def test_provenance_corruption(field: str) -> None:
    frame = synthetic()
    evidence = copy.deepcopy(frame.evidence)
    if field == "rgb_stable":
        evidence[field]["projection_matrix_float32"][0] = 2.0
    elif field == "rgb_material":
        evidence[field]["scene_geoms"] = [{"changed": True}]
    elif field == "camera":
        evidence[field]["world_position"][0] = 1.0
    elif field == "mapping":
        evidence[field][0][1] = 100
    else:
        evidence[field] = {"bad": True} if field != "config_sha256" else "0" * 64
    with pytest.raises((ValueError, KeyError)):
        p.validate_frame(replace(frame, evidence=evidence))


def test_metrics_and_invariance_no_surface_drop() -> None:
    solid = synthetic()
    texture = synthetic(p.APPEARANCES[1])
    result = p.comparison(solid, texture)
    assert result["status"] == "PASS" and len(result["surfaces"]) == 3
    constant = replace(texture, rgb=solid.rgb)
    result = p.comparison(solid, constant)
    assert result["status"] == "FAIL" and result["changed_fraction"] == 0.0
    assert all(s["std"] < 0.025 for s in result["surfaces"])
    # The corner diagonal does not participate in the declared four-neighbour erosion.
    mask = np.ones((p.HEIGHT, p.WIDTH), dtype=np.bool_)
    mask[0, 0] = False
    eroded = p.interior(mask)
    assert eroded[1, 1] and not eroded[0, 10]
    changed = texture.rgb.copy()
    changed[:, 106:] = 100
    assert p.comparison(solid, replace(texture, rgb=changed))["status"] == "FAIL"
    assert not p.repeat_equal(solid, replace(solid, rgb=changed))


def test_permissions_chronology_and_unknown_owner() -> None:
    class Provider:
        calls = 0

        def frame(self, index: int) -> p.Frame:
            self.calls += 1
            return synthetic(index=index)

    provider = Provider()
    ecological = p.ObservationView(provider, ModalityPermissionSet.ecological_only(), 0)
    for operation in (
        lambda: ecological.rgb(0),
        lambda: ecological.metric_depth(0),
        lambda: ecological.instrumentation(0),
        lambda: ecological.raster(1),
    ):
        with pytest.raises(PermissionError):
            operation()
    assert provider.calls == 0
    observed = json.loads(ecological.boundaries(0))
    assert {edge["ownership"] for edge in observed["edges"]} == {"unknown"}
    assert "raw" not in observed and "depth" not in observed
    all_view = p.ObservationView(provider, ModalityPermissionSet.all_modalities(), 0)
    assert all_view.metric_depth(0).dtype == np.float32
    with pytest.raises(ValueError):
        p.ObservationView(provider, cast(Any, {}), 0)


def test_derivation_rejects_wrong_ids_depth_and_scene_map() -> None:
    frame = synthetic()
    native = frame.native_id.copy()
    native[0, 0] = [4, 0, 0]
    with pytest.raises(ValueError):
        p.derive_pair(native, frame.native_depth, ((1, 0, 5),), NEAR, FAR)
    invalid = frame.native_depth.copy()
    invalid[0, 0] = np.nan
    with pytest.raises(ValueError):
        p.derive_pair(frame.native_id, invalid, ((1, 0, 5), (2, 1, 5), (3, 2, 5)), NEAR, FAR)
    with pytest.raises(ValueError):
        p.derive_pair(frame.native_id, frame.native_depth, ((1, 0, 4),), NEAR, FAR)


def builder(family: str) -> Any:
    filename = "single_occluder.py" if family == "single_occluder" else "corridor.py"
    function = "build_scene_xml" if family == "single_occluder" else "build_corridor_scene_xml"
    source = ast.parse((ROOT / "src/epsbench/sim" / filename).read_text())
    node = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == function)
    # Execute only the source string builder, with native/annotation modules absent.
    for arg in node.args.args:
        arg.annotation = None
    node.returns = None
    module = ast.Module(body=[node], type_ignores=[])
    scope: dict[str, Any] = {"_appearance": lambda config, plan: plan}
    exec(
        compile(ast.fix_missing_locations(module), "<source-builder-compatibility>", "exec"), scope
    )
    return scope[function]


def test_record_free_builder_source_compatibility() -> None:
    # XML string checks only; no model compilation, rays or apparatus outcome screening.
    for family in p.FAMILIES:
        plan = p.visual_plan(family, p.APPEARANCES[1])
        camera = SimpleNamespace(
            before_lateral=-0.35, forward=-3.0, height=1.25, field_of_view_degrees=55.0
        )
        config = SimpleNamespace(camera=camera, render=SimpleNamespace(width=160, height=120))
        geometry = SimpleNamespace(
            width=3.0,
            length=6.0,
            wall_height=5.0,
            camera_lateral_position=0.0,
            camera_before_forward_position=0.5,
            camera_height=1.25,
            field_of_view_degrees=55.0,
        )
        xml = (
            builder(family)(config, plan)
            if family == "single_occluder"
            else builder(family)(config, geometry, plan)
        )
        p.validate_xml(family, p.APPEARANCES[1], xml)
        with pytest.raises(ValueError):
            p.validate_xml(family, p.APPEARANCES[1], xml.replace('fovy="55.0"', 'fovy="50.0"'))
    # Structural protocol remains compatible with existing concrete plan dataclass by static typing.
    import epsbench.diagnostics.paired_appearance_native as native

    with pytest.raises(PermissionError, match="before SDK import"):
        native.NativeCapture(
            ROOT,
            "single_occluder",
            p.APPEARANCES[0],
            source_head="a" * 40,
            source_tree="b" * 40,
            progress=lambda *args: None,
        )


def test_ci_exact_isolation_and_narrow_capture_source() -> None:
    ci = (ROOT / ".github/workflows/ci.yml").read_text()
    assert ci.count("github.head_ref == 'codex/paired-appearance-diagnostic-20261006')") == 4
    assert "python scripts/check_paired_appearance_source.py" in ci
    source = (ROOT / "src/epsbench/diagnostics/paired_appearance_native.py").read_text()
    assert "counterfactual" not in source and "compute_analytic_transport" not in source
    assert "self.renderer.render()" in source and "pair = self.paired.capture()" in source
    # The separately authorized execution package now owns the driver; this adapter stays narrow.


def test_codec_expansion_hash_and_duplicate_json_rejection() -> None:
    import zipfile

    frame = synthetic()
    encoded = p.encode(frame)
    with np.load(io.BytesIO(encoded), allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    arrays["rgb"][0, 0] = 1
    stream = io.BytesIO()
    np.savez(stream, **arrays)
    with pytest.raises(ValueError, match="hash corruption"):
        p.decode(stream.getvalue())
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for i in range(10):
            archive.writestr(str(i) + ".npy", b"0" * p.ENDPOINT_CAP)
    with pytest.raises(ValueError, match="expanded endpoint"):
        p.decode(stream.getvalue())
    with pytest.raises(ValueError, match="duplicate"):
        p.strict_json(b'{"x":1,"x":1}')
    with pytest.raises(ValueError, match="nonfinite"):
        p.strict_json(b'{"x":NaN}')


def test_missing_matrix_is_inconclusive_and_observed_negative_preserved() -> None:
    assert p.assess({})["status"] == "INCONCLUSIVE"
    solid = synthetic()
    texture = synthetic(p.APPEARANCES[1])
    # A different independently valid ID image changes the declared mask/lattice: FAIL.
    native = texture.native_id.copy()
    native[0, 0] = [2, 0, 0]
    raw, depth = p.derive_pair(
        native, texture.native_depth, ((1, 0, 5), (2, 1, 5), (3, 2, 5)), NEAR, FAR
    )
    opaque = np.zeros_like(raw)
    for r, label, _ in p.mapping("single_occluder", (0, 1, 2)):
        opaque[raw == r] = label
    different = replace(
        texture,
        native_id=native,
        raw=raw,
        depth=depth,
        opaque=opaque,
        controlled=opaque > 0,
        horizontal=opaque[:, :-1] != opaque[:, 1:],
        vertical=opaque[:-1] != opaque[1:],
    )
    assert p.comparison(solid, different) == {"status": "FAIL", "reason": "appearance invariance"}
    tiny = solid.opaque == int(solid.opaque[0, 0])
    tiny[:] = False
    tiny[0, 0] = True
    assert int(p.interior(tiny).sum()) == 0


def test_pure_lazy_progress_preserves_completed_bytes_and_bounds() -> None:
    # Invoke only source progress forwarding; no constructor, SDK, context or draw.
    from epsbench.diagnostics.paired_appearance_native import NativeCapture

    instance = object.__new__(NativeCapture)
    instance.next_index = 0
    events = []
    instance.progress = lambda stage, index, value: events.append((stage, index, value))
    value = (b"0" * (p.HEIGHT * p.WIDTH * 3), b"0" * (p.HEIGHT * p.WIDTH * 4))
    instance._paired_progress("read_complete", value)
    assert events == [("paired_read_complete", 0, value)]
    with pytest.raises(ValueError):
        instance._paired_progress("read_complete", (b"", b""))
    with pytest.raises(ValueError):
        instance._paired_progress("draw_input", b"0" * 65537)


def synthetic_corridor(appearance: str, index: int) -> p.Frame:
    template = synthetic(appearance, index)
    evidence = copy.deepcopy(template.evidence)
    geometry = p.expected_geometry("corridor")
    raw = np.repeat(np.arange(4, dtype=np.int32), 40)[None, :].repeat(p.HEIGHT, axis=0)
    native = np.zeros((*raw.shape, 3), dtype=np.uint8)
    native[..., 0] = raw + 1
    remap = p.mapping("corridor", (0, 1, 2, 3))
    opaque = np.zeros_like(raw)
    for r, label, _ in remap:
        opaque[raw == r] = label
    compiled = evidence["compiled"]
    compiled["raw_geom_ids"] = dict(zip(p.SURFACES["corridor"], (0, 1, 2, 3), strict=True))
    compiled["raw_geom_world_positions"] = {n: v[0] for n, v in geometry.items()}
    compiled["raw_geom_compiled_sizes"] = {n: v[1] for n, v in geometry.items()}
    compiled["raw_geom_types"] = {n: v[2] for n, v in geometry.items()}
    compiled["raw_geom_world_rotations_row_major"] = {
        n: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0] for n in geometry
    }
    compiled["camera_world_position"] = [0.0, 0.5, 1.25]
    compiled["camera_world_rotation_row_major"] = [1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0]
    evidence["camera"] = {
        "world_position": [0.0, (0.5, 1.2)[index], 1.25],
        "rotation_row_major": [1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0],
        "fovy": 55.0,
    }
    evidence["action"] = [0.7, 0.0, 0.0]
    evidence["mapping"] = [list(v) for v in remap]
    evidence["scene_map"] = [[i + 1, i, 5] for i in range(4)]
    evidence["appearance_record"] = p.visual_plan("corridor", appearance).record
    for key in ("rgb_stable", "paired_stable"):
        state = evidence[key]
        state["ngeom"] = 4
        state["scene_map"] = [{"segid_plus_one": i + 1, "objid": i, "objtype": 5} for i in range(4)]
    evidence["far"] = 30.0
    _, depth = p.derive_pair(
        native, template.native_depth, tuple(tuple(v) for v in evidence["scene_map"]), NEAR, 30.0
    )
    for key in ("rgb_stable", "paired_stable"):
        evidence[key]["far"] = 30.0
        evidence[key]["scene_geometry"] = p.expected_draw_geometry(compiled)
        evidence[key]["scene_cameras"] = p.expected_scene_cameras(evidence["camera"], NEAR, 30.0)
    evidence["rgb_material"] = p.expected_material("corridor", appearance, compiled)
    evidence["paired_material"] = copy.deepcopy(evidence["rgb_material"])
    return replace(
        template,
        family="corridor",
        depth=depth,
        native_id=native,
        raw=raw,
        opaque=opaque,
        controlled=opaque > 0,
        horizontal=opaque[:, :-1] != opaque[:, 1:],
        vertical=opaque[:-1] != opaque[1:],
        evidence=evidence,
    )


def test_complete_synthetic_matrix_and_repeat_failure() -> None:
    # Both families here are mocked label strips, never native candidate outcomes.
    frames = {
        context: tuple(
            (
                synthetic(app, index)
                if family == "single_occluder"
                else synthetic_corridor(app, index)
            )
            for index in range(2)
        )
        for context in p.contexts()
        for family, app, repeat in (context,)
    }
    typed = cast(dict[tuple[p.Family, str, int], tuple[p.Frame, p.Frame]], frames)
    result = p.assess(typed)
    assert result["status"] == "PASS" and result["contexts"] == 8 and result["endpoints"] == 16
    assert len(result["cells"]) == 8
    key: tuple[p.Family, str, int] = ("corridor", p.APPEARANCES[0], 1)
    before, after = typed[key]
    rgb = before.rgb.copy()
    rgb[0, 0] = 1
    typed[key] = (replace(before, rgb=rgb), after)
    assert p.assess(typed)["status"] == "FAIL"


@pytest.mark.parametrize("family", p.FAMILIES)
@pytest.mark.parametrize("appearance", p.APPEARANCES)
@pytest.mark.parametrize(
    "corruption", ("geometry", "camera", "empty_material", "light", "texture", "material")
)
def test_common_draw_corruption_rejects(family: p.Family, appearance: str, corruption: str) -> None:
    frame = (
        synthetic(appearance) if family == "single_occluder" else synthetic_corridor(appearance, 0)
    )
    evidence = copy.deepcopy(frame.evidence)
    if corruption in ("geometry", "camera"):
        field = "scene_geometry" if corruption == "geometry" else "scene_cameras"
        for key in ("rgb_stable", "paired_stable"):
            evidence[key][field][0]["pos"][0] = 123.0
    else:
        for key in ("rgb_material", "paired_material"):
            value = evidence[key]
            if corruption == "empty_material":
                evidence[key] = {"model": {}, "scene_geoms": [], "scene_lights": []}
            elif corruption == "light":
                value["scene_lights"][0]["diffuse"][0] = 0.9
            elif corruption == "texture":
                value["model"]["tex_data_sha256"] = "0" * 64
            else:
                value["scene_geoms"][0]["specular"] = 0.5
    with pytest.raises(ValueError):
        p.validate_frame(replace(frame, evidence=evidence))


@pytest.mark.parametrize(
    "field", ("projection_matrix_float32", "modelview_matrix_float32", "scene_flags")
)
def test_cross_appearance_actual_state_drift_rejects(field: str) -> None:
    solid, texture = synthetic(), synthetic(p.APPEARANCES[1])
    evidence = copy.deepcopy(texture.evidence)
    for key in ("rgb_stable", "paired_stable"):
        evidence[key][field][0 if field != "scene_flags" else 1] = (
            2 if field != "scene_flags" else 1
        )
    changed = replace(texture, evidence=evidence)
    p.validate_frame(changed)
    with pytest.raises(ValueError, match="cross-appearance actual"):
        p.comparison(solid, changed)


@pytest.mark.parametrize("field", ("geom_rgba", "mat_texrepeat", "light_ambient", "cam_ipd"))
def test_common_model_plan_corruption_rejects(field: str) -> None:
    frame = synthetic(p.APPEARANCES[1])
    evidence = copy.deepcopy(frame.evidence)
    for key in ("rgb_material", "paired_material"):
        value = evidence[key]["model"][field]
        if isinstance(value[0], list):
            value[0][0] = 0.123
        else:
            value[0] = 0.123
    with pytest.raises(ValueError, match="closed actual material"):
        p.validate_frame(replace(frame, evidence=evidence))


def test_closed_material_native_extraction_without_sdk() -> None:
    # Distinct explicit fake SDK arrays; only the read-only extraction helper is called.
    from epsbench.diagnostics.paired_appearance_native import _material_state

    frame = synthetic(p.APPEARANCES[1])
    material = frame.evidence["paired_material"]
    model = SimpleNamespace(
        **{k: np.asarray(v) for k, v in material["model"].items() if k != "tex_data_sha256"}
    )
    model.tex_data = np.concatenate([p.brick(slot).reshape(-1) for slot in p.slots(frame.family)])
    scene = SimpleNamespace(
        ngeom=3,
        nlight=1,
        geoms=[SimpleNamespace(**g) for g in material["scene_geoms"]],
        lights=[SimpleNamespace(**g) for g in material["scene_lights"]],
    )
    assert _material_state(model, SimpleNamespace(scene=scene)) == material
    assert material["model"]["mat_texid"][0] == [-1, 0, -1, -1, -1, -1, -1, -1, -1, -1]
    assert material["scene_geoms"][0]["rgba"] == [1.0] * 4
    assert material["scene_lights"][0]["attenuation"] == [0.0] * 3
    assert len(canonical_json_bytes(frame.evidence)) < 65536


def test_invalid_fixed_evidence_is_inconclusive_in_full_matrix() -> None:
    frames = {
        context: tuple(
            synthetic(app, i) if family == "single_occluder" else synthetic_corridor(app, i)
            for i in range(2)
        )
        for context in p.contexts()
        for family, app, _ in (context,)
    }
    key: tuple[p.Family, str, int] = ("single_occluder", p.APPEARANCES[1], 0)
    frame, after = frames[key]
    evidence = copy.deepcopy(frame.evidence)
    for field in ("rgb_material", "paired_material"):
        evidence[field] = {"model": {}, "scene_geoms": [], "scene_lights": []}
    frames[key] = (replace(frame, evidence=evidence), after)
    assert (
        p.assess(cast(dict[tuple[p.Family, str, int], tuple[p.Frame, p.Frame]], frames))["status"]
        == "INCONCLUSIVE"
    )


@pytest.mark.parametrize("family", p.FAMILIES)
@pytest.mark.parametrize("appearance", p.APPEARANCES)
def test_corrected_metadata_stays_inside_existing_caps(family: p.Family, appearance: str) -> None:
    frame = (
        synthetic(appearance) if family == "single_occluder" else synthetic_corridor(appearance, 0)
    )
    p.validate_frame(frame)
    assert len(canonical_json_bytes(frame.evidence)) <= 65536
    assert (
        len(
            canonical_json_bytes(
                {"stable": frame.evidence["rgb_stable"], "material": frame.evidence["rgb_material"]}
            )
        )
        <= 65536
    )
    for raw in p.visual_plan(family, appearance).asset_bytes.values():
        image = Image.open(io.BytesIO(raw))
        assert image.mode == "RGB" and image.size == (128, 128)
        assert "srgb" not in image.info and "icc_profile" not in image.info


def test_canonical_asset_bytes_pixels_and_original_root() -> None:
    assets = p.canonical_assets()
    assert [len(assets[slot]) for slot in range(4)] == [339, 340, 342, 340]
    expected_arrays = (
        "9b5d0c25756b0ddfa6e17e169c03c19b1435099904c341d688b9fe81a0c4cc0c",
        "87060a3a8a7ab010e55e570de38a8fe9469fc7734227e4ddba25d2130ab49782",
        "742a9bd269849b84ce39d5337606fea47e9798e121ff926911ff3d8ddfc6798f",
        "dbf364b0d4aa9370c7db1c913a9f96d2284e97303b4e91371fbb9d16b542a304",
    )
    from epsbench.utils.canonical import logical_array_hash

    for slot, raw in assets.items():
        with Image.open(io.BytesIO(raw)) as image:
            pixels = np.asarray(image)
            assert image.mode == "RGB" and pixels.dtype == np.uint8
            assert logical_array_hash(pixels) == expected_arrays[slot]
            assert np.array_equal(pixels, p.brick(slot))
    records = [p.visual_plan(f, a).record for f in p.FAMILIES for a in p.APPEARANCES]
    assert sha256_bytes(canonical_json_bytes(records)) == (
        "8a049cc723851fd7b4c7253351ae042985bebcec3448e452ffd7b7099b46799e"
    )


@pytest.mark.parametrize("damage", ["missing", "corrupt", "extra", "pixels", "mode"])
def test_canonical_asset_denial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str
) -> None:
    assets = p.canonical_assets()
    for slot, raw in assets.items():
        (tmp_path / f"brick-slot-{slot}.png").write_bytes(raw)
    monkeypatch.setattr(p, "CANONICAL_ASSET_DIRECTORY", tmp_path)
    victim = tmp_path / "brick-slot-3.png"
    if damage == "missing":
        victim.unlink()
    elif damage == "extra":
        (tmp_path / "extra.png").write_bytes(b"unexpected")
    elif damage == "corrupt":
        victim.write_bytes(b"not a PNG")
    else:
        pixels = p.brick(3).copy()
        if damage == "pixels":
            pixels[0, 0, 0] ^= 1
            image = Image.fromarray(pixels)
        else:
            image = Image.fromarray(pixels).convert("RGBA")
        stream = io.BytesIO()
        image.save(stream, format="PNG")
        raw = stream.getvalue()
        victim.write_bytes(raw)
        # Bypass only the byte gate to exercise the independent decoded contract.
        hashes = (*p.CANONICAL_ASSET_SHA256[:3], sha256_bytes(raw))
        monkeypatch.setattr(p, "CANONICAL_ASSET_SHA256", hashes)
    # The solid plan must also deny an incomplete inventory before construction.
    with pytest.raises(ValueError, match="canonical appearance asset"):
        p.visual_plan("single_occluder", p.APPEARANCES[0])
