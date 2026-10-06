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


def synthetic(appearance: str = p.APPEARANCES[0], index: int = 0) -> p.Frame:
    # Arbitrary three vertical label regions; not an evaluation of either candidate scene.
    raw = np.zeros((p.HEIGHT, p.WIDTH), dtype=np.int32)
    raw[:, 53:106] = 1
    raw[:, 106:] = 2
    native = np.zeros((*raw.shape, 3), dtype=np.uint8)
    native[..., 0] = raw + 1
    native_depth = np.full(raw.shape, 0.5, dtype=np.float32)
    scene_map = ((1, 0, 5), (2, 1, 5), (3, 2, 5))
    _, depth = p.derive_pair(native, native_depth, scene_map, 0.1, 10.0)
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
            "near": 0.1,
            "far": 10.0,
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
    stable["scene_cameras"] = [
        {
            "pos": [0.0, 0.0, 0.0],
            "forward": [0.0, 0.0, -1.0],
            "up": [0.0, 1.0, 0.0],
            "frustum_near": 0.1,
            "frustum_far": 10.0,
            "frustum_top": 1.0,
            "frustum_bottom": -1.0,
            "frustum_center": 0.0,
            "frustum_width": 2.0,
            "orthographic": 0,
        }
        for _ in range(2)
    ]
    stable["scene_geometry"] = [
        {
            "type": 0,
            "objid": i,
            "objtype": 5,
            "segid": i,
            "category": 0,
            "dataid": 0,
            "pos": v[0],
            "mat": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            "size": v[1],
        }
        for i, v in enumerate(geometry.values())
    ]
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
    rgb_state = copy.deepcopy(stable)
    rgb_state.update({"segment_enabled": False, "idcolor_enabled": False})
    rgb_state["scene_flags"][8:10] = [0, 0]
    material = {"model": {"synthetic": True}, "scene_geoms": [], "scene_lights": []}
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
        "near": 0.1,
        "far": 10.0,
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
        native, frame.native_depth, ((1, 0, 5), (2, 1, 5), (3, 2, 5)), 0.1, 10.0
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
        p.derive_pair(native, frame.native_depth, ((1, 0, 5),), 0.1, 10.0)
    invalid = frame.native_depth.copy()
    invalid[0, 0] = np.nan
    with pytest.raises(ValueError):
        p.derive_pair(frame.native_id, invalid, ((1, 0, 5), (2, 1, 5), (3, 2, 5)), 0.1, 10.0)
    with pytest.raises(ValueError):
        p.derive_pair(frame.native_id, frame.native_depth, ((1, 0, 4),), 0.1, 10.0)


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

    with pytest.raises(RuntimeError, match="forbidden"):
        native.NativeCapture(
            ROOT,
            "single_occluder",
            p.APPEARANCES[0],
            source_head="a" * 40,
            source_tree="b" * 40,
            progress=lambda *args: None,
        )


def test_ci_exact_isolation_and_no_capture_driver() -> None:
    ci = (ROOT / ".github/workflows/ci.yml").read_text()
    assert ci.count("github.head_ref == 'codex/paired-appearance-diagnostic-20261006')") == 4
    assert "python scripts/check_paired_appearance_source.py" in ci
    source = (ROOT / "src/epsbench/diagnostics/paired_appearance_native.py").read_text()
    assert "counterfactual" not in source and "compute_analytic_transport" not in source
    assert "self.renderer.render()" in source and "pair = self.paired.capture()" in source
    assert not (ROOT / "scripts/paired_appearance_execution.py").exists()


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
        native, texture.native_depth, ((1, 0, 5), (2, 1, 5), (3, 2, 5)), 0.1, 10.0
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
        state["scene_geometry"] = [
            {
                "type": 0,
                "objid": i,
                "objtype": 5,
                "segid": i,
                "category": 0,
                "dataid": 0,
                "pos": v[0],
                "mat": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                "size": v[1],
            }
            for i, v in enumerate(geometry.values())
        ]
        state["scene_map"] = [{"segid_plus_one": i + 1, "objid": i, "objtype": 5} for i in range(4)]
    return replace(
        template,
        family="corridor",
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
