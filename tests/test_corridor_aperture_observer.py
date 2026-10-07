"""Execute extracted strict observation source with local fakes, never import SDKs."""

from __future__ import annotations

import ast
import builtins
import copy
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def extract(path: str, names: set[str], namespace: dict[str, Any]) -> None:
    source = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    nodes = [
        node
        for node in source.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
    ]
    assert {node.name for node in nodes} == names
    exec(
        compile(ast.Module(body=cast(list[ast.stmt], nodes), type_ignores=[]), path, "exec"),
        namespace,
    )


class FakeGL:
    GL_RENDERBUFFER_BINDING = 1
    GL_RENDERBUFFER = 2
    GL_RENDERBUFFER_WIDTH = 3
    GL_RENDERBUFFER_HEIGHT = 4
    GL_SAMPLE_BUFFERS = 5
    GL_SAMPLES = 6

    def __init__(self, width: int, height: int) -> None:
        self.bound = 77
        self.storage: dict[int, list[int]] = {10: [width, height], 11: [width, height]}
        self.integers: dict[int, Any] = {5: 0, 6: 0}
        self.errors = [0, 0]
        self.storage_error = False
        self.bindings: list[int] = []

    def glGetError(self) -> int:
        return self.errors.pop(0)

    def glGetIntegerv(self, key: int) -> Any:
        return self.bound if key == self.GL_RENDERBUFFER_BINDING else self.integers[key]

    def glBindRenderbuffer(self, target: int, identity: int) -> None:
        assert target == self.GL_RENDERBUFFER
        self.bound = identity
        self.bindings.append(identity)

    def glGetRenderbufferParameteriv(self, target: int, key: int) -> int:
        assert target == self.GL_RENDERBUFFER
        if self.storage_error:
            raise RuntimeError("controlled storage query failure")
        return self.storage[self.bound][0 if key == self.GL_RENDERBUFFER_WIDTH else 1]


@pytest.fixture
def observer() -> dict[str, Any]:
    width, height = 128, 96
    gl = FakeGL(width, height)
    calls: list[str] = []
    context_type = type("FakeContext", (), {"__module__": "mujoco.osmesa"})
    context = context_type()
    context.make_current = lambda: calls.append("current")
    renderer = SimpleNamespace(
        width=width,
        height=height,
        _model=SimpleNamespace(vis=SimpleNamespace(quality=SimpleNamespace(offsamples=0))),
        _gl_context=context,
        _mjr_context=SimpleNamespace(offWidth=width, offHeight=height, offSamples=0),
        _scene=SimpleNamespace(camera=[SimpleNamespace(orthographic=0)]),
    )
    values: dict[str, Any] = {
        "renderer": renderer,
        "gl": gl,
        "calls": calls,
        "backend": "osmesa",
        "inspection_error": False,
        "native_error": False,
        "provenance": {
            "offscreen_attachments": {
                "offFBO": {
                    "present": True,
                    "draw_framebuffer_samples": 0,
                    "color0": {"object_name": 10, "samples": 0},
                },
                "offFBO_r": {"present": False},
            },
            "attachment_format": "strict fake format",
            "attachment_component_type": "strict fake component type",
            "gl_samples": 0,
            "gl_vendor": "controlled",
            "gl_renderer": "controlled",
            "gl_version": "controlled",
        },
        "depth": {"object_name": 11, "samples": 0},
    }

    def inspect(target: Any) -> dict[str, Any]:
        assert target is renderer
        calls.append("inspect")
        if values["inspection_error"]:
            raise RuntimeError("controlled strict attachment rejection")
        return copy.deepcopy(values["provenance"])

    def native(target: Any) -> dict[str, Any]:
        assert target is renderer
        calls.append("native")
        if values["native_error"]:
            raise RuntimeError("controlled packing/current-binding rejection")
        return {"scene_cameras": [{}]}

    def imports(name: str, *args: Any, **kwargs: Any) -> Any:
        # Supply fake dependencies locally: no native module enters sys.modules.
        modules = {
            "OpenGL": SimpleNamespace(GL=gl),
            "epsbench.diagnostics.gl_provenance": SimpleNamespace(
                inspect_mujoco_offscreen_attachments=inspect
            ),
            "epsbench.diagnostics.mujoco_runner": SimpleNamespace(
                _observed_backend=lambda target: values["backend"]
            ),
            "epsbench.diagnostics.revision_mujoco": SimpleNamespace(
                _study_depth_attachments=lambda target: {"offFBO": values["depth"]}
            ),
            "epsbench.diagnostics.osmesa_joint0_qualification": SimpleNamespace(
                **{key: namespace[key] for key in observer_names if key in namespace}
            ),
            "epsbench.diagnostics.shared_raster_capture": SimpleNamespace(
                observe_native_read_state=native
            ),
        }
        assert name in modules, "unexpected extracted-source dependency: " + name
        return modules[name]

    namespace: dict[str, Any] = {
        "__builtins__": vars(builtins) | {"__import__": imports},
        "Any": Any,
        "Mapping": Mapping,
        "cast": cast,
        "np": np,
        "_scene_geometry": lambda target: {"controlled": True},
    }
    observer_names = {
        "QualificationFailure",
        "observe_zero_sample_osmesa",
        "observe_zero_sample_osmesa_raster",
        "context_runtime_binding",
    }
    extract("src/epsbench/diagnostics/osmesa_joint0_qualification.py", observer_names, namespace)
    extract("src/epsbench/sim/canonical_paired.py", {"observe_canonical_paired_state"}, namespace)
    values["source"] = namespace
    return values


def observe(values: dict[str, Any], width: int = 128, height: int = 96) -> dict[str, Any]:
    renderer = values["renderer"]
    return cast(
        dict[str, Any],
        values["source"]["observe_zero_sample_osmesa_raster"](
            renderer, renderer._model, width, height, expected_width=128, expected_height=96
        ),
    )


def test_default_canonical_observer_routes_explicit_aperture_raster(
    observer: dict[str, Any],
) -> None:
    state = observer["source"]["observe_canonical_paired_state"](observer["renderer"])
    assert state["context_runtime"]["color_storage_dimensions"] == [128, 96]
    assert state["context_runtime"]["depth_storage_dimensions"] == [128, 96]
    assert state["scene_geometry"] == {"controlled": True}
    assert state["scene_cameras"] == [{"orthographic": 0}]
    assert observer["calls"] == ["native", "current", "inspect"]
    assert observer["gl"].bound == 77
    assert observer["gl"].bindings == [10, 77, 11, 77]


def test_historical_wrapper_rejects_aperture(observer: dict[str, Any]) -> None:
    renderer = observer["renderer"]
    with pytest.raises(observer["source"]["QualificationFailure"], match="provenance invalid"):
        observer["source"]["observe_zero_sample_osmesa"](renderer, renderer._model, 128, 96)


def test_historical_wrapper_keeps_160x120(observer: dict[str, Any]) -> None:
    renderer = observer["renderer"]
    renderer._mjr_context.offWidth, renderer._mjr_context.offHeight = 160, 120
    observer["gl"].storage = {10: [160, 120], 11: [160, 120]}
    facts = observer["source"]["observe_zero_sample_osmesa"](renderer, renderer._model, 160, 120)
    assert facts["color_storage_dimensions"] == facts["depth_storage_dimensions"] == [160, 120]


@pytest.mark.parametrize("width,height", [(160, 120), (127, 96), (128, 95)])
def test_requested_mismatch(observer: dict[str, Any], width: int, height: int) -> None:
    with pytest.raises(observer["source"]["QualificationFailure"]):
        observe(observer, width, height)


@pytest.mark.parametrize("identity,axis", [(10, 0), (10, 1), (11, 0), (11, 1)])
def test_each_storage_mismatch(observer: dict[str, Any], identity: int, axis: int) -> None:
    observer["gl"].storage[identity][axis] += 1
    with pytest.raises(observer["source"]["QualificationFailure"]):
        observer["source"]["observe_canonical_paired_state"](observer["renderer"])
    assert observer["gl"].bound == 77


@pytest.mark.parametrize("field", ["offWidth", "offHeight", "offSamples"])
def test_mjr_mismatch(observer: dict[str, Any], field: str) -> None:
    mjr = observer["renderer"]._mjr_context
    setattr(mjr, field, getattr(mjr, field) + 1)
    with pytest.raises(observer["source"]["QualificationFailure"]):
        observe(observer)


@pytest.mark.parametrize(
    "failure",
    [
        "backend",
        "context",
        "model_samples",
        "sample_buffers",
        "samples",
        "before_error",
        "after_error",
        "main_absent",
        "draw_samples",
        "color_samples",
        "depth_samples",
        "resolve",
        "color_identity",
        "depth_identity",
        "nonscalar",
        "context_operation",
    ],
)
def test_existing_provenance_rejections(observer: dict[str, Any], failure: str) -> None:
    renderer, gl = observer["renderer"], observer["gl"]
    main = observer["provenance"]["offscreen_attachments"]["offFBO"]
    if failure == "backend":
        observer["backend"] = "glfw"
    elif failure == "context":
        renderer._gl_context = SimpleNamespace(make_current=lambda: None)
    elif failure == "model_samples":
        renderer._model.vis.quality.offsamples = 1
    elif failure in {"sample_buffers", "samples"}:
        gl.integers[5 if failure == "sample_buffers" else 6] = 1
    elif failure in {"before_error", "after_error"}:
        gl.errors[0 if failure == "before_error" else 1] = 1
    elif failure == "main_absent":
        main["present"] = False
    elif failure == "draw_samples":
        main["draw_framebuffer_samples"] = 1
    elif failure == "color_samples":
        main["color0"]["samples"] = 1
    elif failure == "depth_samples":
        observer["depth"]["samples"] = 1
    elif failure == "resolve":
        observer["provenance"]["offscreen_attachments"]["offFBO_r"]["present"] = True
    elif failure == "color_identity":
        main["color0"]["object_name"] = 0
    elif failure == "depth_identity":
        observer["depth"]["object_name"] = 0
    elif failure == "nonscalar":
        gl.integers[5] = [0, 0]
    elif failure == "context_operation":
        renderer._gl_context = None
    with pytest.raises(observer["source"]["QualificationFailure"]):
        observe(observer)
    assert gl.bound == 77


@pytest.mark.parametrize("failure", ["storage", "inspection"])
def test_dependency_failure_propagates_and_restores(observer: dict[str, Any], failure: str) -> None:
    observer["gl"].storage_error = failure == "storage"
    observer["inspection_error"] = failure == "inspection"
    with pytest.raises(RuntimeError, match="controlled"):
        observer["source"]["observe_canonical_paired_state"](observer["renderer"])
    assert observer["gl"].bound == 77


@pytest.mark.parametrize("width,height", [(0, 96), (128, -1), (True, 96), (128, 96.0)])
def test_invalid_expected_raster(observer: dict[str, Any], width: Any, height: Any) -> None:
    renderer = observer["renderer"]
    with pytest.raises(observer["source"]["QualificationFailure"], match="dimensions invalid"):
        observer["source"]["observe_zero_sample_osmesa_raster"](
            renderer, renderer._model, 128, 96, expected_width=width, expected_height=height
        )
    assert observer["calls"] == []


def test_historical_requested_raster_cannot_mask_storage(observer: dict[str, Any]) -> None:
    renderer = observer["renderer"]
    with pytest.raises(observer["source"]["QualificationFailure"]):
        observer["source"]["observe_zero_sample_osmesa"](renderer, renderer._model, 160, 120)


def test_canonical_binding_requires_complete_provenance(observer: dict[str, Any]) -> None:
    del observer["provenance"]["attachment_format"]
    with pytest.raises(observer["source"]["QualificationFailure"], match="binding is incomplete"):
        observer["source"]["observe_canonical_paired_state"](observer["renderer"])


def test_invalid_native_state_cannot_be_repaired_by_context_query(observer: dict[str, Any]) -> None:
    observer["native_error"] = True
    with pytest.raises(RuntimeError, match="packing/current-binding rejection"):
        observer["source"]["observe_canonical_paired_state"](observer["renderer"])
    assert observer["calls"] == ["native"]
    assert observer["gl"].bindings == []
