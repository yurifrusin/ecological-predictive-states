"""Controlled runtime facts and extracted default call chain; never import the SDK."""

from __future__ import annotations

import ast
import builtins
import hashlib
import json
import os
import platform
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest

from epsbench.diagnostics import corridor_aperture_runtime as r
from epsbench.diagnostics.corridor_aperture import config_root, encode


@pytest.fixture
def facts(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    binding = dict(
        source_head="a" * 40,
        source_tree="b" * 40,
        configuration_root=config_root(),
        image="sha256:" + "c" * 64,
        purpose=r.PURPOSE,
    )
    manifest = encode({k: v for k, v in binding.items() if k != "image"})
    binding["manifest_sha256"] = hashlib.sha256(manifest).hexdigest()
    values: dict[str, Any] = {
        "binding": binding,
        "head": binding["source_head"],
        "tree": binding["source_tree"],
        "dirty": "",
        "docker": True,
        "platform": "Linux",
        "manifest": manifest,
        "memory.max": str(4 * 1024**3),
        "memory.swap.max": "0",
        "pids.max": "64",
        "cpu.max": "100000 100000",
    }
    monkeypatch.setattr(
        os,
        "environ",
        {
            "EPS_APERTURE_RUNTIME": "docker_candidate_v1",
            "EPS_APERTURE_BINDING": encode(binding).decode(),
            "EPS_APERTURE_IMAGE": binding["image"],
            "MUJOCO_GL": "osmesa",
            "PYOPENGL_PLATFORM": "osmesa",
        },
    )
    monkeypatch.setattr(platform, "system", lambda: values["platform"])
    monkeypatch.setattr(Path, "is_file", lambda path: values["docker"])
    monkeypatch.setattr(Path, "stat", lambda path: SimpleNamespace(st_size=len(values["manifest"])))
    monkeypatch.setattr(Path, "read_bytes", lambda path: values["manifest"])
    monkeypatch.setattr(Path, "read_text", lambda path, **kw: values[path.name])

    def git(command: list[str], **kw: Any) -> Any:
        return SimpleNamespace(
            stdout=(
                values["head"] + "\n" + values["tree"]
                if command[3] == "rev-parse"
                else values["dirty"]
            )
        )

    monkeypatch.setattr(subprocess, "run", git)
    return values


def test_prospective_candidate(facts: dict[str, Any]) -> None:
    assert r.aperture_rejection() is None


@pytest.mark.parametrize(
    "field",
    ["source_head", "source_tree", "configuration_root", "image", "purpose", "manifest_sha256"],
)
@pytest.mark.parametrize("missing", [True, False])
def test_binding_changed_or_missing(facts: dict[str, Any], field: str, missing: bool) -> None:
    binding = facts["binding"].copy()
    if missing:
        del binding[field]
    else:
        binding[field] = "changed"
    os.environ["EPS_APERTURE_BINDING"] = encode(binding).decode()
    assert not r.aperture_candidate()


@pytest.mark.parametrize(
    "field,value",
    [
        ("head", "d" * 40),
        ("tree", "d" * 40),
        ("dirty", " M source.py"),
        ("manifest", b"{}"),
        ("docker", False),
        ("platform", "Windows"),
        ("memory.max", str(8 * 1024**3)),
        ("memory.swap.max", "1"),
        ("pids.max", "65"),
        ("cpu.max", "200000 100000"),
        ("cpu.max", "100000 0"),
        ("cpu.max", "max 100000"),
    ],
)
def test_observed_facts_reject(facts: dict[str, Any], field: str, value: Any) -> None:
    facts[field] = value
    assert not r.aperture_candidate()


@pytest.mark.parametrize(
    "field",
    [
        "EPS_APERTURE_RUNTIME",
        "EPS_APERTURE_BINDING",
        "EPS_APERTURE_IMAGE",
        "MUJOCO_GL",
        "PYOPENGL_PLATFORM",
    ],
)
def test_missing_environment(facts: dict[str, Any], field: str) -> None:
    del os.environ[field]
    assert not r.aperture_candidate()


@pytest.mark.parametrize(
    "field",
    [
        "WSL_INTEROP",
        "WSL_DISTRO_NAME",
        "EPS_A1_RUNTIME",
        "EPS_CAUSAL_RUNTIME",
        "EPS_PAIRED_APPEARANCE_RUNTIME",
        "EPS_RENDERER_DISCRIMINATOR_RUNTIME",
    ],
)
def test_old_marker_mixture_reject(facts: dict[str, Any], field: str) -> None:
    os.environ[field] = "present"
    assert not r.aperture_candidate()


def default_chain() -> dict[str, Any]:
    # Parse repository source without importing epsbench.sim or any native module.
    source = Path(__file__).resolve().parents[1] / "src/epsbench/sim/canonical_paired.py"
    module = ast.parse(builtins.open(source, encoding="utf-8").read())
    names = {"CanonicalPairedRenderer", "CanonicalPairedCaptureError", "require_supported_runtime"}
    nodes: list[ast.stmt] = [
        n for n in module.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names
    ]
    nodes += [
        n
        for n in module.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "SUPPORTED_RUNTIME" for t in n.targets)
    ]
    routes = dict(
        docker_candidate=False,
        causal_candidate=False,
        paired_appearance_candidate=False,
        candidate_runtime=False,
    )

    def imports(name: str, *args: Any, **kw: Any) -> Any:
        if name == r.__name__:
            return r
        assert name in {
            "epsbench.diagnostics.a1_docker_runtime",
            "epsbench.diagnostics.causal_history_runtime",
            "epsbench.diagnostics.paired_appearance_runtime",
            "epsbench.diagnostics.renderer_discriminator",
        }
        return SimpleNamespace(**{k: (lambda key=k: routes[key]) for k in routes})

    versions = {"PyOpenGL": "3.1.10", "glfw": "2.10.2"}
    namespace: dict[str, Any] = {
        "__builtins__": vars(builtins) | {"__import__": imports},
        "os": os,
        "platform": SimpleNamespace(system=lambda: "Linux", python_version=lambda: "3.11.15"),
        "mujoco": SimpleNamespace(
            __version__="3.12.0", mjr_render=None, mjr_readPixels=None, mjr_setBuffer=None
        ),
        "np": SimpleNamespace(__version__="2.4.6"),
        "threading": threading,
        "observe_canonical_paired_state": object(),
        "importlib": SimpleNamespace(metadata=SimpleNamespace(version=lambda name: versions[name])),
        "routes": routes,
        "versions": versions,
    }
    code = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            *nodes,
        ],
        type_ignores=[],
    )
    # Future import is compiler-only and is not routed through the fake importer.
    namespace["__builtins__"]["__import__"] = lambda name, *args, **kw: (
        builtins.__import__(name, *args, **kw)
        if name == "__future__"
        else imports(name, *args, **kw)
    )
    exec(compile(ast.fix_missing_locations(code), str(source), "exec"), namespace)
    return namespace


def test_default_call_chain_missing_then_admitted(facts: dict[str, Any]) -> None:
    chain = default_chain()
    marker = os.environ.pop("EPS_APERTURE_RUNTIME")
    with pytest.raises(chain["CanonicalPairedCaptureError"], match="aperture_marker_absent"):
        chain["CanonicalPairedRenderer"](object())
    os.environ["EPS_APERTURE_RUNTIME"] = marker
    chain["CanonicalPairedRenderer"](
        object()
    )  # No custom observer; actual default constructor/gate.


@pytest.mark.parametrize(
    "component,value",
    [
        ("python", "3.11.14"),
        ("platform", "Windows"),
        ("mujoco", "3.11.0"),
        ("numpy", "2.4.5"),
        ("PyOpenGL", "3.1.9"),
        ("glfw", "2.10.1"),
        ("backend", "egl"),
    ],
)
def test_default_common_runtime_reject(facts: dict[str, Any], component: str, value: str) -> None:
    chain = default_chain()
    backend = "osmesa"
    if component == "python":
        chain["platform"].python_version = lambda: value
    elif component == "platform":
        chain["platform"].system = lambda: value
    elif component in ("mujoco", "numpy"):
        chain["mujoco" if component == "mujoco" else "np"].__version__ = value
    elif component == "backend":
        backend = value
    else:
        chain["versions"][component] = value
    with pytest.raises(chain["CanonicalPairedCaptureError"]):
        chain["require_supported_runtime"](backend)


@pytest.mark.parametrize(
    "route",
    [
        "docker_candidate",
        "causal_candidate",
        "paired_appearance_candidate",
        "candidate_runtime",
        "WSL_INTEROP",
        "WSL_DISTRO_NAME",
    ],
)
def test_historical_route_noninterference(facts: dict[str, Any], route: str) -> None:
    chain = default_chain()
    for field in ("EPS_APERTURE_RUNTIME", "EPS_APERTURE_BINDING", "EPS_APERTURE_IMAGE"):
        os.environ.pop(field)
    if route.startswith("WSL"):
        os.environ[route] = "historical"
    else:
        chain["routes"][route] = True
    chain["CanonicalPairedRenderer"](object())


@pytest.mark.parametrize(
    "route",
    [
        "docker_candidate",
        "causal_candidate",
        "paired_appearance_candidate",
        "candidate_runtime",
        "WSL_INTEROP",
        "WSL_DISTRO_NAME",
    ],
)
def test_aperture_cannot_fall_back(facts: dict[str, Any], route: str) -> None:
    chain = default_chain()
    os.environ["EPS_APERTURE_BINDING"] = "invalid"
    if route.startswith("WSL"):
        os.environ[route] = "historical"
    else:
        chain["routes"][route] = True
    with pytest.raises(chain["CanonicalPairedCaptureError"]):
        chain["CanonicalPairedRenderer"](object())


@pytest.mark.parametrize(
    "field",
    [
        "WSL_INTEROP",
        "WSL_DISTRO_NAME",
        "EPS_A1_RUNTIME",
        "EPS_CAUSAL_RUNTIME",
        "EPS_PAIRED_APPEARANCE_RUNTIME",
        "EPS_RENDERER_DISCRIMINATOR_RUNTIME",
    ],
)
def test_valid_aperture_mixed_markers_default_reject(facts: dict[str, Any], field: str) -> None:
    chain = default_chain()
    os.environ[field] = "historical"
    with pytest.raises(chain["CanonicalPairedCaptureError"], match="aperture_mixed_runtime"):
        chain["CanonicalPairedRenderer"](object())


@pytest.mark.parametrize(
    "kind",
    [
        "marker_only",
        "duplicate_key",
        "noncanonical",
        "oversize",
        "image_mismatch",
        "config_mismatch",
    ],
)
def test_binding_fail_closed(facts: dict[str, Any], kind: str) -> None:
    if kind == "marker_only":
        os.environ.pop("EPS_APERTURE_BINDING")
        os.environ.pop("EPS_APERTURE_IMAGE")
    elif kind == "duplicate_key":
        raw = os.environ["EPS_APERTURE_BINDING"]
        os.environ["EPS_APERTURE_BINDING"] = '{"purpose":"bad",' + raw[1:]
    elif kind == "noncanonical":
        os.environ["EPS_APERTURE_BINDING"] += "\n"
    elif kind == "oversize":
        os.environ["EPS_APERTURE_BINDING"] = " " * 4097
    elif kind == "image_mismatch":
        os.environ["EPS_APERTURE_IMAGE"] = "sha256:" + "d" * 64
    else:
        binding = facts["binding"] | {"configuration_root": "d" * 64}
        os.environ["EPS_APERTURE_BINDING"] = encode(binding).decode()
    assert not r.aperture_candidate()


@pytest.mark.parametrize("failure", ["manifest", "cgroup", "git", "timeout"])
def test_unavailable_facts_fail_closed(
    facts: dict[str, Any], monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    def unavailable(*args: Any, **kw: Any) -> Any:
        if failure == "timeout":
            raise subprocess.TimeoutExpired("git", 5)
        raise OSError("controlled unavailable public runtime fact")

    if failure == "manifest":
        monkeypatch.setattr(Path, "read_bytes", unavailable)
    elif failure == "cgroup":
        monkeypatch.setattr(Path, "read_text", unavailable)
    else:
        monkeypatch.setattr(subprocess, "run", unavailable)
    assert r.aperture_rejection() == "aperture_facts_unavailable"


def test_no_historical_candidate_still_rejects(facts: dict[str, Any]) -> None:
    chain = default_chain()
    for field in ("EPS_APERTURE_RUNTIME", "EPS_APERTURE_BINDING", "EPS_APERTURE_IMAGE"):
        os.environ.pop(field)
    with pytest.raises(chain["CanonicalPairedCaptureError"]):
        chain["CanonicalPairedRenderer"](object())


def _fake_mujoco(position: tuple[float, ...], rotation: tuple[float, ...]) -> Any:
    operations: list[str] = []

    class CameraPositions:
        def __init__(self) -> None:
            self.values = [0.0, 2.0, 1.0]

        def __setitem__(self, key: tuple[int, int], value: float) -> None:
            row, column = key
            assert row == 0
            self.values[column] = value

        def __getitem__(self, key: tuple[int, int]) -> float:
            row, column = key
            assert row == 0
            return self.values[column]

    class Model:
        ngeom = 9
        ncam = 1

        def __init__(self) -> None:
            self.cam_pos = CameraPositions()

        @staticmethod
        def from_xml_string(xml: str) -> Any:
            operations.append("compile")
            assert "aperture_camera" in xml
            model = Model()
            MujocoFake.models.append(model)
            return model

    class Data:
        def __init__(self, model: Any) -> None:
            operations.append("data")
            self.cam_xpos = [position]
            self.cam_xmat = [SimpleNamespace(flat=rotation)]

    class MJObject:
        mjOBJ_GEOM = "geom"
        mjOBJ_CAMERA = "camera"

    class MujocoFake:
        __version__ = "fake-3.12.0"
        MjModel = Model
        MjData = Data
        mjtObj = MJObject
        calls: ClassVar[list[str]] = operations
        models: ClassVar[list[Any]] = []

        @staticmethod
        def mj_name2id(model: Any, kind: str, name: str) -> int:
            operations.append("name")
            if kind == "camera":
                return 0
            return (
                "floor",
                "left",
                "end",
                "right_bottom",
                "right_top",
                "right_front",
                "right_pier",
                "right_back",
                "target",
            ).index(name)

        @staticmethod
        def mj_forward(model: Any, data: Any) -> None:
            operations.append("forward")

        @staticmethod
        def Renderer(*args: Any, **kwargs: Any) -> Any:
            pytest.fail("compile-only diagnostic constructed a renderer")

    return MujocoFake


def test_camera_compile_diagnostic_is_privileged_bound_and_renderer_free(
    facts: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    from epsbench.diagnostics import corridor_aperture_capture as c
    from epsbench.diagnostics.corridor_aperture import config_root
    from epsbench.schema import ModalityPermissionSet

    fake = _fake_mujoco(
        (0.0, 2.0000000000000004, 1.0),
        (1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0),
    )
    monkeypatch.setattr(c, "_load_mujoco", lambda: fake)
    expected_source = c.SourceBinding("a" * 40, "b" * 40, config_root())
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **kwargs: SimpleNamespace(
            stdout=(expected_source.head + "\n" + expected_source.tree)
            if command[-2:] == ["HEAD", "HEAD^{tree}"]
            else ""
        ),
    )
    retained: list[tuple[str, bytes]] = []
    diagnostic = c.CameraCompilationDiagnostic(
        ModalityPermissionSet.all_modalities(),
        enabled=True,
        expected_source=expected_source,
        retain=lambda name, payload: retained.append((name, payload)),
    )
    payload = diagnostic.run()
    result = json.loads(payload)
    assert result["validity"] == "unvalidated_privileged_compiled_camera_pose"
    assert result["source"] == {
        "head": expected_source.head,
        "tree": expected_source.tree,
        "configuration_root": config_root(),
    }
    assert result["dependency"] == "mujoco:fake-3.12.0"
    assert result["observed"]["position_hex"][1] == (2.0000000000000004).hex()
    assert result["component_equal"]["position"] == [True, False, True]
    assert result["component_equal"]["orientation"] == [True] * 9
    assert retained == [("camera-pose-0.json", payload)]
    assert fake.calls == ["compile"] + ["name"] * 10 + ["data", "forward"]
    assert fake.models[0].cam_pos[0, 1] == 2
    with pytest.raises(PermissionError, match="single-use"):
        diagnostic.run()
    assert retained == [("camera-pose-0.json", payload)]
    assert fake.calls == ["compile"] + ["name"] * 10 + ["data", "forward"]


def test_camera_compile_diagnostic_denies_before_sdk_and_fails_closed_on_retention(
    facts: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    from epsbench.diagnostics import corridor_aperture_capture as c
    from epsbench.diagnostics.corridor_aperture import config_root
    from epsbench.schema import ModalityPermissionSet

    monkeypatch.setattr(c, "_load_mujoco", lambda: pytest.fail("SDK reached before authorization"))
    source = c.SourceBinding("a" * 40, "b" * 40, config_root())
    with pytest.raises(PermissionError, match="disabled by default"):
        c.CameraCompilationDiagnostic(
            ModalityPermissionSet.all_modalities(), expected_source=source, retain=lambda *_: None
        )
    with pytest.raises(PermissionError, match="typed aperture permissions"):
        c.CameraCompilationDiagnostic(
            ModalityPermissionSet.ecological_only(),
            enabled=True,
            expected_source=source,
            retain=lambda *_: None,
        )
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **kwargs: SimpleNamespace(
            stdout="d" * 40 + "\n" + "e" * 40 if command[-2:] == ["HEAD", "HEAD^{tree}"] else ""
        ),
    )
    diagnostic = c.CameraCompilationDiagnostic(
        ModalityPermissionSet.all_modalities(),
        enabled=True,
        expected_source=source,
        retain=lambda *_: None,
    )
    with pytest.raises(PermissionError, match="source identity/cleanliness"):
        diagnostic.run()
    with pytest.raises(ValueError, match="finite compiled camera operands"):
        c._camera_record(0, (0.0, float("nan"), 1.0), (1.0,) * 9, source, "mujoco:fake")


def test_camera_compile_diagnostic_retention_failure_is_fatal(
    facts: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    from epsbench.diagnostics import corridor_aperture_capture as c
    from epsbench.diagnostics.corridor_aperture import config_root
    from epsbench.schema import ModalityPermissionSet

    source = c.SourceBinding("a" * 40, "b" * 40, config_root())
    fake = _fake_mujoco((0.0, 2.0, 1.0), (1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0))
    monkeypatch.setattr(c, "_load_mujoco", lambda: fake)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **kwargs: SimpleNamespace(
            stdout=source.head + "\n" + source.tree
            if command[-2:] == ["HEAD", "HEAD^{tree}"]
            else ""
        ),
    )

    def fail_sink(*_: Any) -> None:
        raise OSError("controlled sink failure")

    diagnostic = c.CameraCompilationDiagnostic(
        ModalityPermissionSet.all_modalities(),
        enabled=True,
        expected_source=source,
        retain=fail_sink,
    )
    with pytest.raises(c.RetentionFailure, match="camera pose"):
        diagnostic.run()
    operations = list(fake.calls)
    with pytest.raises(PermissionError, match="single-use"):
        diagnostic.run()
    assert fake.calls == operations


def test_native_capture_retains_camera_operands_before_frame_validation() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "src/epsbench/diagnostics/corridor_aperture_capture.py"
    )
    module = ast.parse(source.read_text(encoding="utf-8"))
    adapter = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "NativeAdapter"
    )
    frame_method = next(
        node for node in adapter.body if isinstance(node, ast.FunctionDef) and node.name == "frame"
    )
    build = next(
        node
        for node in frame_method.body
        if isinstance(node, ast.FunctionDef) and node.name == "build"
    )
    calls = {
        node.func.id: node.lineno
        for node in ast.walk(build)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert calls["_retain_camera_record"] < calls["Frame"] < calls["verify_pair"]
