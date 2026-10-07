"""Controlled runtime facts and extracted default call chain; never import the SDK."""

from __future__ import annotations

import ast
import builtins
import hashlib
import os
import platform
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

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
