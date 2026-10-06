from __future__ import annotations

import copy
import os
import platform
import runpy
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from epsbench.diagnostics import paired_appearance as p
from epsbench.diagnostics import paired_appearance_execution as e
from epsbench.diagnostics import paired_appearance_runtime as r
from epsbench.utils.canonical import canonical_json_bytes
from tests.test_paired_appearance import synthetic, synthetic_corridor

ROOT = Path(__file__).resolve().parents[1]
HOST = runpy.run_path(str(ROOT / "scripts/paired_appearance_execution.py"))


def binding(purpose: str = r.NATIVE, token: str = "f" * 32) -> r.AppearanceExecutionBinding:
    return r.AppearanceExecutionBinding.model_validate(
        {
            "preparation": r.preparation(ROOT, "a" * 40, "b" * 40).model_dump(mode="json"),
            "image": "sha256:" + "c" * 64,
            "purpose": purpose,
            "output_id": "e" * 32,
            "token": token,
        }
    )


def decision(b: r.AppearanceExecutionBinding) -> r.LaunchDecision:
    return r.LaunchDecision(
        binding_root=b.root,
        approved=True,
        purpose=b.purpose,
        authorization="Separate prospective test-only synthetic decision",
    )


def sink_at(tmp_path: Path) -> e.ByteBoundedSink:
    b = binding()
    root = tmp_path / b.output_id
    e.consume_attempt(root, b, decision(b))
    return e.ByteBoundedSink(root, b)


def fixture(context: int, index: int) -> p.Frame:
    family, appearance, _ = p.contexts()[context]
    return (
        synthetic(appearance, index)
        if family == "single_occluder"
        else synthetic_corridor(appearance, index)
    )


def emit(progress: Any, frame: p.Frame, stop: str | None = None) -> None:
    # Independent fixed producer callback vocabulary; do not mirror implementation mistakes.
    stages = (
        "endpoint_attempt",
        "rgb_attempt",
        "rgb_read_complete",
        "rgb_state_complete",
        "paired_draw_input",
        "paired_draw_attempt",
        "paired_draw_complete",
        "paired_draw_output",
        "paired_read_input",
        "paired_read_attempt",
        "paired_read_complete",
        "paired_read_output",
        "endpoint_complete",
    )
    assert e.STAGES == stages
    for stage in stages:
        value: Any = None
        if stage == "rgb_read_complete":
            value = frame.rgb
        elif stage == "rgb_state_complete":
            value = canonical_json_bytes(
                {"stable": frame.evidence["rgb_stable"], "material": frame.evidence["rgb_material"]}
            )
        elif stage in e.METADATA:
            value = canonical_json_bytes(frame.evidence["paired_stable"])
        elif stage == "paired_read_complete":
            value = (np.flipud(frame.native_id).tobytes(), np.flipud(frame.native_depth).tobytes())
        elif stage == "endpoint_complete":
            value = frame
        progress(stage, frame.index, value)
        if stage == stop:
            raise RuntimeError("source-only injected interruption")


class FakeCapture:
    def __init__(
        self, context: int, progress: Any, opened: list[int], closed: list[int], change: Any = None
    ):
        self.context, self.progress, self.closed, self.change = context, progress, closed, change
        opened.append(context)

    def capture(self, index: int) -> p.Frame:
        frame = fixture(self.context, index)
        if self.change:
            frame = self.change(self.context, index, frame)
        emit(self.progress, frame)
        return frame

    def close(self) -> None:
        self.closed.append(self.context)


@pytest.mark.parametrize(
    "field,value",
    [
        ("purpose", "causal_history_native_v1"),
        ("token", False),
        ("image", "latest"),
        ("policy", "larger"),
        ("extra", True),
        ("output_id", 1),
    ],
)
def test_closed_binding_denials(field: str, value: Any) -> None:
    bad = binding().model_dump(mode="json")
    bad[field] = value
    with pytest.raises(ValidationError):
        r.AppearanceExecutionBinding.model_validate(bad)


def test_decision_strict_and_exact() -> None:
    b = binding()
    bad = decision(b).model_dump(mode="json")
    for value in (1, "true", False):
        with pytest.raises((ValidationError, PermissionError)):
            r.require_decision(b, r.LaunchDecision.model_validate({**bad, "approved": value}))
    with pytest.raises(PermissionError):
        r.require_decision(binding(token="0" * 32), decision(b))
    with pytest.raises(ValueError):
        r.parse(r.AppearanceExecutionBinding, b'{"purpose":"x","purpose":"y"}')


def test_native_denied_before_sdk_and_runtime_conflicts(monkeypatch: pytest.MonkeyPatch) -> None:
    from epsbench.diagnostics.paired_appearance_native import NativeCapture

    for b in (None, binding(r.DUMMY)):
        with pytest.raises(PermissionError):
            NativeCapture(
                ROOT,
                "single_occluder",
                p.APPEARANCES[0],
                source_head="a" * 40,
                source_tree="b" * 40,
                progress=lambda *args: None,
                binding=b,
            )
    monkeypatch.setenv(
        "EPS_PAIRED_APPEARANCE_BINDING",
        canonical_json_bytes(binding(r.DUMMY).model_dump(mode="json")).decode(),
    )
    assert not r.paired_appearance_candidate()
    monkeypatch.setenv("EPS_A1_RUNTIME", "docker_candidate_v1")
    assert not r.paired_appearance_candidate()
    assert (
        "or paired_appearance_candidate()"
        in (ROOT / "src/epsbench/sim/canonical_paired.py").read_text()
    )
    for retained in (
        "or docker_candidate()",
        "or causal_candidate()",
        'os.environ.get("WSL_INTEROP")',
    ):
        assert retained in (ROOT / "src/epsbench/sim/canonical_paired.py").read_text()


def test_exclusive_attempt_no_relaunch_or_wrong_path(tmp_path: Path) -> None:
    b = binding()
    root = tmp_path / b.output_id
    e.consume_attempt(root, b, decision(b))
    with pytest.raises(FileExistsError):
        e.consume_attempt(root, b, decision(b))
    with pytest.raises(ValueError):
        e.consume_attempt(tmp_path / "wrong", b, decision(b))
    sink = e.ByteBoundedSink(root, b)
    with pytest.raises(ValueError, match="resume"):
        e.ByteBoundedSink(root, b)
    with pytest.raises(ValueError):
        sink.begin_context(True)


def test_full_synthetic_source_matrix_retained_replay(tmp_path: Path) -> None:
    sink = sink_at(tmp_path)
    opened: list[int] = []
    closed: list[int] = []
    result = e.run_matrix(
        sink, lambda c, q: FakeCapture(c, q, opened, closed), time.monotonic() + 60
    )
    assert result["status"] == "CAPTURE_COMPLETE"
    assert result["cleanup_pending"] is True  # Driver alone never claims owned host cleanup.
    assert opened == closed == list(range(8))
    retained = e.replay(sink.root, sink.binding)
    assert retained["status"] == "COMPLETE"
    assert result["counts"] == {k: 16 for k in sink.counts}
    for slot in range(16):
        assert (
            sum(v.stat().st_size for v in (sink.root / f"endpoints/e{slot:02d}").iterdir()) <= p.MIB
        )
    assert sum(v.stat().st_size for v in sink.root.rglob("*") if v.is_file()) <= 18 * p.MIB
    assert not list(sink.root.rglob("*.npz"))
    assert len(list(sink.root.rglob("rgb.bin"))) == 16


@pytest.mark.parametrize("kind", ("repeat", "texture", "invalid"))
def test_earliest_stop_preserves_valid_negative(tmp_path: Path, kind: str) -> None:
    sink = sink_at(tmp_path)
    opened: list[int] = []
    closed: list[int] = []

    def change(c: int, i: int, f: p.Frame) -> p.Frame:
        if i == 0 and c == (1 if kind == "repeat" else 2):
            if kind == "invalid":
                evidence = copy.deepcopy(f.evidence)
                evidence["camera"]["fovy"] = 99.0
                return replace(f, evidence=evidence)
            rgb = f.rgb.copy()
            rgb[:] = 200 if kind == "texture" else 201
            return replace(f, rgb=rgb)
        return f

    result = e.run_matrix(
        sink, lambda c, q: FakeCapture(c, q, opened, closed, change), time.monotonic() + 60
    )
    assert result["status"] == ("INCONCLUSIVE" if kind == "invalid" else "FAIL")
    assert opened == closed == (list(range(2)) if kind == "repeat" else list(range(3)))
    assert result["counts"]["rgb_attempt"] == (3 if kind == "repeat" else 5)
    assert result["completed_endpoints"] < 16


@pytest.mark.parametrize(
    "stage", ("rgb_read_complete", "paired_read_complete", "paired_read_output")
)
def test_immediate_prefix_on_source_interrupt(tmp_path: Path, stage: str) -> None:
    sink = sink_at(tmp_path)
    sink.begin_context(0)
    with pytest.raises(RuntimeError):
        emit(sink.progress, fixture(0, 0), stage)
    retained = e.replay(sink.root, sink.binding)
    assert retained["status"] == "PREFIX"
    assert (sink.root / "endpoints/e00/rgb.bin").read_bytes() == fixture(0, 0).rgb.tobytes()
    if stage != "rgb_read_complete":
        assert (sink.root / "endpoints/e00/read_id.bin").stat().st_size == 57600
    assert not retained["frames"]


@pytest.mark.parametrize("fault", ("write", "flush"))
def test_sink_fault_poison_partial_counted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    sink = sink_at(tmp_path)

    def fail(path: Path, payload: bytes) -> None:
        if fault == "write":
            path.write_bytes(payload[:1])
        else:
            path.write_bytes(payload)
        raise OSError("injected " + fault)

    monkeypatch.setattr(e, "exclusive_file", fail)
    with pytest.raises(OSError):
        sink.put("endpoints/e00/raw.bin", b"abc")
    assert sink.poisoned and sink.used[0] == 3
    assert (sink.root / "endpoints/e00/raw.bin").exists()
    with pytest.raises(OSError):
        sink.put("later.json", b"{}")


def test_caps_paths_metadata_and_torn_event(tmp_path: Path) -> None:
    sink = sink_at(tmp_path)
    with pytest.raises(ValueError, match="endpoint byte"):
        sink.put("endpoints/e00/raw.bin", b"x" * (p.MIB + 1))
    assert sink.poisoned and not (sink.root / "endpoints/e00/raw.bin").exists()
    other = tmp_path / "other"
    other.mkdir()
    sink = sink_at(other)
    with pytest.raises(ValueError):
        sink.put("../escape.bin", b"x")
    third = tmp_path / "third"
    third.mkdir()
    sink = sink_at(third)
    sink.begin_context(0)
    with pytest.raises(ValueError):
        sink.progress("rgb_attempt", 0, None)
    assert sink.poisoned


def test_retained_corruption_rejects_preserves_prefix(tmp_path: Path) -> None:
    sink = sink_at(tmp_path)
    sink.begin_context(0)
    emit(sink.progress, fixture(0, 0))
    emit(sink.progress, fixture(0, 1))
    (sink.root / "endpoints/e01/rgb.bin").write_bytes(b"x")
    retained = e.replay(sink.root, sink.binding)
    assert retained["status"] == "INCONCLUSIVE"
    assert set(retained["frames"]) == {(0, 0)}


def test_watchdog_before_factory_and_terminal_fault(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sink = sink_at(tmp_path)

    def denied(*args: Any) -> e.Capture:
        raise AssertionError("factory must not run")

    result = e.run_matrix(sink, denied, time.monotonic() - 1)
    assert result["status"] == "INCONCLUSIVE" and result["counts"]["rgb_attempt"] == 0
    second = tmp_path / "second"
    second.mkdir()
    sink = sink_at(second)
    monkeypatch.setattr(
        sink, "terminal", lambda value: (_ for _ in ()).throw(OSError("terminal flush"))
    )
    result = e.run_matrix(sink, denied, time.monotonic() - 1)
    assert result["status"] == "INCONCLUSIVE" and "retention_error" in result


def test_tiny_four_dummy_source_only_and_ci_isolation() -> None:
    assert e.DUMMY_CASES == ("normal", "interrupted", "deadline", "overbudget")
    assert e.DUMMY_TOTAL + e.HOST_ACTUAL == p.MIB
    source = (ROOT / "src/epsbench/diagnostics/paired_appearance_execution.py").read_text()
    assert "def run_dummy" in source
    # Never call run_dummy: real once-only qualification remains held.
    ci = (ROOT / ".github/workflows/ci.yml").read_text()
    assert ci.count("github.head_ref == 'codex/paired-appearance-execution-20261006' ||") == 4
    assert "python scripts/check_paired_appearance_execution_source.py" in ci
    assert "github.head_ref != 'codex/paired-appearance-execution-20261006'" in ci


def test_commands_deadline_before_process(monkeypatch: pytest.MonkeyPatch) -> None:
    commands = HOST["Commands"]("forbidden-no-process", 10)
    monkeypatch.setattr(
        HOST["subprocess"],
        "Popen",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("process forbidden")),
    )
    with pytest.raises(TimeoutError):
        commands.command(["inspect"], time.monotonic() - 1)


def test_owned_denial_precedes_delete() -> None:
    b = binding()
    with pytest.raises(PermissionError):
        HOST["owned"]({"Id": "0" * 64, "Name": "/unowned", "Config": {"Labels": {}}}, b)
    args = HOST["create_arguments"](b, Path("C:/owned").absolute(), None, time.time() + 275)
    assert "--memory=2g" in args and "--memory-swap=2g" in args
    assert "--tmpfs=/tmp:rw,noexec,nosuid,nodev,size=8m" in args
    assert "--read-only" in args and "--network=none" in args
    assert "--log-driver=none" in args
    with pytest.raises(ValueError):
        HOST["create_arguments"](binding(r.DUMMY), ROOT, None, time.time() + 275)


def container_info(b: r.AppearanceExecutionBinding, output: Path, wall: float) -> dict[str, Any]:
    return {
        "Id": "d" * 64,
        "Name": "/eps-appearance-" + b.token,
        "Image": b.image,
        "Config": {
            "Labels": {
                HOST["LABEL"]: b.token,
                "eps.paired-appearance.binding": b.root,
                **{
                    "eps.appearance." + k.replace("_", "-"): str(v)
                    for k, v in b.preparation.model_dump().items()
                },
            },
            "Cmd": ["python", "scripts/paired_appearance_execution.py", "--inside"],
            "Entrypoint": None,
            "Env": [
                k + "=" + v
                for k, v in {
                    **HOST["environment"](b),
                    "EPS_PAIRED_APPEARANCE_WORK_DEADLINE_UNIX": str(wall),
                }.items()
            ],
        },
        "Mounts": [{"Type": "bind", "Source": str(output), "Destination": "/output", "RW": True}],
        "HostConfig": {
            "Mounts": [
                {"Type": "bind", "Source": str(output), "Target": "/output", "ReadOnly": False}
            ],
            "Binds": None,
            "Privileged": False,
            "ReadonlyRootfs": True,
            "NetworkMode": "none",
            "CapDrop": ["ALL"],
            "CapAdd": None,
            "SecurityOpt": ["no-new-privileges"],
            "LogConfig": {"Type": "none"},
            "Memory": 2 * 1024**3,
            "MemorySwap": 2 * 1024**3,
            "NanoCpus": 2_000_000_000,
            "CpusetCpus": "0,1",
            "PidsLimit": 64,
            "ShmSize": p.MIB,
            "Tmpfs": {"/tmp": "rw,noexec,nosuid,nodev,size=8m"},
            "RestartPolicy": {"Name": "no"},
        },
    }


@pytest.mark.parametrize(
    "mutation", ("image", "mount", "memory", "network", "label", "environment")
)
def test_prestart_independent_confinement_denials(tmp_path: Path, mutation: str) -> None:
    b, wall = binding(), time.time() + 275
    output = tmp_path / b.output_id
    info = container_info(b, output, wall)
    HOST["inspect_confinement"](info, b, output, None, wall)
    if mutation == "image":
        info["Image"] = "sha256:" + "0" * 64
    elif mutation == "mount":
        info["Mounts"][0]["Source"] = str(tmp_path / "other")
    elif mutation == "memory":
        info["HostConfig"]["Memory"] = 8 * 1024**3
    elif mutation == "network":
        info["HostConfig"]["NetworkMode"] = "host"
    elif mutation == "label":
        info["Config"]["Labels"]["eps.appearance.asset-root"] = "0" * 64
    else:
        info["Config"]["Env"].append("EPS_CAUSAL_RUNTIME=docker_candidate_v1")
    with pytest.raises((ValueError, PermissionError)):
        HOST["inspect_confinement"](info, b, output, None, wall)


@pytest.mark.parametrize("cleanup_failure,negative", ((False, False), (True, False), (True, True)))
def test_owned_host_cleanup_and_provisional_result(
    tmp_path: Path, cleanup_failure: bool, negative: bool
) -> None:
    b = binding()
    group = tmp_path / "group"
    opened: list[int] = []
    closed: list[int] = []

    class FakeCommands:
        used = 0

        def __init__(self) -> None:
            self.info: dict[str, Any] = {}
            self.calls: list[tuple[list[str], float]] = []

        def command(self, args: list[str], deadline: float) -> str:
            self.calls.append((args, deadline))
            assert deadline > time.monotonic()
            if args[0] == "create":
                wall = float(
                    next(
                        v.split("=", 1)[1]
                        for v in args
                        if v.startswith("EPS_PAIRED_APPEARANCE_WORK_DEADLINE_UNIX=")
                    )
                )
                self.info = container_info(b, group / b.output_id, wall)
                return str(self.info["Id"])
            if args[0] == "start":
                sink = e.ByteBoundedSink(group / b.output_id, b)

                def change(c: int, i: int, frame: p.Frame) -> p.Frame:
                    if negative and c == 1 and i == 0:
                        return replace(frame, rgb=np.full_like(frame.rgb, 201))
                    return frame

                e.run_matrix(
                    sink,
                    lambda c, q: FakeCapture(c, q, opened, closed, change),
                    time.monotonic() + 60,
                )
                return ""
            if args[0] == "inspect":
                if "{{json .State}}" in args:
                    return canonical_json_bytes(
                        {"Running": False, "ExitCode": 0, "OOMKilled": False}
                    ).decode()
                return canonical_json_bytes(self.info).decode()
            if args[0] == "rm":
                assert args == ["rm", "--force", "d" * 64]
                if cleanup_failure:
                    raise TimeoutError("simulated cleanup failure")
                return "d" * 64
            raise AssertionError("unexpected control call")

    commands = FakeCommands()
    result = HOST["host_run"](b, decision(b), group, commands)
    assert result["retained_result"]["status"] == ("FAIL" if negative else "CAPTURE_COMPLETE")
    assert result["apparatus_status"] == ("FAIL" if negative else "PASS")
    assert result["operational_status"] == ("INCONCLUSIVE" if cleanup_failure else "COMPLETE")
    assert result["status"] == ("INCONCLUSIVE" if cleanup_failure else "PASS")
    assert result["cleaned"] is not cleanup_failure
    assert opened == closed == list(range(2 if negative else 8))
    work_deadlines = {
        deadline for args, deadline in commands.calls if args[0] in ("create", "start")
    }
    assert len(work_deadlines) == 1
    with pytest.raises(FileExistsError):
        HOST["host_run"](b, decision(b), group, commands)


def test_command_overflow_and_timeout_without_external_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import io
    import subprocess

    class Process:
        def __init__(self, data: bytes, timeout: bool = False):
            self.stdout = io.BytesIO(data)
            self.returncode: int | None = 0
            self.timeout = timeout
            self.killed = False

        def kill(self) -> None:
            self.killed = True

        def wait(self, timeout: float) -> int:
            assert 0 < timeout <= 15
            if self.timeout:
                raise subprocess.TimeoutExpired("mock-client", timeout)
            return 0

    overflow = Process(b"x" * 20)
    monkeypatch.setattr(HOST["subprocess"], "Popen", lambda *args, **kwargs: overflow)
    commands = HOST["Commands"]("never-executed", 10)
    with pytest.raises(RuntimeError):
        commands.command(["inspect"], time.monotonic() + 20)
    assert overflow.killed and commands.used <= 10
    timed = Process(b"", True)
    monkeypatch.setattr(HOST["subprocess"], "Popen", lambda *args, **kwargs: timed)
    with pytest.raises(TimeoutError):
        HOST["Commands"]("never-executed", 10).command(["inspect"], time.monotonic() + 20)
    assert timed.killed


def test_distinct_runtime_resource_facts_without_native(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    b = binding()
    env = HOST["environment"](b)
    for key in ("EPS_A1_RUNTIME", "EPS_CAUSAL_RUNTIME", "WSL_INTEROP", "WSL_DISTRO_NAME"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    original_is_file = Path.is_file
    original_read = Path.read_text

    def file(path: Path) -> bool:
        return True if path.name == ".dockerenv" else original_is_file(path)

    values = {
        "memory.max": str(2 * 1024**3),
        "memory.swap.max": "0",
        "pids.max": "64",
        "cpu.max": "200000 100000",
    }

    def read(path: Path, *args: Any, **kwargs: Any) -> str:
        return values[path.name] if path.name in values else original_read(path, *args, **kwargs)

    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(Path, "is_file", file)
    monkeypatch.setattr(Path, "read_text", read)
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {0, 1}, raising=False)
    monkeypatch.setattr(
        os,
        "statvfs",
        lambda path: SimpleNamespace(f_blocks=(8 if path == "/tmp" else 1) * 256, f_frsize=4096),
        raising=False,
    )
    assert r.paired_appearance_candidate()
    for key in (
        "EPS_PAIRED_APPEARANCE_IMAGE",
        "EPS_PAIRED_APPEARANCE_ROOT",
        "EPS_PAIRED_APPEARANCE_SOURCE_TREE",
        "PYOPENGL_PLATFORM",
        "LP_NUM_THREADS",
    ):
        original = env[key]
        monkeypatch.setenv(key, "wrong")
        assert not r.paired_appearance_candidate()
        monkeypatch.setenv(key, original)
    values["cpu.max"] = "0 0"
    assert not r.paired_appearance_candidate()


def test_closed_component_names_and_metadata_root(tmp_path: Path) -> None:
    sink = sink_at(tmp_path)
    with pytest.raises(ValueError, match="component path"):
        sink.put("unplanned.json", b"{}")
    other = tmp_path / "other"
    other.mkdir()
    sink = sink_at(other)
    sink.begin_context(0)
    with pytest.raises(RuntimeError):
        emit(sink.progress, fixture(0, 0), stop="rgb_read_complete")
    with pytest.raises(ValueError, match="metadata mapping"):
        sink.progress("rgb_state_complete", 0, b"1")
    assert sink.poisoned
