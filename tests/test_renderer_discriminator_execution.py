"""Guarded adapter fake-command/entry contracts; never invokes Docker or dummy bodies."""

from __future__ import annotations

import copy
import io
import runpy
import time
from pathlib import Path
from typing import Any

import pytest

from epsbench.diagnostics import renderer_discriminator as d
from epsbench.utils.canonical import canonical_json_bytes
from tests.test_paired_appearance_execution import binding as old_binding
from tests.test_paired_appearance_execution import container_info as old_info
from tests.test_renderer_discriminator import Capture, binding, decision

ROOT = Path(__file__).resolve().parents[1]
E = runpy.run_path(str(ROOT / "scripts/renderer_discriminator_execution.py"))
G = E["host_run"].__globals__


def info_for(b: d.Binding, dec: d.Decision, output: Path, wall: float) -> dict[str, Any]:
    info = old_info(old_binding(), output, wall)
    info["Name"] = "/eps-renderer-" + b.token
    info["Image"] = b.image
    info["Config"] = {
        "Labels": {E["LABEL"]: b.token, E["BINDING_LABEL"]: b.root, **E["image_labels"](b)},
        "Cmd": ["python", "scripts/renderer_discriminator_execution.py", "--inside"],
        "Entrypoint": None,
        "Env": [k + "=" + v for k, v in E["environment"](b, dec, wall).items()],
    }
    return info


@pytest.mark.parametrize(
    "mutation", ["memory", "network", "mount", "label", "decision", "mixed", "duplicate"]
)
def test_exact_prestart_confinement_and_decision_transport(tmp_path: Path, mutation: str) -> None:
    b = binding(d.NATIVE)
    dec = decision(b)
    wall = time.time() + 275.0
    output = tmp_path / b.output_id
    info = info_for(b, dec, output, wall)
    E["inspect_confinement"](info, b, dec, output, None, wall)
    if mutation == "memory":
        info["HostConfig"]["Memory"] += 1
    elif mutation == "network":
        info["HostConfig"]["NetworkMode"] = "host"
    elif mutation == "mount":
        info["Mounts"][0]["Source"] = str(tmp_path / "unowned")
    elif mutation == "label":
        info["Config"]["Labels"]["eps.renderer-discriminator.asset-root"] = "0" * 64
    elif mutation == "decision":
        info["Config"]["Env"] = [
            v for v in info["Config"]["Env"] if not v.startswith(E["DECISION_ENV"] + "=")
        ]
    elif mutation == "mixed":
        info["Config"]["Env"].append("EPS_PAIRED_APPEARANCE_RUNTIME=")
    else:
        info["Config"]["Env"].append(info["Config"]["Env"][0])
    with pytest.raises(PermissionError):
        E["inspect_confinement"](info, b, dec, output, None, wall)


@pytest.mark.parametrize(
    "fault", [None, "cleanup", "control", "nonzero", "image", "ownership", "overflow"]
)
def test_owned_host_compact_recomputed_results_and_failure_preservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str | None
) -> None:
    b = binding(d.NATIVE)
    dec = decision(b)
    root = tmp_path / "group"
    monkeypatch.setitem(G, "verify_source", lambda *a: ("a" * 40, "b" * 40))

    class Fake:
        used = 0

        def __init__(self) -> None:
            self.calls: list[tuple[list[str], float, bool]] = []
            self.info: dict[str, Any] = {}

        def command(self, args: list[str], deadline: float, *, cleanup: bool = False) -> str:
            self.calls.append((args, deadline, cleanup))
            assert deadline > time.monotonic()
            if args[0] == "image":
                return canonical_json_bytes(
                    {
                        "Id": b.image,
                        "Config": {
                            "Labels": {} if fault == "image" else E["image_labels"](b),
                            "Env": [],
                        },
                    }
                ).decode()
            if args[0] == "create":
                assert (root / "control/actual/attempt.json").is_file()
                assert not any((root / b.output_id).iterdir())
                wall = float(
                    next(v.split("=", 1)[1] for v in args if v.startswith(E["DEADLINE_ENV"] + "="))
                )
                self.info = info_for(b, dec, root / b.output_id, wall)
                return "d" * 64
            if args[0] == "start":
                # Synthetic producer with a dummy-purpose Sink and explicit test-owned transport.
                db = binding()
                sink = d.Sink(tmp_path / "synthetic-source", db, decision(db))
                sink.root, sink.binding = root / b.output_id, b
                sink.put(
                    "consumed.json",
                    canonical_json_bytes(
                        {
                            "binding": b.model_dump(mode="json"),
                            "decision": dec.model_dump(mode="json"),
                            "binding_root": b.root,
                        }
                    ),
                )
                result = d._drive(Capture(sink, control_fault=True), sink)
                assert result["status"] == "DIAGNOSTIC_COMPLETE"
                return ""
            if args[0] == "inspect":
                if E["POLL"] in args:
                    if fault == "control":
                        raise TimeoutError("synthetic control failure")
                    return "false 2 false" if fault == "nonzero" else "false 0 false"
                if fault == "ownership" and cleanup:
                    info = copy.deepcopy(self.info)
                    info["Name"] = "/unowned"
                    return canonical_json_bytes(info).decode()
                return canonical_json_bytes(self.info).decode()
            if args[0] == "rm":
                assert args == ["rm", "--force", "d" * 64]
                if fault == "cleanup":
                    raise TimeoutError("synthetic cleanup failure")
                if fault == "overflow":
                    self.used = E["HOST_ACTUAL"]
                return "d" * 64
            raise AssertionError(args)

    commands = Fake()
    if fault == "overflow":
        with pytest.raises(ValueError, match="allowance"):
            E["host_run"](b, dec, root, commands)
        assert not (root / "control/actual/result.json").exists()
        return
    result = E["host_run"](b, dec, root, commands)
    assert len(canonical_json_bytes(result)) <= 16384
    assert "arms" not in result and "scientific_result" not in result
    if fault == "image":
        assert not any(a[0][0] in ("create", "start", "rm") for a in commands.calls)
        assert result["status"] == "INCONCLUSIVE"
    else:
        assert result["acquisition_status"] == "DIAGNOSTIC_COMPLETE"
        assert result["arm_suitability"]["original_default"] == "FAIL"
        assert "CONTROL_NOT_SUPPORTED" in result["solid_sampling_controls"]
        assert result["report_reference"]["path"] == "report.json"
        assert result["status"] == ("DIAGNOSTIC_COMPLETE" if fault is None else "INCONCLUSIVE")
    with pytest.raises(FileExistsError):
        E["host_run"](b, dec, root, commands)
    assert (root / "control/actual/attempt.json").is_file()
    assert len({deadline for _, deadline, cleanup in commands.calls if not cleanup}) == 1


def test_inherited_deadline_and_core_hook_only_shorten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    b = binding()
    sink = d.Sink(tmp_path / "deadline", b, decision(b))
    capture = Capture(sink)
    result = d._drive(capture, sink, time.monotonic() - 1.0)
    assert result["status"] == "INCONCLUSIVE" and not capture.visited and capture.closed
    monkeypatch.setattr(E["time"], "time", lambda: 100.0)
    monkeypatch.setattr(E["time"], "monotonic", lambda: 200.0)
    assert E["inherited_deadline"](binding(d.NATIVE), 150.0) == 250.0
    for value in (99.0, 376.0, float("inf")):
        with pytest.raises(TimeoutError):
            E["inherited_deadline"](binding(d.NATIVE), value)


def test_inside_sink_anchor_precedes_native_and_dummy_denies_sdk(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    b = binding(d.NATIVE)
    dec = decision(b)
    wall = time.time() + 200.0
    env = E["environment"](b, dec, wall)
    monkeypatch.setattr(E["os"], "environ", env)
    monkeypatch.setitem(G, "verify_source", lambda *a: ("a" * 40, "b" * 40))
    monkeypatch.setitem(G, "bounded_json", lambda path: b.preparation.model_dump(mode="json"))
    monkeypatch.setitem(G, "inside_runtime", lambda: None)
    root = tmp_path / "output"
    root.mkdir()
    real_path = E["Path"]
    monkeypatch.setitem(G, "Path", lambda value: root if value == "/output" else real_path(value))
    order: list[str] = []

    class Sink:
        def __init__(self, path: Path, bound: d.Binding, chosen: d.Decision):
            assert path == root and not any(path.iterdir()) and bound == b and chosen == dec
            (path / "consumed.json").write_bytes(canonical_json_bytes(chosen.model_dump()))
            order.append("sink")

    def native(repo: Path, bound: d.Binding, sink: Any) -> Any:
        assert order == ["sink"] and (root / "consumed.json").is_file()
        order.append("native")
        return object()

    monkeypatch.setattr(d, "Sink", Sink)
    monkeypatch.setattr(d, "NativeCapture", native)
    monkeypatch.setattr(
        d, "_drive", lambda capture, sink, deadline: {"status": "DIAGNOSTIC_COMPLETE"}
    )
    assert E["inside"](None) == 0 and order == ["sink", "native"]
    with pytest.raises(PermissionError, match="empty"):
        E["inside"](None)
    with pytest.raises(PermissionError):
        E["task"](b, dec, "normal")
    with pytest.raises(PermissionError):
        E["task"](binding(), decision(binding()), None)


def test_command_overflow_charges_discarded_observed_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    class Process:
        stdout = io.BytesIO(b"x" * 20000)
        returncode = 0

        def kill(self) -> None:
            pass

        def wait(self, timeout: float) -> None:
            pass

    monkeypatch.setattr(E["subprocess"], "Popen", lambda *a, **kw: Process())
    commands = E["Commands"]("no-process", 65536)
    with pytest.raises(RuntimeError):
        commands.command(["inspect"], time.monotonic() + 5)
    assert commands.used == 20000  # Kept output is only16KiB; the discarded suffix still counts.


def test_recipe_isolated_pins_and_old_defaults_unchanged() -> None:
    recipe = (ROOT / "docker/renderer-discriminator/Dockerfile").read_text()
    old = (ROOT / "docker/paired-appearance/Dockerfile").read_text()
    assert [line for line in recipe.splitlines() if line.startswith("FROM ")] == [
        line for line in old.splitlines() if line.startswith("FROM ")
    ]
    assert (
        "eps.appearance." not in recipe
        and "renderer_discriminator_execution.py --prepare" in recipe
    )
    assert "renderer_illumination_sampling_execution_v1" in recipe and "SCENE_ROOT" in recipe
    assert E["HOST_ACTUAL"] + E["HOST_DUMMY"] == 1024**2


def test_preparation_manifest_is_exact_and_exclusive_without_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    b = binding()
    monkeypatch.setitem(G, "verify_source", lambda *a: ("a" * 40, "b" * 40))
    monkeypatch.setattr(E["os"], "environ", {"SOURCE_HEAD": "a" * 40, "SOURCE_TREE": "b" * 40})
    monkeypatch.setattr(E["sys"], "argv", ["entry", "--prepare"])
    written: list[tuple[Path, bytes]] = []
    monkeypatch.setitem(G, "exclusive_file", lambda path, data: written.append((path, data)))
    assert E["main"]() == 0
    assert written == [
        (
            Path("/preparation/renderer.json"),
            canonical_json_bytes(b.preparation.model_dump(mode="json")),
        )
    ]
    E["os"].environ["SOURCE_TREE"] = "0" * 40
    with pytest.raises(PermissionError):
        E["main"]()
    assert len(written) == 1


def test_mixed_or_invalid_documents_deny_before_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    called: list[bool] = []
    monkeypatch.setitem(G, "verify_source", lambda *a: called.append(True))
    b = binding(d.NATIVE)
    dec = decision(b)
    bad = dec.model_copy(update={"purpose": d.DUMMY})
    with pytest.raises(PermissionError):
        E["host_run"](b, bad, tmp_path / "bad", object())
    with pytest.raises(PermissionError):
        E["host_run"](old_binding(), dec, tmp_path / "old", object())
    monkeypatch.setattr(E["os"], "environ", {"EPS_PAIRED_APPEARANCE_RUNTIME": ""})
    with pytest.raises(PermissionError):
        E["host_run"](b, dec, tmp_path / "mixed", object())
    assert not called
    for value in ("old-image", "embedded-authority"):
        info: dict[str, Any] = {
            "Id": b.image,
            "Config": {"Labels": E["image_labels"](b), "Env": []},
        }
        if value == "old-image":
            info["Id"] = "sha256:" + "0" * 64
        else:
            info["Config"]["Env"] = [d.ENV + "={}"]
        with pytest.raises(PermissionError):
            E["image_admission"](info, b)


def test_dummy_entry_guard_and_finite_body_are_separate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    b = binding()
    dec = decision(b)
    wall = time.time() + 30.0
    monkeypatch.setattr(E["os"], "environ", E["environment"](b, dec, wall))
    monkeypatch.setitem(G, "verify_source", lambda *a: ("a" * 40, "b" * 40))
    monkeypatch.setitem(G, "bounded_json", lambda path: b.preparation.model_dump(mode="json"))
    monkeypatch.setitem(G, "inside_runtime", lambda: None)
    root = tmp_path / "dummy-output"
    root.mkdir()
    real_path = E["Path"]
    monkeypatch.setitem(G, "Path", lambda value: root if value == "/output" else real_path(value))
    guarded: list[str] = []

    class Guard:
        def find_spec(self, *args: Any, **kwargs: Any) -> None:
            return None

    monkeypatch.setattr(
        E["runpy"],
        "run_path",
        lambda *a: {"Guard": Guard, "check_loaded": lambda: guarded.append("check")},
    )

    def body(path: Path, case: str, deadline: float) -> None:
        assert path == root and case == "normal" and deadline > time.monotonic()
        assert (root / "consumed.json").is_file()
        guarded.append("finite-body-stub")

    monkeypatch.setitem(G, "run_dummy", body)
    monkeypatch.setattr(
        d, "NativeCapture", lambda *a: (_ for _ in ()).throw(AssertionError("native denied"))
    )
    previous = list(E["sys"].meta_path)
    try:
        assert E["inside"]("normal") == 0
    finally:
        E["sys"].meta_path[:] = previous
    assert guarded == ["check", "finite-body-stub", "check"]


def test_host_deadline_reserves_cleanup_and_receipt_without_reset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    b = binding()
    dec = decision(b)
    clock = [100.0]
    monkeypatch.setitem(G, "verify_source", lambda *a: ("a" * 40, "b" * 40))
    monkeypatch.setattr(E["time"], "monotonic", lambda: clock[0])
    monkeypatch.setattr(E["time"], "time", lambda: 1000.0 + clock[0])
    root = tmp_path / "timed"

    class Fake:
        used = 0

        def __init__(self) -> None:
            self.calls: list[tuple[list[str], float]] = []

        def command(self, args: list[str], deadline: float, *, cleanup: bool = False) -> str:
            self.calls.append((args, deadline))
            if args[0] == "image":
                return canonical_json_bytes(
                    {"Id": b.image, "Config": {"Labels": E["image_labels"](b), "Env": []}}
                ).decode()
            if args[0] == "create":
                self.wall = float(
                    next(v.split("=", 1)[1] for v in args if v.startswith(E["DEADLINE_ENV"] + "="))
                )
                self.info = info_for(b, dec, root / "control/deadline" / b.output_id, self.wall)
                self.info["Config"]["Cmd"].extend(("--case", "deadline"))
                return "d" * 64
            if args[0] == "start":
                return ""
            if args[0] == "inspect" and E["POLL"] in args:
                clock[0] = 136.0
                raise TimeoutError("bounded synthetic work deadline")
            if args[0] == "inspect":
                return canonical_json_bytes(self.info).decode()
            if args[0] == "rm":
                assert args[-1] == "d" * 64 and deadline == 155.0
                return "d" * 64
            raise AssertionError(args)

    commands = Fake()
    result = E["host_run"](b, dec, root, commands, "deadline")
    assert result["cleaned"] and result["operational_status"] == "INCONCLUSIVE"
    assert commands.wall == 1135.0
    assert {
        deadline
        for args, deadline in commands.calls
        if args[0] != "rm" and E["CLEANUP"] not in args
    } == {135.0}
    assert result["elapsed_seconds"] == 36.0


def test_execution_ci_is_source_only() -> None:
    ci = (ROOT / ".github/workflows/ci.yml").read_text()
    assert ci.count("github.head_ref == 'codex/renderer-discriminator-execution-20261006' ||") == 4
    assert "python scripts/check_renderer_execution_source.py" in ci
    assert "github.head_ref != 'codex/renderer-discriminator-execution-20261006'" in ci


def test_dummy_resource_entry_never_admits_native(monkeypatch: pytest.MonkeyPatch) -> None:
    b = binding()
    env = E["environment"](b, decision(b), time.time() + 30.0)
    monkeypatch.setattr(E["os"], "environ", env)
    monkeypatch.setattr(d, "environment_binding", lambda: b)
    assert not d.candidate_runtime()  # Purpose denies before resource observations can promote it.
    with pytest.raises(PermissionError, match="before SDK access"):
        d.require_native(ROOT, b)
    with pytest.raises(PermissionError):
        d.NativeCapture(ROOT, b, None)  # type: ignore[arg-type]


def test_unlisted_host_bytes_deny_before_create_and_are_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    b = binding(d.NATIVE)
    dec = decision(b)
    root = tmp_path / "group"
    root.mkdir()
    unknown = root / "unlisted-notes.bin"
    unknown.write_bytes(b"charged")
    monkeypatch.setitem(G, "verify_source", lambda *a: ("a" * 40, "b" * 40))

    class Fake:
        used = 0

        def command(self, *a: Any, **kw: Any) -> str:
            raise AssertionError("create denied")

    with pytest.raises(ValueError, match="namespace"):
        E["host_run"](b, dec, root, Fake())
    assert unknown.read_bytes() == b"charged"
    assert (root / "control/actual/attempt.json").is_file()


def test_core_deadline_extension_cannot_exceed_existing_ceiling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    b = binding()
    sink = d.Sink(tmp_path / "ceiling", b, decision(b))
    capture = Capture(sink)
    monkeypatch.setattr(time, "monotonic", lambda: sink.started + 301.0)
    result = d._drive(capture, sink, sink.started + 1000.0)
    assert not capture.visited and capture.closed and result["status"] == "INCONCLUSIVE"
