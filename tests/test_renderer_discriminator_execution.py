"""Guarded adapter fake-command/entry contracts; never invokes Docker or dummy bodies."""

from __future__ import annotations

import copy
import io
import json
import re
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
    "fault",
    [
        None,
        "cleanup",
        "control",
        "nonzero",
        "image",
        "ownership",
        "overflow",
        "corrupt_report",
        "missing_report",
        "corrupt_terminal",
        "missing_terminal",
        "inconsistent_terminal",
        "invalid_endpoint",
        "invalid_event",
        "partial",
    ],
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
                result = d._drive(
                    Capture(sink, control_fault=True, fail=5 if fault == "partial" else None), sink
                )
                assert result["status"] == (
                    "INCONCLUSIVE" if fault == "partial" else "DIAGNOSTIC_COMPLETE"
                )
                if fault in ("corrupt_report", "corrupt_terminal", "invalid_endpoint"):
                    name = {
                        "corrupt_report": "report.json",
                        "corrupt_terminal": "terminal.json",
                        "invalid_endpoint": "endpoints/e00/rgb.bin",
                    }[fault]
                    file = sink.root / name
                    file.write_bytes(
                        b"{}" if fault == "corrupt_terminal" else file.read_bytes() + b" "
                    )
                elif fault in ("missing_report", "missing_terminal"):
                    (
                        sink.root
                        / ("report.json" if fault == "missing_report" else "terminal.json")
                    ).unlink()
                elif fault == "inconsistent_terminal":
                    terminal = E["bounded_json"](sink.root / "terminal.json")
                    terminal["counts"]["rgb_attempt"] = 15
                    (sink.root / "terminal.json").write_bytes(canonical_json_bytes(terminal))
                elif fault == "invalid_event":
                    (sink.root / "events/0000.json").write_bytes(b"{}")
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
    elif fault in ("invalid_endpoint", "invalid_event"):
        assert result["measurement_status"] == "UNVERIFIED"
        assert "verified_measurement_endpoints" not in result and "arm_suitability" not in result
        assert result["acquisition_status"] == result["operational_status"] == "INCONCLUSIVE"
        with pytest.raises(ValueError):
            d.replay(root / b.output_id, b)
    elif fault == "partial":
        assert result["measurement_status"] == "VERIFIED_PREFIX"
        assert result["verified_measurement_endpoints"] == 5
        assert result["arm_suitability"] == {"original_default": "FAIL"}
        assert result["solid_sampling_controls"] == []
        assert result["acquisition_status"] == result["status"] == "INCONCLUSIVE"
    else:
        assert result["measurement_status"] == "VERIFIED_COMPLETE"
        assert result["verified_measurement_endpoints"] == 16
        assert result["arm_suitability"]["original_default"] == "FAIL"
        assert "CONTROL_NOT_SUPPORTED" in result["solid_sampling_controls"]
        final_fault = fault in (
            "corrupt_report",
            "missing_report",
            "corrupt_terminal",
            "missing_terminal",
            "inconsistent_terminal",
        )
        if final_fault:
            assert (
                result["bundle_integrity_status"] == result["acquisition_status"] == "INCONCLUSIVE"
            )
            assert result["operational_status"] == result["status"] == "INCONCLUSIVE"
            assert "retention_error" in result and "report_reference" not in result
            if fault != "missing_terminal":
                if fault == "inconsistent_terminal":
                    # Prefix components remain intact; host exact report counts deny acceptance.
                    assert len(d.replay(root / b.output_id, b)) == 16
                else:
                    with pytest.raises((ValueError, FileNotFoundError)):
                        d.replay(root / b.output_id, b)
        else:
            assert result["acquisition_status"] == "DIAGNOSTIC_COMPLETE"
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


# Independent expected validator payload: this deliberately does not reuse HOST_FIELDS.
INSPECT_HOST_FIELDS = {
    "Binds",
    "Privileged",
    "ReadonlyRootfs",
    "NetworkMode",
    "CapDrop",
    "CapAdd",
    "SecurityOpt",
    "Memory",
    "MemorySwap",
    "NanoCpus",
    "CpusetCpus",
    "PidsLimit",
    "ShmSize",
    "Tmpfs",
    "LogConfig",
    "RestartPolicy",
    "Mounts",
}


def fake_projection(template: str, full: dict[str, Any]) -> dict[str, Any]:
    """Evaluate the requested Go-template subset against full fake daemon objects.

    Fixtures use daemon JSON names; convert the two aliases to typed input fields first.
    Missing typed source fields raise; this models no raw-JSON fallback or omitempty defaults.
    Range retains every mount entry. Real CLI compatibility needs separate actual evidence.
    """
    typed = copy.deepcopy(full)
    if "Id" in typed:
        typed["ID"] = typed.pop("Id")
    if "HostConfig" in typed and "NanoCpus" in typed["HostConfig"]:
        typed["HostConfig"]["NanoCPUs"] = typed["HostConfig"].pop("NanoCpus")

    def lookup(path: str, item: dict[str, Any] | None = None) -> Any:
        value = typed if path.startswith(".") else item
        for key in path.removeprefix("$m.").removeprefix(".").split("."):
            assert value is not None
            value = value[key]
        return value

    def scalar(text: str, item: dict[str, Any] | None = None) -> str:
        return re.sub(
            r"\{\{json ([.$\w]+)\}\}",
            lambda match: json.dumps(lookup(match[1], item), separators=(",", ":")),
            text,
        )

    def array(match: re.Match[str]) -> str:
        entries = lookup(match[1])
        assert type(entries) is list
        return ",".join(scalar(match[2], entry) for entry in entries)

    text = re.sub(
        r"\{\{range \$i, \$m := ([.\w]+)\}\}\{\{if \$i\}\},\{\{end\}\}(.*?)\{\{end\}\}",
        array,
        template,
    )
    assert "{{" not in scalar(text)
    return json.loads(scalar(text))  # type: ignore[no-any-return]


def compact_fixture(tmp_path: Path) -> tuple[d.Binding, d.Decision, Path, float, dict[str, Any]]:
    b = binding()
    dec = decision(b).model_copy(
        update={
            "authorization": "Synthetic source-only authority. " + "a" * 224,
        }
    )
    output = tmp_path / b.output_id
    wall = time.time() + 35.0
    full = info_for(b, dec, output, wall)
    full["Config"]["Cmd"].extend(("--case", "normal"))
    return b, dec, output, wall, full


def test_projection_requests_every_consumed_field_and_excludes_daemon_metadata(
    tmp_path: Path,
) -> None:
    b, dec, output, wall, full = compact_fixture(tmp_path)
    full["GraphDriver"] = {"Data": {"LowerDir": "x" * 100000}}
    full["Config"]["UnconsumedDefaults"] = "x" * 100000
    full["HostConfig"]["UnconsumedDefaults"] = "x" * 100000
    full["HostConfig"]["LogConfig"]["Config"] = {"ignored": "x" * 100000}
    full["Mounts"][0]["Propagation"] = "x" * 100000
    full["HostConfig"]["Mounts"][0]["BindOptions"] = {"ignored": "x" * 100000}
    projected = fake_projection(E["CONTAINER_INSPECT"], full)
    assert set(projected) == {"Id", "Name", "Image", "Config", "HostConfig", "Mounts"}
    assert set(projected["Config"]) == {"Labels", "Env", "Cmd", "Entrypoint"}
    assert projected["Config"]["Labels"] == full["Config"]["Labels"]
    assert projected["Config"]["Env"] == full["Config"]["Env"]
    assert set(projected["HostConfig"]) == INSPECT_HOST_FIELDS
    assert projected["HostConfig"]["LogConfig"] == {"Type": "none"}
    assert projected["HostConfig"]["RestartPolicy"] == {"Name": "no"}
    assert set(projected["HostConfig"]["Mounts"][0]) == {"Type", "Source", "Target", "ReadOnly"}
    assert set(projected["Mounts"][0]) == {"Type", "Source", "Destination", "RW"}
    # Same admission result before/after projection; large unused fields cannot occupy the reader.
    E["inspect_confinement"](full, b, dec, output, "normal", wall)
    E["inspect_confinement"](projected, b, dec, output, "normal", wall)
    image = {"Id": b.image, "Config": full["Config"], "RootFS": {"Layers": ["x" * 100000]}}
    image["Config"] = {**image["Config"], "Env": ["PATH=/usr/local/bin:/usr/bin"]}
    small_image = fake_projection(E["IMAGE_INSPECT"], image)
    assert small_image == {
        "Id": b.image,
        "Config": {
            "Labels": full["Config"]["Labels"],
            "Env": ["PATH=/usr/local/bin:/usr/bin"],
        },
    }
    E["image_admission"](small_image, b)
    cleanup = fake_projection(E["CLEANUP"], full)
    assert set(cleanup) == {"Id", "Name", "Config", "Mounts"}
    assert cleanup["Config"] == {"Labels": full["Config"]["Labels"]}
    E["owned"](cleanup, b, full["Id"])
    E["sole_mount"](cleanup, output)
    assert len(canonical_json_bytes(projected)) < 6000


def leaf_paths(value: dict[str, Any], prefix: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
    paths: list[tuple[Any, ...]] = []
    for key, child in value.items():
        path = (*prefix, key)
        if (
            key in ("Labels", "Env", "Tmpfs")
            or not isinstance(child, (dict, list))
            or child is None
        ):
            paths.append(path)
        elif isinstance(child, dict):
            paths.extend(leaf_paths(child, path))
        elif child and isinstance(child[0], dict):
            for index, item in enumerate(child):
                paths.extend(leaf_paths(item, (*path, index)))
        else:
            paths.append(path)
    return paths


def test_typed_selectors_preserve_json_keys_and_explicit_writable_mount(tmp_path: Path) -> None:
    b, _, _, _, full = compact_fixture(tmp_path)
    image = {"Id": b.image, "Config": {"Labels": E["image_labels"](b), "Env": []}}
    for key, fixture in (("IMAGE_INSPECT", image), ("CONTAINER_INSPECT", full), ("CLEANUP", full)):
        template = E[key]
        assert "{{json .ID}}" in template and "{{json .Id}}" not in template
        projected = fake_projection(template, fixture)
        assert projected["Id"] == fixture["Id"] and "ID" not in projected
        with pytest.raises(KeyError, match="Id"):
            fake_projection(template.replace("json .ID}", "json .Id}"), fixture)
    container = E["CONTAINER_INSPECT"]
    assert "{{json .HostConfig.NanoCPUs}}" in container
    assert "{{json .HostConfig.NanoCpus}}" not in container
    projected = fake_projection(container, full)
    assert projected["HostConfig"]["NanoCpus"] == 2_000_000_000
    assert "NanoCPUs" not in projected["HostConfig"]
    assert projected["HostConfig"]["Mounts"][0]["ReadOnly"] is False
    with pytest.raises(KeyError, match="NanoCpus"):
        fake_projection(container.replace(".HostConfig.NanoCPUs}", ".HostConfig.NanoCpus}"), full)
    # The fake evaluator must not manufacture a writable fact from an omitted field.
    omitted = copy.deepcopy(full)
    del omitted["HostConfig"]["Mounts"][0]["ReadOnly"]
    with pytest.raises(KeyError, match="ReadOnly"):
        fake_projection(container, omitted)


def test_all_projected_required_fields_missing_or_mutated_deny(tmp_path: Path) -> None:
    b, dec, output, wall, full = compact_fixture(tmp_path)
    compact = fake_projection(E["CONTAINER_INSPECT"], full)
    for path in leaf_paths(compact):
        for missing in (True, False):
            changed = copy.deepcopy(compact)
            parent = changed
            for key in path[:-1]:
                parent = parent[key]
            if missing:
                del parent[path[-1]]
            else:
                parent[path[-1]] = ["unexpected"]
            with pytest.raises((PermissionError, TypeError, ValueError, KeyError)):
                E["inspect_confinement"](changed, b, dec, output, "normal", wall)
    for section, key in (
        ("HostConfig", "Binds"),
        ("HostConfig", "CapAdd"),
        ("Config", "Entrypoint"),
    ):
        malformed: Any
        for malformed in (False, "", {}, [1]):
            changed = copy.deepcopy(compact)
            changed[section][key] = malformed
            with pytest.raises(PermissionError):
                E["inspect_confinement"](changed, b, dec, output, "normal", wall)
    changed = copy.deepcopy(compact)
    changed["HostConfig"]["SecurityOpt"].append(1)
    with pytest.raises(PermissionError):
        E["inspect_confinement"](changed, b, dec, output, "normal", wall)
    for field in ("Id", "Config", "HostConfig", "Image", "Mounts"):
        changed = copy.deepcopy(compact)
        del changed[field]
        with pytest.raises(PermissionError):
            E["inspect_confinement"](changed, b, dec, output, "normal", wall)
    for env in (None, ["no-equals"], [1], ["PATH=x", "PATH=x"], ["EPS_A1_TOKEN=x"]):
        with pytest.raises(PermissionError):
            E["image_admission"](
                {
                    "Id": b.image,
                    "Config": {
                        "Labels": E["image_labels"](b),
                        "Env": env,
                    },
                },
                b,
            )
    for path in (("Id",), ("Config", "Labels"), ("Config", "Env")):
        image: dict[str, Any] = {
            "Id": b.image,
            "Config": {"Labels": E["image_labels"](b), "Env": []},
        }
        parent = image if len(path) == 1 else image[path[0]]
        del parent[path[-1]]
        with pytest.raises(PermissionError):
            E["image_admission"](image, b)


@pytest.mark.parametrize("polls,overflow", [(1, False), (70, False), (1, True)])
def test_real_reader_cumulative_lifecycle_and_reserved_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, polls: int, overflow: bool
) -> None:
    b = binding()
    dec = decision(b).model_copy(
        update={
            "authorization": "Synthetic source-only authority. " + "a" * 224,
        }
    )
    root = tmp_path / "reader-group"
    clock = [time.monotonic()]
    monkeypatch.setattr(E["time"], "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        E["time"], "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds)
    )
    monkeypatch.setitem(G, "verify_source", lambda *a: ("a" * 40, "b" * 40))
    calls: list[list[str]] = []
    charges: list[int] = []
    observed: list[int] = []
    full: dict[str, Any] = {}
    remaining = [polls]

    class Process:
        returncode = 0

        def __init__(self, payload: bytes):
            index = len(observed)
            observed.append(0)

            class Pipe(io.BytesIO):
                def read(self, size: int | None = -1) -> bytes:
                    chunk = super().read(size)
                    observed[index] += len(chunk)
                    return chunk

            self.stdout = Pipe(payload)

        def kill(self) -> None:
            pass

        def wait(self, timeout: float) -> None:
            assert 0 < timeout <= 15.0

    def popen(args: list[str], **kwargs: Any) -> Process:
        command = args[1:]
        calls.append(command)
        if command[0] == "image":
            # Python/uv image defaults, complete preparation labels, large ignored layer metadata.
            image = {
                "Id": b.image,
                "Config": {
                    "Labels": E["image_labels"](b),
                    "Env": [
                        "PATH=/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin",
                        "LANG=C.UTF-8",
                        "PYTHON_VERSION=3.11.15",
                        "PYTHON_SHA256=" + "0" * 64,
                        "UV_COMPILE_BYTECODE=1",
                        "UV_LINK_MODE=copy",
                    ],
                },
                "RootFS": {"Layers": ["sha256:" + "f" * 64] * 1000},
            }
            value = canonical_json_bytes(fake_projection(command[3], image))
        elif command[0] == "create":
            wall = float(
                next(v.split("=", 1)[1] for v in command if v.startswith(E["DEADLINE_ENV"] + "="))
            )
            full.update(info_for(b, dec, root / "control/normal" / b.output_id, wall))
            full["Config"]["Cmd"].extend(("--case", "normal"))
            full["Config"]["Env"].extend(
                [
                    "PATH=/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin",
                    "LANG=C.UTF-8",
                    "PYTHON_VERSION=3.11.15",
                    "PYTHON_SHA256=" + "0" * 64,
                    "UV_COMPILE_BYTECODE=1",
                    "UV_LINK_MODE=copy",
                ]
            )
            full["GraphDriver"] = {"Data": "x" * 100000}
            full["HostConfig"]["UnusedDaemonDefaults"] = "x" * 100000
            if overflow:
                # Required complete environment cannot be silently filtered to fit the allowance.
                full["Config"]["Env"].append("UNRELATED_IMAGE_ENV=" + "x" * 4096)
            value = b"d" * 64 + b"\n"
        elif command[0] == "start":
            value = b"d" * 64 + b"\n"
        elif command[0] == "inspect" and E["POLL"] in command:
            remaining[0] -= 1
            value = b"true 0 false\n" if remaining[0] or polls == 70 else b"false 0 false\n"
        elif command[0] == "inspect":
            value = canonical_json_bytes(fake_projection(command[2], full))
        elif command[0] == "rm":
            assert command == ["rm", "--force", "d" * 64]
            value = b"d" * 64 + b"\n"
        else:
            raise AssertionError(command)
        charges.append(len(value))
        return Process(value)

    monkeypatch.setattr(E["subprocess"], "Popen", popen)
    commands = E["Commands"]("fake-no-external-process", 16384)
    result = E["host_run"](b, dec, root, commands, "normal")
    assert commands.cap == 16384 and commands.cleanup_reserve == 8192
    assert result["cleaned"] and calls[-1] == ["rm", "--force", "d" * 64]
    assert E["CLEANUP"] in calls[-2]
    assert commands.used == sum(observed)
    assert result["command_bytes_charged"] == sum(observed)
    assert (root / "control/normal/attempt.json").is_file()
    assert result["status"] == "INCONCLUSIVE"  # No body was invoked and no consumed anchor exists.
    if overflow:
        assert not any(call[0] == "start" for call in calls)
        assert "overflow" in result["reason"]
        assert commands.used > 8192 and commands.used < 16384
    else:
        assert sum(charges[:-2]) < 8192
        assert len([call for call in calls if E["POLL"] in call]) == polls
        assert any(call[0] == "start" for call in calls)
        if polls == 70:
            assert result["elapsed_seconds"] == 35.0
            assert result["reason"] == "work watchdog expired"
    print(
        f"synthetic cumulative polls={polls} overflow={overflow} "
        f"generated={charges} observed={observed} total={sum(observed)}"
    )


def test_compact_ci_route_preserves_historical_jobs() -> None:
    ci = (ROOT / ".github/workflows/ci.yml").read_text()
    branch = "codex/renderer-discriminator-compact-inspection-20261006"
    assert ci.count(f"github.head_ref == '{branch}' ||") == 5
    assert f"github.head_ref != '{branch}'" in ci
    assert "python scripts/check_renderer_execution_source.py" in ci


def test_typed_selector_ci_route_stays_source_only() -> None:
    ci = (ROOT / ".github/workflows/ci.yml").read_text()
    branch = "codex/renderer-typed-selectors-20261006"
    assert ci.count(f"github.head_ref == '{branch}' ||") == 5
    assert f"github.head_ref != '{branch}'" in ci
    assert "python scripts/check_renderer_execution_source.py" in ci
