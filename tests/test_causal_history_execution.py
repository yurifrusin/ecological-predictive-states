"""Finite causal lifecycle/controller synthetic checks; native imports stay forbidden."""

from __future__ import annotations

import ast
import subprocess
import sys
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from epsbench.diagnostics import causal_history_lifecycle as life
from epsbench.diagnostics.a1_retention import MIB, recover
from epsbench.diagnostics.causal_history_audit import MaskScore
from epsbench.diagnostics.causal_history_execution import (
    BUDGET,
    MAX_FILES,
    MAX_MATERIALIZED,
    Binding,
    CaptureRetention,
    DockerController,
    FiniteArchive,
    HostReceipts,
    create_arguments,
    history,
    initialize,
    maximum_bytes,
    replay,
    verify_prefix,
)
from epsbench.diagnostics.causal_history_native import Capture, Progress, produce_sequence
from epsbench.diagnostics.causal_history_sequence import (
    CONFIG_SHA256,
    HEIGHT,
    MEMBERS,
    WIDTH,
    canonical,
    digest,
    parse,
)
from tests.test_causal_history_sequence import CONFIG, FakeBackend, encode

SOURCE = CONFIG.parents[2]
BINDING = Binding("1" * 40, "2" * 40, CONFIG_SHA256, "sha256:" + "3" * 64)
RENDERER = {"runtime": "synthetic-not-qualified"}
BOUND = life.BoundSource(
    head=BINDING.source_head, tree=BINDING.source_tree, image=BINDING.image, renderer=RENDERER
)


class ThreeTokens(FakeBackend):
    def __init__(self, xml: str):
        super().__init__(xml)
        self.raw[:, 120:] = 30

    def capture(self, position: float) -> Capture:
        f = super().capture(position)
        return replace(
            f, boundaries=(), evidence=f.evidence.model_copy(update={"raw_boundaries": ()})
        )


def six(sink: FiniteArchive) -> None:
    for member in MEMBERS:
        result = produce_sequence(CONFIG.read_bytes(), member, factory=ThreeTokens)
        manifest, artifacts = encode(result, member)
        for path, data in artifacts.items():
            sink.put(f"sequences/{member}/{path}", data)
        sink.put(f"sequences/{member}/manifest.json", manifest)


def test_synthetic_six_seal_expose_evaluate_without_regeneration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sink = FiniteArchive(tmp_path / "archive", BINDING.root, BUDGET)
    try:
        six(sink)
        reader = life.BoundedReader(sink)
        reads: list[str] = []
        original = reader.read

        def read(path: str, ceiling: int = 3 * MIB) -> bytes:
            reads.append(path)
            return original(path, ceiling)

        monkeypatch.setattr(reader, "read", read)
        seal = life.seal_forecasts(BOUND, reader)
        assert life.EXPOSURE_PATH not in sink.used
        assert not any(
            f"/{member}/frame-{MEMBERS[member][5]}/" in p for member in MEMBERS for p in reads
        )
        with pytest.raises(KeyError):
            life.VerifiedTargets(
                seal, life.ExposureReceipt(BINDING.root, seal.sha256, "f" * 64), reader
            )
        exposure = life.expose_targets(seal, reader)

        def forbidden(*args: Any) -> Any:
            raise AssertionError("post-exposure regeneration")

        monkeypatch.setattr(life, "persistence", forbidden)
        monkeypatch.setattr(life, "extrapolate", forbidden)
        result = life.evaluate(seal, exposure, reader)
        life.inspect(seal, exposure, reader)
        assert len(reader.read("inspection/current-target-rgb.bin")) == 691200
        assert result["member_count"] == 6 and result["case_count"] == 12
        assert result["all_four"] == {"persistence": "INCONCLUSIVE", "extrapolate": "INCONCLUSIVE"}
        private = parse(reader.read("results/private.json"))
        assert len(private["cases"]) == 12 and len(private["pairs"]) == 3
        assert result["repeat_pass"] is True
        with pytest.raises(ValueError, match="reseal"):
            life.seal_forecasts(BOUND, reader)
        with pytest.raises(ValueError, match="already exposed"):
            life.expose_targets(seal, reader)
        assert set(result) == {
            "member_count",
            "case_count",
            "repeat_pass",
            "all_four",
            "pair3",
            "phase_gate_effect",
        }
    finally:
        sink.close()


def test_missing_member_prevents_seal_and_release_retains_reason(tmp_path: Path) -> None:
    sink = FiniteArchive(tmp_path / "archive", BINDING.root, BUDGET)
    with pytest.raises(KeyError):
        life.seal_forecasts(BOUND, life.BoundedReader(sink))
    assert life.SEAL_PATH not in sink.used and life.EXPOSURE_PATH not in sink.used
    root = sink.history()
    sink.close()
    state = recover(tmp_path / "archive", BINDING.root, BUDGET, root)
    assert state.sequence == 2  # STUDY + retained terminal failure, no invented member.
    with pytest.raises(ValueError, match="terminal failed"):
        FiniteArchive(tmp_path / "archive", BINDING.root, BUDGET, expected_history=root)


def test_wrong_source_and_unobserved_background_denied(tmp_path: Path) -> None:
    for wrong in (True, False):
        sink = FiniteArchive(tmp_path / str(wrong), BINDING.root, BUDGET)
        try:
            for member in MEMBERS:
                manifest, artifacts = encode(member=member)
                for path, data in artifacts.items():
                    sink.put(f"sequences/{member}/{path}", data)
                sink.put(f"sequences/{member}/manifest.json", manifest)
            source = BOUND.model_copy(update={"head": "a" * 40}) if wrong else BOUND
            with pytest.raises(ValueError, match=r"binding|observed association"):
                life.seal_forecasts(source, life.BoundedReader(sink))
            assert life.SEAL_PATH not in sink.used
        finally:
            sink.close()


def test_actual_mask_unknown_rational_and_metadata_validation() -> None:
    common: dict[str, Any] = dict(
        member=next(iter(MEMBERS)),
        context="a" * 64,
        rule="extrapolate",
        token="surface-0000000000000001",
        status="UNKNOWN_MOTION",
        mask=None,
        mask_sha256=None,
        last_index=0,
        motion=None,
        cumulative=(1, 1),
        displacement=None,
        shift=None,
        clipped_pixels=None,
    )
    r = life.ForecastRecord(**common)
    f = r.forecast(lambda _: pytest.fail("UNKNOWN read mask"))
    assert f.mask is None and f.status == "UNKNOWN_MOTION"
    with pytest.raises(ValueError, match="rational"):
        life.ForecastRecord(**{**common, "cumulative": (2, 2)})
    with pytest.raises(ValueError, match="mask"):
        life.ForecastRecord(**{**common, "mask": "invented.bin"})
    binary = np.zeros((HEIGHT, WIDTH), dtype=np.bool_).tobytes()
    r = life.ForecastRecord(
        **{
            **common,
            "status": "MASK",
            "mask": "f.bin",
            "mask_sha256": digest(binary),
            "displacement": ((0, 1), (0, 1)),
            "shift": (0, 0),
            "clipped_pixels": 0,
        }
    )
    with pytest.raises(ValueError, match="bytes/hash/shape"):
        r.forecast(lambda _: bytes([2]) * (HEIGHT * WIDTH))


def test_all_four_complete_valid_wrong_unknown_and_missing() -> None:
    exact = MaskScore("MASK", True, 0, Fraction(1))
    wrong = MaskScore("MASK", False, 1, Fraction(0))
    unknown = MaskScore("UNKNOWN_MOTION", None, None, None)
    assert life.all_four((exact,) * 4, valid=True, qualified=True) == "PASS"
    for item in (wrong, unknown):
        assert life.all_four((exact,) * 3 + (item,), valid=True, qualified=True) == "FAIL"
    for scores, valid, qualified in (
        ((exact,) * 3, True, True),
        ((exact,) * 4, False, True),
        ((exact,) * 4, True, False),
    ):
        assert life.all_four(scores, valid=valid, qualified=qualified) == "INCONCLUSIVE"


def test_incremental_capture_failure_keeps_first_frame_and_operations(tmp_path: Path) -> None:
    sink = FiniteArchive(tmp_path / "archive", BINDING.root, BUDGET)
    member = next(iter(MEMBERS))
    retention = CaptureRetention(sink, CONFIG.read_bytes(), member, BINDING, RENDERER)
    backends = []

    class Interrupted(ThreeTokens):
        def capture(self, position: float) -> Capture:
            if len(self.positions) == 1:
                raise RuntimeError("second capture failed")
            return super().capture(position)

    def factory(xml: str) -> Interrupted:
        backend = Interrupted(xml)
        backends.append(backend)
        return backend

    try:
        with pytest.raises(RuntimeError, match="second capture"):
            produce_sequence(CONFIG.read_bytes(), member, factory=factory, progress=retention)
        assert backends[0].draws == ["rgb", "paired"] and backends[0].closed
        assert f"sequences/{member}/frame-0/rgb.bin" in sink.committed
        assert f"partial/{member}/frame-0-frame_operations.json" in sink.committed
        assert f"sequences/{member}/manifest.json" not in sink.used
        evidence = [
            parse(life.BoundedReader(sink).read(p)) for p in sink.committed if "progress-" in p
        ]
        assert evidence[-1]["stage"] == "failure" and evidence[-1]["failure"] == [
            "capture",
            "RuntimeError",
        ]
    finally:
        sink.close()


def test_progress_failure_never_adds_capture_and_snapshots_owned() -> None:
    backends = []

    def factory(xml: str) -> ThreeTokens:
        b = ThreeTokens(xml)
        backends.append(b)
        return b

    def progress(event: Progress) -> None:
        if event.stage == "frame":
            with pytest.raises(ValueError):
                event.value.rgb.setflags(write=True)
            raise OSError("failed synchronous callback")

    with pytest.raises(OSError, match="synchronous callback"):
        produce_sequence(
            CONFIG.read_bytes(), next(iter(MEMBERS)), factory=factory, progress=progress
        )
    assert backends[0].draws == ["rgb", "paired"] and backends[0].closed


def test_partial_flow_failure_early_native_read_and_no_duplicate_charge(tmp_path: Path) -> None:
    sink = FiniteArchive(tmp_path / "archive", BINDING.root, BUDGET)
    member = next(iter(MEMBERS))
    retention = CaptureRetention(sink, CONFIG.read_bytes(), member, BINDING, RENDERER)

    class Interrupted(ThreeTokens):
        calls = 0

        def transport(self, before: Capture, after: Capture) -> tuple[Any, Any]:
            self.calls += 1
            if self.calls == 2:
                raise ValueError("second flow failure")
            return super().transport(before, after)

    try:
        with pytest.raises(ValueError, match="second flow"):
            produce_sequence(CONFIG.read_bytes(), member, factory=Interrupted, progress=retention)
        assert len(retention.frames) == 4 and len(retention.flows) == 1
        assert f"sequences/{member}/flow-0/vectors.bin" in sink.committed
        before = sink.staging_reserved
        path = f"sequences/{member}/flow-0/vectors.bin"
        life.put_once(sink, path, life.BoundedReader(sink).read(path))
        assert sink.staging_reserved == before
        with pytest.raises(ValueError, match="failed capture"):
            retention.encode(partial=False)
        assert f"sequences/{member}/manifest.json" not in sink.used
    finally:
        sink.close()


def test_early_rgb_and_paired_bytes_survive_validation_failure(tmp_path: Path) -> None:
    sink = FiniteArchive(tmp_path / "early", BINDING.root, BUDGET)
    member = next(iter(MEMBERS))
    retention = CaptureRetention(sink, CONFIG.read_bytes(), member, BINDING, RENDERER)
    try:
        retention(Progress("rgb_read", 0, np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)))
        retention(
            Progress(
                "paired_read_complete",
                0,
                (b"\0" * (HEIGHT * WIDTH * 3), b"\0" * (HEIGHT * WIDTH * 4)),
            )
        )
        retention(Progress("failure", 0, ("capture-validation", "ValueError")))
        assert f"sequences/{member}/frame-0/rgb.bin" in sink.committed
        assert f"partial/{member}/frame-0-native-depth.bin" in sink.committed
        assert f"sequences/{member}/manifest.json" not in sink.used
        with pytest.raises(ValueError, match="terminal"):
            retention(Progress("rgb_attempt", 1))
    finally:
        sink.close()


def test_bounded_direct_reads_and_journal_materialization_charged(tmp_path: Path) -> None:
    sink = FiniteArchive(tmp_path / "archive", BINDING.root, BUDGET)
    try:
        sink.put("control/test.json", b"{}")
        reader = life.BoundedReader(sink)
        before = sink.staging_reserved
        assert reader.read("control/test.json") == b"{}" and sink.staging_reserved == before
        assert reader.control("control/test.json") == b"{}" and sink.staging_reserved == before + 2
        with pytest.raises(ValueError, match="bounded artifact"):
            reader.read("control/test.json", 1)
        reader.journal_reads = life.MAX_JOURNAL_READS
        with pytest.raises(ValueError, match="count exhausted"):
            reader.control("control/test.json")
        sink.start("dummy/oversized")
        with pytest.raises(ValueError, match="bound"):
            sink.reserve_staging(3 * MIB + 1)
    finally:
        sink.close()


def test_failed_launch_consumes_reservation_and_retains_exact_history(tmp_path: Path) -> None:
    archive, path = tmp_path / "archive", tmp_path / "receipts"
    receipts = HostReceipts(path, BINDING, archive, initial=True)
    expected = initialize(archive, BINDING, receipts)

    class FailedCreate(DockerController):
        def command(self, args: list[str], deadline: float, timeout: float = 15) -> str:
            assert replay(self.receipts.records, self.binding, self.path).reserved == 300
            raise RuntimeError("no actual external command")

    try:
        controller = FailedCreate("held", archive, BINDING, receipts)
        with pytest.raises(RuntimeError, match="terminal retained"):
            controller.launch("compile-only", expected, receipts.root)
        state = replay(receipts.records, BINDING, archive)
        assert state.failed and state.reserved == 300 and state.task == "compile-only"
        sink = FiniteArchive(archive, BINDING.root, BUDGET, expected_history=history(archive))
        assert "operations/compile-only-reserved.json" in sink.committed
        sink.close()
        with pytest.raises(ValueError, match="stale/incomplete"):
            controller.launch("compile-only", history(archive), receipts.root)
    finally:
        receipts.close()


def test_stale_prefix_fixed_order_changed_roots_and_purposes(tmp_path: Path) -> None:
    archive, path = tmp_path / "archive", tmp_path / "receipts"
    receipts = HostReceipts(path, BINDING, archive, initial=True)
    try:
        expected = initialize(archive, BINDING, receipts)
        controller = DockerController("held", archive, BINDING, receipts)
        for task, root in (("compile-only", "f" * 64), ("seal", receipts.root)):
            with pytest.raises(ValueError, match=r"stale|unordered"):
                controller.launch(task, expected, root)
        assert len(receipts.records) == 2
        with pytest.raises(ValueError, match="roots"):
            replay(receipts.records, BINDING, tmp_path / "other")
        dummy = replace(BINDING, purpose="causal_history_dummy_v1", dummy_task="dummy-complete")
        assert dummy.root != BINDING.root and dummy.tasks == ("dummy-complete",)
        with pytest.raises(ValueError, match="task must"):
            replace(BINDING, dummy_task="dummy-complete")
        args = create_arguments("n", "t", BINDING, archive, "compile-only", expected)
        assert "--env=EPS_CAUSAL_RUNTIME=docker_candidate_v1" in args
        assert not any("EPS_A1" in a or "WSL" in a for a in args)
        with archive.open("ab") as stream:
            stream.write(b"torn")
        with pytest.raises(ValueError, match="stale"):
            controller.launch("compile-only", expected, receipts.root)
    finally:
        receipts.close()


def test_proof_limits_and_static_paired_call_order() -> None:
    bound = maximum_bytes()
    assert bound["staging"] == MAX_MATERIALIZED == 240 * MIB
    assert bound["total"] < 1024 * MIB and bound["archive_including_terminal"] < BUDGET.archive
    assert bound["reserved_seconds"] == 2700 and MAX_FILES == 2048
    # Native guard forbids importing this module. Static review binds call order.
    tree = ast.parse((SOURCE / "src/epsbench/sim/canonical_paired.py").read_text())
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "CanonicalPairedRenderer"
    )
    capture = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "capture")
    calls = [
        n.func.attr
        for n in ast.walk(capture)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
    ]
    assert calls.count("_render") == 1 and calls.count("_read") == 1
    assert calls.count("_observe") == 4


@pytest.mark.parametrize("failure", (None, "timeout", "overflow", "cleanup", "prefix"))
def test_owned_controller_completion_and_terminal_failure(
    tmp_path: Path, failure: str | None
) -> None:
    archive, path = tmp_path / "archive", tmp_path / "receipts"
    receipts = HostReceipts(path, BINDING, archive, initial=True)
    expected = initialize(archive, BINDING, receipts)

    class Synthetic(DockerController):
        def command(self, args: list[str], deadline: float, timeout: float = 15) -> str:
            operation = args[0]
            state = replay(self.receipts.records, self.binding, self.path)
            assert state.task == "compile-only" and state.reserved == 300
            result = ""
            if operation == "create":
                result = "c" * 64
            elif operation == "inspect":
                result = canonical(
                    {
                        "Id": "c" * 64,
                        "Name": "/" + str(state.name),
                        "Image": BINDING.image,
                        "Config": {"Labels": {"eps.causal-history.execution-owner": state.token}},
                        "HostConfig": {
                            "Mounts": [
                                {
                                    "Source": str(archive.resolve()),
                                    "Type": "bind",
                                    "Target": "/retained/history",
                                }
                            ],
                            "Binds": None,
                            "Memory": 8 * 1024**3,
                            "MemorySwap": 8 * 1024**3,
                            "NanoCpus": 2_000_000_000,
                            "CpusetCpus": "0,1",
                            "PidsLimit": 64,
                            "ReadonlyRootfs": True,
                            "NetworkMode": "none",
                            "LogConfig": {"Type": "none"},
                            "CapDrop": ["ALL"],
                            "Privileged": False,
                            "SecurityOpt": ["no-new-privileges"],
                            "ShmSize": MIB,
                            "Tmpfs": {"/output": "rw,noexec,nosuid,nodev,size=255m"},
                            "RestartPolicy": {"Name": "no"},
                        },
                        "Mounts": [
                            {
                                "Source": str(archive.resolve()),
                                "Type": "bind",
                                "Destination": "/retained/history",
                                "RW": True,
                            }
                        ],
                        "State": {"Running": False, "ExitCode": 0, "OOMKilled": False},
                    }
                ).decode()
            elif operation == "start" and failure not in {"timeout", "overflow"}:
                sink = FiniteArchive(
                    archive, BINDING.root, BUDGET, expected_history=history(archive)
                )
                sink.put("operations/compile-only-complete.json", b'{"complete":true}')
                sink.close()
                if failure == "prefix":
                    content = archive.read_bytes()
                    changed = content.replace(BINDING.root.encode(), ("f" * 64).encode(), 1)
                    assert changed != content
                    archive.write_bytes(changed)
            failed = (operation == "start" and failure in {"timeout", "overflow"}) or (
                operation == "rm" and failure == "cleanup"
            )
            self.receipts.append(
                {
                    "kind": "COMMAND",
                    "operation": operation,
                    "exit": 1 if failed else 0,
                    "timeout": failure == "timeout" and failed,
                    "overflow": failure == "overflow" and failed,
                    "stdout": result,
                    "stderr": "",
                }
            )
            if failed:
                raise RuntimeError("synthetic unsuccessful command")
            return result

    try:
        controller = Synthetic("held", archive, BINDING, receipts)
        if failure is None:
            after = controller.launch("compile-only", expected, receipts.root)
            state = replay(receipts.records, BINDING, archive)
            assert state.history == after and state.index == 1 and state.task is None
            # Old receipt/history pair cannot authorize the next capture.
            with pytest.raises(ValueError, match="stale"):
                controller.launch("capture-chf-v1-p1-a", expected, receipts.root)
        else:
            with pytest.raises(RuntimeError, match="terminal retained"):
                controller.launch("compile-only", expected, receipts.root)
            assert replay(receipts.records, BINDING, archive).failed
        assert receipts.records[-1]["kind"] == ("COMPLETE" if failure is None else "FAILURE")
    finally:
        receipts.close()


def test_exact_prefix_and_runtime_purpose_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from epsbench.diagnostics.causal_history_runtime import causal_candidate

    path = tmp_path / "prefix"
    path.write_bytes(b"prefixsuffix")
    verify_prefix(path, 6, digest(b"prefix"))
    with pytest.raises(ValueError, match="changed"):
        verify_prefix(path, 6, digest(b"other!"))
    with pytest.raises(ValueError, match="rolled back"):
        verify_prefix(path, 20, digest(b"prefix"))
    for marker in (None, "docker_candidate_v1", "a1_native_study_v1"):
        if marker is None:
            monkeypatch.delenv("EPS_CAUSAL_RUNTIME", raising=False)
        else:
            monkeypatch.setenv("EPS_CAUSAL_RUNTIME", marker)
        monkeypatch.setenv("EPS_A1_RUNTIME", "docker_candidate_v1")
        assert causal_candidate() is False


def test_changed_host_anchor_denied_before_reservation(tmp_path: Path) -> None:
    archive, path = tmp_path / "archive", tmp_path / "receipts"
    receipts = HostReceipts(path, BINDING, archive, initial=True)
    expected = initialize(archive, BINDING, receipts)
    try:
        with path.open("ab") as stream:
            stream.write(b"foreign suffix\n")
        controller = DockerController("held", archive, BINDING, receipts)
        with pytest.raises(ValueError, match="changed or replaced"):
            controller.launch("compile-only", expected, receipts.root)
        assert len(receipts.records) == 2 and receipts.poisoned
    finally:
        receipts.close()


def test_dummy_controller_entry_imports_without_native_producer() -> None:
    # Import-only fresh interpreter probe, never a dummy task/container/qualification.
    code = """
import importlib.abc, sys, runpy
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + ".") for name in (
            "mujoco", "OpenGL", "glfw", "epsbench.sim", "epsbench.data.generate",
            "epsbench.diagnostics.causal_history_native",
        )):
            raise RuntimeError("forbidden dummy producer/native import: " + fullname)
sys.meta_path.insert(0, Guard())
sys.path.insert(0, "src")
import epsbench.diagnostics.causal_history_execution
runpy.run_path("scripts/causal_history_entry.py")
"""
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=SOURCE, timeout=20, check=True, capture_output=True
    )
    assert result.stdout == result.stderr == b""
