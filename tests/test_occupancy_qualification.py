"""Handwritten noncandidate geometry and fake providers only."""

from __future__ import annotations

import ast
import json
from fractions import Fraction as Q
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from epsbench.diagnostics.boundary_observation import VisibleRaster
from epsbench.diagnostics.occupancy_collection import (
    FAILURE_BYTES,
    MEMBERS,
    TOTAL_BYTES,
    Qualification,
    Retention,
    encode,
    flatten,
    inspect,
    unflatten,
)
from epsbench.diagnostics.occupancy_reference import (
    Audit,
    Footprint,
    Solid,
    Status,
    audit,
    continuous_cover,
    face_hit,
    footprint,
    hull,
    status,
)
from epsbench.diagnostics.visible_forecast_contract import Command

TOKEN = "surface-00000000000000ab"
OTHER = "surface-00000000000000ac"


def solid(x0: int, x1: int, y0: int, y1: int, z0: int, z1: int) -> Solid:
    return Solid((Q(x0), Q(y0), Q(z0)), (Q(x1), Q(y1), Q(z1)))


def test_faces_inside_outside_and_edge() -> None:
    s = solid(-2, 2, 4, 5, -2, 2)
    assert face_hit((Q(0), Q(0), Q(0)), (Q(0), Q(1), Q(0)), s) == (Q(4), False)
    assert face_hit((Q(3), Q(0), Q(0)), (Q(0), Q(1), Q(0)), s) == (None, False)
    assert face_hit((Q(2), Q(0), Q(0)), (Q(0), Q(1), Q(0)), s) == (Q(4), True)
    assert face_hit((Q(0), Q(9, 2), Q(0)), (Q(0), Q(1), Q(0)), s) == (Q(1, 2), False)


def test_continuous_hull_and_positive_domain() -> None:
    origin = (Q(0), Q(0), Q(0))
    f = footprint(origin, solid(-2, 2, 4, 6, -2, 2))
    assert f.domain and f.depths == (Q(4), Q(6))
    assert len(f.polygon) == 4
    assert continuous_cover(origin, f, solid(-2, 2, 2, 3, -2, 2))
    assert not continuous_cover(origin, f, solid(-1, 1, 2, 3, -1, 1))  # tangent
    assert not footprint(origin, solid(-2, 2, -1, 2, -2, 2)).domain
    assert len(hull(((Q(0), Q(0)), (Q(1), Q(0)), (Q(0), Q(1)), (Q(0), Q(0))))) == 3


def test_exact_audit_tie_and_support_counts() -> None:
    calls = []
    s = solid(-2, 2, 4, 6, -2, 2)
    result = audit((Q(0), Q(0), Q(3)), (s, s), (Q(1), Q(1)), 4, lambda: calls.append(1))
    assert result.ties and calls
    assert result.supports[0][0] >= result.supports[0][1]
    support = status(result, 1, True, (Q(0), Q(0), Q(3)), s)
    assert support.cause == "UNKNOWN_DOMAIN"


def record(
    polygon: tuple[tuple[Q, Q], ...],
    full: int,
    clipped: int,
    visible: int,
    boundary: bool = False,
    tie: bool = False,
) -> Audit:
    f = Footprint(polygon, (Q(4), Q(6)), True)
    return Audit(
        (tuple([3] * visible + [0] * (4 - visible)),),
        ((0, 0),) if tie else (),
        (),
        ((0, 0, False), (full, clipped, boundary)),
        (f, f),
    )


@pytest.mark.parametrize(
    ("x0", "x1", "clipped", "visible", "expected"),
    [
        (-3, -2, 0, 0, "COMPLETE_FRAME_LOSS"),
        (-2, 0, 2, 0, "PARTIAL_FRAME_LOSS"),
        (-2, 0, 2, 1, "PARTIAL_FRAME_LOSS"),
    ],
)
def test_whole_footprint_clipping(
    x0: int, x1: int, clipped: int, visible: int, expected: str
) -> None:
    polygon = ((Q(x0), Q(-1, 2)), (Q(x1), Q(-1, 2)), (Q(x1), Q(1, 2)), (Q(x0), Q(1, 2)))
    r = record(polygon, 4, clipped, visible)
    s = status(r, 3, True, (Q(0), Q(0), Q(0)), solid(-10, 10, 2, 3, -10, 10))
    assert s.cause == expected and s.full_centres == 4
    if clipped > visible:
        assert "MIXED_FRAME_LOSS_AND_OCCLUSION" in s.flags


def test_strict_pure_hidden_boundary_and_tie() -> None:
    polygon = ((Q(-1, 4), Q(-1, 4)), (Q(1, 4), Q(-1, 4)), (Q(1, 4), Q(1, 4)), (Q(-1, 4), Q(1, 4)))
    fg = solid(-2, 2, 2, 3, -2, 2)
    origin = (Q(0), Q(0), Q(0))
    assert (
        status(record(polygon, 4, 4, 0), 3, True, origin, fg).cause
        == "COMPLETE_IN_FRUSTUM_OCCLUSION"
    )
    assert status(record(polygon, 4, 4, 0, boundary=True), 3, True, origin, fg).cause == "UNKNOWN"
    assert status(record(polygon, 4, 4, 0, tie=True), 3, True, origin, fg).cause == "UNKNOWN"
    assert status(record(polygon, 4, 4, 0), 3, False, origin, fg).cause == "UNKNOWN"


def test_closed_status_fields() -> None:
    with pytest.raises(ValueError):
        Status("PASS", (), 0, 0, 0, Footprint((), None, False), False, False)
    with pytest.raises(ValueError):
        Status("UNKNOWN", (), True, 0, 0, Footprint((), None, False), False, False)


class Fake:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.controller: Qualification | None = None
        self.bad = False

    def prefix(self, unit: str, index: int) -> VisibleRaster:
        self.calls.append(f"prefix-{unit}-{index}")
        return VisibleRaster(index, np.array([[1, 0]], dtype=np.int32), ((1, TOKEN),))

    def target(self, member: str) -> VisibleRaster:
        assert self.controller is not None and self.controller.sealed
        assert len(self.calls) >= 6
        root = self.controller.retained.root
        assert len(list(root.glob("forecast-*.json"))) == 8
        self.calls.append("target-" + member)
        if self.bad:
            raise ValueError("synthetic producer failed")
        return VisibleRaster(2, np.array([[0, 2]], dtype=np.int32), ((2, OTHER),))


def setup(root: Path, fake: Fake | None = None) -> tuple[Qualification, Fake]:
    provider = Fake() if fake is None else fake
    command = (Q(0), Q(3, 7), Q(0))
    commands: dict[str, tuple[tuple[Command, ...], Command]] = {
        m: ((command,), (Q(0), Q(i, 9), Q(0))) for i, m in enumerate(MEMBERS)
    }
    q = Qualification(provider, Retention(root), {"synthetic": True}, commands)
    provider.controller = q
    return q, provider


def test_global_seal_synthetic_smoke_validation_inspection(tmp_path: Path) -> None:
    q, provider = setup(tmp_path / "private")
    with pytest.raises(PermissionError):
        q.target("R")
    assert provider.calls == []
    reports = q.run()
    assert len(provider.calls) == 10
    assert list(reports) == list(MEMBERS)
    assert all(
        report["N"] == 2 and report["omitted_new_tokens"] == 1
        for rs in reports.values()
        for report in rs.values()
    )
    receipt = inspect(q.retained.root)
    assert receipt["status"] == "COMPLETE" and receipt["retained_bytes"] < TOTAL_BYTES
    print(encode(receipt).decode())
    with pytest.raises(RuntimeError):
        q.run()


def test_tampered_forecast_denied_before_target(tmp_path: Path) -> None:
    q, provider = setup(tmp_path / "private")
    original = q.target

    def tamper(member: str) -> VisibleRaster:
        (q.retained.root / "forecast-R-1.json").write_bytes(b"{}")
        return original(member)

    q.target = tamper  # type: ignore[method-assign]
    with pytest.raises(ValueError):
        q.run()
    assert not any(c.startswith("target") for c in provider.calls)
    failure = json.loads((q.retained.root / "failure.json").read_bytes())
    assert failure["pending"] == list(MEMBERS) and failure["stage"] == "TARGET"


def test_failure_pending_and_no_retry(tmp_path: Path) -> None:
    fake = Fake()
    fake.bad = True
    q, _ = setup(tmp_path / "private", fake)
    with pytest.raises(ValueError):
        q.run()
    assert inspect(q.retained.root)["status"] == "FAILED_CLOSED"
    assert json.loads((q.retained.root / "failure.json").read_bytes())["pending"] == list(MEMBERS)
    with pytest.raises(RuntimeError):
        q.run()


def test_deadline_call_and_byte_bounds(tmp_path: Path) -> None:
    now = [0.0]
    retention = Retention(tmp_path / "bounded", lambda: now[0])
    for _ in range(10):
        retention.producer_call()
    with pytest.raises(RuntimeError):
        retention.producer_call()
    retention.used = TOTAL_BYTES - FAILURE_BYTES
    with pytest.raises(RuntimeError):
        retention.write("extra.json", b"x")
    assert not (retention.root / "extra.json").exists()
    now[0] = 60
    with pytest.raises(RuntimeError):
        retention.check()
    retention.fail(["R"], RuntimeError("deadline"))
    assert (retention.root / "failure.json").exists()


def test_fresh_path_and_append_only(tmp_path: Path) -> None:
    retention = Retention(tmp_path / "once")
    retention.write("x.json", b"{}")
    with pytest.raises(ValueError):
        retention.write("x.json", b"{}")
    with pytest.raises(ValueError):
        Retention(retention.root)
    with pytest.raises(ValueError):
        retention.write("../escape.json", b"{}")


def test_static_source_binding_without_import_or_geometry() -> None:
    path = Path("src/epsbench/diagnostics/occupancy_producer.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: dict[str, Any] = {
        n.targets[0].id: n.value
        for n in tree.body
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
    }
    assert ast.literal_eval(names["SOURCE_HEAD"]) == "bf9d2f80caa92184232f70c101d835263433b323"
    assert (
        ast.literal_eval(names["PROPOSAL_SHA256"])
        == "753f461c597ea5e58c29c63cfec0bf92403d6a39ac53a4c1905537fc67571494"
    )
    table = next(
        n.value
        for n in tree.body
        if isinstance(n, ast.AnnAssign)
        and isinstance(n.target, ast.Name)
        and n.target.id == "TABLE"
    )
    assert isinstance(table, ast.Dict)
    assert [ast.literal_eval(k) for k in table.keys if k] == list(MEMBERS)
    # AST assertions only: no evaluation/allocation of prospective positions.
    assert "Q(57, 8)" in ast.unparse(table) and "Q(63, 10)" in ast.unparse(table)


def test_flat_buffer_fidelity_and_corruption(tmp_path: Path) -> None:
    q, _ = setup(tmp_path / "flat")
    q.run()
    from epsbench.diagnostics.occupancy_collection import LIMITS
    from epsbench.diagnostics.visible_forecast_contract import REQUIRED, CausalInput
    from epsbench.schema import ModalityPermissionSet

    source = CausalInput.from_bytes(
        (q.retained.root / "input-R.json").read_bytes(),
        ModalityPermissionSet(allowed=REQUIRED),
        LIMITS,
    )
    data = flatten(source)
    assert unflatten(data, source).canonical_bytes() == source.canonical_bytes()
    with pytest.raises(ValueError):
        unflatten(data[:-1] + b"\x02", source)
    with pytest.raises(ValueError):
        unflatten(data + b"\x00", source)


def test_write_fault_preserves_pending(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    q, provider = setup(tmp_path / "fault")
    write = q.retained.write

    def fault(name: str, data: bytes, failure: bool = False) -> str:
        if name == "forecast-F-2.json":
            raise OSError("synthetic retained write failure")
        return write(name, data, failure)

    monkeypatch.setattr(q.retained, "write", fault)
    with pytest.raises(OSError):
        q.run()
    assert not any(c.startswith("target") for c in provider.calls)
    failure = json.loads((q.retained.root / "failure.json").read_bytes())
    assert failure["failure_kind"] == "IO" and failure["pending"] == list(MEMBERS)
    assert len(list(q.retained.root.glob("forecast-*.json"))) == 5


def test_bound_artifact_inspection_fails_on_change(tmp_path: Path) -> None:
    q, _ = setup(tmp_path / "changed")
    q.run()
    (q.retained.root / "reports.json").write_bytes(b"{}")
    with pytest.raises(ValueError, match="bound retained artifact"):
        inspect(q.retained.root)


def test_actual_modules_denied_before_collection() -> None:
    import importlib

    for name in (
        "epsbench.diagnostics.occupancy_producer",
        "epsbench.diagnostics.causal_history_fixture",
    ):
        with pytest.raises(RuntimeError, match="forbidden"):
            importlib.import_module(name)


def test_exact_occupancy_ci_route_and_no_default_check() -> None:
    workflow = Path(".github/workflows/ci.yml").read_text()
    branch = "codex/occupancy-qualification-20261007"
    for job in ("a1-source", "quality", "qualify-wgl", "qualify-osmesa"):
        section = workflow.split("  " + job + ":", 1)[1].split("    runs-on:", 1)[0]
        assert branch in section and "codex/a1-integration-base-20261005" in section
    assert "github.head_ref != '" + branch + "'" in workflow
    assert "run: uv run --locked python scripts/check_occupancy_source.py" in workflow


def test_pinned_producer_hash_static_only() -> None:
    import hashlib

    tree = ast.parse(Path("src/epsbench/diagnostics/occupancy_producer.py").read_text())
    declared = next(
        ast.literal_eval(n.value)
        for n in tree.body
        if isinstance(n, ast.Assign)
        and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id == "PRODUCER_SHA256"
    )
    raw = (
        Path("src/epsbench/diagnostics/causal_history_fixture.py")
        .read_bytes()
        .replace(b"\r\n", b"\n")
    )
    assert hashlib.sha256(raw).hexdigest() == declared


def test_no_predictor_after_seal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    q, provider = setup(tmp_path / "no-readout")
    original = provider.target

    def target(member: str) -> VisibleRaster:
        import epsbench.diagnostics.occupancy_collection as module

        def denied(*args: Any, **kwargs: Any) -> None:
            raise AssertionError("predictor ran after seal")

        monkeypatch.setattr(module, "forecast", denied)
        return original(member)

    monkeypatch.setattr(provider, "target", target)
    assert len(q.run()) == 4


def test_offline_inspection_requires_all_flat_files(tmp_path: Path) -> None:
    q, _ = setup(tmp_path / "missing-flat")
    q.run()
    (q.retained.root / "flat-R.bin").unlink()
    with pytest.raises(ValueError, match="sealed artifact missing"):
        inspect(q.retained.root)


def test_reparse_flag_denied(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    from epsbench.diagnostics.occupancy_collection import reject_reparse

    original = Path.lstat

    def lstat(path: Path) -> Any:
        if path == tmp_path:
            return SimpleNamespace(st_file_attributes=1024, st_mode=0)
        return original(path)

    monkeypatch.setattr(Path, "lstat", lstat)
    with pytest.raises(ValueError, match="reparse"):
        reject_reparse(tmp_path)


def test_current_strata_and_reappearance_keep_unknown_inventory(tmp_path: Path) -> None:
    class Absent(Fake):
        def prefix(self, unit: str, index: int) -> VisibleRaster:
            self.calls.append(f"prefix-{unit}-{index}")
            return VisibleRaster(
                index,
                np.array([[1 if index == 0 else 0, 0]], dtype=np.int32),
                ((1, TOKEN),) if index == 0 else (),
            )

        def target(self, member: str) -> VisibleRaster:
            self.calls.append("target-" + member)
            return VisibleRaster(2, np.array([[1, 0]], dtype=np.int32), ((1, TOKEN),))

    q, _ = setup(tmp_path / "reappearance", Absent())
    reports = q.run()
    for rules in reports.values():
        for report in rules.values():
            assert report["N"] == 2 and report["current_strata"]["current_visible"] == []
            assert len(report["current_strata"]["current_absent"]) == 1
            assert report["reappearance"] == [{"token": TOKEN, "reappeared": True}]
        assert rules["k-frame-agreement"]["U"] == 2


def test_inspection_cap_precedes_reads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "large"
    root.mkdir()
    with (root / "large.bin").open("wb") as f:
        f.truncate(TOTAL_BYTES + 1)

    def denied(path: Path) -> bytes:
        raise AssertionError("oversized evidence read before cap")

    monkeypatch.setattr(Path, "read_bytes", denied)
    with pytest.raises(ValueError, match="before artifact reads"):
        inspect(root)


def test_failure_receipt_fault_preserves_initiating_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    q, _ = setup(tmp_path / "two-faults")

    def fault(name: str, data: bytes, failure: bool = False) -> str:
        if failure:
            raise OSError("receipt also failed")
        raise ValueError("initiating source failure")

    monkeypatch.setattr(q.retained, "write", fault)
    with pytest.raises(ValueError, match="initiating") as error:
        q.run()
    assert isinstance(error.value.__cause__, OSError)
