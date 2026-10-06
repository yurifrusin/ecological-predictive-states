from __future__ import annotations

import copy
import time
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from epsbench.diagnostics import paired_appearance as p
from epsbench.diagnostics import paired_appearance_execution as e
from epsbench.diagnostics import paired_appearance_runtime as r
from epsbench.diagnostics.appearance_study import CONTRAST_V1, PAIRED_V1, require_spec, select_study
from epsbench.diagnostics.paired_appearance_native import NativeCapture
from epsbench.utils.canonical import canonical_json_bytes, logical_array_hash, sha256_bytes
from tests.test_paired_appearance import synthetic, synthetic_corridor
from tests.test_paired_appearance_execution import HOST, container_info, decision, emit

ROOT = Path(__file__).resolve().parents[1]
STUDY = CONTRAST_V1


def binding() -> r.AppearanceExecutionBinding:
    return r.AppearanceExecutionBinding(
        preparation=r.preparation(ROOT, "a" * 40, "b" * 40, STUDY),
        image="sha256:" + "c" * 64,
        purpose=STUDY.native_purpose,
        output_id="e" * 32,
        token="f" * 32,
    )


def fixture(context: int, index: int) -> p.Frame:
    family, appearance, _ = p.contexts(STUDY)[context]
    return (
        synthetic(appearance, index, STUDY)
        if family == "single_occluder"
        else synthetic_corridor(appearance, index, STUDY)
    )


class Capture:
    def __init__(
        self,
        context: int,
        progress: Any,
        opened: list[int],
        *,
        negative: bool = False,
        close_fault: bool = False,
    ):
        self.context, self.progress = context, progress
        self.negative, self.close_fault = negative, close_fault
        opened.append(context)

    def capture(self, index: int) -> p.Frame:
        frame = fixture(self.context, index)
        if self.negative and self.context == 1 and index == 0:
            frame = replace(frame, rgb=np.full_like(frame.rgb, 201))
        emit(self.progress, frame)
        return frame

    def close(self) -> None:
        if self.close_fault and self.context == 1:
            raise OSError("synthetic close failure after valid negative")


def test_fixed_calibration_and_old_contract_are_distinct() -> None:
    assert select_study(STUDY.schema) is STUDY
    assert select_study(PAIRED_V1.schema) is PAIRED_V1
    assert PAIRED_V1.config_file_hash == r.CONFIG_FILE
    assert p.fixed_config(PAIRED_V1.config_file_bytes) == p.FIXED
    assert p.canonical_assets() == p.canonical_assets(PAIRED_V1)
    assert p.config_bytes() == p.config_bytes(PAIRED_V1)
    assert p.contexts() == p.contexts(PAIRED_V1)
    old_preparation = r.preparation(ROOT, "a" * 40, "b" * 40)
    assert (
        old_preparation.config_root
        == "714a447a0cd9da0ce856d3e3466d90a25f7d74d2dd13fab6537f85140652aaae"
    )
    assert (
        old_preparation.asset_root
        == "8a049cc723851fd7b4c7253351ae042985bebcec3448e452ffd7b7099b46799e"
    )
    assert (
        old_preparation.protection_root
        == "defe33f7d4758f1e2ca8aba1bdd285c13a8288f61a88360d795181e4bdc6535b"
    )
    assert (
        old_preparation.membership_root
        == "55c8dc6d0e1ce9e96f64ab3c09852417aba6754081cacc41412c35417b3867c6"
    )
    old_frame = synthetic()
    assert (
        sha256_bytes(canonical_json_bytes(old_frame.evidence))
        == "361826c4edf3f2e9806b8d9ae18d047463d3db8cfaa76c86d3dd2ec52b1270b5"
    )
    assert (
        sha256_bytes(p.encode(old_frame))
        == "6bc0510fe88e33835352eaba90435c445c47cd5cd91fc2f73b5771af761fc1ba"
    )

    assert p.seed_domain("single_occluder")["root"] == 2026100601
    assert p.seed_domain("corridor")["root"] == 2026100602
    assert p.seed_domain("single_occluder", STUDY)["root"] == 2026100603
    assert p.seed_domain("corridor", STUDY)["root"] == 2026100604
    old, new = PAIRED_V1.fixed, STUDY.fixed
    for key in ("schema", "roots", "appearances", "palette"):
        old.pop(key)
        new.pop(key)
    assert old == new
    assert len(p.contexts(STUDY)) == 8
    assert STUDY.appearances == ("dev_calibration_solid_v1", "dev_calibration_brick_v1")
    pixels = p.brick(0, STUDY)
    assert set(map(tuple, pixels.reshape(-1, 3))) == {(224, 224, 224), (32, 32, 32)}
    assert all(np.array_equal(pixels, p.brick(i, STUDY)) for i in range(4))
    assert len(set(p.canonical_assets(STUDY).values())) == 1
    assert len(STUDY.asset_names) == 1
    assert (
        STUDY.config_file_hash == "ee9d3a3591bc0ae3aaac4767d51db31e1e92c83b81b86adae81b397d2773947f"
    )
    assert (
        logical_array_hash(pixels)
        == "9114be5304fa8f4424eb32c137dd69fadf84da19b0f45a84fcd5f924ecaf527f"
    )
    receipt = p.protection_check(ROOT, STUDY)
    assert len(receipt["inputs"]) == 12 and "18/21" in receipt["historical_limit"]
    old_values = {v for family in p.FAMILIES for v in p.seed_domain(family).values()}
    assert not set(receipt["fresh_seed_values"]) & old_values
    assert p.retention_bound()["total_with_two_exports_and_host_metadata"] == 55 * p.MIB


def test_immutable_closed_selection_and_exact_configs() -> None:
    with pytest.raises(FrozenInstanceError):
        STUDY.schema = "arbitrary"  # type: ignore[misc]  # Deliberate immutable rejection.
    with pytest.raises(ValueError):
        require_spec(replace(STUDY))
    with pytest.raises(ValueError):
        select_study("unknown")
    private_copy = STUDY.fixed
    private_copy["roots"][0] += 1
    assert STUDY.fixed["roots"][0] == 2026100603
    for bad in (private_copy, {**STUDY.fixed, "extra": 1}, PAIRED_V1.fixed):
        with pytest.raises(ValueError):
            p.fixed_config(canonical_json_bytes(bad), STUDY)
    with pytest.raises(ValueError):
        p.config_bytes(replace(STUDY))


@pytest.mark.parametrize("field", ("version", "config_file", "config_root", "membership_root"))
def test_cross_domain_preparation_rejection(field: str) -> None:
    new = binding().preparation.model_dump(mode="json")
    old = r.preparation(ROOT, "a" * 40, "b" * 40).model_dump(mode="json")
    new[field] = old[field]
    with pytest.raises(ValidationError):
        r.Preparation.model_validate(new)


def test_wrong_purpose_or_membership_denied_before_native() -> None:
    b = binding()
    mixed = b.model_dump(mode="json")
    mixed["purpose"] = PAIRED_V1.native_purpose
    with pytest.raises(ValidationError):
        r.AppearanceExecutionBinding.model_validate(mixed)
    with pytest.raises(PermissionError):
        NativeCapture(
            ROOT,
            "single_occluder",
            PAIRED_V1.appearances[0],
            source_head="a" * 40,
            source_tree="b" * 40,
            progress=lambda *args: None,
            binding=b,
        )
    mixed["purpose"] = STUDY.dummy_purpose
    dummy = r.AppearanceExecutionBinding.model_validate(mixed)
    with pytest.raises(PermissionError):
        NativeCapture(
            ROOT,
            "single_occluder",
            STUDY.appearances[0],
            source_head="a" * 40,
            source_tree="b" * 40,
            progress=lambda *args: None,
            binding=dummy,
        )
    altered = decision(b).model_copy(update={"purpose": PAIRED_V1.native_purpose})
    with pytest.raises(PermissionError):
        r.require_decision(b, altered)


@pytest.mark.parametrize("family", p.FAMILIES)
def test_new_serialization_validation_and_inspection(family: p.Family) -> None:
    ordinal = 0 if family == "single_occluder" else 4
    frame = fixture(ordinal, 0)
    p.validate_frame(frame, STUDY)
    with pytest.raises(ValueError):
        p.validate_frame(frame)
    encoded = p.encode(frame, STUDY)
    decoded = p.decode(encoded, STUDY)
    assert decoded.study is STUDY and decoded.evidence == frame.evidence
    assert p.inspect(decoded, STUDY).startswith(b"\x89PNG")
    with pytest.raises(ValueError):
        p.decode(encoded)
    assert p.comparison(frame, fixture(ordinal + 2, 0), STUDY)["status"] == "PASS"


def test_new_retention_full_synthetic_matrix(tmp_path: Path) -> None:
    b = binding()
    output = tmp_path / b.output_id
    e.consume_attempt(output, b, decision(b))
    sink = e.ByteBoundedSink(output, b)
    opened: list[int] = []
    result = e.run_matrix(sink, lambda c, q: Capture(c, q, opened), time.monotonic() + 60)
    assert result["status"] == "CAPTURE_COMPLETE" and opened == list(range(8))
    retained = e.replay(output, b)
    assert retained["status"] == "COMPLETE"
    assert all(frame.study is STUDY for frame in retained["frames"].values())
    assert result["counts"] == {k: 16 for k in sink.counts}
    assert sum(v.stat().st_size for v in output.rglob("*") if v.is_file()) <= 18 * p.MIB


@pytest.mark.parametrize(
    "negative,close_fault,cleanup_fault", ((False, False, False), (True, True, True))
)
def test_host_assessment_and_retained_negative(
    tmp_path: Path, negative: bool, close_fault: bool, cleanup_fault: bool
) -> None:
    b = binding()
    group = tmp_path / "group"
    opened: list[int] = []

    class FakeCommands:
        used = 0

        def __init__(self) -> None:
            self.info: dict[str, Any] = {}

        def command(self, args: list[str], deadline: float, *, cleanup: bool = False) -> str:
            if args[0] == "create":
                wall = float(
                    next(
                        v.split("=", 1)[1]
                        for v in args
                        if v.startswith("EPS_PAIRED_APPEARANCE_WORK_DEADLINE_UNIX=")
                    )
                )
                self.info = container_info(b, group / b.output_id, wall)
                return "d" * 64
            if args[0] == "start":
                sink = e.ByteBoundedSink(group / b.output_id, b)
                result = e.run_matrix(
                    sink,
                    lambda c, q: Capture(c, q, opened, negative=negative, close_fault=close_fault),
                    time.monotonic() + 60,
                )
                assert result["apparatus_status"] == ("FAIL" if negative else "INCONCLUSIVE")
                return ""
            if args[0] == "inspect":
                return (
                    ("false 2 false" if close_fault else "false 0 false")
                    if HOST["POLL"] in args
                    else canonical_json_bytes(self.info).decode()
                )
            if args[0] == "rm":
                if cleanup_fault:
                    raise TimeoutError("synthetic owned cleanup failure")
                return "d" * 64
            raise AssertionError("unsupported synthetic command")

    result = HOST["host_run"](b, decision(b), group, FakeCommands())
    if negative:
        assert result["status"] == result["operational_status"] == "INCONCLUSIVE"
        assert result["apparatus_status"] == "FAIL"
        assert result["scientific_result"] == {
            "status": "FAIL",
            "reason": "exact regeneration mismatch",
        }
        assert opened == [0, 1]
    else:
        assert result["status"] == result["apparatus_status"] == "PASS"
        assert result["operational_status"] == "COMPLETE"
        assert opened == list(range(8))


def test_new_image_labels_and_environment_are_closed(tmp_path: Path) -> None:
    b = binding()
    wall = time.time() + 275
    output = tmp_path / b.output_id
    info = container_info(b, output, wall)
    HOST["inspect_confinement"](info, b, output, None, wall)
    for change in ("image", "version", "config-file", "asset-root"):
        bad = copy.deepcopy(info)
        if change == "image":
            bad["Image"] = "sha256:" + "0" * 64
        else:
            bad["Config"]["Labels"]["eps.appearance." + change] = "wrong"
        with pytest.raises(ValueError):
            HOST["inspect_confinement"](bad, b, output, None, wall)


def test_exact_ci_source_route() -> None:
    ci = (ROOT / ".github/workflows/ci.yml").read_text()
    assert ci.count("github.head_ref == 'codex/appearance-contrast-calibration-20261006')") == 4
    assert "github.head_ref != 'codex/appearance-contrast-calibration-20261006'" in ci
    assert "github.head_ref == 'codex/appearance-contrast-calibration-20261006' }}" in ci


def test_native_config_and_image_denial_precedes_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    b = binding()
    monkeypatch.setattr(r, "environment_binding", lambda: b)
    monkeypatch.setattr(r, "paired_appearance_candidate", lambda: True)
    for field in ("config_file", "config_root", "asset_root", "protection_root", "membership_root"):
        altered_preparation = b.preparation.model_copy(update={field: "0" * 64})
        altered = b.model_copy(update={"preparation": altered_preparation})
        monkeypatch.setattr(r, "environment_binding", lambda current=altered: current)
        with pytest.raises(PermissionError, match="config/assets/protection"):
            NativeCapture(
                ROOT,
                "single_occluder",
                STUDY.appearances[0],
                source_head="a" * 40,
                source_tree="b" * 40,
                progress=lambda *args: None,
                binding=altered,
            )
    monkeypatch.setattr(r, "environment_binding", lambda: b)
    altered = b.model_copy(update={"image": "sha256:" + "0" * 64})
    with pytest.raises(PermissionError, match="unsupported exact"):
        NativeCapture(
            ROOT,
            "single_occluder",
            STUDY.appearances[0],
            source_head="a" * 40,
            source_tree="b" * 40,
            progress=lambda *args: None,
            binding=altered,
        )
