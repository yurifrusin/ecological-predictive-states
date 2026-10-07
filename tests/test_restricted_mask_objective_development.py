"""Handwritten objective arithmetic and fake retained records; never fits."""

import base64
import json
import math
import struct
from pathlib import Path
from typing import Any, cast

import pytest
import torch

from epsbench.diagnostics import restricted_mask_objective_development as d
from epsbench.diagnostics import restricted_mask_training as original
from epsbench.diagnostics.restricted_mask_readers import (
    SPECS,
    VERSION,
    ArithmeticFailure,
    Reader,
    configure,
    tensor,
)
from epsbench.utils.canonical import canonical_json_bytes


def test_equal_present_mean_and_background_gradient() -> None:
    configure()
    logits = torch.zeros((3, 32, 32), dtype=torch.float64, requires_grad=True)
    labels = torch.full((32, 32), 2, dtype=torch.int64)
    labels[0, 0] = 0
    value = d.balanced_loss(logits, labels)
    background_sum = 0.0
    for _ in range(1023):
        background_sum += math.log(3)
    assert float(value.detach()) == (math.log(3) + background_sum / 1023) / 2
    value.backward()  # type: ignore[no-untyped-call]
    assert logits.grad is not None
    # Present rare class and background each have half of the example's mass.
    assert float(logits.grad[0, 0, 0]) == pytest.approx(-1 / 3)
    assert float(logits.grad[1, 0, 0]) == pytest.approx(1 / 6)  # Absent NEW still competes.
    assert float(logits.grad[2, 1, 1]) == pytest.approx(-1 / (3 * 1023))
    assert float(logits.grad[1, 1, 1]) > 0


def test_scalar_class_means_not_pixel_mean() -> None:
    configure()
    logits = torch.zeros((3, 32, 32), dtype=torch.float64)
    logits[0, 0, 0] = 1
    labels = torch.full((32, 32), 2, dtype=torch.int64)
    labels[0, 0] = 0
    rare = math.log(math.e + 2) - 1
    background_sum = 0.0
    for _ in range(1023):
        background_sum += math.log(3)
    expected = (rare + background_sum / 1023) / 2
    assert float(d.balanced_loss(logits, labels)) == pytest.approx(expected, abs=1e-14)
    assert float(d.balanced_loss(logits, labels)) != float(original.loss(logits, labels))


def test_renaming_and_single_present_class() -> None:
    configure()
    logits = tensor([1.0, -1.0, 0.5]).reshape(3, 1, 1).expand(3, 32, 32).clone()
    labels = torch.zeros((32, 32), dtype=torch.int64)
    labels[0, :] = 1
    renamed = labels.clone()
    renamed[labels == 0] = 1
    renamed[labels == 1] = 0
    assert float(d.balanced_loss(logits, labels)) == float(
        d.balanced_loss(logits[[1, 0, 2]], renamed)
    )
    labels.fill_(2)
    assert float(d.balanced_loss(logits, labels)) == float(original.loss(logits, labels))


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite(bad: float) -> None:
    with pytest.raises(ArithmeticFailure):
        d.balanced_loss(
            torch.full((3, 32, 32), bad, dtype=torch.float64),
            torch.zeros((32, 32), dtype=torch.int64),
        )


def test_invalid_labels() -> None:
    with pytest.raises(ValueError):
        d.balanced_loss(
            torch.zeros((3, 32, 32), dtype=torch.float64),
            torch.full((32, 32), 3, dtype=torch.int64),
        )


def payload(shape: tuple[int, ...], value: float = 0.0) -> dict[str, Any]:
    return {
        "shape": list(shape),
        "bytes": base64.b64encode(struct.pack("<d", value) * math.prod(shape)).decode(),
    }


@pytest.fixture(scope="module")
def records() -> dict[str, bytes]:
    predictions = {
        n: canonical_json_bytes(
            {"version": VERSION + ":prediction", "arm": "E", "logits": payload((n, 32, 32))}
        )
        for n in (3, 4)
    }
    records = {name: predictions[3 if int(name[-6]) <= 2 else 4] for name in d.LOGIT_NAMES}
    weights = {}
    for name, inputs, outputs, kernel in SPECS:
        weights[name + ".weight"] = payload(
            (outputs, inputs, kernel, kernel) if kernel else (outputs, inputs)
        )
        weights[name + ".bias"] = payload((outputs,))
    records["checkpoint.json"] = canonical_json_bytes(
        {
            "version": VERSION + ":checkpoint",
            "arm": "E",
            "step": 200,
            "learning_rate": 0.01,
            "weights": weights,
        }
    )
    return records


def test_complete_exact_numeric_match_metadata_may_differ(records: dict[str, bytes]) -> None:
    current = dict(records)
    record = json.loads(current[d.LOGIT_NAMES[0]])
    record["binding_sha256"] = "new-source-metadata"
    current[d.LOGIT_NAMES[0]] = canonical_json_bytes(record)
    assert d.compare_control(records, current) == {
        "status": "MATCH",
        "logit_arrays": 800,
        "weight_tensors": 22,
    }


@pytest.mark.parametrize(
    "fault", ["missing", "extra", "shape", "nan", "signed_zero", "final_weights", "step"]
)
def test_first_mismatch_and_complete_membership(records: dict[str, bytes], fault: str) -> None:
    current = dict(records)
    name = d.LOGIT_NAMES[0]
    if fault == "missing":
        del current[name]
    elif fault == "extra":
        current["logits-201-1.json"] = current[name]
    elif fault in ("shape", "nan", "signed_zero"):
        value = json.loads(current[name])
        value["logits"] = payload(
            (4 if fault == "shape" else 3, 32, 32), float("nan") if fault == "nan" else -0.0
        )
        current[name] = canonical_json_bytes(value)
    else:
        value = json.loads(current["checkpoint.json"])
        if fault == "step":
            value["step"] = 199
        else:
            value["weights"]["global.fc2.bias"] = payload((2,), 1.0)
        current["checkpoint.json"] = canonical_json_bytes(value)
    result = d.compare_control(records, current)
    assert result["status"] == "MISMATCH"
    assert result["first_mismatch"]["record"]
    assert result["logit_arrays"] == (800 if fault in ("final_weights", "step") else 0)


def receipt(criterion: str = "NONZERO_ERROR") -> dict[str, Any]:
    return {
        "status": "COMPLETE",
        "updates": 200,
        "operation_artifact_retained": True,
        "record_kind": "EXTERNAL_CLOSURE_RECEIPT",
        "local_criterion": criterion,
    }


def test_control_failure_stops_balanced_and_closure_overrides_local() -> None:
    matched = {"status": "MATCH", "logit_arrays": 800, "weight_tensors": 22}
    control = receipt()
    assert d.balanced_permitted(control, matched)  # Nonzero original remains eligible.
    assert (
        d.pair_result(control, matched, receipt("ZERO_ERROR"))["development_discriminator"] is True
    )
    for change in (
        {"status": "INCONCLUSIVE"},
        {"updates": 199},
        {"operation_artifact_retained": False},
    ):
        failed = control | change
        assert not d.balanced_permitted(failed, matched)
        assert d.pair_result(failed, matched, None)["balanced"] == "NOT_ATTEMPTED"
    assert not d.balanced_permitted(control, {"status": "MISMATCH"})
    assert (
        d.pair_result(control, matched, receipt("ZERO_ERROR") | {"status": "INCONCLUSIVE"})[
            "status"
        ]
        == "INCONCLUSIVE"
    )
    assert (
        d.pair_result(receipt("ZERO_ERROR"), matched, receipt("ZERO_ERROR"))[
            "development_discriminator"
        ]
        is False
    )
    assert d.pair_result(control, matched, receipt())["development_discriminator"] is False


def test_late_closure_failure_keeps_descriptive_result() -> None:
    result = original.close_readiness(
        receipt("ZERO_ERROR"), lambda _: None, 0, clock=lambda: 61, measure=lambda: 1
    )
    assert result["status"] == "INCONCLUSIVE"
    assert result["local_criterion"] == "ZERO_ERROR"
    assert not d.balanced_permitted(
        result, {"status": "MATCH", "logit_arrays": 800, "weight_tensors": 22}
    )


def test_source_guard_denies_real_entries() -> None:
    with pytest.raises(RuntimeError, match="fit denied"):
        d.public_development("control", "0" * 40, Path("must-not-exist"))
    with pytest.raises(RuntimeError, match="fit denied"):
        Reader("E", bytes(32))
    with pytest.raises(RuntimeError, match="fit denied"):
        original.Adam.update(cast(original.Adam, None), cast(Reader, None))
    with pytest.raises(RuntimeError, match="fit denied"):
        d.balanced_update(cast(Reader, None), cast(original.Adam, None), ())


def test_fixed_trajectory_never_stops_at_early_zero() -> None:
    calls = []

    def retained(index: int) -> bool:
        calls.append(index)
        return index in (1, 200)

    assert d.fixed_trajectory(retained) == 1
    assert calls == list(range(1, 201))
    calls.clear()

    def failure(index: int) -> bool:
        calls.append(index)
        if index == 197:
            raise OSError("fake late retention failure")
        return True

    with pytest.raises(OSError, match="fake late retention"):
        d.fixed_trajectory(failure)
    assert calls == list(range(1, 198))


def test_public_literal_smoke_validation_inspection() -> None:
    from epsbench.diagnostics.restricted_mask_fixtures import fixtures

    cases = fixtures("0" * 40)
    assert len(cases) == 4
    for case in cases:
        source = case.projection.revalidate()
        snapshot = case.fetch(case.projection)
        lookup = dict(snapshot.identities)
        known = tuple(
            (
                token,
                snapshot.segmentation
                == next(label for label, name in lookup.items() if name == token),
            )
            for token in source.inventory
        )
        new = snapshot.segmentation == len(source.inventory) + 1
        saved = case.projection.save(known, new)
        example = original.Example(case.projection, saved, original.Supervision.PUBLIC_FIXTURE)
        labels = example.labels()
        assert labels.shape == (32, 32)
        report = case.projection.evaluate(saved, original._Retained(snapshot)).report
        assert report["E"] == report["U"] == 0
        assert report["C"] == report["N"] == (len(source.inventory) + 1) * 1024
        assert report["first_observed_pixels"] == 4
        with pytest.raises(PermissionError):
            case.fetch(case.projection)


def test_guard_importlib_actual_import_denied() -> None:
    import importlib
    import importlib.util

    assert importlib.util.find_spec("transformers") is None
    with pytest.raises(RuntimeError, match="forbidden"):
        importlib.import_module("transformers")
