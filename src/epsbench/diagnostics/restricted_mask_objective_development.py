"""Prospective exposed-public objective development; no operation runs on import."""

from __future__ import annotations

import base64
import math
import struct
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch import Tensor

from epsbench.diagnostics import restricted_mask_training as original
from epsbench.diagnostics.restricted_mask_readers import (
    SPECS,
    VERSION,
    Reader,
    candidate,
    finite,
    prediction,
    reduce,
)
from epsbench.diagnostics.visible_forecast_contract import _json, _keys
from epsbench.utils.canonical import canonical_json_bytes

Objective = Literal["control", "balanced"]
VERSION_DEVELOPMENT = "restricted-mask-public-objective-development-v1"
LOGIT_NAMES = tuple(
    f"logits-{step:03d}-{case}.json" for step in range(1, 201) for case in range(1, 5)
)
COMPARISON_NAMES = (*LOGIT_NAMES, "checkpoint.json")


def balanced_loss(logits: Tensor, labels: Tensor) -> Tensor:
    """Row-major means within present classes; value-sorted mean across classes."""
    finite(logits)
    if (
        logits.ndim != 3
        or tuple(logits.shape[1:]) != (32, 32)
        or labels.dtype != torch.int64
        or labels.device.type != "cpu"
        or tuple(labels.shape) != (32, 32)
        or bool((labels < 0).any())
        or bool((labels >= logits.shape[0]).any())
    ):
        raise ValueError("literal complete class labels required")
    maximum = finite(logits.max(dim=0).values)
    exponential = finite(torch.exp(finite(logits - maximum)))
    normalizer = finite(torch.log(reduce(exponential)))
    truth = logits.gather(0, labels.unsqueeze(0)).squeeze(0)
    pixels = finite(maximum + normalizer - truth).reshape(1024)
    flat = labels.reshape(1024)
    means = []
    for index in range(logits.shape[0]):
        selected = pixels[flat == index]  # Boolean selection preserves row-major order.
        if selected.numel():
            means.append(finite(reduce(selected, ordered=False) / float(selected.numel())))
    return finite(reduce(torch.stack(means)) / float(len(means)))


def balanced_update(
    reader: Reader, optimizer: original.Adam, examples: tuple[original.Example, ...]
) -> float:
    """Only the separately admitted four-example public balanced update."""
    if (
        len(examples) != 4
        or reader.arm != "E"
        or optimizer.learning_rate != 0.01
        or any(
            type(e) is not original.Example or e.access is not original.Supervision.PUBLIC_FIXTURE
            for e in examples
        )
    ):
        raise PermissionError("literal E public development supervision required")
    labels = tuple(e.labels() for e in examples)  # All validation precedes model graphs.
    reader.zero_grad(set_to_none=True)
    terms = []
    for example, target in zip(examples, labels, strict=True):
        logits = reader(example.projection, example.projection.access)
        if logits is None:
            raise ValueError("UNKNOWN cannot supply complete supervision")
        terms.append(balanced_loss(logits, target))
    mean = finite(reduce(torch.stack(terms), ordered=False) / 4.0)
    mean.backward()  # type: ignore[no-untyped-call]
    for _, parameter in reader.named_scalars():
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
    optimizer.update(reader)
    return float(mean.detach())


def _payload(value: Any, shape: tuple[int, ...]) -> bytes:
    _keys(value, {"shape", "bytes"})
    if (
        type(value["shape"]) is not list
        or any(type(v) is not int for v in value["shape"])
        or value["shape"] != list(shape)
        or type(value["bytes"]) is not str
    ):
        raise ValueError("exact binary64 shape required")
    raw = base64.b64decode(value["bytes"], validate=True)
    if len(raw) != math.prod(shape) * 8 or base64.b64encode(raw).decode() != value["bytes"]:
        raise ValueError("exact canonical binary64 length required")
    if any(not math.isfinite(v[0]) for v in struct.iter_unpack("<d", raw)):
        raise ValueError("finite numeric payload required")
    return raw


def _record(raw: bytes, kind: str) -> dict[str, Any]:
    value = _json(raw)
    if canonical_json_bytes(value) != raw or value.get("version") != VERSION + ":" + kind:
        raise ValueError("canonical original numeric record required")
    if value.get("arm") != "E":
        raise ValueError("E numeric record required")
    return value


def compare_control(
    historical: Mapping[str, bytes], current: Mapping[str, bytes]
) -> dict[str, Any]:
    """Pure retained-only comparison. Caller MUST first verify the frozen archive hash.

    Keys are exact arm-directory basenames, all 800 logits and checkpoint.json.
    Other retained artifacts may be present, but no extra logits are permitted.
    Metadata bindings may change; numeric shapes, ordering and finite LE64 bytes may not.
    """
    checked = 0
    name = "membership"
    try:
        for records in (historical, current):
            if (
                set(k for k in records if k.startswith("logits-")) != set(LOGIT_NAMES)
                or "checkpoint.json" not in records
            ):
                raise ValueError("complete exact 200-by-four logit membership required")
        for name in LOGIT_NAMES:
            case = int(name[-6])
            shape = (3 if case <= 2 else 4, 32, 32)
            old = _payload(_record(historical[name], "prediction")["logits"], shape)
            new = _payload(_record(current[name], "prediction")["logits"], shape)
            if old != new:
                raise ValueError("binary64 logit payload differs")
            checked += 1
        name = "checkpoint.json"
        checkpoints = tuple(
            _record(records[name], "checkpoint") for records in (historical, current)
        )
        for record in checkpoints:
            if (
                type(record.get("step")) is not int
                or record["step"] != 200
                or record.get("learning_rate") != 0.01
            ):
                raise ValueError("exact final checkpoint required")
        shapes: dict[str, tuple[int, ...]] = {}
        for layer, inputs, outputs, kernel in SPECS:
            shapes[layer + ".weight"] = (
                (outputs, inputs, kernel, kernel) if kernel else (outputs, inputs)
            )
            shapes[layer + ".bias"] = (outputs,)
        for record in checkpoints:
            _keys(record["weights"], set(shapes))
        for tensor_name, tensor_shape in shapes.items():
            name = "checkpoint.json:" + tensor_name
            old = _payload(checkpoints[0]["weights"][tensor_name], tensor_shape)
            new = _payload(checkpoints[1]["weights"][tensor_name], tensor_shape)
            if old != new:
                raise ValueError("binary64 final weight payload differs")
        return {"status": "MATCH", "logit_arrays": checked, "weight_tensors": len(shapes)}
    except Exception as exc:
        return {
            "status": "MISMATCH",
            "logit_arrays": checked,
            "first_mismatch": {"record": name, "type": type(exc).__name__, "message": str(exc)},
        }


def balanced_permitted(control: Mapping[str, Any], comparison: Mapping[str, Any]) -> bool:
    return (
        control.get("status") == "COMPLETE"
        and control.get("updates") == 200
        and control.get("operation_artifact_retained") is True
        and control.get("record_kind") == "EXTERNAL_CLOSURE_RECEIPT"
        and comparison == {"status": "MATCH", "logit_arrays": 800, "weight_tensors": 22}
    )


def pair_result(
    control: Mapping[str, Any], comparison: Mapping[str, Any], balanced: Mapping[str, Any] | None
) -> dict[str, Any]:
    """Descriptive pair summary; qualified external receipts remain caller obligations."""
    intact = balanced_permitted(control, comparison)
    complete = (
        intact
        and balanced is not None
        and balanced.get("status") == "COMPLETE"
        and balanced.get("updates") == 200
        and balanced.get("operation_artifact_retained") is True
        and balanced.get("record_kind") == "EXTERNAL_CLOSURE_RECEIPT"
    )
    return {
        "status": "COMPLETE" if complete else "INCONCLUSIVE",
        "balanced": "COMPLETE"
        if complete
        else "NOT_ATTEMPTED"
        if balanced is None
        else "INCONCLUSIVE",
        "development_discriminator": bool(
            complete
            and balanced is not None
            and balanced.get("local_criterion") == "ZERO_ERROR"
            and control.get("local_criterion") == "NONZERO_ERROR"
        )
        if complete
        else None,
    }


def fixed_trajectory(retain_step: Callable[[int], bool]) -> int | None:
    """Complete all 200 retained steps; zero error never stops this trajectory."""
    earliest = None
    for index in range(1, 201):
        zero = retain_step(index)
        if type(zero) is not bool:
            raise ValueError("complete step's literal neutral-zero observation required")
        if zero and earliest is None:
            earliest = index
    return earliest


def public_development(
    objective: Objective,
    source_head: str,
    output: Path,
    *,
    control: Mapping[str, Any] | None = None,
    comparison: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """One later-admitted E fit. No launch authority; all real entries denied in CI.

    Admitted wrapper verifies source/environment/archive, persists external closure
    and inclusive accounting, and calls balanced only after intact matching control.
    """
    if objective not in ("control", "balanced"):
        raise ValueError("literal public objective required")
    if objective == "balanced" and (
        control is None or comparison is None or not balanced_permitted(control, comparison)
    ):
        raise PermissionError("intact exactly matching control required before balanced")
    from epsbench.diagnostics.restricted_mask_fixtures import fixtures, public_key

    sink = original._Sink(output)
    start = time.monotonic()
    result: dict[str, Any] = {
        "version": VERSION_DEVELOPMENT,
        "objective": objective,
        "status": "INCONCLUSIVE",
        "local_criterion": "UNRESOLVED",
        "updates": 0,
        "earliest_zero": None,
    }

    def check() -> None:
        result["peak_working_set"] = max(result.get("peak_working_set", 0), original.memory_check())
        if time.monotonic() - start > 60:
            raise TimeoutError("public development wall deadline")

    try:
        cases = fixtures(source_head)
        reader = Reader("E", public_key())
        optimizer = original.Adam(reader, 0.01)
        examples = []
        retained = []
        for case in cases:
            snapshot = case.fetch(case.projection)
            retained.append(original._Retained(snapshot))
            source = case.projection.revalidate()
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
            examples.append(
                original.Example(
                    case.projection,
                    case.projection.save(known, new),
                    original.Supervision.PUBLIC_FIXTURE,
                )
            )
            sink.write(
                f"fixture-{case.number}.json",
                canonical_json_bytes(
                    {
                        "binding": _json(case.projection.binding_bytes()),
                        "features": _json(case.projection.feature_bytes()),
                        "prefix": _json(source.canonical_bytes()),
                        "state": _json(case.projection._state),
                        "truth": _json(examples[-1].target),
                    }
                ),
            )
        sink.write("initial.json", original.checkpoint(reader, optimizer))
        updater = original.update if objective == "control" else balanced_update

        def retain_step(index: int) -> bool:
            check()
            value = updater(reader, optimizer, tuple(examples))
            result["updates"] = index
            reports = []
            for case, cached, example in zip(cases, retained, examples, strict=True):
                check()
                saved = prediction(reader, case.projection, case.projection.access)
                sink.write(f"logits-{index:03d}-{case.number}.json", saved)
                masks = candidate(saved, case.projection, case.projection.access)
                report = case.projection.evaluate(masks, cached).report
                labels = example.labels().numpy()
                payload = _json(saved)["logits"]
                a = np.frombuffer(base64.b64decode(payload["bytes"]), dtype="<f8").reshape(
                    payload["shape"]
                )
                maxima = a.max(axis=0)
                winners = a.argmax(axis=0)
                winners[np.count_nonzero(a == maxima, axis=0) != 1] = a.shape[0] - 1
                n = a.shape[0] - 2
                channels = [
                    {
                        "channel": row["channel"],
                        "target_pixels": row["target_pixels"],
                        "predicted_pixels": int(np.count_nonzero(winners == j)),
                        "error": row["error"],
                    }
                    for j, row in enumerate(report["channels"])
                ]
                channels.append(
                    {
                        "channel": "BACKGROUND",
                        "target_pixels": int(np.count_nonzero(labels == n + 1)),
                        "predicted_pixels": int(np.count_nonzero(winners == n + 1)),
                    }
                )
                reports.append(
                    {
                        "case": case.number,
                        **{k: report[k] for k in ("N", "C", "U", "E")},
                        "known_errors": sum(row["error"] for row in channels[:n]),
                        "new_false_positive": int(np.count_nonzero((winners == n) & (labels != n))),
                        "new_false_negative": int(np.count_nonzero((winners != n) & (labels == n))),
                        "channels": channels,
                    }
                )
            zero = all(report["E"] == 0 and report["U"] == 0 for report in reports)
            sink.write(
                f"report-{index:03d}.json",
                canonical_json_bytes({"step": index, "loss": value, "cases": reports}),
            )
            result["local_criterion"] = "ZERO_ERROR" if zero else "NONZERO_ERROR"
            return zero

        result["earliest_zero"] = fixed_trajectory(retain_step)
        sink.write("checkpoint.json", original.checkpoint(reader, optimizer))
        check()
        result["status"] = "COMPLETE"
    except Exception as exc:
        result["first_failure"] = {"type": type(exc).__name__, "message": str(exc)}
        result["status"] = "INCONCLUSIVE"
    closed = original.close_readiness(
        result, lambda raw: sink.write("terminal.json", raw, terminal=True), start
    )
    closed["retained_bytes"] = sink.count
    return closed


# Existing reviewed outer measurement hook for the separately admitted wrapper.
memory_check = original.memory_check
