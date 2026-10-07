"""Frozen shared trainer and non-executable-by-default checkpoint policy."""

from __future__ import annotations

import struct
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor

from epsbench.diagnostics.restricted_learning_contract import (
    ARCHITECTURE,
    CONDITIONS,
    Fit,
    digest,
    sha,
)
from epsbench.diagnostics.restricted_learning_membership import INIT_DOMAIN
from epsbench.diagnostics.restricted_learning_sampling import commitment
from epsbench.diagnostics.restricted_model_export import Example, TrainingMaterial
from epsbench.diagnostics.restricted_model_resources import (
    SCALAR_CONVENTION,
    WorkTrace,
    observed_peak_bytes,
    provenance,
    scalar_work,
)
from epsbench.diagnostics.restricted_models import RestrictedModel
from epsbench.diagnostics.visible_forecast_contract import _json, _keys
from epsbench.utils.canonical import canonical_json_bytes


@dataclass(frozen=True)
class ModelSource:
    head: str
    tree: str
    launch: str

    def __post_init__(self) -> None:
        sha(self.launch)
        if any(
            type(v) is not str or len(v) != 40 or any(c not in "0123456789abcdef" for c in v)
            for v in (self.head, self.tree)
        ):
            raise ValueError("exact separately accepted model source required")


def prediction(model: RestrictedModel, example: Example) -> Tensor:
    if type(example) is not Example:
        raise ValueError("typed sanitized example required")
    state = model.initial_state()
    for observation in example.observations:
        state = model.advance(state, observation)
    return model.readout(state, example.announced)


def brier(model: RestrictedModel, example: Example) -> Tensor:
    probabilities = prediction(model, example)
    strata = []
    for visible in (0, 1):
        values = [
            (probabilities[i] - float(y)) ** 2
            for i, y in enumerate(example.truth)
            if example.observations[-1].visible[i] == visible
        ]
        if values:
            total = values[0]
            for value in values[1:]:
                total = total + value
            strata.append(total / len(values))
    if not strata:
        raise ValueError("nonempty observed inventory required for learning loss")
    result = strata[0]
    for value in strata[1:]:
        result = result + value
    return result / len(strata)


class Adam:
    def __init__(self, model: RestrictedModel) -> None:
        self.parameters = tuple(model.parameters())
        self.m = tuple(torch.zeros_like(p) for p in self.parameters)
        self.v = tuple(torch.zeros_like(p) for p in self.parameters)
        self.k = 0

    def step(self) -> None:
        self.k += 1
        with torch.no_grad():
            for p, m, v in zip(self.parameters, self.m, self.v, strict=True):
                if p.grad is None or not torch.isfinite(p.grad).all():
                    raise ValueError("connected finite gradients required; zero is valid")
                m.copy_(0.9 * m + 0.1 * p.grad)
                v.copy_(0.999 * v + 0.001 * (p.grad * p.grad))
                mhat, vhat = m / (1 - 0.9**self.k), v / (1 - 0.999**self.k)
                p.sub_(0.001 * mhat / (torch.sqrt(vhat) + 1e-8))
                if not torch.isfinite(p).all():
                    raise ValueError("nonfinite optimizer weight")
                p.grad = None


def update(
    model: RestrictedModel, adam: Adam, batch: tuple[Example, ...], trace: WorkTrace
) -> float:
    if type(batch) is not tuple or len(batch) != 16:
        raise ValueError("frozen batch16 required")
    trace.phase = "forward"
    losses = []
    for example in batch:
        # Both prediction and loss operators are charged separately.
        probabilities = prediction(model, example)
        trace.phase = "loss"
        strata = []
        for visible in (0, 1):
            values = [
                (probabilities[i] - float(y)) ** 2
                for i, y in enumerate(example.truth)
                if example.observations[-1].visible[i] == visible
            ]
            if values:
                total = values[0]
                for value in values[1:]:
                    total = total + value
                strata.append(total / len(values))
        if not strata:
            raise ValueError("empty training inventory")
        total = strata[0]
        for value in strata[1:]:
            total = total + value
        losses.append(total / len(strata))
        trace.phase = "forward"
    trace.phase = "loss"
    total = losses[0]
    for value in losses[1:]:
        total = total + value
    loss = total / 16
    if not torch.isfinite(loss):
        raise ValueError("nonfinite loss")
    trace.phase = "backward"
    loss.backward()  # type: ignore[no-untyped-call]
    trace.phase = "adam"
    adam.step()
    return float(loss.detach())


def checkpoint(
    model: RestrictedModel, fit: Fit, source: ModelSource, material: TrainingMaterial, updates: int
) -> bytes:
    if updates != 1000 or fit.condition != model.condition or fit.budget != material.budget:
        raise PermissionError("only frozen final checkpoint after1000 updates")
    header = {
        "architecture": ARCHITECTURE,
        "fit": fit.key,
        "source": source.__dict__,
        "data": material.data_root,
        "schedule": material.schedule_root,
        "seal_a": material.seal_a,
        "updates": updates,
        "initialization": commitment(
            INIT_DOMAIN,
            material.initialization_values[int(fit.initialization[-1])].to_bytes(8, "little"),
        ),
        "labels": material.labels_root,
        "order": digest(canonical_json_bytes(list(material.order))),
        "tensors": [[n, list(p.shape)] for n, p in model.weights.items()],
    }
    raw = canonical_json_bytes(header)
    arrays = []
    for p in model.weights.values():
        if p.dtype != torch.float32 or p.device.type != "cpu" or not torch.isfinite(p).all():
            raise ValueError("finite float32 checkpoint only")
        arrays.append(p.detach().numpy().astype("<f4").tobytes())
    return b"EPSMODEL1" + struct.pack("<Q", len(raw)) + raw + b"".join(arrays)


def validate_checkpoint(data: bytes, expected: dict[str, Any], condition: str) -> None:
    if type(data) is not bytes or len(data) > 1024**2 or len(data) < 17 or data[:9] != b"EPSMODEL1":
        raise ValueError("bounded explicit binary checkpoint required")
    length = struct.unpack("<Q", data[9:17])[0]
    raw = data[17 : 17 + length]
    header = _json(raw)
    _keys(
        header,
        {
            "architecture",
            "fit",
            "source",
            "data",
            "schedule",
            "seal_a",
            "updates",
            "initialization",
            "labels",
            "order",
            "tensors",
        },
    )
    if (
        header != expected
        or raw != canonical_json_bytes(header)
        or header["architecture"] != ARCHITECTURE
        or header["updates"] != 1000
        or header["tensors"] != RestrictedModel.tensor_layout(condition)
    ):
        raise ValueError("complete canonical checkpoint binding/layout differs")
    count = 99984 if condition == "dense" else 99913
    payload = data[17 + length :]
    if len(payload) != count * 4 or not np.isfinite(np.frombuffer(payload, dtype="<f4")).all():
        raise ValueError("exact complete finite float32 checkpoint payload required")


def restore(data: bytes, model: RestrictedModel, expected: dict[str, Any]) -> None:
    validate_checkpoint(data, expected, model.condition)
    if type(data) is not bytes or len(data) > 1024**2 or data[:9] != b"EPSMODEL1" or len(data) < 17:
        raise ValueError("bounded explicit binary checkpoint required; no pickle")
    length = struct.unpack("<Q", data[9:17])[0]
    header = _json(data[17 : 17 + length])
    _keys(
        header,
        {
            "architecture",
            "fit",
            "source",
            "data",
            "schedule",
            "seal_a",
            "updates",
            "initialization",
            "labels",
            "order",
            "tensors",
        },
    )
    if (
        header != expected
        or header["architecture"] != ARCHITECTURE
        or header["updates"] != 1000
        or header["tensors"] != [[n, list(p.shape)] for n, p in model.weights.items()]
    ):
        raise ValueError("checkpoint exact source/data/fit/schema binding differs")
    offset = 17 + length
    decoded = []
    with torch.no_grad():
        for p in model.weights.values():
            count = p.numel() * 4
            raw = data[offset : offset + count]
            if len(raw) != count:
                raise ValueError("truncated checkpoint")
            tensor = torch.from_numpy(
                np.frombuffer(raw, dtype="<f4").copy().reshape(tuple(p.shape))
            )
            if not torch.isfinite(tensor).all():
                raise ValueError("nonfinite checkpoint")
            decoded.append((p, tensor))
            offset += count
    if offset != len(data):
        raise ValueError("extra checkpoint payload")
    if canonical_json_bytes(header) != data[17 : 17 + length]:
        raise ValueError("canonical checkpoint header required")
    with torch.no_grad():
        for p, tensor in decoded:
            p.copy_(tensor)


def _fit_setup(
    condition: str, seed: int
) -> tuple[RestrictedModel, Adam, WorkTrace, dict[str, Any], float, float]:
    started_cpu, started_wall = time.process_time(), time.monotonic()
    model = RestrictedModel(condition, seed)
    adam = Adam(model)
    return model, adam, WorkTrace(), provenance(), started_cpu, started_wall


def train(
    material: TrainingMaterial, fit: Fit, source: ModelSource
) -> tuple[bytes, dict[str, Any]]:
    """Explicit later-admitted launch only; source tests never invoke this1000-update fit."""
    if (
        type(fit) is not Fit
        or type(material) is not TrainingMaterial
        or type(source) is not ModelSource
        or fit.condition not in CONDITIONS
        or fit.budget != material.budget
    ):
        raise PermissionError("typed committed material and explicit separate launch required")
    if torch.get_num_threads() != 1 or not torch.are_deterministic_algorithms_enabled():
        raise PermissionError("recorded deterministic single-thread CPU runtime required")
    seed = material.initialization_values[int(fit.initialization[-1])]
    model, adam, trace, runtime, started_cpu, started_wall = _fit_setup(fit.condition, seed)
    saved_bytes = 0
    batch_saved_bytes = 0
    peak_saved_upperbound = 0

    def pack(tensor: Tensor) -> Tensor:
        nonlocal saved_bytes, batch_saved_bytes
        saved_bytes += tensor.numel() * tensor.element_size()
        batch_saved_bytes += tensor.numel() * tensor.element_size()
        return tensor

    with trace, torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        for k in range(1000):
            if (
                time.process_time() - started_cpu > 2 * 3600
                or time.monotonic() - started_wall > 2 * 3600
            ):
                raise RuntimeError(
                    "cooperative fit planning deadline; partial work must be retained by controller"
                )
            batch = tuple(material.examples[i] for i in material.order[k * 16 : (k + 1) * 16])
            batch_saved_bytes = 0
            update(model, adam, batch, trace)
            peak_saved_upperbound = max(peak_saved_upperbound, batch_saved_bytes)
            peak = observed_peak_bytes()
            if peak is not None and peak > 4 * 1024**3:
                raise RuntimeError("observed process memory planning limit")
    payload = checkpoint(model, fit, source, material, adam.k)
    report = {
        "fit": fit.key,
        "model_source": source.__dict__,
        "initialization": commitment(INIT_DOMAIN, seed.to_bytes(8, "little")),
        "labels": material.labels_root,
        "order": digest(canonical_json_bytes(list(material.order))),
        **scalar_work(fit.condition, len(material.order), adam.k),
        "python_scalar_scope": SCALAR_CONVENTION,
        "optimizer": {
            "name": "Adam",
            "lr": 0.001,
            "beta1": 0.9,
            "beta2": 0.999,
            "epsilon": 1e-8,
            "weight_decay": 0,
            "clipping": False,
            "amsgrad": False,
            "dtype": "float32",
        },
        "trace": trace.payload(),
        "used_parameters": sum(p.numel() for p in model.parameters()),
        "gradient_connected": True,
        "state_floats": 384,
        "recurrent_bytes": 1536,
        "parameter_bytes": sum(p.numel() * 4 for p in model.parameters()),
        "gradient_bytes": sum(p.numel() * 4 for p in model.parameters()),
        "adam_moment_bytes": sum(p.numel() * 8 for p in model.parameters()),
        "saved_activation_bytes_cumulative_not_peak": saved_bytes,
        "saved_activation_peak_conservative_bytes": peak_saved_upperbound,
        "buffers": {
            "current_mask_bytes": 12288,
            "shared_features_bytes": 192,
            "current_boundary_bytes": 48,
            "flags_bytes": 24,
            "commands_bytes": 24,
            "candidate_context_bytes": 4240 if fit.condition != "dense" else 0,
            "prediction_bytes": 12,
            "batch_examples": 16,
            "schedule_index_bytes": len(material.order) * 8,
            "lawful_material_mask_bytes": sum(
                len(m)
                for e in (*material.examples, *material.development)
                for o in e.observations
                for m in o.masks
            ),
        },
        "observed_mask_encodings": sum(
            sum(sum(o.present) for o in material.examples[i].observations) for i in material.order
        ),
        "trainer_batch_examples": 16,
        "prediction_values": 3,
        "shared_encoder": True,
        "streams": 6,
        "schedule": material.schedule_root,
        "data": material.data_root,
        "updates": adam.k,
        "batch": 16,
        "checkpoint": digest(payload),
        "provenance": runtime,
        "memory_peak": observed_peak_bytes(),
        "memory_complete": platform_memory_complete(),
        "elapsed_cpu": time.process_time() - started_cpu,
        "elapsed_wall": time.monotonic() - started_wall,
    }
    return payload, report


def platform_memory_complete() -> bool:
    # Windows PeakWorkingSetSize is process-lifetime peak, including construction and tracing.
    return observed_peak_bytes() is not None


def cheap_control(
    condition: str,
    example: Example,
    frequencies: dict[tuple[tuple[float, float, float], int], float] | None = None,
) -> tuple[float, ...]:
    if condition not in ("persistence", "absent", "visible", "half", "train-frequency"):
        raise ValueError("fixed cheap control required")
    if condition == "train-frequency" and frequencies is None:
        raise PermissionError("train-only fitted frequency table required")
    return tuple(
        float(example.observations[-1].visible[i])
        if condition == "persistence"
        else 0.0
        if condition == "absent"
        else 1.0
        if condition == "visible"
        else 0.5
        if condition == "half"
        else (frequencies or {}).get((example.announced, example.observations[-1].visible[i]), 0.5)
        for i in range(len(example.truth))
    )


def train_frequencies(
    examples: tuple[Example, ...],
) -> dict[tuple[tuple[float, float, float], int], float]:
    counts: dict[tuple[tuple[float, float, float], int], list[int]] = {}
    for example in examples:
        for i, truth in enumerate(example.truth):
            key = (example.announced, example.observations[-1].visible[i])
            row = counts.setdefault(key, [0, 0])
            row[0] += int(truth)
            row[1] += 1
    return {key: (row[0] + 1) / (row[1] + 2) for key, row in counts.items()}
