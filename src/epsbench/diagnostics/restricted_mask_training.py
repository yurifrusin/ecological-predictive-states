"""Common bounded trainer and public one-shot entry; never fits on import or in CI."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from epsbench.diagnostics.neutral_observation_target import Candidate
from epsbench.diagnostics.restricted_mask_projection import Projection
from epsbench.diagnostics.restricted_mask_readers import (
    VERSION,
    ArithmeticFailure,
    Arm,
    Reader,
    candidate,
    finite,
    pack,
    prediction,
    reduce,
    restore,
    unpack,
    validate,
    weights,
)
from epsbench.diagnostics.visible_forecast_contract import _json, _keys, integer
from epsbench.utils.canonical import canonical_json_bytes, sha256_bytes


class Supervision(Enum):
    TRAINING = "training"
    PUBLIC_FIXTURE = "public-fixture"


@dataclass(frozen=True)
class Example:
    """Explicit training-authorized supervision, never a target provider callback.

    The caller's qualifier must attest training split access. This type does not
    establish external provenance or admit any empirical operation by itself.
    """

    projection: Projection
    target: bytes  # Projection-wrapped canonical complete known/NEW masks.
    access: Supervision

    def __post_init__(self) -> None:
        if type(self.access) is not Supervision or type(self.target) is not bytes:
            raise PermissionError("typed supervised access and immutable target bytes required")

    def labels(self) -> Tensor:
        if type(self.access) is not Supervision:
            raise PermissionError("typed supervised access required")
        source = self.projection.revalidate()
        validate(self.projection, self.projection.access)
        p = _json(self.target)
        _keys(p, {"version", "binding_sha256", "candidate"})
        if (
            p["version"] != "restricted-mask-projection-v1:candidate"
            or p["binding_sha256"] != sha256_bytes(self.projection.binding_bytes())
            or self.target != canonical_json_bytes(p)
        ):
            raise ValueError("training truth binding/canonical bytes differ")
        raw = canonical_json_bytes(p["candidate"])
        target = Candidate.from_bytes(raw, source)
        if raw != target.canonical_bytes():
            raise ValueError("canonical training truth required")
        channels = (*target.channels.known, ("NEW", target.channels.new))
        if any(mask is None for _, mask in channels):
            raise ValueError("complete supervision required")
        labels = np.full((32, 32), len(channels), dtype=np.int64)
        for i, (_, mask) in enumerate(channels):
            assert mask is not None
            labels[mask] = i
        return torch.from_numpy(labels)


def loss(logits: Tensor, labels: Tensor) -> Tensor:
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
    pixels = finite(maximum + normalizer - truth)
    return finite(reduce(pixels.reshape(1024), ordered=False) / 1024.0)


class Adam:
    """Literal unfused binary64 Adam with staged all-or-nothing finite updates."""

    def __init__(self, reader: Reader, learning_rate: float = 0.001) -> None:
        if learning_rate not in (0.001, 0.01):
            raise ValueError("only declared comparative/readiness learning rates")
        self.learning_rate = learning_rate
        self.step = 0
        self.m = {name: torch.zeros_like(p) for name, p in reader.named_scalars()}
        self.v = {name: torch.zeros_like(p) for name, p in reader.named_scalars()}

    def update(self, reader: Reader) -> None:
        t = self.step + 1
        staged = []
        for name, p in reader.named_scalars():
            finite(p)
            if p.grad is None:
                raise ValueError("every registered scalar requires gradient")
            g = finite(p.grad)
            m = finite(0.9 * finite(self.m[name]) + (1.0 - 0.9) * g)
            v = finite(0.999 * finite(self.v[name]) + (1.0 - 0.999) * g * g)
            mh = finite(m / (1.0 - 0.9**t))
            vh = finite(v / (1.0 - 0.999**t))
            replacement = finite(p - self.learning_rate * mh / (torch.sqrt(vh) + 1e-8))
            staged.append((name, p, m.detach(), v.detach(), replacement.detach()))
        with torch.no_grad():
            for name, p, m, v, replacement in staged:
                p.copy_(replacement)
                self.m[name], self.v[name] = m, v
        self.step = t


def checkpoint(reader: Reader, optimizer: Adam) -> bytes:
    return canonical_json_bytes(
        {
            "version": VERSION + ":checkpoint",
            "arm": reader.arm,
            "step": optimizer.step,
            "learning_rate": optimizer.learning_rate,
            "weights": json.loads(weights(reader)),
            "m": {name: pack(p) for name, p in optimizer.m.items()},
            "v": {name: pack(p) for name, p in optimizer.v.items()},
        }
    )


def load_checkpoint(saved: bytes, reader: Reader) -> Adam:
    p = _json(saved)
    _keys(p, {"version", "arm", "step", "learning_rate", "weights", "m", "v"})
    integer(p["step"])
    if (
        p["version"] != VERSION + ":checkpoint"
        or p["arm"] != reader.arm
        or p["step"] > 400
        or canonical_json_bytes(p) != saved
    ):
        raise ValueError("checkpoint contract differs")
    opt = Adam(reader, p["learning_rate"])
    names = {name for name, _ in reader.named_scalars()}
    _keys(p["m"], names)
    _keys(p["v"], names)
    m = {name: unpack(p["m"][name], tuple(value.shape)) for name, value in reader.named_scalars()}
    v = {name: unpack(p["v"][name], tuple(value.shape)) for name, value in reader.named_scalars()}
    if any(bool((value < 0).any()) for value in v.values()):
        raise ValueError("negative Adam second moment")
    if p["step"] == 0 and any(bool((value != 0).any()) for value in (*m.values(), *v.values())):
        raise ValueError("initial moments must be zero")
    restore(reader, canonical_json_bytes(p["weights"]))  # All moment checks precede mutation.
    opt.m, opt.v, opt.step = m, v, p["step"]
    return opt


def update(reader: Reader, optimizer: Adam, examples: tuple[Example, ...]) -> float:
    if len(examples) not in (4, 8):
        raise ValueError("declared committed minibatch size required")
    # Complete validation before any model graph construction.
    required = (
        Supervision.PUBLIC_FIXTURE if optimizer.learning_rate == 0.01 else Supervision.TRAINING
    )
    if any(type(e) is not Example or e.access is not required for e in examples):
        raise PermissionError("minibatch supervision capability differs")
    labels = tuple(e.labels() for e in examples)
    reader.zero_grad(set_to_none=True)
    terms = []
    for e, target in zip(examples, labels, strict=True):
        logits = reader(e.projection, e.projection.access)
        if logits is None:
            raise ValueError("UNKNOWN training example cannot supply complete objective")
        terms.append(loss(logits, target))
    mean = finite(reduce(torch.stack(terms), ordered=False) / float(len(terms)))
    mean.backward()  # type: ignore[no-untyped-call]
    # Structurally inactive shared parameters still have explicit zero gradients.
    for _, p in reader.named_scalars():
        if p.grad is None:
            p.grad = torch.zeros_like(p)
    optimizer.update(reader)
    return float(mean.detach())


def fit_comparative(
    reader: Reader,
    batches: tuple[tuple[Example, ...], ...],
    development: Callable[[Reader], Fraction],
    retain: Callable[[str, bytes], None],
    check: Callable[[], None],
) -> bytes:
    """Later-only 400-step fit. No final provider/split exists in this interface.

    An admitted driver supplies frozen shared minibatch order and an evaluator
    restricted to the fixed development groups. It must verify external split
    rights and seal all final candidates before final scoring, outside this API.
    """
    if len(batches) != 400 or any(len(b) != 8 for b in batches):
        raise ValueError("400 committed eight-decision minibatches required")
    if any(
        type(e) is not Example or e.access is not Supervision.TRAINING for b in batches for e in b
    ):
        raise PermissionError("comparative training access required; final labels denied")
    opt = Adam(reader)
    retain("initial.json", checkpoint(reader, opt))
    best: tuple[Fraction, bytes] | None = None
    start = time.monotonic()
    for index, batch in enumerate(batches, 1):
        check()
        memory_check()
        if time.monotonic() - start > 300:
            raise TimeoutError("comparative wall deadline")
        value = update(reader, opt, batch)
        retain(f"loss-{index:03d}.json", canonical_json_bytes({"step": index, "loss": value}))
        if index in (100, 200, 300, 400):
            saved = checkpoint(reader, opt)
            retain(f"checkpoint-{index}.json", saved)
            metric = development(reader)
            if type(metric) is not Fraction or not 0 <= metric <= 1:
                raise ValueError("exact development group-mean neutral metric required")
            retain(
                f"development-{index}.json",
                canonical_json_bytes({"step": index, "error": str(metric)}),
            )
            if best is None or metric < best[0]:
                best = metric, saved
    check()
    if time.monotonic() - start > 300:
        raise TimeoutError("comparative wall deadline at closure")
    assert best is not None
    retain("selected.json", best[1])
    check()
    memory_check()
    if time.monotonic() - start > 300:
        raise TimeoutError("comparative wall deadline after retention")
    return best[1]


def memory_check() -> int:
    """Measure this process peak working set; cooperative, not host enforcement."""
    import os

    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("faults", wintypes.DWORD),
                ("peak", ctypes.c_size_t),
                ("working", ctypes.c_size_t),
                ("pool_peak", ctypes.c_size_t),
                ("pool", ctypes.c_size_t),
                ("nonpaged_peak", ctypes.c_size_t),
                ("nonpaged", ctypes.c_size_t),
                ("pagefile", ctypes.c_size_t),
                ("pagefile_peak", ctypes.c_size_t),
            ]

        info = Counters()
        info.cb = ctypes.sizeof(info)
        get_current = ctypes.windll.kernel32.GetCurrentProcess
        get_current.restype = wintypes.HANDLE
        query = ctypes.windll.psapi.GetProcessMemoryInfo
        query.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
        if not query(get_current(), ctypes.byref(info), info.cb):
            raise OSError("working-set measurement unavailable")
        peak = int(info.peak)
    else:
        import resource
        import sys

        peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)  # type: ignore[attr-defined]
        if sys.platform != "darwin":
            peak *= 1024
    if peak > 1 << 30:
        raise MemoryError("restricted process working-set budget")
    return peak


class _Sink:
    def __init__(self, root: Path) -> None:
        root.mkdir(exist_ok=False)
        self.root = root
        self.count = 0

    def write(self, name: str, raw: bytes, *, terminal: bool = False) -> None:
        # Two 192MiB originals + first archives +128MiB review =896MiB.
        bound = 192 * (1 << 20) - (0 if terminal else 1 << 16)
        if self.count + len(raw) > bound:
            raise OSError("inclusive retained/archive reservation exhausted")
        import os

        self.count += len(raw)  # Failed partial writes keep their full reservation.
        with (self.root / name).open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())


class _Retained:
    """Offline evaluator of a previously single-fetched immutable public target."""

    def __init__(self, snapshot: Any) -> None:
        from epsbench.diagnostics.boundary_observation import VisibleRaster

        self.snapshot = VisibleRaster(
            snapshot.sequence_index, snapshot.segmentation, snapshot.identities
        )

    def raster(self, index: int) -> Any:
        if index != self.snapshot.sequence_index:
            raise PermissionError("retained target chronology differs")
        return self.snapshot


def public_readiness(arm: Arm, source_head: str, output: Path) -> dict[str, Any]:
    """ONE later-admitted operation per arm, exclusive output; CI must deny this entry.

    This callable is not launch authority. The coordinator must bind verified
    source, explicit one-time admission and retained operation limits beforehand.
    Failures preserve first evidence; neither this function nor CI retries a fit.
    """
    from epsbench.diagnostics.restricted_mask_fixtures import fixtures, public_key

    sink = _Sink(output)
    start = time.monotonic()
    cpu_start = time.process_time()
    fitting_started = False
    result: dict[str, Any] = {"arm": arm, "status": "INCONCLUSIVE", "updates": 0}

    def check() -> None:
        result["peak_working_set"] = max(result.get("peak_working_set", 0), memory_check())
        if time.monotonic() - start > 60:
            raise TimeoutError("public readiness wall deadline")

    try:
        cases = fixtures(source_head)
        reader = Reader(arm, public_key())
        opt = Adam(reader, 0.01)
        examples = []
        retained = []
        for case in cases:
            source = case.projection.revalidate()
            snapshot = case.fetch(case.projection)
            retained.append(_Retained(snapshot))
            lookup = dict(case.target.identities)
            known = tuple(
                (
                    name,
                    case.target.segmentation
                    == next(label for label, token in lookup.items() if token == name),
                )
                for name in source.inventory
            )
            new = case.target.segmentation == len(source.inventory) + 1
            examples.append(
                Example(
                    case.projection, case.projection.save(known, new), Supervision.PUBLIC_FIXTURE
                )
            )
            sink.write(
                f"fixture-{case.number}.json",
                canonical_json_bytes(
                    {
                        "binding": json.loads(case.projection.binding_bytes()),
                        "features": json.loads(case.projection.feature_bytes()),
                        "prefix": json.loads(source.canonical_bytes()),
                        "state": json.loads(case.projection._state),
                        "truth": json.loads(examples[-1].target),
                    }
                ),
            )
        sink.write("initial.json", checkpoint(reader, opt))
        for index in range(1, 201):
            check()
            fitting_started = True
            value = update(reader, opt, tuple(examples))
            result["updates"] = index
            errors = []
            for case, cached in zip(cases, retained, strict=True):
                check()
                saved = prediction(reader, case.projection, case.projection.access)
                sink.write(f"logits-{index:03d}-{case.number}.json", saved)
                masks = candidate(saved, case.projection, case.projection.access)
                report = case.projection.evaluate(masks, cached).report
                errors.append(int(report["E"]))
            sink.write(
                f"report-{index:03d}.json",
                canonical_json_bytes(
                    {
                        "step": index,
                        "loss": value,
                        "errors": errors,
                    }
                ),
            )
            if not any(errors):
                sink.write("checkpoint.json", checkpoint(reader, opt))
                check()
                result["status"] = "PASS"
                break
        else:
            sink.write("checkpoint.json", checkpoint(reader, opt))
            check()
            result["status"] = "NO_GO"
    except Exception as exc:
        result["status"] = (
            "NO_GO" if fitting_started and isinstance(exc, ArithmeticFailure) else "INCONCLUSIVE"
        )
        result["first_failure"] = {"type": type(exc).__name__, "message": str(exc)}
    result["cpu_seconds"] = time.process_time() - cpu_start
    result["wall_seconds"] = time.monotonic() - start
    result["retained_bytes_before_terminal"] = sink.count
    sink.write("terminal.json", canonical_json_bytes(result), terminal=True)
    return result
