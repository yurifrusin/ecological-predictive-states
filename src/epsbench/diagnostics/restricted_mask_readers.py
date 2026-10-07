"""Restricted CPU binary64 readers; no geometry, collection or fitting on import."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import re
from fractions import Fraction
from typing import Any, Literal

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from epsbench.diagnostics.neutral_observation_target import Evaluation
from epsbench.diagnostics.restricted_mask_projection import Projection
from epsbench.diagnostics.visible_forecast_contract import RasterProvider, _json, _keys, permissions
from epsbench.schema import ModalityPermissionSet
from epsbench.utils.canonical import canonical_json_bytes, sha256_bytes

VERSION = "restricted-mask-readers-v2"
Arm = Literal["E", "L"]
SPECS = (
    ("node.conv1", 3, 8, 5),
    ("node.conv2", 8, 8, 3),
    ("union.conv1", 2, 8, 5),
    ("union.conv2", 8, 8, 3),
    ("node.pool", 18, 16, 0),
    ("message.fc1", 35, 16, 0),
    ("message.fc2", 16, 16, 0),
    ("known.fc1", 70, 8, 1),
    ("known.fc2", 8, 1, 1),
    ("global.fc1", 74, 8, 1),
    ("global.fc2", 8, 2, 1),
)


def configure() -> None:
    if torch.__version__ != "2.10.0+cpu":
        raise ValueError("pinned CPU torch required")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.mkldnn.enabled = False  # type: ignore[assignment]


class ArithmeticFailure(ValueError):
    """Finite-domain failure in a model arithmetic operation."""


def finite(value: Tensor) -> Tensor:
    if value.dtype != torch.float64 or value.device.type != "cpu":
        raise ValueError("CPU binary64 tensor required")
    if not bool(torch.isfinite(value).all()):
        raise ArithmeticFailure("nonfinite restricted arithmetic")
    return value


class _OrderedSum(torch.autograd.Function):
    """Literal sequential reduction; the formal derivative of each summand is one.

    Only this reduction uses a custom autograd primitive. NumPy add.accumulate
    evaluates the prescribed recurrence; np.sum's alternative reduction is avoided.
    Sorting affects rounding order, not the ordinary sum derivative used by training.
    """

    @staticmethod
    def forward(ctx: Any, values: Tensor, ordered: bool) -> Tensor:
        finite(values)
        ctx.shape = tuple(values.shape)
        a = values.detach().numpy()
        if ordered and a.shape[0]:
            zero_rank = np.where((a == 0) & np.signbit(a), 0, 1)
            indices = np.lexsort((zero_rank, a), axis=0)
            a = np.take_along_axis(a, indices, axis=0)
        a = np.concatenate((np.zeros((1, *a.shape[1:]), dtype=np.float64), a), axis=0)
        try:
            with np.errstate(over="raise", invalid="raise"):
                result = np.asarray(np.add.accumulate(a, axis=0, dtype=np.float64)[-1]).copy()
        except FloatingPointError as exc:
            raise ArithmeticFailure("nonfinite sequential reduction") from exc
        return finite(torch.from_numpy(result))

    @staticmethod
    def backward(ctx: Any, gradient: Tensor) -> tuple[Tensor, None]:
        return finite(gradient).unsqueeze(0).expand(ctx.shape), None


def reduce(values: Tensor, *, ordered: bool = True) -> Tensor:
    return finite(_OrderedSum.apply(values, ordered))  # type: ignore[no-untyped-call]


def tensor(value: Any) -> Tensor:
    return finite(torch.tensor(value, dtype=torch.float64, device="cpu"))


def broadcast(value: Tensor) -> Tensor:
    return value.reshape(-1, 1, 1).expand(-1, 32, 32)


def commands(values: list[str]) -> list[float]:
    result = []
    for value in values:
        q = Fraction(value)
        with np.errstate(over="raise", invalid="raise"):
            f = float(np.float32(float(q)))
        if not math.isfinite(f) or Fraction(f) != q:
            raise ValueError("command must be exactly binary32 representable")
        result.append(0.0 if q == 0 else f)
    return result


def validate(projection: Projection, access: ModalityPermissionSet) -> dict[str, Any]:
    permissions(access)
    if type(projection) is not Projection:
        raise ValueError("typed immutable Projection required")
    source = projection.revalidate()
    if source.shape != (32, 32) or len(source.frames) != 2 or len(source.inventory) > 3:
        raise ValueError("restricted two-frame 32x32 domain with at most three nodes")
    f: dict[str, Any] = json.loads(projection.feature_bytes())
    commands(f["executed_action"] + f["announced_action"])
    return f


class Layer(nn.Module):
    def __init__(self, name: str, inputs: int, outputs: int, kernel: int, key: bytes) -> None:
        super().__init__()
        shape = (outputs, inputs, kernel, kernel) if kernel else (outputs, inputs)
        area = kernel * kernel if kernel else 1
        scale = math.sqrt(6.0 / ((inputs + outputs) * area))
        weights = []
        for k in range(math.prod(shape)):
            raw = hashlib.sha256(key + f"|{name}.weight|{k}".encode()).digest()
            u = (int.from_bytes(raw[:8], "big") >> 11) / 2**53
            weights.append((2.0 * u - 1.0) * scale)
        self.weight = nn.Parameter(tensor(weights).reshape(shape))
        self.bias = nn.Parameter(torch.zeros(outputs, dtype=torch.float64))
        self.kernel = kernel

    def forward(self, value: Tensor) -> Tensor:
        finite(value)
        finite(self.weight)
        finite(self.bias)
        if self.kernel:
            result = F.conv2d(value.unsqueeze(0), self.weight, self.bias, padding=self.kernel // 2)
            return finite(result.squeeze(0))
        return finite(F.linear(value, self.weight, self.bias))


class Reader(nn.Module):
    def __init__(self, arm: Arm, key: bytes) -> None:
        super().__init__()
        configure()
        if arm not in ("E", "L") or type(key) is not bytes or len(key) != 32:
            raise ValueError("literal arm and 32-byte initialization key required")
        self.arm = arm
        self.layers = nn.ModuleDict(
            {
                name.replace(".", "_"): Layer(name, inputs, outputs, kernel, key)
                for name, inputs, outputs, kernel in SPECS
            }
        )
        if sum(p.numel() for p in self.parameters()) != 4531:
            raise ValueError("exact parameter count differs")

    def named_scalars(self) -> tuple[tuple[str, Tensor], ...]:
        return tuple(
            (name + suffix, getattr(self.layers[name.replace(".", "_")], suffix[1:]))
            for name, _, _, _ in SPECS
            for suffix in (".weight", ".bias")
        )

    def layer(self, name: str, value: Tensor, *, relu: bool = True) -> Tensor:
        value = self.layers[name.replace(".", "_")](value)
        return finite(F.relu(value) if relu else value)

    def forward(self, projection: Projection, access: ModalityPermissionSet) -> Tensor | None:
        permissions(access)
        if self.arm not in ("E", "L"):
            raise ValueError("literal routing arm required")
        for _, parameter in self.named_scalars():
            finite(parameter)
        f = validate(projection, access)
        if not all(e["available"] for e in f["endpoints"]):
            return None
        cmd = tensor(commands(f["executed_action"] + f["announced_action"]))
        endpoint = tensor([v for e in f["endpoints"] for v in (int(e["available"]), e["age"])])
        n = len(f["nodes"])
        hs, zs, ds = [], [], []
        for node in f["nodes"]:
            masks = tensor([node["current"], node["previous"], node["difference"]])
            h = self.layer("node.conv2", self.layer("node.conv1", masks))
            pool = reduce(h.permute(1, 2, 0).reshape(1024, 8), ordered=False) / 1024.0
            d = torch.cat(
                (
                    tensor(
                        [
                            int(node["status"] == "VISIBLE"),
                            int(node["status"] == "REMEMBERED_ABSENT"),
                            node["first_seen_age"],
                            node["last_seen_age"],
                        ]
                    ),
                    cmd,
                )
            )
            hs.append(h)
            ds.append(d)
            zs.append(self.layer("node.pool", torch.cat((pool, d))))
        ms = []
        for i in range(n):
            endpoints = []
            for e in f["endpoints"]:
                terms = []
                for j in range(n):
                    if i == j:
                        continue
                    bit = int(sorted((i, j)) in e["pairs"])
                    message = self.layer(
                        "message.fc2",
                        self.layer(
                            "message.fc1",
                            torch.cat(
                                (
                                    zs[i],
                                    zs[j],
                                    tensor([bit, int(e["available"]), e["age"]]),
                                )
                            ),
                        ),
                    )
                    terms.append(finite(message * (bit if self.arm == "E" else 1)))
                endpoints.append(reduce(torch.stack(terms) if terms else tensor([]).reshape(0, 16)))
            ms.append(torch.cat(endpoints))
        known = []
        for h, z, d, message in zip(hs, zs, ds, ms, strict=True):
            features = torch.cat((h, broadcast(torch.cat((z, message, d, endpoint)))))
            known.append(self.layer("known.fc2", self.layer("known.fc1", features), relu=False))
        hu = self.layer(
            "union.conv2",
            self.layer(
                "union.conv1",
                tensor(
                    [
                        f["current_union"],
                        f["previous_union"],
                    ]
                ),
            ),
        )
        sumh = reduce(torch.stack(hs) if hs else tensor([]).reshape(0, 8, 32, 32))
        sumz = reduce(torch.stack(zs) if zs else tensor([]).reshape(0, 16))
        summ = reduce(torch.stack(ms) if ms else tensor([]).reshape(0, 32))
        global_features = torch.cat((hu, sumh, broadcast(torch.cat((sumz, summ, cmd, endpoint)))))
        globals_ = self.layer("global.fc2", self.layer("global.fc1", global_features), relu=False)
        return finite(torch.cat((*known, globals_)))


def pack(value: Tensor) -> dict[str, Any]:
    a = finite(value).detach().numpy().astype("<f8", copy=False)
    return {"shape": list(a.shape), "bytes": base64.b64encode(a.tobytes(order="C")).decode()}


def unpack(payload: dict[str, Any], shape: tuple[int, ...]) -> Tensor:
    _keys(payload, {"shape", "bytes"})
    if (
        type(payload["shape"]) is not list
        or any(type(v) is not int for v in payload["shape"])
        or payload["shape"] != list(shape)
        or type(payload["bytes"]) is not str
    ):
        raise ValueError("exact tensor shape/encoding required")
    raw = base64.b64decode(payload["bytes"], validate=True)
    if len(raw) != math.prod(shape) * 8 or base64.b64encode(raw).decode() != payload["bytes"]:
        raise ValueError("canonical binary64 length/padding required")
    return finite(torch.from_numpy(np.frombuffer(raw, dtype="<f8").copy().reshape(shape)))


def weights(reader: Reader) -> bytes:
    return canonical_json_bytes({name: pack(value) for name, value in reader.named_scalars()})


def restore(reader: Reader, saved: bytes) -> None:
    p = _json(saved)
    _keys(p, {name for name, _ in reader.named_scalars()})
    staged = [
        (value, unpack(p[name], tuple(value.shape))) for name, value in reader.named_scalars()
    ]
    if canonical_json_bytes(p) != saved:
        raise ValueError("canonical weights required")
    with torch.no_grad():
        for value, replacement in staged:
            value.copy_(replacement)


def hard_candidate(projection: Projection, logits: Tensor | None) -> bytes:
    if logits is None:
        return projection.save(tuple((name, None) for name in projection.alignment), None)
    a = finite(logits).detach().numpy()
    if a.shape != (len(projection.alignment) + 2, 32, 32):
        raise ValueError("active logits shape differs")
    maxima = a.max(axis=0)
    winner = a.argmax(axis=0)
    winner[np.count_nonzero(a == maxima, axis=0) != 1] = a.shape[0] - 1
    return projection.save(
        tuple((name, winner == i) for i, name in enumerate(projection.alignment)),
        winner == len(projection.alignment),
    )


def prediction(reader: Reader, projection: Projection, access: ModalityPermissionSet) -> bytes:
    with torch.no_grad():
        logits = reader(projection, access)
    return canonical_json_bytes(
        {
            "version": VERSION + ":prediction",
            "arm": reader.arm,
            "weights_sha256": sha256_bytes(weights(reader)),
            "binding_sha256": sha256_bytes(projection.binding_bytes()),
            "logits": None if logits is None else pack(logits),
            "candidate": json.loads(hard_candidate(projection, logits)),
        }
    )


def candidate(saved: bytes, projection: Projection, access: ModalityPermissionSet) -> bytes:
    f = validate(projection, access)  # Before parsing or later evaluator fetch.
    p = _json(saved)
    _keys(p, {"version", "arm", "weights_sha256", "binding_sha256", "logits", "candidate"})
    if (
        p["version"] != VERSION + ":prediction"
        or p["arm"] not in ("E", "L")
        or p["binding_sha256"] != sha256_bytes(projection.binding_bytes())
        or type(p["weights_sha256"]) is not str
        or re.fullmatch(r"[0-9a-f]{64}", p["weights_sha256"]) is None
    ):
        raise ValueError("prediction context differs")
    unavailable = not all(e["available"] for e in f["endpoints"])
    if (p["logits"] is None) != unavailable:
        raise ValueError("UNKNOWN must match unavailable relation context")
    logits = None if unavailable else unpack(p["logits"], (len(projection.alignment) + 2, 32, 32))
    actual = hard_candidate(projection, logits)
    if actual != canonical_json_bytes(p["candidate"]) or canonical_json_bytes(p) != saved:
        raise ValueError("saved masks must equal retained binary64 decision")
    return actual


def evaluate_prediction(
    saved: bytes,
    projection: Projection,
    access: ModalityPermissionSet,
    expected_weights: str,
    provider: RasterProvider,
) -> Evaluation:
    masks = candidate(saved, projection, access)
    if _json(saved)["weights_sha256"] != expected_weights:
        raise ValueError("retained selected-weight binding differs")
    return projection.evaluate(masks, provider)
