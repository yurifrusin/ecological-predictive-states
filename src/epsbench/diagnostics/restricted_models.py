"""Frozen restricted CPU models; no data loading or launch side effects."""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn

from epsbench.diagnostics.restricted_learning_contract import (
    BIJECTIONS,
    CONDITIONS,
    Observation,
    StreamingView,
)

DOMAIN = b"M0-occupancy-development-v1/init-v1"


def configure_cpu() -> None:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)


def initialized(name: str, rows: int, columns: int, seed: int) -> Tensor:
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("unsigned initialization seed required")
    encoded = name.encode("utf-8")
    prefix = struct.pack("<Q", len(DOMAIN)) + DOMAIN
    prefix += struct.pack("<Q", len(encoded)) + encoded + struct.pack("<Q", seed)
    bound = 1 / math.sqrt(columns)
    values = []
    for i in range(rows * columns):
        integer = int.from_bytes(
            hashlib.sha256(prefix + struct.pack("<Q", i)).digest()[:8], "little"
        )
        u = (integer + 0.5) / 2**64
        values.append((2 * u - 1) * bound)
    return torch.tensor(values, dtype=torch.float32).reshape(rows, columns)


@dataclass(frozen=True)
class State:
    streams: tuple[Tensor, ...]
    index: int
    handles: tuple[str, ...]
    flags: tuple[tuple[int, int], ...]
    first_seen: tuple[int, ...] = ()


class RestrictedModel(nn.Module):
    """Weights are shared across six fixed streams; masks never enter readout."""

    def __init__(self, condition: str, initialization_seed: int) -> None:
        super().__init__()
        if condition not in CONDITIONS:
            raise ValueError("frozen learned condition required")
        self.condition = condition
        self.weights = nn.ParameterDict()
        self._names: dict[str, str] = {}
        self._seed = initialization_seed
        self._dense("encoder/layer1", 1024, 32)
        self._dense("encoder/layer2", 32, 16)
        if condition == "dense":
            self._gru_parameters("comparator/joint-GRU", 69, 64)
            self._dense("comparator/joint-head/layer1", 73, 529)
            self._dense("comparator/joint-head/layer2", 529, 3)
        else:
            self._dense("candidate/sender", 32, 16)
            self._gru_parameters("candidate/token-GRU", 53, 16)
            self._gru_parameters("candidate/global-GRU", 23, 16)
            self._dense("candidate/context/layer1", 35, 1060)
            self._dense("candidate/context/layer2", 1060, 16)
            self._dense("candidate/token-head/layer1", 34, 154)
            self._dense("candidate/token-head/layer2", 154, 1)
        expected = 99984 if condition == "dense" else 99913
        if sum(p.numel() for p in self.parameters()) != expected:
            raise ValueError("frozen used parameter inventory differs")

    @staticmethod
    def tensor_layout(condition: str) -> list[list[object]]:
        if condition not in CONDITIONS:
            raise ValueError("frozen learned condition required")
        layout: list[list[object]] = []

        def dense(name: str, d: int, h: int) -> None:
            layout.extend(
                [[name.replace("/", "__") + "__W", [h, d]], [name.replace("/", "__") + "__b", [h]]]
            )

        def gru(name: str, d: int, h: int) -> None:
            for gate in ("r", "z", "n"):
                for kind, shape in (("W", [h, d]), ("U", [h, h]), ("bi", [h]), ("bh", [h])):
                    layout.append([name.replace("/", "__") + "__" + kind + gate, shape])

        dense("encoder/layer1", 1024, 32)
        dense("encoder/layer2", 32, 16)
        if condition == "dense":
            gru("comparator/joint-GRU", 69, 64)
            dense("comparator/joint-head/layer1", 73, 529)
            dense("comparator/joint-head/layer2", 529, 3)
        else:
            dense("candidate/sender", 32, 16)
            gru("candidate/token-GRU", 53, 16)
            gru("candidate/global-GRU", 23, 16)
            dense("candidate/context/layer1", 35, 1060)
            dense("candidate/context/layer2", 1060, 16)
            dense("candidate/token-head/layer1", 34, 154)
            dense("candidate/token-head/layer2", 154, 1)
        return layout

    def _add(self, name: str, value: Tensor) -> None:
        key = name.replace("/", "__")
        self._names[name] = key
        self.weights[key] = nn.Parameter(value)

    def _dense(self, name: str, d: int, h: int) -> None:
        self._add(name + "/W", initialized(name + "/W", h, d, self._seed))
        self._add(name + "/b", torch.zeros(h, dtype=torch.float32))

    def _gru_parameters(self, name: str, d: int, h: int) -> None:
        for gate in ("r", "z", "n"):
            self._add(name + "/W" + gate, initialized(name + "/W" + gate, h, d, self._seed))
            self._add(name + "/U" + gate, initialized(name + "/U" + gate, h, h, self._seed))
            self._add(name + "/bi" + gate, torch.zeros(h, dtype=torch.float32))
            self._add(name + "/bh" + gate, torch.zeros(h, dtype=torch.float32))

    def weight(self, name: str) -> Tensor:
        return cast(Tensor, self.weights[self._names[name]])

    def dense(self, name: str, x: Tensor) -> Tensor:
        return self.weight(name + "/W") @ x + self.weight(name + "/b")

    def gru(self, name: str, x: Tensor, h: Tensor) -> Tensor:
        def affine(gate: str, recurrent: bool) -> Tensor:
            return self.weight(name + ("/U" if recurrent else "/W") + gate) @ (
                h if recurrent else x
            ) + self.weight(name + ("/bh" if recurrent else "/bi") + gate)

        r = torch.sigmoid(affine("r", False) + affine("r", True))
        z = torch.sigmoid(affine("z", False) + affine("z", True))
        n = torch.tanh(affine("n", False) + r * affine("n", True))
        return (1 - z) * n + z * h

    def initial_state(self) -> State:
        return State(tuple(torch.zeros(64, dtype=torch.float32) for _ in BIJECTIONS), -1, (), ())

    @staticmethod
    def validate_state(state: State) -> None:
        if (
            type(state) is not State
            or type(state.streams) is not tuple
            or len(state.streams) != 6
            or type(state.index) is not int
            or state.index not in (-1, 0, 1, 2)
        ):
            raise ValueError("owned fixed streaming state required")
        if (
            type(state.handles) is not tuple
            or len(state.handles) > 3
            or len(set(state.handles)) != len(state.handles)
            or type(state.flags) is not tuple
        ):
            raise ValueError("immutable observed inventory/flags required")
        if (
            type(state.first_seen) is not tuple
            or len(state.first_seen) != len(state.handles)
            or any(type(i) is not int or not 0 <= i <= state.index for i in state.first_seen)
        ):
            raise ValueError("causal state first-seen metadata required")
        if state.index == -1:
            if state.handles or state.flags:
                raise ValueError("empty initial state inventory required")
        elif (
            len(state.flags) != 3
            or any(
                type(f) is not tuple
                or len(f) != 2
                or any(type(v) is not int or v not in (0, 1) for v in f)
                for f in state.flags
            )
            or tuple(f[0] for f in state.flags)
            != tuple(int(i < len(state.handles)) for i in range(3))
        ):
            raise ValueError("exact immutable three-slot flags required")
        for value in state.streams:
            if (
                type(value) is not Tensor
                or value.shape != (64,)
                or value.dtype != torch.float32
                or value.device.type != "cpu"
                or not torch.isfinite(value).all()
            ):
                raise ValueError("six finite float32 CPU states required")

    def advance(self, state: State, observation: Observation) -> State:
        self.validate_state(state)
        if type(observation) is not Observation or type(state) is not State:
            raise ValueError("typed causal observation/state required")
        if observation.index != state.index + 1 or observation.index > 2:
            raise PermissionError("exact sequential three-step update required")
        if observation.first_seen[: len(state.first_seen)] != state.first_seen or any(
            observation.first_seen[i] != observation.index or not observation.visible[i]
            for i in range(len(state.handles), len(observation.handles))
        ):
            raise ValueError("new tokens must be currently visible and first-seen now")
        if observation.handles[: len(state.handles)] != state.handles:
            raise ValueError("causal arrival inventory changed")
        if len(state.streams) != 6 or any(
            s.shape != (64,)
            or s.dtype != torch.float32
            or s.device.type != "cpu"
            or not torch.isfinite(s).all()
            for s in state.streams
        ):
            raise ValueError("six finite float32 CPU states required")
        features = []
        for i in range(3):
            if observation.present[i]:
                mask = torch.tensor(list(observation.masks[i]), dtype=torch.float32)
                features.append(
                    torch.tanh(
                        self.dense("encoder/layer2", torch.relu(self.dense("encoder/layer1", mask)))
                    )
                )
            else:
                features.append(torch.zeros(16, dtype=torch.float32))
        e = torch.tensor([float(v) for v in observation.executed], dtype=torch.float32)
        if self.condition == "action-zero":
            e = torch.zeros(3, dtype=torch.float32)
        boundary = {
            (i, j): torch.tensor([float(v) for v in observation.boundary[k]], dtype=torch.float32)
            for k, (i, j) in enumerate((i, j) for i in range(3) for j in range(3) if i != j)
        }
        streams = []
        for permutation, old in zip(BIJECTIONS, state.streams, strict=True):
            if self.condition == "memory-reset":
                old = old * 0
            x = [features[i] for i in permutation]
            p = [observation.present[i] for i in permutation]
            v = [observation.visible[i] for i in permutation]
            flags = torch.tensor([f for i in range(3) for f in (p[i], v[i])], dtype=torch.float32)
            edges = [
                boundary[(permutation[i], permutation[j])]
                for i in range(3)
                for j in range(3)
                if i != j
            ]
            if self.condition == "dense":
                new = self.gru("comparator/joint-GRU", torch.cat((*x, flags, *edges, e)), old)
            else:
                h = [old[i * 16 : (i + 1) * 16] for i in range(3)]
                g = old[48:]
                senders = [
                    p[j] * torch.tanh(self.dense("candidate/sender", torch.cat((h[j], x[j]))))
                    for j in range(3)
                ]
                nodes = []
                for i in range(3):
                    products = []
                    for j in range(3):
                        if i != j:
                            b = boundary[(permutation[i], permutation[j])]
                            products.append(((b[0] + b[1]) / 2) * senders[j])
                    message = (p[i] / 2) * (products[0] + products[1])
                    node_input = torch.cat((x[i], message, g, flags[2 * i : 2 * i + 2], e))
                    nodes.append(p[i] * self.gru("candidate/token-GRU", node_input, h[i]))
                pooled = (nodes[0] + nodes[1] + nodes[2]) / 3
                bh = edges[0][0]
                bv = edges[0][1]
                for b in edges[1:]:
                    bh = bh + b[0]
                    bv = bv + b[1]
                summaries = torch.stack(
                    (
                        torch.tensor(sum(p) / 3, dtype=torch.float32),
                        torch.tensor(sum(v) / 3, dtype=torch.float32),
                        bh / 6,
                        bv / 6,
                    )
                )
                global_state = self.gru(
                    "candidate/global-GRU", torch.cat((pooled, e, summaries)), g
                )
                new = torch.cat((*nodes, global_state))
            streams.append(new)
        if any(not torch.isfinite(s).all() for s in streams):
            raise ValueError("nonfinite recurrent state")
        return State(
            tuple(streams),
            observation.index,
            observation.handles,
            tuple(zip(observation.present, observation.visible, strict=True)),
            observation.first_seen,
        )

    def readout(self, state: State, announced: tuple[float, float, float]) -> Tensor:
        self.validate_state(state)
        if (
            state.index != 2
            or type(announced) is not tuple
            or len(announced) != 3
            or any(type(v) is not float or not math.isfinite(v) for v in announced)
        ):
            raise PermissionError("readout requires all three updates and finite announced command")
        a = torch.tensor(announced, dtype=torch.float32)
        if self.condition == "action-zero":
            a = torch.zeros(3, dtype=torch.float32)
        total = torch.zeros(3, dtype=torch.float32)
        for permutation, s in zip(BIJECTIONS, state.streams, strict=True):
            flags = torch.tensor(
                [f for i in permutation for f in state.flags[i]], dtype=torch.float32
            )
            if self.condition == "dense":
                q = torch.sigmoid(
                    self.dense(
                        "comparator/joint-head/layer2",
                        torch.relu(
                            self.dense("comparator/joint-head/layer1", torch.cat((s, a, flags)))
                        ),
                    )
                )
            else:
                nodes = [s[i * 16 : (i + 1) * 16] for i in range(3)]
                pooled = (nodes[0] + nodes[1] + nodes[2]) / 3
                c = torch.tanh(
                    self.dense(
                        "candidate/context/layer2",
                        torch.relu(
                            self.dense("candidate/context/layer1", torch.cat((pooled, s[48:], a)))
                        ),
                    )
                )
                q = torch.cat(
                    tuple(
                        torch.sigmoid(
                            self.dense(
                                "candidate/token-head/layer2",
                                torch.relu(
                                    self.dense(
                                        "candidate/token-head/layer1",
                                        torch.cat((nodes[i], c, flags[2 * i : 2 * i + 2])),
                                    )
                                ),
                            )
                        )
                        for i in range(3)
                    )
                )
            remapped = torch.stack(
                tuple(q[permutation.index(i)] * state.flags[i][0] for i in range(3))
            )
            total = total + remapped
        result = total / 6
        if not torch.isfinite(result).all() or bool(((result < 0) | (result > 1)).any()):
            raise ValueError("invalid Bernoulli probabilities")
        return result

    def forward(self, view: StreamingView) -> tuple[State, Tensor]:
        if type(view) is not StreamingView:
            raise ValueError("typed causal streaming view required")
        state = self.initial_state()
        for _ in range(3):
            state = self.advance(state, view.next_observation())
        a = tuple(float(v) for v in view.announced)
        return state, self.readout(state, (a[0], a[1], a[2]))
