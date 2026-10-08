"""Frozen PR84 eee49ff public N=2 untrained reference; regression only, never fitting."""

from __future__ import annotations

from typing import cast

import numpy as np
import torch
from torch import Tensor, nn

from epsbench.diagnostics.oracle_organization_contract import OracleInput

SEED = 271828


class Common(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Registration order is the frozen paired initialization order.
        self.encoder = nn.Linear(1024, 16)
        self.head = nn.Sequential(nn.Linear(33, 16), nn.Tanh(), nn.Linear(16, 1))
        self.frame = nn.Linear(20, 16)

    def encode(self, x: OracleInput) -> Tensor:
        masks = torch.tensor(x.masks.astype(np.float32))
        flags = torch.tensor(np.stack((x.present, x.observed), axis=-1), dtype=torch.float32)
        times = torch.arange(3, dtype=torch.float32)[:, None, None].expand(3, 2, 1)
        commands = torch.tensor((0, *x.executed), dtype=torch.float32)[:, None, None]
        commands = commands.expand(3, 2, 1)
        return torch.tanh(
            self.frame(
                torch.cat(
                    (torch.tanh(self.encoder(masks.flatten(2))), flags, times, commands), dim=-1
                )
            )
        )

    def output(self, state: Tensor, x: OracleInput) -> Tensor:
        global_state = state.mean(dim=0, keepdim=True).expand(2, 16)
        arms = []
        for command in x.announced:
            q = torch.full((2, 1), float(command))
            arms.append(self.head(torch.cat((state, global_state, q), dim=-1)).squeeze(-1))
        return torch.stack(arms, dim=-1)


class ContactLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.message = nn.Linear(16, 16)
        self.gate = nn.GRUCell(48, 16)
        self.availability = nn.Linear(1, 16, bias=False)

    def forward(self, local: Tensor, state: Tensor, contact: Tensor, available: Tensor) -> Tensor:
        messages = contact @ torch.tanh(self.message(state)) / 2
        messages = messages + self.availability(available.mean(dim=-1, keepdim=True))
        context = state.mean(dim=0, keepdim=True).expand_as(state)
        return cast(Tensor, self.gate(torch.cat((local, messages, context), dim=-1), state))


class Ecological(Common):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([ContactLayer(), ContactLayer()])
        initialize(self)

    def forward(self, inputs: OracleInput) -> Tensor:
        x = inputs.checked()
        local = self.encode(x)
        state = torch.zeros((2, 16))
        for t in range(3):
            contact = torch.tensor(x.contacts[t], dtype=torch.float32)
            available = torch.tensor(x.available[t], dtype=torch.float32)
            for layer in self.layers:
                state = layer(local[t], state, contact, available)
        return self.output(state, x)


class AttentionLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.qkv = nn.Linear(16, 48)
        self.projection = nn.Linear(16, 16)
        # Per head: association equality, same-frame contact, same-frame availability.
        self.relations = nn.Linear(3, 4, bias=False)
        self.norm1 = nn.LayerNorm(16)
        self.ff = nn.Sequential(nn.Linear(16, 64), nn.GELU(), nn.Linear(64, 16))
        self.norm2 = nn.LayerNorm(16)

    def forward(self, tokens: Tensor, relations: Tensor) -> Tensor:
        qkv = self.qkv(tokens).reshape(6, 3, 4, 4).permute(1, 2, 0, 3)
        q, k, v = qkv.unbind(0)
        scores = q @ k.transpose(-1, -2) / 2
        scores = scores + self.relations(relations).permute(2, 0, 1)
        attended = (scores.softmax(dim=-1) @ v).transpose(0, 1).reshape(6, 16)
        tokens = self.norm1(tokens + self.projection(attended))
        return cast(Tensor, self.norm2(tokens + self.ff(tokens)))


class Generic(Common):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([AttentionLayer(), AttentionLayer()])
        initialize(self)

    @staticmethod
    def relations(x: OracleInput) -> Tensor:
        values = torch.zeros((6, 6, 3))
        for a in range(6):
            for b in range(6):
                ta, ra = divmod(a, 2)
                tb, rb = divmod(b, 2)
                values[a, b, 0] = float(ra == rb)
                if ta == tb:
                    values[a, b, 1] = float(x.contacts[ta, ra, rb])
                    values[a, b, 2] = float(x.available[ta, ra, rb])
        return values

    def forward(self, inputs: OracleInput) -> Tensor:
        x = inputs.checked()
        tokens = self.encode(x).reshape(6, 16)
        relations = self.relations(x)
        for layer in self.layers:
            tokens = layer(tokens, relations)
        return self.output(tokens.reshape(3, 2, 16).mean(dim=0), x)


def initialize(model: Common) -> None:
    """Separate same-seed generators; common encoder/head/frame first, then layers."""
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter.ndim == 2:
                nn.init.xavier_uniform_(parameter, gain=1, generator=generator)
            elif name.endswith("weight") and ".norm" in name:
                parameter.fill_(1)
            else:
                parameter.zero_()


def parameter_inventory(model: Common) -> dict[str, int]:
    return {name: p.numel() for name, p in model.named_parameters() if p.requires_grad}
