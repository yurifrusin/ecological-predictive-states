"""Inert prospective readiness recipe and non-fitting public storage inspection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor
from torch.utils._python_dispatch import TorchDispatchMode

from epsbench.diagnostics.oracle_organization_contract import OracleInput, public_fixtures
from epsbench.diagnostics.oracle_organization_models import Common, Ecological, Generic


@dataclass(frozen=True)
class Recipe:
    seed: int = 271828
    updates: int = 200
    learning_rate: float = 0.001
    betas: tuple[float, float] = (0.9, 0.999)
    epsilon: float = 1e-8
    weight_decay: float = 0.0001
    total_seconds: int = 300
    input_bytes_per_case: int = 65536
    derived_bytes_per_case: int = 262144
    maximum_parameters: int = 32768
    maximum_relative_difference: float = 0.05
    final_bce_ceiling: float = 0.1


RECIPE = Recipe()


def readiness() -> None:
    """Intentionally inert until exact-source review and separate operation adoption."""
    raise PermissionError("source preparation only; readiness fits are not adopted")


def policy(probabilities: tuple[float, float]) -> int:
    """Return arm 0/1 or abstention 2; ties with abstention abstain."""
    if any(not 0 <= p <= 1 for p in probabilities):
        raise ValueError("finite probabilities required")
    costs = tuple(5 - 4 * p for p in probabilities)
    if min(costs) >= 2:
        return 2
    return 0 if costs[0] <= costs[1] else 1


def regret(probabilities: tuple[float, float], success: tuple[int, int]) -> int:
    if any(type(s) is not int or s not in (0, 1) for s in success):
        raise ValueError("binary teacher targets required")
    costs = (5 - 4 * success[0], 5 - 4 * success[1], 2)
    return costs[policy(probabilities)] - min(costs)


class AllocationLedger(TorchDispatchMode):
    """Retain distinct output storages: conservative cumulative tensor allocation.

    Views/aliases count once. Parameter storages are separate. This measures CPU
    tensor storage, not native allocator bookkeeping, BLAS workspaces or host RSS.
    Retention itself intentionally overstates live storage. No host cap is claimed.
    """

    def __init__(self, model: Common) -> None:
        super().__init__()
        self.excluded = {p.untyped_storage().data_ptr() for p in model.parameters()}
        self.storages: dict[int, Any] = {}

    def __torch_dispatch__(self, func: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
        result = func(*args, **(kwargs or {}))

        def record(value: Any) -> None:
            if isinstance(value, Tensor):
                storage = value.untyped_storage()
                pointer = storage.data_ptr()
                if pointer not in self.excluded:
                    self.storages[pointer] = storage
            elif isinstance(value, (tuple, list)):
                for item in value:
                    record(item)

        record(result)
        return result

    @property
    def bytes(self) -> int:
        return sum(int(s.nbytes()) for s in self.storages.values())


def profile(model: Common, inputs: OracleInput) -> dict[str, int]:
    ledger = AllocationLedger(model)
    with ledger:
        output = model(inputs)
    if not bool(torch.isfinite(output).all()):
        raise ValueError("nonfinite source profile")
    # checked() owns Boolean copies; astype(float32) is a separate NumPy temporary.
    # Torch copies, float flags and all dispatch outputs are already in the ledger.
    # Conservative NumPy validation scratch allowance: 49,152 bytes exceeds
    # int64 region-overlap reduction (24,576), comparison (3,072), all four
    # Boolean neighbour products over three frames (11,904), and other tiny
    # flag/relation products. Copies and the float32 cast are separate below.
    validation_scratch_bound = 49152
    numpy_temporaries = (
        inputs.storage_bytes()
        - 32
        + inputs.masks.size * 4
        + validation_scratch_bound
        + inputs.present.size * 2
    )
    return {
        "input_bytes": inputs.storage_bytes(),
        "torch_cumulative_bytes": ledger.bytes,
        "numpy_temporary_bytes": numpy_temporaries,
        "derived_conservative_bytes": ledger.bytes + numpy_temporaries,
        "validation_scratch_bound": validation_scratch_bound,
        "parameter_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
    }


def source_report() -> dict[str, Any]:
    models = {"E": Ecological(), "G": Generic()}
    counts = {name: sum(p.numel() for p in model.parameters()) for name, model in models.items()}
    relative = abs(counts["E"] - counts["G"]) / min(counts.values())
    if max(counts.values()) > RECIPE.maximum_parameters or relative > 0.05:
        raise ValueError("prospective parameter parity failure; fit prohibited")
    profiles = {name: profile(model, public_fixtures()[0].inputs) for name, model in models.items()}
    if any(
        p["input_bytes"] > RECIPE.input_bytes_per_case
        or p["derived_conservative_bytes"] > RECIPE.derived_bytes_per_case
        for p in profiles.values()
    ):
        raise ValueError("prospective storage accounting failure; fit prohibited")
    return {
        "counts": counts,
        "relative_difference": relative,
        "profiles": profiles,
        "scope": "untrained public software checks only; no empirical gate effect",
    }
