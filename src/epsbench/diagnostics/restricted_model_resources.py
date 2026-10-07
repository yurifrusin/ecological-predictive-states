"""Observed CPU tensor work, distinct from the frozen planning proxy."""

from __future__ import annotations

import platform
import time
from collections import Counter
from typing import Any

import torch
from torch import Tensor
from torch.utils._python_dispatch import TorchDispatchMode


class WorkTrace(TorchDispatchMode):
    """Trace the executed dispatcher, including autograd and explicit Adam.

    Named nonlinear evaluations are units, not processor instruction counts.
    Unclassified arithmetic denies a matching claim; structural operators are retained.
    """

    def __init__(self) -> None:
        super().__init__()
        self.phase = "forward"
        self.work: Counter[str] = Counter()
        self.operators: Counter[str] = Counter()
        self.unknown: Counter[str] = Counter()
        self.arithmetic: Counter[str] = Counter()
        self.temporary_output_bytes: Counter[str] = Counter()

    def __torch_dispatch__(self, func: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
        result = func(*args, **(kwargs or {}))
        name = str(func)
        self.operators[self.phase + ":" + name] += 1
        tensors = (
            [result]
            if isinstance(result, Tensor)
            else [v for v in result if isinstance(v, Tensor)]
            if isinstance(result, (list, tuple))
            else []
        )
        n = sum(t.numel() for t in tensors)
        self.temporary_output_bytes[self.phase] += sum(
            t.numel() * t.element_size() for t in tensors
        )
        stem = name.split(".")[1]
        cost = 0
        if stem in ("mv", "mm", "bmm"):
            cost = 2 * args[0].shape[-1] * n
        elif stem in (
            "add",
            "sub",
            "rsub",
            "mul",
            "div",
            "neg",
            "sqrt",
            "pow",
            "sigmoid",
            "tanh",
            "relu",
            "reciprocal",
            "add_",
            "mul_",
            "div_",
            "sub_",
        ):
            cost = n
        elif stem in ("sigmoid_backward", "tanh_backward"):
            cost = 3 * n
        elif stem == "threshold_backward":
            cost = n
        elif stem in ("sum", "mean"):
            cost = max(0, args[0].numel() - n) + (n if stem == "mean" else 0)
        elif stem not in {
            "view",
            "reshape",
            "_unsafe_view",
            "slice",
            "select",
            "t",
            "transpose",
            "permute",
            "expand",
            "unsqueeze",
            "squeeze",
            "cat",
            "stack",
            "clone",
            "detach",
            "lift_fresh",
            "empty",
            "empty_like",
            "zeros",
            "zeros_like",
            "ones",
            "ones_like",
            "new_zeros",
            "copy_",
            "fill_",
            "zero_",
            "_to_copy",
            "to",
            "isfinite",
            "abs",
            "ne",
            "eq",
            "lt",
            "gt",
            "le",
            "ge",
            "all",
            "any",
            "bitwise_and",
            "bitwise_or",
            "_local_scalar_dense",
            "slice_backward",
            "select_backward",
            "unbind",
            "as_strided",
            "as_strided_scatter",
            "where",
        }:
            self.unknown[self.phase + ":" + name] += 1
        self.work[self.phase] += int(cost)
        self.arithmetic[self.phase + ":" + name] += int(cost)
        return result

    def payload(self) -> dict[str, Any]:
        return {
            "convention": "executed-dispatch-v1; nonlinear unit; not CPU instructions",
            "work": dict(self.work),
            "arithmetic_by_operator": dict(self.arithmetic),
            "operators": dict(self.operators),
            "unclassified": dict(self.unknown),
            "temporary_output_bytes_cumulative_not_peak": dict(self.temporary_output_bytes),
        }


def provenance() -> dict[str, Any]:
    return {
        "device": "cpu",
        "cuda": torch.version.cuda,
        "cpu": platform.processor(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "framework": str(torch.__version__),
        "runtime_blas": torch.__config__.show(),
        "threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "cpu_seconds": time.process_time(),
        "wall_seconds": time.monotonic(),
    }


def observed_peak_bytes() -> int | None:
    if platform.system() != "Windows":
        return None
    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
            (name, ctypes.c_size_t)
            for name in (
                "peak_working",
                "working",
                "peak_paged",
                "paged",
                "peak_nonpaged",
                "nonpaged",
                "pagefile",
                "peak_pagefile",
            )
        ]

    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    loader = getattr(ctypes, "windll", None)
    if not isinstance(loader, ctypes.LibraryLoader):
        return None
    try:
        current = loader.LoadLibrary("kernel32").GetCurrentProcess
        api = loader.LoadLibrary("psapi").GetProcessMemoryInfo
    except (AttributeError, OSError):
        return None
    current.restype = wintypes.HANDLE
    current.argtypes = ()
    api.argtypes = (wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD)
    if not api(current(), ctypes.byref(counters), counters.cb):
        return None
    return int(counters.peak_working)


SCALAR_CONVENTION = "model-normalization-and-Adam-v1; arithmetic and named power units"


def scalar_work(condition: str, examples: int, updates: int) -> dict[str, int]:
    from epsbench.diagnostics.restricted_learning_contract import CONDITIONS

    if (
        condition not in CONDITIONS
        or type(examples) is not int
        or type(updates) is not int
        or examples < 0
        or updates < 0
    ):
        raise ValueError("known condition and nonnegative integer execution counts required")
    # Per stream/step: three p/2 divisions; two sum(3 terms, starting0)/3 (4 each).
    candidate_forward = 6 * 3 * (3 + 2 * (3 + 1))
    tensors = 20 if condition == "dense" else 38
    return {
        "python_scalar_forward_units": 0 if condition == "dense" else candidate_forward * examples,
        "python_scalar_adam_bias_correction_units": updates * tensors * 2 * (1 + 1),
        "python_scalar_adam_step_units": updates,
    }


def valid_measurement(record: dict[str, Any]) -> bool:
    """Check observed schema/work consistency; external immutable receipts authenticate bytes.

    This is not a cryptographic proof of honest execution in a hostile same-process caller.
    Exact-source independent review plus retained trace/launch/checkpoint establishes provenance.
    """
    try:
        required = {
            "fit",
            "model_source",
            "initialization",
            "labels",
            "order",
            "optimizer",
            "trace",
            "used_parameters",
            "gradient_connected",
            "state_floats",
            "recurrent_bytes",
            "parameter_bytes",
            "gradient_bytes",
            "adam_moment_bytes",
            "saved_activation_bytes_cumulative_not_peak",
            "saved_activation_peak_conservative_bytes",
            "buffers",
            "observed_mask_encodings",
            "trainer_batch_examples",
            "prediction_values",
            "shared_encoder",
            "streams",
            "schedule",
            "data",
            "updates",
            "batch",
            "checkpoint",
            "provenance",
            "memory_peak",
            "memory_complete",
            "python_scalar_forward_units",
            "python_scalar_adam_bias_correction_units",
            "python_scalar_adam_step_units",
            "python_scalar_scope",
            "elapsed_cpu",
            "elapsed_wall",
        }
        if not required <= record.keys():
            return False
        from epsbench.diagnostics.restricted_learning_contract import Fit, sha
        from epsbench.diagnostics.restricted_model_training import ModelSource

        budget, initialization, condition = record["fit"].split("/")
        fit = Fit(int(budget), initialization, condition)
        ModelSource(**record["model_source"])
        expected_scalars = scalar_work(condition, 16000, 1000)
        if record["python_scalar_scope"] != SCALAR_CONVENTION or any(
            type(record[k]) is not int or record[k] != value
            for k, value in expected_scalars.items()
        ):
            return False
        for name in ("initialization", "labels", "order", "schedule", "data", "checkpoint"):
            sha(record[name])
        expected = 99984 if fit.condition == "dense" else 99913
        if (
            record["used_parameters"],
            record["parameter_bytes"],
            record["gradient_bytes"],
            record["adam_moment_bytes"],
        ) != (expected, expected * 4, expected * 4, expected * 8):
            return False
        if (
            (
                record["state_floats"],
                record["recurrent_bytes"],
                record["streams"],
                record["updates"],
                record["batch"],
                record["trainer_batch_examples"],
                record["prediction_values"],
            )
            != (384, 1536, 6, 1000, 16, 16, 3)
            or record["shared_encoder"] is not True
            or record["gradient_connected"] is not True
        ):
            return False
        if record["optimizer"] != {
            "name": "Adam",
            "lr": 0.001,
            "beta1": 0.9,
            "beta2": 0.999,
            "epsilon": 1e-8,
            "weight_decay": 0,
            "clipping": False,
            "amsgrad": False,
            "dtype": "float32",
        }:
            return False
        runtime = record["provenance"]
        if (
            runtime["threads"] != 1
            or runtime["interop_threads"] != 1
            or runtime["deterministic"] is not True
            or runtime["device"] != "cpu"
            or runtime["cuda"] is not None
            or any(
                type(runtime[k]) is not str or not runtime[k]
                for k in ("cpu", "os", "python", "framework", "runtime_blas")
            )
        ):
            return False
        if (
            record["memory_complete"] is not True
            or type(record["memory_peak"]) is not int
            or not 0 < record["memory_peak"] <= 4 * 1024**3
        ):
            return False
        import math

        if any(
            type(record[k]) is not float or not math.isfinite(record[k]) or record[k] <= 0
            for k in ("elapsed_cpu", "elapsed_wall")
        ):
            return False
        buffers = record["buffers"]
        if set(buffers) != {
            "current_mask_bytes",
            "shared_features_bytes",
            "current_boundary_bytes",
            "flags_bytes",
            "commands_bytes",
            "candidate_context_bytes",
            "prediction_bytes",
            "batch_examples",
            "schedule_index_bytes",
            "lawful_material_mask_bytes",
        } or any(type(v) is not int or v < 0 for v in buffers.values()):
            return False
        if (
            buffers["current_mask_bytes"] != 12288
            or buffers["shared_features_bytes"] != 192
            or buffers["batch_examples"] != 16
            or buffers["schedule_index_bytes"] != 128000
        ):
            return False
        if (
            type(record["saved_activation_peak_conservative_bytes"]) is not int
            or record["saved_activation_peak_conservative_bytes"] <= 0
        ):
            return False
        trace = record["trace"]
        if (
            trace["unclassified"]
            or trace["convention"] != "executed-dispatch-v1; nonlinear unit; not CPU instructions"
        ):
            return False
        for phase in ("forward", "backward", "adam", "loss"):
            work = trace["work"][phase]
            observed = sum(
                v for k, v in trace["arithmetic_by_operator"].items() if k.startswith(phase + ":")
            )
            if type(work) is not int or work <= 0 or observed != work:
                return False
        n = record["observed_mask_encodings"]
        if type(n) is not int or not 0 <= n <= 9 * 16000:
            return False
        expected_mv = 2 * n + (120 if fit.condition == "dense" else 534) * 16000
        if (
            trace["operators"].get("forward:aten.mv.default") != expected_mv
            or trace["operators"].get("adam:aten.sub_.Tensor")
            != (20 if fit.condition == "dense" else 38) * 1000
        ):
            return False
        return True
    except (KeyError, ValueError, TypeError, AttributeError):
        return False


def matching_totals(record: dict[str, Any]) -> tuple[int, int]:
    forward = int(record["trace"]["work"]["forward"] + record["python_scalar_forward_units"])
    total = sum(int(record["trace"]["work"][p]) for p in ("forward", "backward", "adam", "loss"))
    total += sum(
        int(record[k])
        for k in (
            "python_scalar_forward_units",
            "python_scalar_adam_bias_correction_units",
            "python_scalar_adam_step_units",
        )
    )
    return forward, total


def matched_reports(left: dict[str, Any], right: dict[str, Any]) -> str:
    if not valid_measurement(left) or not valid_measurement(right):
        return "INCONCLUSIVE"
    if (left["used_parameters"], right["used_parameters"]) != (99913, 99984):
        return "INCONCLUSIVE"
    for name in (
        "state_floats",
        "schedule",
        "data",
        "labels",
        "order",
        "initialization",
        "model_source",
        "optimizer",
        "updates",
        "batch",
        "observed_mask_encodings",
    ):
        if left[name] != right[name]:
            return "INCONCLUSIVE"
    if left["provenance"] != right["provenance"]:
        # Volatile timing origins are reported but do not define runtime identity.
        keys = (
            "cpu",
            "os",
            "python",
            "framework",
            "runtime_blas",
            "threads",
            "interop_threads",
            "deterministic",
            "device",
            "cuda",
        )
        if any(left["provenance"][k] != right["provenance"][k] for k in keys):
            return "INCONCLUSIVE"
    a, x = matching_totals(left)
    b, y = matching_totals(right)
    if max(a, b) / min(a, b) > 1.1 or max(x, y) / min(x, y) > 1.1:
        return "INCONCLUSIVE"
    return "MATCHED_OBSERVED_DISPATCH_ONLY"
