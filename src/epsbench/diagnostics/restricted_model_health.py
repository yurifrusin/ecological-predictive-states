"""Restricted health only: canonical selected input, fixed child and durable parent.

No operation on import. Private archive ACLs and a later exact admission remain required.
"""

from __future__ import annotations

import math
import os
import struct
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from epsbench.diagnostics import restricted_health_runtime
from epsbench.diagnostics.restricted_health_runtime import OwnedProcess, filtered_environment
from epsbench.diagnostics.restricted_learning_contract import (
    ARCHITECTURE,
    CONDITIONS,
    Observation,
    boundaries,
    digest,
    sha,
)
from epsbench.diagnostics.restricted_learning_membership import MembershipLock
from epsbench.diagnostics.restricted_learning_retention import reject_reparse
from epsbench.diagnostics.restricted_learning_sampling import shuffled
from epsbench.diagnostics.restricted_model_export import Example, ReadOnlyExport, TrainingMaterial
from epsbench.diagnostics.restricted_model_resources import (
    SCALAR_CONVENTION,
    WorkTrace,
    observed_peak_bytes,
    provenance,
    scalar_work,
)
from epsbench.diagnostics.restricted_model_training import Adam, ModelSource, brier, update
from epsbench.diagnostics.restricted_models import RestrictedModel
from epsbench.diagnostics.visible_forecast_contract import _json, _keys
from epsbench.utils.canonical import canonical_json_bytes

VERSION = "EPS-HEALTH-ONLY-1"
SCHEDULE_DOMAIN = "M0-occupancy-development-v1/schedule-v1"
PACKET_LIMIT, RESULT_LIMIT = 32768, 4 * 1024**2
CPU_LIMIT, RAM_LIMIT, DISK_LIMIT = 2480, 4 * 1024**3, 1024**3
PLANNING_RESERVE_CPU, PACKAGING_RESERVE = 1120, 256 * 1024**2
UPDATES = 200
_USED = False


def schedules(lock: MembershipLock, seed: bytes) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Reproduce existing commitments; no split allocation/producer import."""
    if type(lock) is not MembershipLock or type(seed) is not bytes or len(seed) != 32:
        raise ValueError("committed membership and retained schedule bytes required")
    result = []
    for budget in (16, 64):
        identities = lock.nested_train if budget == 16 else tuple(g.identity for g in lock.train)
        counters = dict.fromkeys(identities, 0)
        orders = {
            g: shuffled(tuple(range(8)), SCHEDULE_DOMAIN + "/decisions/" + g, seed)
            for g in identities
        }
        output = []
        for cycle in range(16000 // budget):
            for g in shuffled(
                identities, SCHEDULE_DOMAIN + f"/budget-{budget}/cycle-{cycle}", seed
            ):
                k = orders[g][counters[g] % 8]
                output.append(f"{g}/prefix-{k // 4}/action-{k % 4}")
                counters[g] += 1
        if digest(canonical_json_bytes(output)) != dict(lock.schedule_commitments)[budget]:
            raise ValueError("reconstructed schedule differs from original commitment")
        result.append(tuple(output))
    return result[0], result[1]


def example_payload(example: Example) -> dict[str, Any]:
    if type(example) is not Example:
        raise ValueError("lawful selected Example required")
    return {
        "observations": [
            {
                "index": o.index,
                "handles": list(o.handles),
                "first_seen": list(o.first_seen),
                "masks": [m.hex() for m in o.masks],
                "present": list(o.present),
                "visible": list(o.visible),
                "executed": [str(v) for v in o.executed],
            }
            for o in example.observations
        ],
        "announced": list(example.announced),
        "truth": list(example.truth),
    }


def decode_example(p: dict[str, Any]) -> Example:
    _keys(p, {"observations", "announced", "truth"})
    if type(p["observations"]) is not list or len(p["observations"]) != 3:
        raise ValueError("three immutable observations required")
    observations = []
    for o in p["observations"]:
        _keys(o, {"index", "handles", "first_seen", "masks", "present", "visible", "executed"})
        if any(
            type(o[k]) is not list
            for k in ("handles", "first_seen", "masks", "present", "visible", "executed")
        ):
            raise ValueError("exact observation arrays required")
        if len(o["masks"]) != 3 or any(type(m) is not str or len(m) != 2048 for m in o["masks"]):
            raise ValueError("bounded hex masks required")
        masks = cast(tuple[bytes, bytes, bytes], tuple(bytes.fromhex(m) for m in o["masks"]))
        if [m.hex() for m in masks] != o["masks"]:
            raise ValueError("canonical masks required")
        if len(o["executed"]) != 3 or any(type(v) is not str for v in o["executed"]):
            raise ValueError("canonical rational command required")
        executed = cast(
            tuple[Fraction, Fraction, Fraction], tuple(Fraction(v) for v in o["executed"])
        )
        if [str(v) for v in executed] != o["executed"]:
            raise ValueError("canonical rational command required")
        observations.append(
            Observation(
                o["index"],
                tuple(o["handles"]),
                tuple(o["first_seen"]),
                masks,
                tuple(o["present"]),
                tuple(o["visible"]),
                boundaries(masks),
                executed,
            )
        )
    if type(p["announced"]) is not list or type(p["truth"]) is not list:
        raise ValueError("exact example arrays required")
    return Example(
        (observations[0], observations[1], observations[2]),
        tuple(p["announced"]),
        tuple(p["truth"]),
    )


def packet(material: TrainingMaterial, source: ModelSource, collection: tuple[str, str]) -> bytes:
    if (
        type(material) is not TrainingMaterial
        or material.budget != 16
        or type(source) is not ModelSource
    ):
        raise PermissionError("authenticated nested16 export and exact health source required")
    raw = canonical_json_bytes(
        {
            "version": VERSION,
            "architecture": ARCHITECTURE,
            "source": source.__dict__,
            "collection": list(collection),
            "seal_a": material.seal_a,
            "data": material.data_root,
            "schedule": material.schedule_root,
            "labels": material.labels_root,
            "ordinal": 0,
            "initialization": "init-0",
            "seed": material.initialization_values[0],
            "example": example_payload(material.examples[0]),
        }
    )
    parse_packet(raw, digest(raw))
    return raw


def parse_packet(raw: bytes, expected: str) -> tuple[Example, int, dict[str, Any]]:
    sha(expected)
    if type(raw) is not bytes or len(raw) > PACKET_LIMIT or digest(raw) != expected:
        raise ValueError("bounded authenticated packet required")
    p = _json(raw)
    _keys(
        p,
        {
            "version",
            "architecture",
            "source",
            "collection",
            "seal_a",
            "data",
            "schedule",
            "labels",
            "ordinal",
            "initialization",
            "seed",
            "example",
        },
    )
    _keys(p["source"], {"head", "tree", "launch"})
    ModelSource(**p["source"])
    if type(p["collection"]) is not list or len(p["collection"]) != 2:
        raise ValueError("distinct collection identity required")
    ModelSource(p["collection"][0], p["collection"][1], p["source"]["launch"])
    for name in ("seal_a", "data", "schedule", "labels"):
        sha(p[name])
    if (
        p["version"] != VERSION
        or p["architecture"] != ARCHITECTURE
        or type(p["ordinal"]) is not int
        or p["ordinal"] != 0
        or p["initialization"] != "init-0"
        or type(p["seed"]) is not int
        or not 0 <= p["seed"] < 2**64
    ):
        raise ValueError("fixed health selection/init/policy required")
    example = decode_example(p["example"])
    if canonical_json_bytes(p) != raw or example_payload(example) != p["example"]:
        raise ValueError("strict canonical packet required")
    return example, p["seed"], p


def floor(example: Example, condition: str) -> float:
    if condition not in CONDITIONS:
        raise ValueError("frozen health condition required")
    absent = [
        int(y) for i, y in enumerate(example.truth) if not example.observations[-1].visible[i]
    ]
    if condition != "memory-reset" or not absent:
        return 0.0
    mean = sum(absent) / len(absent)
    return mean * (1 - mean) / (2 if len(absent) < len(example.truth) else 1)


def criterion(initial: float, final: float, minimum: float) -> bool:
    if any(
        type(v) is not float or not math.isfinite(v) or not 0 <= v <= 1
        for v in (initial, final, minimum)
    ):
        raise ValueError("finite bounded losses/floor required")
    excess = initial - minimum
    return final <= minimum + 0.01 and (excess <= 0.01 or final - minimum <= 0.1 * excess)


def health_state(model: RestrictedModel, adam: Adam) -> str:
    """Ineligible explicit binary weights + both moments, never EPSMODEL1."""
    arrays = (*model.parameters(), *adam.m, *adam.v)
    raw = b"EPSHEALTH1" + struct.pack("<Q", adam.k)
    for p in arrays:
        if p.dtype != torch.float32 or p.device.type != "cpu" or not torch.isfinite(p).all():
            raise ValueError("finite health-only state required")
        raw += p.detach().numpy().astype("<f4").tobytes()
    return raw.hex()


def validate_state(value: str, condition: str) -> None:
    if type(value) is not str or len(value) > 3 * 100000 * 4 * 2 + 34:
        raise ValueError("bounded health-only state required")
    data = bytes.fromhex(value)
    count = 99984 if condition == "dense" else 99913
    if (
        data.hex() != value
        or data[:10] != b"EPSHEALTH1"
        or data[10:18] != struct.pack("<Q", UPDATES)
        or len(data) != 18 + 3 * count * 4
        or not np.isfinite(np.frombuffer(data[18:], dtype="<f4")).all()
    ):
        raise ValueError("complete finite ineligible health-only state required")


def learner(raw: bytes, expected: str, condition: str) -> bytes:
    """Later-admitted isolated child only; never called by source tests."""
    started_cpu, started_wall = time.process_time(), time.monotonic()
    if condition not in CONDITIONS:
        raise ValueError("frozen condition required")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    example, seed, p = parse_packet(raw, expected)
    model, trace = RestrictedModel(condition, seed), WorkTrace()
    adam = Adam(model)
    with trace:
        with torch.no_grad():
            initial = float(brier(model, example))
        for _ in range(UPDATES):
            if (
                time.process_time() - started_cpu > CPU_LIMIT
                or time.monotonic() - started_wall > CPU_LIMIT
            ):
                raise RuntimeError("cooperative health deadline")
            update(model, adam, (example,) * 16, trace)
            peak = observed_peak_bytes()
            if peak is not None and peak > RAM_LIMIT:
                raise RuntimeError("observed health process memory limit")
        trace.phase = "forward"
        with torch.no_grad():
            final = float(brier(model, example))
    result = {
        "version": VERSION,
        "packet": expected,
        "condition": condition,
        "updates": adam.k,
        "source": p["source"],
        "initial": initial,
        "final": final,
        "floor": floor(example, condition),
        "connected_finite": True,
        "state": health_state(model, adam),
        "trace": trace.payload(),
        "scalar": scalar_work(condition, UPDATES * 16 + 2, UPDATES),
        "scalar_convention": SCALAR_CONVENTION,
        "provenance": provenance(),
        "elapsed_cpu": time.process_time() - started_cpu,
        "elapsed_wall": time.monotonic() - started_wall,
        "peak_ram": observed_peak_bytes(),
    }
    return canonical_json_bytes(result)


def validate_result(raw: bytes, payload: bytes, condition: str) -> dict[str, Any]:
    if type(raw) is not bytes or len(raw) > RESULT_LIMIT:
        raise ValueError("bounded complete child result required")
    example, _, packet_data = parse_packet(payload, digest(payload))
    p = _json(raw)
    _keys(
        p,
        {
            "version",
            "packet",
            "condition",
            "updates",
            "source",
            "initial",
            "final",
            "floor",
            "connected_finite",
            "state",
            "trace",
            "scalar",
            "scalar_convention",
            "provenance",
            "elapsed_cpu",
            "elapsed_wall",
            "peak_ram",
        },
    )
    if (
        canonical_json_bytes(p) != raw
        or p["version"] != VERSION
        or p["packet"] != digest(payload)
        or p["condition"] != condition
        or type(p["updates"]) is not int
        or p["updates"] != UPDATES
        or p["source"] != packet_data["source"]
        or p["floor"] != floor(example, condition)
        or p["connected_finite"] is not True
    ):
        raise ValueError("fixed child/source/input/update/numerical binding differs")
    validate_state(p["state"], condition)
    criterion(p["initial"], p["final"], p["floor"])
    if (
        p["scalar"] != scalar_work(condition, UPDATES * 16 + 2, UPDATES)
        or type(p["scalar"]) is not dict
        or any(type(v) is not int for v in p["scalar"].values())
        or p["scalar_convention"] != SCALAR_CONVENTION
    ):
        raise ValueError("known full health scalar work required")
    for name in ("elapsed_cpu", "elapsed_wall"):
        if type(p[name]) is not float or not math.isfinite(p[name]) or p[name] <= 0:
            raise ValueError("positive inclusive child observations required")
    if p["peak_ram"] is not None and (
        type(p["peak_ram"]) is not int or not 0 < p["peak_ram"] <= RAM_LIMIT
    ):
        raise ValueError("honest process peak required")
    _keys(
        p["trace"],
        {
            "convention",
            "work",
            "arithmetic_by_operator",
            "operators",
            "unclassified",
            "temporary_output_bytes_cumulative_not_peak",
        },
    )
    if set(p["trace"]["work"]) != {"forward", "backward", "loss", "adam"} or any(
        type(v) is not int or v <= 0 for v in p["trace"]["work"].values()
    ):
        raise ValueError("all health tensor phases required")
    if p["trace"][
        "convention"
    ] != "executed-dispatch-v1; nonlinear unit; not CPU instructions" or any(
        type(p["trace"][k]) is not dict
        or any(type(n) is not str or type(v) is not int or v < 0 for n, v in p["trace"][k].items())
        for k in (
            "work",
            "arithmetic_by_operator",
            "operators",
            "unclassified",
            "temporary_output_bytes_cumulative_not_peak",
        )
    ):
        raise ValueError("closed nonnegative trace convention required")
    trace = p["trace"]
    if not trace["operators"] or any(
        sum(
            v for name, v in trace["arithmetic_by_operator"].items() if name.startswith(phase + ":")
        )
        != units
        for phase, units in trace["work"].items()
    ):
        raise ValueError("complete phase/operator work breakdown required")
    _keys(p["provenance"], set(provenance()))
    runtime = p["provenance"]
    if (
        any(type(runtime[n]) is not int for n in ("threads", "interop_threads"))
        or any(
            type(runtime[n]) is not float or not math.isfinite(runtime[n]) or runtime[n] < 0
            for n in ("cpu_seconds", "wall_seconds")
        )
        or runtime["device"] != "cpu"
        or runtime["cuda"] is not None
        or runtime["threads"] != 1
        or runtime["interop_threads"] != 1
        or runtime["deterministic"] is not True
        or any(
            type(runtime[n]) is not str or not runtime[n]
            for n in ("os", "python", "framework", "runtime_blas")
        )
    ):
        raise ValueError("complete deterministic CPU provenance required")
    return p


@dataclass(frozen=True)
class HealthAdmission:
    source: ModelSource
    archive: str
    seal_a: str
    precommit: str
    admission_cpu_debit: float
    environment_bytes: int
    retained_evidence_bytes: int
    packaging_reserve_bytes: int = PACKAGING_RESERVE

    def __post_init__(self) -> None:
        if type(self.source) is not ModelSource:
            raise ValueError("exact admitted health source required")
        for value in (self.archive, self.seal_a, self.precommit):
            sha(value)
        if (
            type(self.admission_cpu_debit) is not float
            or not math.isfinite(self.admission_cpu_debit)
            or self.admission_cpu_debit < 0
            or any(
                type(v) is not int or v <= 0
                for v in (
                    self.environment_bytes,
                    self.retained_evidence_bytes,
                    self.packaging_reserve_bytes,
                )
            )
        ):
            raise ValueError("retained measured allowance inputs and positive reserve required")
        if self.packaging_reserve_bytes != PACKAGING_RESERVE:
            raise ValueError("fixed256MiB review reserve required")
        if (
            self.admission_cpu_debit + 13 * 60 >= CPU_LIMIT
            or self.environment_bytes
            + self.retained_evidence_bytes
            + self.packaging_reserve_bytes
            + 8 * RESULT_LIMIT
            + 8 * 1024**2
            + 65536
            + PACKET_LIMIT
            >= DISK_LIMIT
        ):
            raise RuntimeError("prospective health allowance conflict; no launch")


@dataclass(frozen=True)
class ProcessOutcome:
    raw: bytes
    cpu: float | None
    wall: float


def child_process(
    payload: bytes, condition: str, directory: Path, remaining: float
) -> ProcessOutcome:
    environment = filtered_environment()
    ready, go = (directory / (condition + suffix) for suffix in ("-runtime.json", "-ack"))
    helper = Path(restricted_health_runtime.__file__).resolve()
    source = digest(helper.read_bytes())
    command = [
        sys.executable,
        str(helper),
        "--learner",
        str(ready),
        str(go),
        source,
        digest(payload),
        condition,
    ]
    started = time.monotonic()
    owned = None
    child = None
    record: dict[str, Any] = {"cpu": None, "exit": None, "identity_complete": False}
    try:
        with (
            (directory / f"{condition}.stdout").open("xb") as output,
            (directory / f"{condition}.stderr").open("xb") as error,
        ):
            child = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=output,
                stderr=error,
                cwd=directory,
                env=environment,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            owned = OwnedProcess(child, ready, remaining)
            restricted_health_runtime.publish(go, source.encode())
            child.communicate(payload, timeout=max(0.001, remaining - (time.monotonic() - started)))
            observation = owned.finish(max(0.001, remaining - (time.monotonic() - started)))
            record.update(observation)
            record.update(
                {
                    "exit": child.returncode,
                    "wall": time.monotonic() - started,
                    "identity_complete": True,
                }
            )
            if child.returncode != 0:
                raise RuntimeError("health child failed; terminal observations retained")
            if observation["peak"] is None or observation["peak"] > RAM_LIMIT:
                raise RuntimeError("actual owned process peak unavailable or over allowance")
    except BaseException as initiating:
        record["error_type"] = type(initiating).__name__
        record["cleanup_complete"] = False
        try:
            if owned is not None:
                record["owned_cleanup"] = owned.cleanup()
                record["cleanup_complete"] = record["owned_cleanup"]["complete"]
            if child is not None and child.poll() is None:
                with (
                    (directory / f"{condition}-cleanup.stdout").open("xb") as out,
                    (directory / f"{condition}-cleanup.stderr").open("xb") as err,
                ):
                    cleanup = subprocess.run(
                        ["taskkill", "/PID", str(child.pid), "/T", "/F"],
                        stdout=out,
                        stderr=err,
                        timeout=20,
                        check=False,
                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                    )
                    record["cleanup_exit"] = cleanup.returncode
                    child.wait(timeout=20)
                    record["launcher_tree_cleanup_complete"] = cleanup.returncode == 0
            if owned is not None:
                record["terminal_after_failure"] = owned.finish(0.001)
        except BaseException as cleanup_error:
            record["cleanup_error_type"] = type(cleanup_error).__name__
        raise
    finally:
        failing = sys.exc_info()[0] is not None
        try:
            _write(directory, condition + "-process.json", canonical_json_bytes(record))
        except BaseException:
            if not failing:
                raise
        finally:
            if owned is not None:
                owned.close()
    path = directory / f"{condition}.stdout"
    if path.stat().st_size > RESULT_LIMIT:
        raise ValueError("child result exceeds bound")
    raw = path.read_bytes()
    result = validate_result(raw, payload, condition)
    actual = next(p for p in record["processes"] if p["pid"] == record["ready"]["pid"])
    if actual["final"]["cpu"] + 1e-6 < max(
        result["elapsed_cpu"], result["provenance"]["cpu_seconds"]
    ):
        raise RuntimeError("actual interpreter lifetime CPU undercounts its self snapshots")
    return ProcessOutcome(raw, record["cpu"], time.monotonic() - started)


def _write(directory: Path, name: str, data: bytes) -> None:
    with (directory / name).open("xb") as output:
        output.write(data)
        output.flush()
        os.fsync(output.fileno())


def _environment_bytes() -> int:
    root = Path(sys.prefix).resolve()
    reject_reparse(root)
    total = 0
    for directory, children, files in os.walk(root, followlinks=False):
        for name in (*children, *files):
            path = Path(directory) / name
            reject_reparse(path)
        total += sum((Path(directory) / name).stat().st_size for name in files)
    return total


def _operate(
    output: Path,
    closed_parent: Path,
    admission: HealthAdmission,
    prepare: Callable[[], bytes],
    execute: Callable[[bytes, str, Path, float], ProcessOutcome],
) -> dict[str, Any]:
    """Fixed parent seam: tests inject only synthetic preparation/execution."""
    started_cpu, started_wall = 0.0, time.monotonic()
    for path in (output, closed_parent):
        reject_reparse(path)
    output, closed_parent = output.resolve(), closed_parent.resolve()
    if (
        output == closed_parent
        or output.is_relative_to(closed_parent)
        or closed_parent.is_relative_to(output)
    ):
        raise PermissionError("new output must be disjoint from whole closed package parent")
    output.mkdir(parents=False, exist_ok=False)
    completed, child_cpu, cpu_complete = [], 0.0, True
    try:
        _write(
            output,
            "pending.json",
            canonical_json_bytes(
                {
                    "version": VERSION,
                    "source": admission.source.__dict__,
                    "conditions": list(CONDITIONS),
                    "health_only": True,
                }
            ),
        )
        initial_environment = _environment_bytes()
        if initial_environment != admission.environment_bytes:
            raise ValueError("environment observation changed since exact admission")
        payload = prepare()
        _, _, p = parse_packet(payload, digest(payload))
        if p["source"] != admission.source.__dict__ or p["seal_a"] != admission.seal_a:
            raise ValueError("prepared packet differs from independently admitted source/SealA")
        _write(output, "packet.json", payload)
        for condition in CONDITIONS:
            remaining = min(
                CPU_LIMIT
                - admission.admission_cpu_debit
                - time.process_time()
                + started_cpu
                - child_cpu,
                CPU_LIMIT - (time.monotonic() - started_wall),
            )
            current_bytes = sum(path.stat().st_size for path in output.iterdir() if path.is_file())
            if (
                admission.environment_bytes
                + admission.retained_evidence_bytes
                + PACKAGING_RESERVE
                + current_bytes
                + 2 * RESULT_LIMIT
                + 65536
                > DISK_LIMIT
            ):
                raise RuntimeError("insufficient retained-byte allowance for next child")
            if remaining <= 13 * 60 / 4:
                raise RuntimeError("insufficient current inclusive allowance for next fixed child")
            outcome = execute(payload, condition, output, remaining)
            if (
                type(outcome) is not ProcessOutcome
                or type(outcome.wall) is not float
                or not math.isfinite(outcome.wall)
                or outcome.wall <= 0
                or (
                    outcome.cpu is not None
                    and (
                        type(outcome.cpu) is not float
                        or not math.isfinite(outcome.cpu)
                        or outcome.cpu <= 0
                    )
                )
            ):
                raise ValueError("honest inclusive process observations required")
            cpu_complete = cpu_complete and outcome.cpu is not None
            child_cpu += outcome.cpu or 0.0
            raw = outcome.raw
            _write(output, condition + ".json", raw)
            result = validate_result(raw, payload, condition)
            if outcome.cpu is not None and outcome.cpu + 1e-6 < result["elapsed_cpu"]:
                raise ValueError("kernel lifetime CPU cannot undercount learner phase")
            completed.append(condition)
            parent_cpu = time.process_time() - started_cpu
            if (
                admission.admission_cpu_debit + parent_cpu + child_cpu > CPU_LIMIT
                or time.monotonic() - started_wall > CPU_LIMIT
            ):
                raise RuntimeError("inclusive health allowance exhausted")
            total = sum(path.stat().st_size for path in output.iterdir() if path.is_file())
            if (
                admission.environment_bytes
                + admission.retained_evidence_bytes
                + admission.packaging_reserve_bytes
                + total
                > DISK_LIMIT
            ):
                raise RuntimeError("inclusive logical health disk allowance exhausted")
            if not criterion(result["initial"], result["final"], result["floor"]):
                raise RuntimeError("fixed final health criterion failed; no retry")
            if (
                not cpu_complete
                or result["peak_ram"] is None
                or result["trace"]["unclassified"]
                or not result["provenance"]["cpu"]
            ):
                break  # Retain INCONCLUSIVE; do not run more costly children with missing evidence.
        terminal_environment = _environment_bytes()
        if (
            terminal_environment
            + admission.retained_evidence_bytes
            + admission.packaging_reserve_bytes
            + sum(path.stat().st_size for path in output.iterdir() if path.is_file())
            + 65536
            > DISK_LIMIT
        ):
            raise RuntimeError("terminal environment/evidence allowance exhausted")
        parent_peak = observed_peak_bytes()
        if parent_peak is not None and parent_peak > RAM_LIMIT:
            raise RuntimeError("observed evaluator memory planning limit")
        results = [_json((output / (c + ".json")).read_bytes()) for c in completed]
        status = (
            "HEALTH_READY"
            if len(completed) == len(CONDITIONS)
            and cpu_complete
            and parent_peak is not None
            and all(
                r["peak_ram"] is not None
                and not r["trace"]["unclassified"]
                and bool(r["provenance"]["cpu"])
                for r in results
            )
            else "INCONCLUSIVE"
        )
        report = {
            "version": VERSION,
            "status": status,
            "health_only": True,
            "completed": completed,
            "remaining": [c for c in CONDITIONS if c not in completed],
            "admission": admission.__dict__ | {"source": admission.source.__dict__},
            "parent_cpu": time.process_time() - started_cpu,
            "child_cpu": child_cpu,
            "child_cpu_complete": cpu_complete,
            "original_actual_health_allocation": 3600,
            "successor_cpu_ceiling": CPU_LIMIT,
            "actual_health_cpu_ceiling": CPU_LIMIT,
            "debited_planning_reserve_cpu": PLANNING_RESERVE_CPU,
            "historical_wp2_cpu": "UNKNOWN",
            "whole_wp2_resource_compliance": "INCONCLUSIVE",
            "elapsed_wall": time.monotonic() - started_wall,
            "parent_peak_ram": parent_peak,
            "environment_bytes_initial": initial_environment,
            "environment_bytes_terminal": terminal_environment,
            "files": {
                path.name: digest(path.read_bytes()) for path in output.iterdir() if path.is_file()
            },
            "payload_bytes_before_manifest": sum(
                path.stat().st_size for path in output.iterdir() if path.is_file()
            ),
        }
        _write(output, "completed.json", canonical_json_bytes(report))
        final_bytes = sum(path.stat().st_size for path in output.iterdir() if path.is_file())
        if (
            terminal_environment
            + admission.retained_evidence_bytes
            + admission.packaging_reserve_bytes
            + final_bytes
            + 1024
            > DISK_LIMIT
            or admission.admission_cpu_debit + time.process_time() - started_cpu + child_cpu
            > CPU_LIMIT
        ):
            raise RuntimeError("post-retention inclusive allowance exhausted")
        _write(
            output,
            "ready.json",
            canonical_json_bytes(
                {
                    "status": status,
                    "final_payload_bytes_before_ready": final_bytes,
                    "parent_cpu_after_manifest_retention": time.process_time() - started_cpu,
                    "wall_after_manifest_retention": time.monotonic() - started_wall,
                }
            ),
        )
        return report
    except BaseException as error:
        try:
            _write(
                output,
                "failure.json",
                canonical_json_bytes(
                    {
                        "version": VERSION,
                        "status": "FAILED_CLOSED",
                        "completed": completed,
                        "error_type": type(error).__name__,
                        "aggregate_root": None,
                        "cpu_complete": False,
                        "parent_cpu": time.process_time() - started_cpu,
                        "reported_child_cpu": child_cpu,
                        "elapsed_wall": time.monotonic() - started_wall,
                    }
                ),
            )
        except BaseException:
            pass  # Existing pending/partial bytes remain; preserve initiating exception.
        raise


def run(
    output: Path,
    closed_parent: Path,
    qualification: Path,
    admission: HealthAdmission,
    health_decision: bytes,
    precommit: bytes,
    schedule_seed: bytes,
    initializations: tuple[int, int, int],
) -> dict[str, Any]:
    """Evaluator only, later exact admission required. Never called by source checks."""
    global _USED
    if _USED:
        raise PermissionError("one health invocation per fresh evaluator process")
    _USED = True

    def prepare() -> bytes:
        if (
            digest(health_decision) != admission.source.launch
            or digest(precommit) != admission.precommit
        ):
            raise PermissionError("external exact health/precommit authority differs")
        reject_reparse(qualification)
        if not qualification.resolve().is_relative_to(closed_parent.resolve()):
            raise PermissionError("qualification must be inside whole protected package")
        source_root = Path(__file__).resolve().parents[3]
        for revision, expected in (
            ("HEAD", admission.source.head),
            ("HEAD^{tree}", admission.source.tree),
        ):
            actual = subprocess.check_output(
                ["git", "rev-parse", revision], cwd=source_root, text=True
            ).strip()
            if actual != expected:
                raise PermissionError("actual health source differs from exact admission")
        if subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=source_root
        ):
            raise PermissionError("health source must be clean")
        exporter = ReadOnlyExport(qualification, admission.archive, admission.seal_a)
        _write(output, "health-decision.bin", health_decision)
        _write(output, "precommit.bin", precommit)
        lock = exporter._lock
        p = _json(precommit)
        if (
            len(initializations) != 3
            or p["membership"] != lock.digest
            or type(schedule_seed) is not bytes
            or len(schedule_seed) != 32
            or len(p["seed_preimages"]) != 7
            or digest(schedule_seed) != p["seed_preimages"][2]
            or any(
                type(v) is not int
                or not 0 <= v < 2**64
                or digest(v.to_bytes(8, "little")) != p["seed_preimages"][4 + i]
                for i, v in enumerate(initializations)
            )
        ):
            raise PermissionError(
                "authenticated original retained schedule/init preimages required"
            )
        materials = exporter.export(schedules(lock, schedule_seed), initializations)
        _write(
            output,
            "selection-proof.json",
            canonical_json_bytes(
                {
                    "ordinal": 0,
                    "initialization": "init-0",
                    "membership": lock.digest,
                    "archive": admission.archive,
                    "precommit": admission.precommit,
                    "schedule_preimage": digest(schedule_seed),
                    "schedule_roots": dict(lock.schedule_commitments),
                    "initialization_commitments": dict(lock.initialization_commitments),
                }
            ),
        )
        return packet(materials[0], admission.source, (lock.source_head, lock.source_tree))

    return _operate(output, closed_parent, admission, prepare, child_process)


def main() -> int:
    if len(sys.argv) != 4 or sys.argv[1] != "--learner":
        raise PermissionError("no default operation; isolated packet entry only")
    raw = sys.stdin.buffer.read(PACKET_LIMIT + 1)
    result = learner(raw, sys.argv[2], sys.argv[3])
    sys.stdout.buffer.write(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
