"""One-shot execution infrastructure for the preserved eight-transition protocol.

This module grants no capture authority. Human receipts bind an exact executable
PR/head; their decisions are made outside this program. The ledger records only
operational events. There is deliberately no next/resume operation.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import ntpath
import os
import platform
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from epsbench.data.paths import open_owned_regular_file
from epsbench.diagnostics.revision_capture import AttemptLock, publish_bytes
from epsbench.utils.canonical import canonical_json_bytes

VERSION = "epsbench.topology_development.eight_transition.v1"
MEMBERSHIP_SHA256 = "aad141cf46d4efd7ad8694d4c0a8bf1468f30480fec231f77e6d2a5ca092a895"
PROTOCOL_SHA256 = "dfc078b8c6bae076754c6359018a3cace91eaf5d420ac7c0fa96d4fa602141ec"
LOCK_SHA256 = "d8fbbd09590dd2c937db822668d168ed73947d79e3772de7dce1e311b89ebafc"
BASENAME = "eps-topology-eight-transition-v1-20261003"
RUNTIME = {
    "python": "3.11.15",
    "mujoco": "3.12.0",
    "numpy": "2.4.6",
    "PyOpenGL": "3.1.10",
    "glfw": "2.10.2",
}
ENVIRONMENT = {
    "MUJOCO_GL": "osmesa",
    "PYOPENGL_PLATFORM": "osmesa",
    "LIBGL_ALWAYS_SOFTWARE": "1",
    "GALLIUM_DRIVER": "llvmpipe",
    "LP_NUM_THREADS": "2",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
}


class TopologyExecutionFailure(RuntimeError):
    """Integrity failure: preserve evidence and never relaunch this namespace."""


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def owned_read(path: Path) -> bytes:
    with open_owned_regular_file(path.parent, path.name) as item:
        return item.payload


class Envelope(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class EvidenceReference(Envelope):
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def verify(self) -> bytes:
        path = Path(self.path)
        if not path.is_absolute():
            raise TopologyExecutionFailure("evidence reference must be absolute")
        data = owned_read(path)
        if digest(data) != self.sha256:
            raise TopologyExecutionFailure("evidence reference byte identity changed")
        return data


class SourceBinding(Envelope):
    head: str = Field(pattern=r"^[0-9a-f]{40}$")
    tree: str = Field(pattern=r"^[0-9a-f]{40}$")
    origin: str
    files: dict[str, str]


class MachineReceipt(Envelope):
    schema_id: Literal["topology_eight_machine_preflight/v1"] = Field(alias="schema")
    windows_host: Literal["DESKTOP-TPUQMNG"]
    wsl_distribution: str = Field(min_length=1)
    source: str
    output_native: str
    output_wsl: str
    binding: SourceBinding
    runtime_versions: dict[str, str]
    installed_files: dict[str, str]
    osmesa_libraries: dict[str, str]
    operator: str = Field(min_length=1)
    controlled_writer_statement: str = Field(min_length=1)
    resource_agreement: str = Field(min_length=1)
    cpu_headroom: int = Field(ge=2)
    memory_headroom_bytes: int = Field(ge=8 * 1024**3)
    disk_headroom_bytes: int = Field(ge=1024**3)


class HumanReceipt(Envelope):
    schema_id: Literal["topology_eight_human_preflight/v1"] = Field(alias="schema")
    protocol: Literal["epsbench.topology_development.eight_transition.v1"]
    repository: Literal["yurifrusin/ecological-predictive-states"]
    pull_request: int = Field(gt=0)
    binding: SourceBinding
    operator: str = Field(min_length=1)
    owner: str = Field(min_length=1)
    decision: Literal["AUTHORISE_EXACT_FINITE_CAPTURE"]
    decision_text: str = Field(min_length=1)
    protocol_sha256: Literal["dfc078b8c6bae076754c6359018a3cace91eaf5d420ac7c0fa96d4fa602141ec"]
    membership_sha256: Literal["aad141cf46d4efd7ad8694d4c0a8bf1468f30480fec231f77e6d2a5ca092a895"]
    machine: EvidenceReference
    engineering_review: EvidenceReference
    scientific_review: EvidenceReference
    cpu_verification: EvidenceReference
    output_native: str
    output_wsl: str
    no_retry_acknowledged: Literal[True]
    negative_results_retained: Literal[True]
    disclosure_separate: Literal[True]


class LedgerRevision(Envelope):
    schema_id: Literal["topology_eight_operational_ledger/v1"] = Field(alias="schema")
    revision: int = Field(ge=0)
    predecessor_sha256: str | None
    event: Literal["initialised", "reserved", "complete", "released", "failed", "finished"]
    ordinal: int | None
    details: dict[str, Any]


def _git(source: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(source), *args], text=True).strip()


def source_binding(source: Path) -> SourceBinding:
    if source != source.resolve(strict=True) or _git(
        source, "status", "--porcelain=v1", "--untracked-files=all"
    ):
        raise TopologyExecutionFailure("source must be canonical and clean")
    names = _git(source, "ls-files").splitlines()
    files = {name: digest(owned_read(source / name)) for name in names}
    origin = _git(source, "remote", "get-url", "origin")
    if origin != "https://github.com/yurifrusin/ecological-predictive-states.git":
        raise TopologyExecutionFailure("source repository origin differs")
    return SourceBinding(
        head=_git(source, "rev-parse", "HEAD"),
        origin=origin,
        tree=_git(source, "rev-parse", "HEAD^{tree}"),
        files=files,
    )


def fixed_cells(source: Path) -> tuple[dict[str, Any], ...]:
    member = owned_read(source / "docs/protocols/topology-eight-transition-v1-membership.json")
    if digest(member) != MEMBERSHIP_SHA256:
        raise TopologyExecutionFailure("preserved membership byte identity differs")
    value = json.loads(member)
    records: list[dict[str, Any]] = value["records"]
    if len(records) != 8 or [r["ordinal"] for r in records] != list(range(8)):
        raise TopologyExecutionFailure("fixed membership order differs")
    return tuple(
        {
            **r,
            "name": f"{r['ordinal']:02d}-{r['direction']}-"
            f"{'base' if r['profile_id'] == 'legacy_solid_base_v1' else 'alternate'}-"
            f"r{r['repeat_index']}",
        }
        for r in records
    )


def prepare_plan(source: Path) -> dict[str, Any]:
    """Inert prospective source binding: no graphics, namespace or ledger creation."""
    binding = source_binding(source)
    if (
        digest(owned_read(source / "uv.lock")) != LOCK_SHA256
        or digest(owned_read(source / "docs/protocols/topology-eight-transition-v1.md"))
        != PROTOCOL_SHA256
    ):
        raise TopologyExecutionFailure("preserved lock/protocol differs")
    return {
        "schema": VERSION,
        "binding": binding.model_dump(mode="json"),
        "membership_sha256": MEMBERSHIP_SHA256,
        "protocol_sha256": PROTOCOL_SHA256,
        "cells": list(fixed_cells(source)),
        "limits": {"contexts": 8, "pairs": 48},
        "resources": {
            "cpus": 2,
            "memory_bytes": 8 * 1024**3,
            "output_bytes": 1024**3,
            "cell_seconds": 300,
            "total_seconds": 2700,
        },
        "environment": ENVIRONMENT,
        "runtime": RUNTIME,
    }


def runtime_files() -> dict[str, str]:
    """Hash installed SDK/Python artifacts without importing graphics packages."""
    executable = Path(sys.executable).resolve(strict=True)
    files = {str(executable): digest(owned_read(executable))}
    for package in RUNTIME:
        if package == "python":
            continue
        dist = importlib.metadata.distribution(package)
        for relative in dist.files or ():
            path = Path(str(dist.locate_file(relative)))
            if (
                path.suffix.lower() in {".py", ".so", ".dll", ".pyd", ".dylib"}
                or ".so." in path.name
            ):
                absolute = path.resolve(strict=True)
                files[str(absolute)] = digest(owned_read(absolute))
    return files


def validate_preflight(
    source: Path, root: Path, receipt_path: Path
) -> tuple[dict[str, Any], HumanReceipt]:
    """Bind completed human decisions; do not decide reviewer/owner approval here."""
    receipt = HumanReceipt.model_validate_json(owned_read(receipt_path))
    machine = MachineReceipt.model_validate_json(receipt.machine.verify())
    for item in (receipt.engineering_review, receipt.scientific_review, receipt.cpu_verification):
        item.verify()
    plan = prepare_plan(source)
    if (
        machine.binding != receipt.binding
        or receipt.binding.model_dump(mode="json") != plan["binding"]
    ):
        raise TopologyExecutionFailure("preflight exact source binding differs")
    if (
        machine.operator != receipt.operator
        or machine.source != str(source)
        or machine.output_native != receipt.output_native
        or machine.output_wsl != receipt.output_wsl
        or receipt.output_wsl != str(root)
    ):
        raise TopologyExecutionFailure("preflight operator/source/namespace bindings differ")
    if (
        platform.system() != "Linux"
        or not os.environ.get("WSL_INTEROP")
        or os.environ.get("WSL_DISTRO_NAME") != machine.wsl_distribution
    ):
        raise TopologyExecutionFailure("actual WSL runtime does not match target receipt")
    host = subprocess.check_output(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", "$env:COMPUTERNAME"],
        text=True,
    ).strip()
    if host != machine.windows_host:
        raise TopologyExecutionFailure("actual Windows host differs")
    native = receipt.output_native
    if (
        ntpath.splitdrive(native)[0].upper() != "C:"
        or ntpath.normpath(native) != native
        or ntpath.basename(native) != BASENAME
        or root.name != BASENAME
        or not str(root).startswith("/mnt/c/")
        or "\\" not in native
    ):
        raise TopologyExecutionFailure("output is not the fixed native C-drive namespace")
    translated = subprocess.check_output(["wslpath", "-u", native], text=True).strip()
    roundtrip = subprocess.check_output(["wslpath", "-w", str(root)], text=True).strip()
    if translated != str(root) or roundtrip != native:
        raise TopologyExecutionFailure("native/WSL namespace roundtrip differs")
    for parent in (root.parent, *root.parent.parents):
        if parent.is_symlink() or parent != parent.resolve(strict=True):
            raise TopologyExecutionFailure("linked output parent")
    if source == root or source in root.parents or root in source.parents:
        raise TopologyExecutionFailure("study namespace overlaps source")
    if root.exists() or root.is_symlink():
        raise TopologyExecutionFailure(
            "existing study namespace permanently rejects a new invocation"
        )
    if shutil.disk_usage(root.parent).free < 1024**3:
        raise TopologyExecutionFailure("actual output free space below envelope")
    available = next(
        (
            line.split()[1]
            for line in Path("/proc/meminfo").read_text().splitlines()
            if line.startswith("MemAvailable:")
        ),
        None,
    )
    if available is None or int(available) * 1024 < 8 * 1024**3:
        raise TopologyExecutionFailure("actual memory headroom below envelope")
    # Windows reparse aliases cannot be inferred solely from Linux lstat.
    parent_native = ntpath.dirname(native).replace("'", "''")
    command = (
        f"$p=Get-Item -LiteralPath '{parent_native}' -Force; "
        "while($null -ne $p){if(($p.Attributes -band 1024) -ne 0){exit 17};$p=$p.Parent}"
    )
    subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command], check=True
    )
    versions = {
        "python": platform.python_version(),
        **{key: importlib.metadata.version(key) for key in RUNTIME if key != "python"},
    }
    if (
        versions != RUNTIME
        or machine.runtime_versions != versions
        or runtime_files() != machine.installed_files
    ):
        raise TopologyExecutionFailure("installed runtime binding differs")
    if not machine.osmesa_libraries or any(
        digest(owned_read(Path(name))) != sha for name, sha in machine.osmesa_libraries.items()
    ):
        raise TopologyExecutionFailure("OSMesa installed library bindings absent or changed")
    if any(os.environ.get(key) != value for key, value in ENVIRONMENT.items()):
        raise TopologyExecutionFailure("graphics/thread environment differs before import")
    return plan, receipt


def inventory(root: Path) -> list[dict[str, Any]]:
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise TopologyExecutionFailure("linked output artifact")
        if path.is_dir():
            continue
        data = owned_read(path)
        files.append(
            {"path": path.relative_to(root).as_posix(), "bytes": len(data), "sha256": digest(data)}
        )
    return files


def _flush_directory(path: Path) -> None:
    if os.name != "posix":
        raise TopologyExecutionFailure("capture requires POSIX directory durability")
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def flush_artifact_tree(root: Path) -> None:
    """Flush saved producer and analysis files before certifying a cell terminal."""
    directories = [root]
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise TopologyExecutionFailure("linked artifact during durability barrier")
        if path.is_dir():
            directories.append(path)
        else:
            with open_owned_regular_file(root, path.relative_to(root).as_posix()) as item:
                os.fsync(item.handle.fileno())
    for directory in sorted(directories, key=lambda path: len(path.parts), reverse=True):
        _flush_directory(directory)


class OperationalLedger:
    """Only this live invocation owns append state; opening a saved ledger is impossible."""

    def __init__(self, root: Path, publisher: Callable[[Path, bytes], object] = publish_bytes):
        self.root = root
        self.publisher = publisher
        self.revision = 0
        self.previous: str | None = None
        self.records: list[LedgerRevision] = []
        self.stopped = False

    def verify_prefix(self) -> None:
        paths = sorted((self.root / "ledger").glob("*"))
        expected = [self.root / "ledger" / f"revision-{i:04d}.json" for i in range(self.revision)]
        if paths != expected:
            raise TopologyExecutionFailure("saved operational prefix differs")
        for path, record in zip(paths, self.records, strict=True):
            if owned_read(path) != canonical_json_bytes(
                record.model_dump(mode="json", by_alias=True)
            ):
                raise TopologyExecutionFailure("saved operational revision changed")

    def append(
        self,
        event: Literal["initialised", "reserved", "complete", "released", "failed", "finished"],
        ordinal: int | None,
        details: dict[str, Any],
    ) -> Path:
        if self.stopped:
            raise TopologyExecutionFailure("live invocation permanently stopped")
        self.verify_prefix()
        record = LedgerRevision(
            schema="topology_eight_operational_ledger/v1",
            revision=self.revision,
            predecessor_sha256=self.previous,
            event=event,
            ordinal=ordinal,
            details=details,
        )
        payload = canonical_json_bytes(record.model_dump(mode="json", by_alias=True))
        path = self.root / "ledger" / f"revision-{self.revision:04d}.json"
        self.publisher(path, payload)
        # A visible file is insufficient: advance only after the publisher returned.
        self.records.append(record)
        self.previous = digest(payload)
        self.revision += 1
        return path


def _execute_once(
    root: Path,
    plan: dict[str, Any],
    *,
    capture: Callable[[Mapping[str, Any], Path, Mapping[str, object] | None], dict[str, Any]],
    assess: Callable[[Path, Path, Path, Path], dict[str, Any]],
    compare: Callable[[Path, tuple[str, ...]], dict[str, Any]],
    recheck: Callable[[], None],
    phase_start: Callable[[Path, int], None],
    phase_complete: Callable[[Path, int], None],
    publisher: Callable[[Path, bytes], object] = publish_bytes,
    operator_records: Mapping[str, bytes] | None = None,
    flush_artifacts: Callable[[Path], None] = flush_artifact_tree,
) -> dict[str, Any]:
    """Internal injectable lifecycle. Only validated supervised entrypoints call it."""
    if root.exists() or root.is_symlink():
        raise TopologyExecutionFailure("existing namespace; no restart/resume")
    root.mkdir(exist_ok=False)
    ledger = OperationalLedger(root, publisher)
    active: AttemptLock | None = None
    baseline: Mapping[str, object] | None = None
    confirmed: list[tuple[str, list[dict[str, Any]], bytes]] = []

    def cell_inventory(name: str) -> list[dict[str, Any]]:
        result = []
        for relative in (f"datasets/{name}", f"exploratory/{name}"):
            if not (root / relative).is_dir():
                raise TopologyExecutionFailure("required cell artifact directory absent")
            for item in inventory(root / relative):
                result.append({**item, "path": f"{relative}/{item['path']}"})
        for relative in (f"privileged/{name}.json", f"inspections/{name}.png"):
            path = root / relative
            data = owned_read(path)
            result.append({"path": relative, "bytes": len(data), "sha256": digest(data)})
        return result

    def verify_confirmed() -> None:
        for name, files, payload in confirmed:
            if (
                cell_inventory(name) != files
                or owned_read(root / "receipts" / f"{name}.json") != payload
            ):
                raise TopologyExecutionFailure("confirmed cell evidence changed")

    try:
        publisher(root / "plan.json", canonical_json_bytes(plan))
        for name, payload in (operator_records or {}).items():
            if Path(name).name != name or name in {"", ".", ".."}:
                raise TopologyExecutionFailure("invalid operator receipt name")
            publisher(root / "operator" / name, payload)
        ledger.append("initialised", None, {"plan_sha256": digest(canonical_json_bytes(plan))})
        for cell in plan["cells"]:
            ordinal = cell["ordinal"]
            name = cell["name"]
            recheck()
            ledger.verify_prefix()
            verify_confirmed()
            phase_start(root, ordinal)
            for relative in (
                f"datasets/{name}",
                f"receipts/{name}.json",
                f"exploratory/{name}",
                f"privileged/{name}.json",
                f"inspections/{name}.png",
            ):
                if (root / relative).exists() or (root / relative).is_symlink():
                    raise TopologyExecutionFailure("cell output already exists")
            active = AttemptLock(root)
            active.acquire()
            ledger.append("reserved", ordinal, {"cell": name})
            native = capture(cell, root / "datasets" / name, baseline)
            if native["contexts"] != 1 or native["render_pairs"] != 6:
                raise TopologyExecutionFailure("native budget receipt differs")
            if baseline is None:
                baseline = native["runtime"]
            elif native["runtime"] != baseline:
                raise TopologyExecutionFailure("context runtime drift")
            result = assess(
                root / "datasets" / name,
                root / "exploratory" / name,
                root / "privileged" / f"{name}.json",
                root / "inspections" / f"{name}.png",
            )
            recheck()
            files = cell_inventory(name)
            payload = canonical_json_bytes(
                {
                    "schema": VERSION,
                    "ordinal": ordinal,
                    "native": native,
                    "assessment": result,
                    "files": files,
                }
            )
            publisher(root / "receipts" / f"{name}.json", payload)
            flush_artifacts(root)
            terminal = ledger.append("complete", ordinal, {"cell": name})
            active.release_after_success(terminal)
            active = None
            ledger.append("released", ordinal, {"cell": name})
            phase_complete(root, ordinal)
            confirmed.append((name, files, payload))
        verify_confirmed()
        controls = compare(root, tuple(cell["name"] for cell in plan["cells"]))
        publisher(root / "comparisons" / "controls.json", canonical_json_bytes(controls))
        recheck()
        flush_artifacts(root)
        ledger.append("finished", None, {"contexts": 8, "render_pairs": 48})
        publisher(
            root / "evidence-index.json",
            canonical_json_bytes({"schema": VERSION, "files": inventory(root)}),
        )
        return controls
    except BaseException as exc:
        failure = {
            "error_type": type(exc).__name__,
            "error": str(exc),
            "native_partial_receipt": getattr(exc, "native_receipt", None),
            "lock_present": (root / "capture-attempt.lock").exists(),
        }
        try:
            publisher(
                root / "operator" / "failure.json",
                canonical_json_bytes({"schema": VERSION, **failure}),
            )
        except BaseException:
            pass
        try:
            ledger.append(
                "failed",
                None,
                failure,
            )
        except BaseException:
            pass  # Outer supervisor receipt and an unconfirmed prefix remain authoritative.
        ledger.stopped = True
        if active is not None and active.fd is not None:
            os.close(active.fd)
            active.fd = None  # Never unlink a failed lock or fabricate an already-released one.
        raise


def worker(source: Path, root: Path, human: Path) -> dict[str, Any]:
    from epsbench.diagnostics.topology_eight_resources import (
        child_apply_limits,
        mark_cell_complete,
        mark_cell_start,
    )

    limits = child_apply_limits()
    os.chdir(source)
    plan, receipt = validate_preflight(source, root, human)
    machine = MachineReceipt.model_validate_json(receipt.machine.verify())
    from epsbench.diagnostics.topology_eight_assessment import assess_cell, compare_study
    from epsbench.diagnostics.topology_eight_native import capture_cell, validate_saved_native

    def recheck() -> None:
        if source_binding(source) != receipt.binding or any(
            os.environ.get(k) != v for k, v in ENVIRONMENT.items()
        ):
            raise TopologyExecutionFailure("source/environment changed during live invocation")
        if runtime_files() != machine.installed_files or any(
            digest(owned_read(Path(name))) != sha for name, sha in machine.osmesa_libraries.items()
        ):
            raise TopologyExecutionFailure("installed runtime bytes changed during invocation")

    def capture(
        cell: Mapping[str, Any], dataset: Path, baseline: Mapping[str, object] | None
    ) -> dict[str, Any]:
        from epsbench.data.identity import compute_source_provenance_hash
        from epsbench.data.loader import DatasetLoader
        from epsbench.data.provenance import collect_source_provenance
        from epsbench.schema import ModalityPermissionSet

        prospective = collect_source_provenance(source)
        record = {key: value for key, value in cell.items() if key != "name"}
        native = capture_cell(record, dataset, source=source, baseline=baseline)
        runtime = validate_saved_native(dataset, native)
        from epsbench.data.validate import validate_dataset

        manifest = validate_dataset(dataset)
        recorded = DatasetLoader(
            dataset, ModalityPermissionSet.all_modalities()
        ).read_dataset_manifest()
        if recorded != manifest:
            raise TopologyExecutionFailure("dataset changed during independent validation")
        if (
            manifest.source_provenance != prospective
            or manifest.source_provenance_sha256 != compute_source_provenance_hash(prospective)
        ):
            raise TopologyExecutionFailure("generated child source provenance differs")
        return {
            "contexts": len(native["contexts"]),
            "render_pairs": len(native["native_events"]),
            "runtime": runtime,
            "native_receipt": native,
        }

    plan["human_receipt_sha256"] = digest(owned_read(human))
    plan["enforced_child_limits"] = limits
    return _execute_once(
        root,
        plan,
        capture=capture,
        assess=assess_cell,
        compare=compare_study,
        recheck=recheck,
        phase_start=mark_cell_start,
        phase_complete=mark_cell_complete,
        operator_records={
            "human-preflight.json": owned_read(human),
            "machine-preflight.json": receipt.machine.verify(),
        },
    )
