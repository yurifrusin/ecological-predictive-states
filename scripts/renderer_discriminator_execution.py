"""Closed prospective renderer host/preparation/entry; no launch or SDK use on import."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import runpy
import subprocess
import sys
import threading
import time
from pathlib import Path, PureWindowsPath
from typing import Any, BinaryIO, cast

from epsbench.diagnostics import paired_appearance as p
from epsbench.diagnostics import renderer_discriminator as d
from epsbench.diagnostics.paired_appearance_execution import (
    DUMMY_CASES,
    exclusive_file,
    run_dummy,
    safe_path,
)
from epsbench.utils.canonical import canonical_json_bytes

REPOSITORY = Path(__file__).resolve().parents[1]
PREFIX = "EPS_RENDERER_DISCRIMINATOR_"
DECISION_ENV = PREFIX + "DECISION"
DEADLINE_ENV = PREFIX + "WORK_DEADLINE_UNIX"
LABEL = "eps.renderer-discriminator.owner"
BINDING_LABEL = "eps.renderer-discriminator.binding"
IMAGE_PREFIX = "eps.renderer-discriminator."
HOST_ACTUAL, HOST_DUMMY, RECEIPT_CAP = 768 * 1024, 256 * 1024, 16384
POLL = "{{.State.Running}} {{.State.ExitCode}} {{.State.OOMKilled}}"
CLEANUP = (
    '{"Id":{{json .Id}},"Name":{{json .Name}},'
    '"Config":{"Labels":{{json .Config.Labels}}},"Mounts":{{json .Mounts}}}'
)


class Commands:
    """Bounded external commands, not a service; deadlines do not attest daemon termination."""

    def __init__(self, docker: str, cap: int):
        self.docker, self.cap, self.used = docker, cap, 0
        self.cleanup_reserve = min(8192, cap // 2)

    def command(self, args: list[str], deadline: float, *, cleanup: bool = False) -> str:
        command_deadline = min(deadline, time.monotonic() + 15.0)
        remaining = command_deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("external command allowance exhausted")
        limit = self.cap if cleanup else self.cap - self.cleanup_reserve
        if cleanup and args[0] != "rm":
            limit -= 128  # Exact-ID removal output remains admitted after reinspection.
        if self.used >= limit:
            raise ValueError("command output allowance exhausted before launch")
        process = subprocess.Popen(
            [self.docker, *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
        )
        data = bytearray()
        overflow = threading.Event()
        charged = [0]

        def read(pipe: BinaryIO) -> None:
            try:
                while chunk := pipe.read(4096):
                    charged[0] += len(chunk)  # Include observed bytes discarded on overflow.
                    room = min(16384 - len(data), limit - self.used - len(data))
                    data.extend(chunk[: max(0, room)])
                    if len(chunk) > room:
                        overflow.set()
                        process.kill()
                        break
            except BaseException:
                overflow.set()
                process.kill()

        assert process.stdout is not None
        reader = threading.Thread(target=read, args=(process.stdout,), daemon=True)
        reader.start()
        try:
            process.wait(timeout=max(0.001, command_deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            process.kill()
            # No unbounded reap; a stalled host/daemon may leave cleanup incomplete.
            raise TimeoutError("bounded Docker client command timed out") from None
        finally:
            reader.join(timeout=max(0, min(1.0, command_deadline - time.monotonic())))
            if not reader.is_alive():
                process.stdout.close()
            self.used += charged[0]
            if reader.is_alive():
                self.used = max(self.used, limit)  # Unknown trailing bytes poison work capacity.
        if reader.is_alive() or overflow.is_set() or process.returncode != 0:
            raise RuntimeError("command failed/overflow/incomplete output")
        return data.decode("utf-8").strip()


def bounded_json(path: Path, cap: int = RECEIPT_CAP) -> dict[str, Any]:
    safe_path(path)
    if path.stat().st_size > cap:
        raise ValueError("document cap")
    return cast(dict[str, Any], p.strict_json(path.read_bytes()))


def task(binding: d.Binding, decision: d.Decision, case: str | None) -> None:
    if type(binding) is not d.Binding or type(decision) is not d.Decision:
        raise PermissionError("closed renderer typed documents required")
    d.require_decision(binding, decision)
    if (binding.purpose == d.DUMMY) != (case in DUMMY_CASES) or (
        binding.purpose == d.NATIVE and case is not None
    ):
        raise PermissionError("closed renderer native/dummy purpose")


def no_mixed(env: dict[str, str]) -> None:
    if any(
        k.startswith(("EPS_A1_", "EPS_CAUSAL_", "EPS_PAIRED_APPEARANCE_"))
        or k in ("WSL_INTEROP", "WSL_DISTRO_NAME")
        for k in env
    ):
        raise PermissionError("mixed old/new runtime denied")


def verify_source(repository: Path, binding: d.Binding | None = None) -> tuple[str, str]:
    safe_path(repository)
    commands = Commands("git", 3072)

    def git(args: list[str]) -> str:
        return commands.command(
            ["-c", "safe.directory=" + str(repository), "-C", str(repository), *args],
            time.monotonic() + 5,
        )

    head, tree = git(["rev-parse", "HEAD", "HEAD^{tree}"]).splitlines()
    if git(["status", "--porcelain"]) or git(["remote", "get-url", "origin"]) != d.SOURCE_URL:
        raise PermissionError("immutable clean source checkout required")
    if binding is not None and d.preparation(repository, head, tree) != binding.preparation:
        raise PermissionError("source/preparation differs before container creation")
    return head, tree


def image_labels(binding: d.Binding) -> dict[str, str]:
    return {
        IMAGE_PREFIX + k.replace("_", "-"): str(v)
        for k, v in binding.preparation.model_dump().items()
    }


def environment(binding: d.Binding, decision: d.Decision, wall: float) -> dict[str, str]:
    d.require_decision(binding, decision)
    if type(wall) is not float or not math.isfinite(wall):
        raise ValueError("finite inherited work deadline")
    transport = canonical_json_bytes(decision.model_dump(mode="json"))
    serialized = canonical_json_bytes(binding.model_dump(mode="json"))
    if len(transport) > RECEIPT_CAP or len(serialized) > RECEIPT_CAP:
        raise ValueError("typed environment transport cap")
    return {
        PREFIX + "RUNTIME": "docker_candidate_v1",
        PREFIX + "ROOT": binding.root,
        PREFIX + "IMAGE": binding.image,
        d.ENV: serialized.decode(),
        DECISION_ENV: transport.decode(),
        DEADLINE_ENV: str(wall),
        "MUJOCO_GL": "osmesa",
        "PYOPENGL_PLATFORM": "osmesa",
        "LP_NUM_THREADS": "2",
        "OMP_NUM_THREADS": "2",
        "PYTHONDONTWRITEBYTECODE": "1",
        "MESA_SHADER_CACHE_DISABLE": "true",
        "TMPDIR": "/tmp",
        "XDG_CACHE_HOME": "/tmp",
    }


def image_admission(info: dict[str, Any], binding: d.Binding) -> None:
    if info.get("Id") != binding.image or any(
        info.get("Config", {}).get("Labels", {}).get(k) != v
        for k, v in image_labels(binding).items()
    ):
        raise PermissionError("immutable image/source labels differ before create")
    env = info.get("Config", {}).get("Env", [])
    values = dict(v.split("=", 1) for v in env)
    no_mixed(values)
    if any(k.startswith(PREFIX) for k in values):
        raise PermissionError("prepared image cannot embed launch authority")


def create_arguments(
    binding: d.Binding, decision: d.Decision, output: Path, case: str | None, wall: float
) -> list[str]:
    task(binding, decision, case)
    if wall <= time.time():
        raise TimeoutError("work deadline already expired")
    args = [
        "create",
        "--pull=never",
        "--name=eps-renderer-" + binding.token,
        "--label=" + LABEL + "=" + binding.token,
        "--label=" + BINDING_LABEL + "=" + binding.root,
        "--read-only",
        "--network=none",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--log-driver=none",
        "--cpuset-cpus=0,1",
        "--cpus=2",
        "--pids-limit=64",
        "--memory=2g",
        "--memory-swap=2g",
        "--shm-size=1m",
        "--restart=no",
        "--tmpfs=/tmp:rw,noexec,nosuid,nodev,size=8m",
        "--mount",
        f"type=bind,src={output},dst=/output",
    ]
    for key, value in environment(binding, decision, wall).items():
        args.extend(("--env", key + "=" + value))
    args.extend(
        (binding.image, "python", "scripts/renderer_discriminator_execution.py", "--inside")
    )
    if case:
        args.extend(("--case", case))
    return args


def owned(info: dict[str, Any], binding: d.Binding, identity: str | None = None) -> None:
    if (
        type(info.get("Id")) is not str
        or re.fullmatch(r"[0-9a-f]{64}", info["Id"]) is None
        or (identity is not None and identity != info["Id"])
        or info.get("Name") != "/eps-renderer-" + binding.token
        or info.get("Config", {}).get("Labels", {}).get(LABEL) != binding.token
        or info.get("Config", {}).get("Labels", {}).get(BINDING_LABEL) != binding.root
    ):
        raise PermissionError("ownership mismatch; start/delete denied")


def mount_sources(output: Path) -> set[str]:
    accepted = {str(output)}
    windows = PureWindowsPath(output)
    if re.fullmatch(r"[A-Za-z]:", windows.drive):
        tail = windows.drive[0].lower() + "/" + "/".join(windows.parts[1:])
        accepted |= {"/run/desktop/mnt/host/" + tail, "/host_mnt/" + tail}
    return accepted


def sole_mount(info: dict[str, Any], output: Path) -> None:
    mounts = info.get("Mounts", [])
    if (
        len(mounts) != 1
        or mounts[0].get("Type") != "bind"
        or mounts[0].get("Source") not in mount_sources(output)
        or mounts[0].get("Destination") != "/output"
        or mounts[0].get("RW") is not True
    ):
        raise PermissionError("sole owned output mount differs")


def inspect_confinement(
    info: dict[str, Any],
    binding: d.Binding,
    decision: d.Decision,
    output: Path,
    case: str | None,
    wall: float,
) -> None:
    owned(info, binding)
    sole_mount(info, output)
    h = info["HostConfig"]
    definitions = h.get("Mounts", [])
    command = ["python", "scripts/renderer_discriminator_execution.py", "--inside"]
    if case:
        command.extend(("--case", case))
    env_list = info["Config"].get("Env", [])
    env = dict(v.split("=", 1) for v in env_list)
    no_mixed(env)
    expected = environment(binding, decision, wall)
    if len(env) != len(env_list) or {k for k in env if k.startswith(PREFIX)} != {
        k for k in expected if k.startswith(PREFIX)
    }:
        raise PermissionError("duplicate/unrecognized renderer environment")
    if (
        info["Image"] != binding.image
        or len(definitions) != 1
        or definitions[0].get("Type") != "bind"
        or definitions[0].get("Source") != str(output)
        or definitions[0].get("Target") != "/output"
        or definitions[0].get("ReadOnly", False) is not False
        or any(
            type(h.get(k)) is not int
            for k in ("Memory", "MemorySwap", "NanoCpus", "PidsLimit", "ShmSize")
        )
        or h.get("Binds")
        or h.get("Privileged") is not False
        or h.get("ReadonlyRootfs") is not True
        or h.get("NetworkMode") != "none"
        or h.get("CapDrop") != ["ALL"]
        or h.get("CapAdd")
        or "no-new-privileges" not in h.get("SecurityOpt", [])
        or h.get("LogConfig", {}).get("Type") != "none"
        or h.get("Memory") != 2 * 1024**3
        or h.get("MemorySwap") != 2 * 1024**3
        or h.get("NanoCpus") != 2_000_000_000
        or h.get("CpusetCpus") != "0,1"
        or h.get("PidsLimit") != 64
        or h.get("ShmSize") != p.MIB
        or h.get("Tmpfs") != {"/tmp": "rw,noexec,nosuid,nodev,size=8m"}
        or h.get("RestartPolicy", {}).get("Name") != "no"
        or info["Config"].get("Cmd") != command
        or info["Config"].get("Entrypoint")
        or any(env.get(k) != v for k, v in expected.items())
    ):
        raise PermissionError("pre-start confinement differs")
    if any(info["Config"].get("Labels", {}).get(k) != v for k, v in image_labels(binding).items()):
        raise PermissionError("prepared container labels differ")


def inherited_deadline(binding: d.Binding, wall: float) -> float:
    remaining = wall - time.time()
    ceiling = 275 if binding.purpose == d.NATIVE else 35
    if type(wall) is not float or not math.isfinite(wall) or not 0 < remaining <= ceiling:
        raise TimeoutError("expired/extended inherited work deadline")
    return time.monotonic() + remaining


def inside_runtime() -> None:
    """Actual candidate facts, equally required by dummy and native entry."""
    root = Path("/sys/fs/cgroup")
    affinity, filesystem = "sched_getaffinity", "statvfs"
    cpu = (root / "cpu.max").read_text().split()
    if (
        platform.system() != "Linux"
        or not Path("/.dockerenv").is_file()
        or (root / "memory.max").read_text().strip() != str(2 * 1024**3)
        or (root / "memory.swap.max").read_text().strip() != "0"
        or (root / "pids.max").read_text().strip() != "64"
        or len(cpu) != 2
        or int(cpu[1]) <= 0
        or int(cpu[0]) != 2 * int(cpu[1])
        or sorted(getattr(os, affinity)(0)) != [0, 1]
        or any(
            getattr(os, filesystem)(path).f_blocks * getattr(os, filesystem)(path).f_frsize
            != size * p.MIB
            for path, size in (("/tmp", 8), ("/dev/shm", 1))
        )
    ):
        raise PermissionError("inside fixed owned-container resource facts differ")


def inside(case: str | None) -> int:
    no_mixed(dict(os.environ))
    binding = d.environment_binding()
    raw = os.environ.get(DECISION_ENV, "").encode()
    if not raw or len(raw) > RECEIPT_CAP:
        raise PermissionError("bounded typed decision transport required")
    decision = d.Decision.model_validate(p.strict_json(raw))
    task(binding, decision, case)
    wall = float(os.environ[DEADLINE_ENV])
    expected = environment(binding, decision, wall)
    if {k for k in os.environ if k.startswith(PREFIX)} != {
        k for k in expected if k.startswith(PREFIX)
    } or any(os.environ.get(k) != v for k, v in expected.items()):
        raise PermissionError("inside exact environment differs")
    verify_source(REPOSITORY, binding)
    if bounded_json(Path("/preparation/renderer.json")) != binding.preparation.model_dump(
        mode="json"
    ):
        raise PermissionError("prepared manifest differs before native access")
    deadline = inherited_deadline(binding, wall)
    inside_runtime()
    root = Path("/output")
    safe_path(root)
    if not root.is_dir() or any(root.iterdir()):
        raise PermissionError("inside output must initially be empty")
    if binding.purpose == d.DUMMY:
        guard = runpy.run_path(str(REPOSITORY / "scripts/check_a1_source.py"))
        guard["check_loaded"]()
        sys.meta_path.insert(0, guard["Guard"]())
        exclusive_file(
            root / "consumed.json",
            canonical_json_bytes(
                {
                    "binding": binding.model_dump(mode="json"),
                    "decision": decision.model_dump(mode="json"),
                    "binding_root": binding.root,
                }
            ),
        )
        assert case is not None
        try:
            run_dummy(root, case, deadline)
        finally:
            guard["check_loaded"]()
        return 0
    sink = d.Sink(root, binding, decision)  # Sink consumes authority before any native constructor.
    capture = d.NativeCapture(REPOSITORY, binding, sink)
    result = d._drive(capture, sink, deadline)
    return 0 if result["status"] == "DIAGNOSTIC_COMPLETE" else 2


def measurement_summary(frames: list[d.Endpoint]) -> dict[str, Any]:
    """Only complete arms/controls supported by an independently verified prefix."""
    count = len(frames)
    arms = {
        arm: d.suitability(frames[i * 4], frames[i * 4 + 2])["status"]
        for i, arm in enumerate(d.ARMS)
        if count >= (i + 1) * 4
    }
    controls = []
    relations = []
    for left, right, relation in ((0, 2, "sampling_original"), (1, 3, "sampling_unit")):
        if count >= (right + 1) * 4:
            equal = (frames[left * 4].arrays["rgb"] == frames[right * 4].arrays["rgb"]).all()
            controls.append("CONTROL_SUPPORTED" if equal else "CONTROL_NOT_SUPPORTED")
            relations.append(relation)
    return {
        "measurement_status": "VERIFIED_COMPLETE" if count == 16 else "VERIFIED_PREFIX",
        "verified_measurement_endpoints": count,
        "arm_suitability": arms,
        "solid_sampling_controls": controls,
        "verified_control_relations": relations,
    }


def evidence_summary(
    output: Path, binding: d.Binding, assessed: dict[str, Any], count: int
) -> dict[str, Any]:
    result: dict[str, Any] = {"acquisition_status": assessed["status"], "observed_endpoints": count}
    if assessed["status"] == "DIAGNOSTIC_COMPLETE":
        result["arm_suitability"] = {k: v["status"] for k, v in assessed["arms"].items()}
        result["solid_sampling_controls"] = [
            v["status"] for v in assessed["solid_sampling_controls"]
        ]
    terminal = bounded_json(output / "terminal.json")
    report = bounded_json(output / "report.json", p.MIB)
    d.validate_ref(terminal["report"], "report.json")
    if terminal["binding_root"] != binding.root or terminal["status"] != report.get("status"):
        raise ValueError("terminal/report binding or status differs")
    if report.get("status") == "DIAGNOSTIC_COMPLETE":
        if (
            report != assessed
            or terminal["completed"] != 16
            or any(v != 16 for v in terminal["counts"].values())
        ):
            raise ValueError("complete retained report differs from recomputed evidence")
    elif (
        report.get("status") != "INCONCLUSIVE"
        or report.get("planned") != 16
        or type(report.get("endpoints")) is not int
        or not 0 <= report["endpoints"] <= count
        or report.get("absolute_native_uv_transfer_qualification") != "UNRESOLVED"
    ):
        raise ValueError("closed incomplete report differs")
    result["reported_status"] = report["status"]
    result["report_reference"] = terminal["report"]
    return result


def dummy_summary(output: Path, binding: d.Binding, decision: d.Decision, case: str) -> bool:
    anchor = bounded_json(output / "consumed.json")
    if anchor != {
        "binding": binding.model_dump(mode="json"),
        "decision": decision.model_dump(mode="json"),
        "binding_root": binding.root,
    }:
        raise ValueError("dummy consumed binding differs")
    expected = {"rgb.bin": b"tiny-completed-rgb", "pair.bin": b"tiny-completed-id-depth"}
    if case == "normal":
        expected["complete.json"] = b'{"native":false,"status":"COMPLETE"}'
    paths = {"consumed.json", *("dummy-" + case + "/" + name for name in expected)}
    files = set()
    total = 0
    for path in output.rglob("*"):
        safe_path(path)
        if path.is_file():
            name = path.relative_to(output).as_posix()
            files.add(name)
            total += path.stat().st_size
    if files != paths or total > 32 * 1024:
        raise ValueError("closed dummy retention inventory/cap")
    return all(
        (output / ("dummy-" + case) / name).read_bytes() == data for name, data in expected.items()
    )


def joint_budget(root: Path, current: Path, binding: d.Binding, command_bytes: int) -> None:
    """Count physical bytes and discarded-command receipts in the five fixed slots."""
    group_size = (root / "group.json").stat().st_size
    slots = root / "control"
    native_output: Path | None = None
    total = 0
    for slot in slots.iterdir():
        safe_path(slot)
        if not slot.is_dir() or slot.name not in (*DUMMY_CASES, "actual"):
            raise ValueError("closed control slot namespace")
        attempt = bounded_json(slot / "attempt.json")
        prior = d.Binding.model_validate(attempt["binding"])
        chosen = d.Decision.model_validate(attempt["decision"])
        task(prior, chosen, None if slot.name == "actual" else slot.name)
        if prior.preparation != binding.preparation or prior.image != binding.image:
            raise ValueError("control source/image group differs")
        cap = HOST_ACTUAL if slot.name == "actual" else HOST_DUMMY // 4
        if slot.name == "actual":
            native_output = root / prior.output_id
        for child in slot.iterdir():
            safe_path(child)
            if child.name not in (
                "attempt.json",
                "result.json",
                prior.output_id if slot.name != "actual" else "",
            ):
                raise ValueError("closed control component namespace")
        physical = 0
        for file in slot.rglob("*"):
            safe_path(file)
            if file.is_file():
                physical += file.stat().st_size
        if slot == current:
            charged = command_bytes
        elif (slot / "result.json").is_file():
            charged = bounded_json(slot / "result.json")["command_bytes_charged"]
            if type(charged) is not int or charged < 0:
                raise ValueError("typed historical command charge")
        else:
            # Unobservable earlier failed command output consumes the entire share.
            total += cap
            if physical + group_size > cap:
                raise ValueError("physical failed host share")
            continue
        billed = physical + charged + group_size + 6144
        if billed > cap:
            raise ValueError("joint control-slot byte cap")
        total += billed
    if total > p.MIB:
        raise ValueError("joint host byte cap")
    allowed = {root / "group.json", slots}
    if native_output is not None:
        allowed.add(native_output)
    for child in root.iterdir():
        safe_path(child)
        if child not in allowed:
            raise ValueError("closed execution group namespace")
    if native_output is not None and native_output.exists():
        original = 0
        for file in native_output.rglob("*"):
            safe_path(file)
            if file.is_file():
                original += file.stat().st_size
        if original > 18 * p.MIB:
            raise ValueError("separate original byte cap")


def host_run(
    binding: d.Binding,
    decision: d.Decision,
    root: Path,
    commands: Commands,
    case: str | None = None,
) -> dict[str, Any]:
    task(binding, decision, case)
    started = time.monotonic()
    no_mixed(dict(os.environ))
    verify_source(REPOSITORY, binding)
    safe_path(root)
    if any(c in str(root) for c in (",", "\n", "\r")):
        raise PermissionError("ambiguous output bind path")
    root.mkdir(exist_ok=True)
    group = canonical_json_bytes(
        {"preparation": binding.preparation.model_dump(mode="json"), "image": binding.image}
    )
    group_path = root / "group.json"
    if group_path.exists():
        if bounded_json(group_path) != p.strict_json(group):
            raise PermissionError("execution group differs")
    else:
        exclusive_file(group_path, group)
    control = root / "control"
    control.mkdir(exist_ok=True)
    slot = control / (case or "actual")
    slot.mkdir()  # Collision is terminal before any provider/create call.
    allowance, work = (60, 35) if case else (300, 275)
    final_deadline, work_deadline = started + allowance, started + work
    wall = time.time() + max(0, work_deadline - time.monotonic())
    output = slot / binding.output_id if case else root / binding.output_id
    attempt = canonical_json_bytes(
        {
            "binding": binding.model_dump(mode="json"),
            "decision": decision.model_dump(mode="json"),
            "nonrefundable": True,
            "seconds": allowance,
        }
    )
    exclusive_file(slot / "attempt.json", attempt)
    joint_budget(root, slot, binding, commands.used)
    result: dict[str, Any] = {
        "status": "INCONCLUSIVE",
        "acquisition_status": "INCONCLUSIVE",
        "measurement_status": "UNVERIFIED",
        "bundle_integrity_status": "INCONCLUSIVE",
        "binding_root": binding.root,
        "phase_gate_effect": "NONE",
    }
    identity = "eps-renderer-" + binding.token
    cleaned, exited, completed, created = False, False, False, False
    try:
        output.mkdir()  # The sole mount is empty; host never writes the native consumed anchor.
        image = json.loads(
            commands.command(
                ["image", "inspect", "--format", "{{json .}}", binding.image], work_deadline
            )
        )
        image_admission(image, binding)
        created = True  # Even a lost create reply requires exact name/label reinspection.
        created_id = commands.command(
            create_arguments(binding, decision, output, case, wall), work_deadline
        )
        if re.fullmatch(r"[0-9a-f]{64}", created_id) is None:
            raise ValueError("lost/malformed create identity; exact-name cleanup required")
        identity = created_id
        info = json.loads(
            commands.command(["inspect", "--format", "{{json .}}", identity], work_deadline)
        )
        owned(info, binding, identity)
        inspect_confinement(info, binding, decision, output, case, wall)
        commands.command(["start", identity], work_deadline)
        while time.monotonic() < work_deadline:
            fields = commands.command(
                ["inspect", "--format", POLL, identity], work_deadline
            ).split()
            if (
                len(fields) != 3
                or fields[0] not in ("true", "false")
                or fields[2] not in ("true", "false")
                or re.fullmatch(r"-?[0-9]+", fields[1]) is None
            ):
                raise ValueError("closed compact container state")
            if fields[0] == "false":
                exited = True
                completed = fields[1] == "0" and fields[2] == "false"
                result["exit_state"] = {"code": int(fields[1]), "oom": fields[2] == "true"}
                break
            time.sleep(min(0.5, max(0, work_deadline - time.monotonic())))
        else:
            raise TimeoutError("work watchdog expired")
    except Exception as error:
        result["reason"] = str(error)[:1024]
    finally:
        cleanup_deadline = min(final_deadline - 5, time.monotonic() + 20)
        if created:
            try:
                info = json.loads(
                    commands.command(
                        ["inspect", "--format", CLEANUP, identity], cleanup_deadline, cleanup=True
                    )
                )
                owned(info, binding, identity if re.fullmatch(r"[0-9a-f]{64}", identity) else None)
                sole_mount(info, output)
                commands.command(["rm", "--force", info["Id"]], cleanup_deadline, cleanup=True)
                cleaned = True
            except Exception as error:
                result["cleanup_error"] = str(error)[:1024]
        receipt_deadline = min(final_deadline, time.monotonic() + 5)
        try:
            if not (exited or cleaned):
                raise PermissionError("owned termination unconfirmed; live evidence not admitted")
            if time.monotonic() >= receipt_deadline:
                raise TimeoutError("receipt allowance exhausted")
            if case:
                observed = dummy_summary(output, binding, decision, case)
                expected_exit = completed if case == "normal" else not completed
                if observed and expected_exit:
                    result["dummy_observation"] = "DUMMY_OBSERVED"
            else:
                # Retain independently valid measurements even if cleanup/exit/report later failed.
                frames = d.reconstruct_measurements(output, binding)
                result.update(measurement_summary(frames))
                # Final acceptance is separate; failure cannot erase verified measurements.
                d.verify_final_receipt(output, binding, len(frames))
                assessed = d.assess(frames, binding)
                result.update(evidence_summary(output, binding, assessed, len(frames)))
                result["bundle_integrity_status"] = "VERIFIED"
        except Exception as error:
            result["retention_error"] = str(error)[:1024]
        result["cleaned"] = cleaned
        operational = (
            cleaned
            and completed
            and (bool(case) or result.get("reported_status") == "DIAGNOSTIC_COMPLETE")
            and not any(k in result for k in ("reason", "retention_error", "cleanup_error"))
        )
        result["operational_status"] = "COMPLETE" if operational else "INCONCLUSIVE"
        if operational and result.get("reported_status") == "DIAGNOSTIC_COMPLETE":
            result["status"] = "DIAGNOSTIC_COMPLETE"
        if case and cleaned and result.get("dummy_observation") == "DUMMY_OBSERVED":
            result["status"] = "DUMMY_OBSERVED"
        result["elapsed_seconds"] = time.monotonic() - started
        result["overrun"] = (
            result["elapsed_seconds"] > allowance or time.monotonic() > receipt_deadline
        )
        if result["overrun"]:
            result["status"] = result["operational_status"] = "INCONCLUSIVE"
        result["command_bytes_charged"] = commands.used
        result["command_timeline_retained"] = False
        data = canonical_json_bytes(result)
        cap = HOST_DUMMY // 4 if case else HOST_ACTUAL
        existing = 0
        for path in slot.rglob("*"):
            safe_path(path)
            if path.is_file():
                existing += path.stat().st_size
        if (
            len(data) > RECEIPT_CAP
            or existing + len(data) + commands.used + len(group) + 6144 > cap
        ):
            raise ValueError("host/command/source-byte allowance exhausted; receipt not rewritten")
        joint_budget(root, slot, binding, commands.used + len(data))
        exclusive_file(slot / "result.json", data)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--prepare", action="store_true")
    modes.add_argument("--inside", action="store_true")
    parser.add_argument("--case", choices=DUMMY_CASES)
    parser.add_argument("--binding", type=Path)
    parser.add_argument("--decision", type=Path)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--docker", default="docker")
    args = parser.parse_args()
    if args.prepare or args.inside:
        if args.binding or args.decision or args.root:
            raise PermissionError("host documents cannot enter preparation/inside modes")
        if args.prepare:
            if args.case:
                raise PermissionError("preparation cannot select a case")
            head, tree = verify_source(REPOSITORY)
            if (head, tree) != (os.environ["SOURCE_HEAD"], os.environ["SOURCE_TREE"]):
                raise PermissionError("prepared source identity differs")
            manifest = d.preparation(REPOSITORY, head, tree)
            exclusive_file(
                Path("/preparation/renderer.json"),
                canonical_json_bytes(manifest.model_dump(mode="json")),
            )
            return 0
        return inside(args.case)
    if not args.binding or not args.decision or not args.root:
        raise PermissionError("explicit binding/decision/group root required")
    binding = d.Binding.model_validate(bounded_json(args.binding.absolute()))
    decision = d.Decision.model_validate(bounded_json(args.decision.absolute()))
    result = host_run(
        binding,
        decision,
        args.root.absolute(),
        Commands(args.docker, 16384 if args.case else HOST_ACTUAL - 32768),
        args.case,
    )
    print(canonical_json_bytes(result).decode())
    return 0 if result["status"] in ("DIAGNOSTIC_COMPLETE", "DUMMY_OBSERVED") else 2


if __name__ == "__main__":
    raise SystemExit(main())
