"""Thin prospective appearance host/entry. No Docker or native invocation on import."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import runpy
import subprocess
import sys
import threading
import time
from pathlib import Path, PureWindowsPath
from typing import Any, BinaryIO

from epsbench.diagnostics import paired_appearance as p
from epsbench.diagnostics.appearance_study import CONTRAST_V1, PAIRED_V1, select_study
from epsbench.diagnostics.paired_appearance_execution import (
    DUMMY_CASES,
    HOST_ACTUAL,
    consume_attempt,
    consumed,
    exclusive_file,
    replay,
    run_dummy,
    run_native,
    safe_path,
)
from epsbench.diagnostics.paired_appearance_execution import (
    ready as e_ready,
)
from epsbench.diagnostics.paired_appearance_runtime import (
    AppearanceExecutionBinding,
    LaunchDecision,
    environment_binding,
    parse,
    preparation,
    require_decision,
)
from epsbench.utils.canonical import canonical_json_bytes

LABEL = "eps.paired-appearance.owner"
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

        def read(pipe: BinaryIO) -> None:
            while chunk := pipe.read(4096):
                room = min(16384 - len(data), limit - self.used - len(data))
                data.extend(chunk[: max(0, room)])
                if len(chunk) > room:
                    overflow.set()
                    process.kill()
                    break

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
            self.used += len(data)
        if reader.is_alive() or overflow.is_set() or process.returncode != 0:
            raise RuntimeError("command failed/overflow/incomplete output")
        return data.decode("utf-8").strip()


def environment(binding: AppearanceExecutionBinding) -> dict[str, str]:
    return {
        "EPS_PAIRED_APPEARANCE_RUNTIME": "docker_candidate_v1",
        "EPS_PAIRED_APPEARANCE_PURPOSE": binding.purpose,
        "EPS_PAIRED_APPEARANCE_SOURCE_HEAD": binding.preparation.source_head,
        "EPS_PAIRED_APPEARANCE_SOURCE_TREE": binding.preparation.source_tree,
        "EPS_PAIRED_APPEARANCE_IMAGE": binding.image,
        "EPS_PAIRED_APPEARANCE_ROOT": binding.root,
        "EPS_PAIRED_APPEARANCE_BINDING": canonical_json_bytes(
            binding.model_dump(mode="json")
        ).decode(),
        "MUJOCO_GL": "osmesa",
        "PYOPENGL_PLATFORM": "osmesa",
        "LP_NUM_THREADS": "2",
        "OMP_NUM_THREADS": "2",
        "PYTHONDONTWRITEBYTECODE": "1",
        "MESA_SHADER_CACHE_DISABLE": "true",
        "TMPDIR": "/tmp",
        "XDG_CACHE_HOME": "/tmp",
    }


def create_arguments(
    binding: AppearanceExecutionBinding, output: Path, case: str | None, work_deadline_unix: float
) -> list[str]:
    if (binding.purpose == binding.preparation.study.dummy_purpose) != (case in DUMMY_CASES):
        raise ValueError("native/dummy task mismatch")
    args = [
        "create",
        "--pull=never",
        "--name=eps-appearance-" + binding.token,
        "--label=" + LABEL + "=" + binding.token,
        "--label=eps.paired-appearance.binding=" + binding.root,
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
    if not math.isfinite(work_deadline_unix) or work_deadline_unix <= time.time():
        raise ValueError("nonresetting external work deadline required")
    env = {
        **environment(binding),
        "EPS_PAIRED_APPEARANCE_WORK_DEADLINE_UNIX": str(work_deadline_unix),
    }
    for key, value in env.items():
        args.extend(("--env", key + "=" + value))
    args.extend((binding.image, "python", "scripts/paired_appearance_execution.py", "--inside"))
    if case:
        args.extend(("--case", case))
    return args


def owned(
    info: dict[str, Any], binding: AppearanceExecutionBinding, identity: str | None = None
) -> None:
    if (
        type(info.get("Id")) is not str
        or re.fullmatch(r"[0-9a-f]{64}", info["Id"]) is None
        or (identity is not None and info["Id"] != identity)
        or info.get("Name") != "/eps-appearance-" + binding.token
        or info.get("Config", {}).get("Labels", {}).get(LABEL) != binding.token
        or info.get("Config", {}).get("Labels", {}).get("eps.paired-appearance.binding")
        != binding.root
    ):
        raise PermissionError("ownership mismatch; delete/start denied")


def inspect_confinement(
    info: dict[str, Any],
    binding: AppearanceExecutionBinding,
    output: Path,
    case: str | None,
    work_deadline_unix: float,
) -> None:
    owned(info, binding)
    host = info["HostConfig"]
    mounts = info["Mounts"]
    accepted = {str(output)}
    windows = PureWindowsPath(output)
    if re.fullmatch(r"[A-Za-z]:", windows.drive):
        tail = windows.drive[0].lower() + "/" + "/".join(windows.parts[1:])
        accepted |= {"/run/desktop/mnt/host/" + tail, "/host_mnt/" + tail}
    definitions = host.get("Mounts", [])
    command = ["python", "scripts/paired_appearance_execution.py", "--inside"]
    if case:
        command.extend(("--case", case))
    env = info["Config"].get("Env", [])
    env_map = dict(v.split("=", 1) for v in env)
    if (
        info["Image"] != binding.image
        or len(mounts) != 1
        or len(definitions) != 1
        or definitions[0].get("Type") != "bind"
        or definitions[0].get("Source") != str(output)
        or definitions[0].get("Target") != "/output"
        or definitions[0].get("ReadOnly", False) is not False
        or mounts[0].get("Type") != "bind"
        or mounts[0].get("Source") not in accepted
        or mounts[0].get("Destination") != "/output"
        or mounts[0].get("RW") is not True
        or host.get("Binds")
        or host.get("Privileged") is not False
        or host.get("ReadonlyRootfs") is not True
        or host.get("NetworkMode") != "none"
        or host.get("CapDrop") != ["ALL"]
        or host.get("CapAdd")
        or "no-new-privileges" not in host.get("SecurityOpt", [])
        or host.get("LogConfig", {}).get("Type") != "none"
        or host.get("Memory") != 2 * 1024**3
        or host.get("MemorySwap") != 2 * 1024**3
        or host.get("NanoCpus") != 2_000_000_000
        or host.get("CpusetCpus") != "0,1"
        or host.get("PidsLimit") != 64
        or host.get("ShmSize") != 1024**2
        or host.get("Tmpfs") != {"/tmp": "rw,noexec,nosuid,nodev,size=8m"}
        or host.get("RestartPolicy", {}).get("Name") != "no"
        or info["Config"].get("Cmd") != command
        or info["Config"].get("Entrypoint")
        or env_map.get("EPS_PAIRED_APPEARANCE_WORK_DEADLINE_UNIX") != str(work_deadline_unix)
        or any(env_map.get(k) != v for k, v in environment(binding).items())
        or any(
            env_map.get(k)
            for k in ("EPS_A1_RUNTIME", "EPS_CAUSAL_RUNTIME", "WSL_INTEROP", "WSL_DISTRO_NAME")
        )
    ):
        raise ValueError("pre-start confinement differs")
    labels = info["Config"].get("Labels", {})
    if any(
        labels.get("eps.appearance." + k.replace("_", "-")) != str(v)
        for k, v in binding.preparation.model_dump().items()
    ):
        raise ValueError("prepared image labels differ")


def host_run(
    binding: AppearanceExecutionBinding,
    decision: LaunchDecision,
    root: Path,
    commands: Commands,
    case: str | None = None,
) -> dict[str, Any]:
    require_decision(binding, decision)
    if (binding.purpose == binding.preparation.study.dummy_purpose) != (case in DUMMY_CASES):
        raise ValueError("closed task purpose")
    safe_path(root)
    # Reject aliases/mount delimiters before constructing Docker arguments.
    if "," in str(root) or "\n" in str(root):
        raise ValueError("ambiguous bind path")
    root.mkdir(exist_ok=True)
    group = canonical_json_bytes(
        {"preparation": binding.preparation.model_dump(mode="json"), "image": binding.image}
    )
    group_path = root / "group.json"
    if group_path.exists():
        safe_path(group_path)
        if group_path.stat().st_size > 8192 or group_path.read_bytes() != group:
            raise ValueError("execution group preparation/image differs")
    else:
        exclusive_file(group_path, group)
    control = root / "control"
    control.mkdir(exist_ok=True)
    slot = control / (case if case else "actual")
    slot.mkdir()  # At most one native slot and each of four dummy slots, never resume.
    output = slot / binding.output_id if case else root / binding.output_id
    allowance = 60 if case else 300
    started = time.monotonic()
    final_deadline = started + allowance
    work_deadline = started + (35 if case else 275)
    work_deadline_unix = time.time() + (35 if case else 275)
    result: dict[str, Any] = {"status": "INCONCLUSIVE", "binding_root": binding.root}
    identity = "eps-appearance-" + binding.token
    completed = False
    cleaned = False
    result["apparatus_status"] = "INCONCLUSIVE"
    try:
        consume_attempt(output, binding, decision)
        exclusive_file(
            slot / "attempt.json",
            canonical_json_bytes(
                {"binding_root": binding.root, "seconds": allowance, "nonrefundable": True}
            ),
        )
        identity = commands.command(
            create_arguments(binding, output, case, work_deadline_unix), work_deadline
        )
        info = json.loads(
            commands.command(["inspect", "--format", "{{json .}}", identity], work_deadline)
        )
        owned(info, binding, identity)
        inspect_confinement(info, binding, output, case, work_deadline_unix)
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
            state = {
                "Running": fields[0] == "true",
                "ExitCode": int(fields[1]),
                "OOMKilled": fields[2] == "true",
            }
            if state["Running"] is False:
                completed = (
                    type(state["ExitCode"]) is int
                    and state["ExitCode"] == 0
                    and state["OOMKilled"] is False
                )
                result["exit_state"] = state
                break
            time.sleep(min(0.5, max(0, work_deadline - time.monotonic())))
        else:
            raise TimeoutError("work watchdog expired")
        if case:
            tiny = output / ("dummy-" + case)
            valid = (tiny / "rgb.bin").read_bytes() == b"tiny-completed-rgb" and (
                tiny / "pair.bin"
            ).read_bytes() == b"tiny-completed-id-depth"
            if case == "normal":
                valid &= (
                    completed
                    and (tiny / "complete.json").read_bytes()
                    == b'{"native":false,"status":"COMPLETE"}'
                )
            elif case in ("interrupted", "overbudget"):
                valid &= not completed and not (tiny / "overflow.bin").exists()
            result["status"] = "DUMMY_OBSERVED" if valid else "INCONCLUSIVE"
    except Exception as error:
        result["reason"] = str(error)[:1024]
        if case == "deadline" and isinstance(error, TimeoutError):
            # Host checks only the tiny completed prefix; owned termination is still mandatory.
            try:
                tiny = output / "dummy-deadline"
                if (tiny / "rgb.bin").read_bytes() == b"tiny-completed-rgb" and (
                    tiny / "pair.bin"
                ).read_bytes() == b"tiny-completed-id-depth":
                    result["status"] = "DUMMY_OBSERVED"
            except OSError:
                pass
    finally:
        cleanup_deadline = min(final_deadline - 5, time.monotonic() + 20)
        try:
            info = json.loads(
                commands.command(
                    ["inspect", "--format", CLEANUP, identity], cleanup_deadline, cleanup=True
                )
            )
            owned(info, binding, identity if re.fullmatch(r"[0-9a-f]{64}", identity) else None)
            actual_mounts = info.get("Mounts", [])
            accepted = {str(output)}
            windows = PureWindowsPath(output)
            if re.fullmatch(r"[A-Za-z]:", windows.drive):
                tail = windows.drive[0].lower() + "/" + "/".join(windows.parts[1:])
                accepted |= {"/run/desktop/mnt/host/" + tail, "/host_mnt/" + tail}
            if (
                len(actual_mounts) != 1
                or actual_mounts[0].get("Source") not in accepted
                or actual_mounts[0].get("Destination") != "/output"
            ):
                raise PermissionError("owned output differs; delete denied")
            commands.command(["rm", "--force", info["Id"]], cleanup_deadline, cleanup=True)
            cleaned = True
        except Exception as error:
            result["cleanup_error"] = str(error)[:1024]
        if not case:
            # Recover only independently replayable relations, after owned termination.
            # A missing/poisoned terminal writer is never retried.
            try:
                retained = replay(output, binding)
                frames = retained["frames"]
                negative = next(
                    (
                        v
                        for c, i in sorted(frames)
                        if (v := e_ready(frames, c, i, binding.preparation.study))["status"]
                        == "FAIL"
                    ),
                    None,
                )
                result["replay_status"] = retained["status"]
                if negative is not None:
                    result["apparatus_status"] = "FAIL"
                    result["scientific_result"] = negative
                if retained["status"] == "INCONCLUSIVE":
                    result["replay_error"] = retained.get("reason", "invalid retained prefix")
                terminal = output / "terminal.json"
                safe_path(terminal)
                if terminal.stat().st_size > 16384:
                    raise ValueError("terminal result cap")
                report = p.strict_json(terminal.read_bytes())
                if report["binding_root"] != binding.root or report["counts"] != retained["counts"]:
                    raise ValueError("terminal binding/counts differ")
                result["retained_result"] = report
                if (
                    completed
                    and retained["status"] != "INCONCLUSIVE"
                    and report.get("operational_status", "COMPLETE") == "COMPLETE"
                    and "reason" not in result
                ):
                    if negative is not None:
                        result["status"] = "FAIL"
                    elif (
                        report["status"] == "CAPTURE_COMPLETE"
                        and retained["status"] == "COMPLETE"
                        and p.assess(
                            {
                                key: (frames[(i, 0)], frames[(i, 1)])
                                for i, key in enumerate(p.contexts(binding.preparation.study))
                            },
                            study=binding.preparation.study,
                        )["status"]
                        == "PASS"
                    ):
                        result["status"] = "PASS"
                        result["apparatus_status"] = "PASS"
            except Exception as error:
                result["retention_error"] = str(error)[:1024]
                result["status"] = "INCONCLUSIVE"
        if not cleaned:
            result["status"] = "INCONCLUSIVE"
        result["cleaned"] = cleaned
        result["operational_status"] = (
            "COMPLETE"
            if cleaned
            and completed
            and "reason" not in result
            and "retention_error" not in result
            and "replay_error" not in result
            and result.get("retained_result", {}).get("operational_status", "COMPLETE")
            == "COMPLETE"
            else "INCONCLUSIVE"
        )
        result["elapsed_seconds"] = time.monotonic() - started
        result["overrun"] = result["elapsed_seconds"] > allowance
        if result["overrun"]:
            result["status"] = "INCONCLUSIVE"
            result["operational_status"] = "INCONCLUSIVE"
        data = canonical_json_bytes(result)
        cap = 64 * 1024 if case else HOST_ACTUAL
        existing = sum(v.stat().st_size for v in slot.rglob("*") if v.is_file())
        # Actual command bytes count even though discarded; group manifest is also charged.
        if existing + len(data) + commands.used + len(group) > cap or len(data) > 16384:
            raise ValueError("host/dummy allowance exhausted; no further write")
        exclusive_file(slot / "result.json", data)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inside", action="store_true")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--study", choices=(PAIRED_V1.schema, CONTRAST_V1.schema))
    parser.add_argument("--case", choices=DUMMY_CASES)
    parser.add_argument("--binding", type=Path)
    parser.add_argument("--decision", type=Path)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--docker", default="docker")
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    if args.prepare:
        if args.inside or args.case or args.binding or args.decision or args.root:
            raise ValueError("preparation is separate from execution")
        manifest = preparation(
            repository,
            os.environ["SOURCE_HEAD"],
            os.environ["SOURCE_TREE"],
            select_study(args.study or PAIRED_V1.schema),
        )
        exclusive_file(
            Path("/preparation/appearance.json"),
            canonical_json_bytes(manifest.model_dump(mode="json")),
        )
        return 0
    if args.study:
        raise ValueError("execution selects study only from consumed exact binding")
    if args.inside:
        if args.prepare or args.binding or args.decision or args.root:
            raise ValueError("inside uses consumed mounted authority only")
        binding = environment_binding()
        consumed(Path("/output"), binding)
        anchor = p.strict_json(Path("/output/consumed.json").read_bytes())
        decision = LaunchDecision.model_validate(anchor["decision"])
        require_decision(binding, decision)
        external_deadline = float(os.environ["EPS_PAIRED_APPEARANCE_WORK_DEADLINE_UNIX"])
        if not math.isfinite(external_deadline):
            raise ValueError("finite outer watchdog required")
        # Host monotonic watchdog remains authoritative if wall clocks move.
        deadline = time.monotonic() + max(0, external_deadline - time.time())
        if binding.purpose == binding.preparation.study.dummy_purpose:
            # Install the existing source guard before dummy bodies; dummy cannot import SDK.
            guard = runpy.run_path(str(repository / "scripts/check_a1_source.py"))
            guard["check_loaded"]()
            sys.meta_path.insert(0, guard["Guard"]())
            run_dummy(Path("/output"), args.case, deadline)
            guard["check_loaded"]()
            return 0
        if args.case:
            raise ValueError("native cannot take dummy task")
        result = run_native(repository, Path("/output"), binding, decision, deadline)
        return 0 if result["status"] in ("CAPTURE_COMPLETE", "FAIL") else 2
    if not args.binding or not args.decision or not args.root:
        raise ValueError("separate binding/decision/group root required; no defaults authorize")
    binding = parse(AppearanceExecutionBinding, args.binding.read_bytes())
    decision = parse(LaunchDecision, args.decision.read_bytes())
    if not isinstance(binding, AppearanceExecutionBinding) or not isinstance(
        decision, LaunchDecision
    ):
        raise ValueError("document types")
    result = host_run(
        binding,
        decision,
        args.root.absolute(),
        Commands(args.docker, 16 * 1024 if args.case else HOST_ACTUAL - 16384),
        args.case,
    )
    print(canonical_json_bytes(result).decode())
    return 0 if result["status"] in ("PASS", "FAIL", "DUMMY_OBSERVED") else 2


if __name__ == "__main__":
    raise SystemExit(main())
