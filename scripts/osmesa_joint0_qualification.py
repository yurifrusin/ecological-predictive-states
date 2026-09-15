"""Operate the prospectively reviewed finite OSMesa/joint0 qualification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

from epsbench.diagnostics.osmesa_joint0_qualification import (
    QualificationFailure,
    compare_profile_outputs,
    compare_repeat_artifacts,
    fixed_attempts,
    geometry_diagnostics,
    git_binding,
    initialise_ledger,
    run_attempt,
    validate_bindings,
    validate_ledger,
    validate_recorded_attempt,
)


def _binding(source: Path) -> dict[str, object]:
    return git_binding(
        source,
        source / "uv.lock",
        (source / "configs/benchmark_v0.yaml", source / "configs/corridor_v0.yaml"),
        source / "configs/appearance_candidates_v0.yaml",
        source / "configs/evaluation_seed_candidates_v0.yaml",
    )


def _generate(source: Path, attempt: Any, dataset: Path, _adapter: object) -> object:
    from epsbench.appearance import load_appearance_registry, load_evaluation_seed_registry
    from epsbench.config import load_config
    from epsbench.data.generate import generate_dataset

    config_name = "benchmark_v0.yaml" if attempt.family == "single_occluder" else "corridor_v0.yaml"
    config = load_config(source / "configs" / config_name)
    config = config.model_copy(
        update={"appearance": config.appearance.model_copy(update={"profile_id": attempt.profile})}
    )
    if (
        config.seed != 1729
        or config.render.width != 160
        or config.render.height != 120
        or config.appearance.registry_version != "appearance_candidate_registry_v1"
    ):
        raise QualificationFailure("resolved generator configuration differs from the fixed plan")
    registry = load_appearance_registry(source / "configs/appearance_candidates_v0.yaml")
    seeds = load_evaluation_seed_registry(source / "configs/evaluation_seed_candidates_v0.yaml")
    return generate_dataset(
        config,
        4,
        dataset,
        appearance_registry=registry,
        seed_registry=seeds,
        component_topology=False,
    )


def _assessment(output: Path, attempt: Any, dataset: Path, _manifest: object) -> dict[str, object]:
    from epsbench.data.inspect import create_inspection_image
    from epsbench.data.validate import validate_dataset

    validate_dataset(dataset)
    inspection = output / "inspections" / f"{attempt.name}-episode-000000.png"
    inspection.parent.mkdir(parents=True, exist_ok=True)
    create_inspection_image(dataset, 0, inspection)
    result: dict[str, object] = {
        "validate_dataset": "passed",
        "inspection": {"path": inspection.relative_to(output).as_posix()},
        "geometry_diagnostics": geometry_diagnostics(dataset),
    }
    datasets = output / "datasets"
    if attempt.repeat == 1:
        prior = next(
            item
            for item in fixed_attempts()
            if item.family == attempt.family
            and item.profile == attempt.profile
            and item.repeat == 0
        )
        result["repeat_exact_members"] = compare_repeat_artifacts(datasets / prior.name, dataset)
    if attempt.profile == "legacy_solid_alternate_v1":
        base = next(
            item
            for item in fixed_attempts()
            if item.family == attempt.family
            and item.profile == "legacy_solid_base_v1"
            and item.repeat == attempt.repeat
        )
        result["appearance_comparison"] = compare_profile_outputs(datasets / base.name, dataset)
    return result


def _validate_complete(output: Path, source: Path) -> dict[str, object]:
    from epsbench.data.validate import validate_dataset

    records = validate_ledger(output)
    binding = records[0]["binding"]
    validate_bindings(source, binding)
    import mujoco

    from epsbench.diagnostics.osmesa_joint0_qualification import (
        bind_pristine_sdk,
        require_osmesa_linux,
    )

    require_osmesa_linux()
    current_runtime = bind_pristine_sdk(mujoco)
    completed = [record for record in records if record["event"] == "complete"]
    expected_attempts = fixed_attempts()[: len(completed)]
    expected_names = {attempt.name for attempt in expected_attempts}
    dataset_names = (
        {path.name for path in (output / "datasets").iterdir()}
        if (output / "datasets").is_dir()
        else set()
    )
    receipt_names = (
        {path.stem for path in (output / "receipts").glob("*.json")}
        if (output / "receipts").is_dir()
        else set()
    )
    if dataset_names != expected_names or receipt_names != expected_names:
        raise QualificationFailure(
            "recorded dataset/receipt membership differs from completed ledger prefix"
        )
    baseline: dict[str, object] | None = None
    for attempt in fixed_attempts()[: len(completed)]:
        dataset = output / "datasets" / attempt.name
        receipt = output / "receipts" / f"{attempt.name}.json"
        validate_dataset(dataset)
        recorded = validate_recorded_attempt(
            output,
            attempt,
            context_baseline=baseline,
            expected_binding=binding,
        )
        if recorded.get("runtime") != current_runtime:
            raise QualificationFailure("recorded CPU/package runtime binding differs")
        if baseline is None:
            from epsbench.diagnostics.osmesa_joint0_qualification import context_runtime_binding

            baseline = context_runtime_binding(cast(Any, recorded["contexts"])[0])
        details = completed[attempt.ordinal]["details"]
        if details["receipt_sha256"] != __import__(
            "epsbench.diagnostics.osmesa_joint0_qualification", fromlist=["digest_file"]
        ).digest_file(receipt):
            raise QualificationFailure("qualification receipt hash differs from ledger")
        if attempt.repeat == 1:
            prior = next(
                item
                for item in fixed_attempts()
                if item.family == attempt.family
                and item.profile == attempt.profile
                and item.repeat == 0
            )
            compare_repeat_artifacts(output / "datasets" / prior.name, dataset)
        if attempt.profile == "legacy_solid_alternate_v1":
            base = next(
                item
                for item in fixed_attempts()
                if item.family == attempt.family
                and item.profile == "legacy_solid_base_v1"
                and item.repeat == attempt.repeat
            )
            compare_profile_outputs(output / "datasets" / base.name, dataset)
    execution_complete = len(completed) == len(fixed_attempts())
    return {
        "status": "PARTIAL_EVIDENCE_ALIGNMENT_UNSPECIFIED" if execution_complete else "in_progress",
        "execution_complete": execution_complete,
        "completed_attempts": len(completed),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=Path.cwd())
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init")
    init.add_argument("--output-root", type=Path, required=True)
    nxt = sub.add_parser("next")
    nxt.add_argument("--output-root", type=Path, required=True)
    check = sub.add_parser("validate")
    check.add_argument("--output-root", type=Path, required=True)
    sub.add_parser("plan")
    args = parser.parse_args()
    source = args.source_root.resolve()
    if args.command == "plan":
        from epsbench.diagnostics.osmesa_joint0_qualification import plan

        print(json.dumps(plan(), sort_keys=True))
        return
    output = args.output_root.resolve()
    if args.command == "init":
        path = initialise_ledger(output, _binding(source))
        print(json.dumps({"status": "initialised", "ledger": str(path)}, sort_keys=True))
        return
    if args.command == "validate":
        print(json.dumps(_validate_complete(output, source), sort_keys=True))
        return
    records = validate_ledger(output)
    binding = records[0]["binding"]
    validate_bindings(source, binding)
    completed = len([record for record in records if record["event"] == "complete"])
    attempts = fixed_attempts()
    if completed >= len(attempts):
        raise QualificationFailure("qualification plan is already complete")
    attempt = attempts[completed]
    import mujoco

    terminal = run_attempt(
        output,
        attempt,
        source,
        binding,
        lambda dataset, adapter: _generate(source, attempt, dataset, adapter),
        mujoco,
        assess=lambda dataset, manifest: _assessment(output, attempt, dataset, manifest),
    )
    print(
        json.dumps(
            {"status": "complete", "attempt": attempt.name, "ledger": str(terminal)}, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
