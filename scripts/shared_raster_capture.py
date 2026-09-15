"""Operate the reviewed finite shared-raster capture study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

from epsbench.diagnostics.osmesa_joint0_qualification import (
    compare_profile_outputs,
    compare_repeat_artifacts,
    geometry_diagnostics,
    validate_source_linkage,
)
from epsbench.diagnostics.shared_raster_capture import (
    STATUS,
    SharedRasterFailure,
    compare_repeat_pairs,
    digest_file,
    fixed_attempts,
    git_binding,
    initialise_ledger,
    plan,
    run_attempt,
    validate_bindings,
    validate_completed_prefix,
    validate_ledger,
    validate_pair_tree,
    validate_recorded_attempt,
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
    if config.seed != 1729 or config.render.width != 160 or config.render.height != 120:
        raise SharedRasterFailure("resolved generator configuration differs from fixed plan")
    return generate_dataset(
        config,
        4,
        dataset,
        appearance_registry=load_appearance_registry(
            source / "configs/appearance_candidates_v0.yaml"
        ),
        seed_registry=load_evaluation_seed_registry(
            source / "configs/evaluation_seed_candidates_v0.yaml"
        ),
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
        "inspection": {
            "path": inspection.relative_to(output).as_posix(),
            "sha256": digest_file(inspection),
        },
        "validate_dataset": "passed",
        "paired": validate_pair_tree(output / "paired" / attempt.name),
        "analytic_geometry_descriptive_only": geometry_diagnostics(dataset),
    }
    if attempt.repeat == 1:
        prior = next(
            a
            for a in fixed_attempts()
            if a.family == attempt.family and a.profile == attempt.profile and a.repeat == 0
        )
        result["repeat_dataset_exact_members"] = compare_repeat_artifacts(
            output / "datasets" / prior.name, dataset
        )
        result["repeat_pair_exact_members"] = compare_repeat_pairs(
            output / "paired" / prior.name, output / "paired" / attempt.name
        )
    if attempt.profile == "legacy_solid_alternate_v1":
        base = next(
            a
            for a in fixed_attempts()
            if a.family == attempt.family
            and a.profile == "legacy_solid_base_v1"
            and a.repeat == attempt.repeat
        )
        result["appearance_comparison"] = compare_profile_outputs(
            output / "datasets" / base.name, dataset
        )
    return result


def _validate(output: Path, source: Path) -> dict[str, object]:
    from epsbench.data.validate import validate_dataset
    from epsbench.diagnostics.shared_raster_capture import bind_pristine_sdk, require_osmesa_linux

    records = validate_ledger(output)
    binding = cast(dict[str, object], records[0]["binding"])
    validate_bindings(source, binding)
    require_osmesa_linux()
    import mujoco

    runtime = bind_pristine_sdk(mujoco)
    completed = [r for r in records if r["event"] == "complete"]
    attempts = fixed_attempts()[: len(completed)]
    names = {a.name for a in attempts}
    datasets = (
        {p.name for p in (output / "datasets").iterdir()}
        if (output / "datasets").is_dir()
        else set()
    )
    pairs = (
        {p.name for p in (output / "paired").iterdir()} if (output / "paired").is_dir() else set()
    )
    receipts = (
        {p.stem for p in (output / "shared-raster-receipts").glob("*.json")}
        if (output / "shared-raster-receipts").is_dir()
        else set()
    )
    if datasets != names or pairs != names or receipts != names:
        raise SharedRasterFailure(
            "saved dataset/pair/receipt membership differs from completed prefix"
        )
    baseline = validate_completed_prefix(output, records, binding)
    for attempt in attempts:
        dataset = output / "datasets" / attempt.name
        validate_dataset(dataset)
        receipt = validate_recorded_attempt(
            output, attempt, context_baseline=baseline, expected_binding=binding
        )
        if receipt.get("runtime") != runtime:
            raise SharedRasterFailure("recorded runtime binding differs")
        validate_pair_tree(output / "paired" / attempt.name)
        if attempt.repeat == 1:
            prior = next(
                a
                for a in fixed_attempts()
                if a.family == attempt.family and a.profile == attempt.profile and a.repeat == 0
            )
            compare_repeat_artifacts(output / "datasets" / prior.name, dataset)
            compare_repeat_pairs(output / "paired" / prior.name, output / "paired" / attempt.name)
        if attempt.profile == "legacy_solid_alternate_v1":
            base = next(
                a
                for a in fixed_attempts()
                if a.family == attempt.family
                and a.profile == "legacy_solid_base_v1"
                and a.repeat == attempt.repeat
            )
            compare_profile_outputs(output / "datasets" / base.name, dataset)
    complete = len(completed) == len(fixed_attempts())
    return {
        "status": STATUS if complete else "in_progress",
        "execution_complete": complete,
        "completed_attempts": len(completed),
        "context_binding": baseline,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=Path.cwd())
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init")
    init.add_argument("--output-root", type=Path, required=True)
    nxt = commands.add_parser("next")
    nxt.add_argument("--output-root", type=Path, required=True)
    check = commands.add_parser("validate")
    check.add_argument("--output-root", type=Path, required=True)
    commands.add_parser("plan")
    args = parser.parse_args()
    if args.command == "plan":
        print(json.dumps(plan(), sort_keys=True))
        return
    source = validate_source_linkage(args.source_root.resolve())
    output = args.output_root.resolve()
    if args.command == "init":
        path = initialise_ledger(output, git_binding(source))
        print(json.dumps({"status": "initialised", "ledger": str(path)}, sort_keys=True))
        return
    if args.command == "validate":
        print(json.dumps(_validate(output, source), sort_keys=True))
        return
    records = validate_ledger(output)
    binding = cast(dict[str, object], records[0]["binding"])
    validate_bindings(source, binding)
    completed = len([r for r in records if r["event"] == "complete"])
    attempts = fixed_attempts()
    if completed >= len(attempts):
        raise SharedRasterFailure("shared-raster study is already complete")
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
