"""Typer CLI for generate, validate, and inspect operations."""

from collections import Counter
from pathlib import Path
from typing import Annotated

import typer

from epsbench.audit import create_appearance_audit, validate_appearance_audit
from epsbench.config import load_config
from epsbench.data import DatasetLoader, create_inspection_image, generate_dataset, validate_dataset
from epsbench.failure_analysis import create_failure_analysis, validate_failure_analysis
from epsbench.revision import create_revision_audit, validate_revision_audit
from epsbench.schema import DatasetManifest, ModalityPermissionSet

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="EPS-Bench Milestone 0 dataset utilities.",
)


def _print_oracle_evidence(dataset: Path, manifest: DatasetManifest) -> None:
    typer.echo(
        "Oriented boundary identities: "
        + ", ".join(episode.oriented_boundary_sha256 for episode in manifest.episodes)
    )
    typer.echo(
        "Visibility event identities: "
        + ", ".join(episode.visibility_event_sha256 for episode in manifest.episodes)
    )
    loader = DatasetLoader(dataset, ModalityPermissionSet.ecological_only())
    for episode in manifest.episodes:
        boundary = loader.read_oriented_boundaries(episode.episode_index)
        events = loader.read_ecological_visibility_events(episode.episode_index)
        boundary_counts = Counter(element.kind.value for element in boundary.elements)
        before_counts = Counter(int(value) for value in events.before_fate_codes.reshape(-1))
        after_counts = Counter(int(value) for value in events.after_origin_codes.reshape(-1))
        boundary_summary = dict(sorted(boundary_counts.items()))
        typer.echo(
            f"Episode {episode.episode_index} boundary counts {boundary_summary}; "
            f"before-event counts {dict(sorted(before_counts.items()))}; "
            f"after-event counts {dict(sorted(after_counts.items()))}"
        )


@app.command()
def generate(
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False, readable=True)],
    output: Annotated[Path, typer.Option(file_okay=False)],
    episodes: Annotated[int, typer.Option(min=1)] = 1,
) -> None:
    """Generate a deterministic dataset for the configured scene family."""

    try:
        manifest = generate_dataset(load_config(config), episodes, output)
    except Exception as error:
        typer.echo(f"Generation failed: {error}", err=True)
        raise typer.Exit(code=1) from error
    typer.echo(
        f"Generated {len(manifest.episodes)} {manifest.scene_family.value} episode(s); "
        f"dataset logical hash {manifest.dataset_logical_sha256}"
    )
    typer.echo(
        "Analytic transport identities: "
        + ", ".join(episode.analytic_transport_sha256 for episode in manifest.episodes)
    )
    _print_oracle_evidence(output, manifest)


@app.command(name="validate")
def validate_command(
    dataset: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
) -> None:
    """Validate schemas, artifacts, analytic oracles, remapping, and hashes."""

    try:
        manifest = validate_dataset(dataset)
    except Exception as error:
        typer.echo(f"Validation failed: {error}", err=True)
        raise typer.Exit(code=1) from error
    typer.echo(
        f"Valid {manifest.scene_family.value} dataset: {len(manifest.episodes)} episode(s); "
        f"logical hash {manifest.dataset_logical_sha256}"
    )
    typer.echo(
        "Analytic transport identities: "
        + ", ".join(episode.analytic_transport_sha256 for episode in manifest.episodes)
    )
    _print_oracle_evidence(dataset, manifest)


@app.command(name="inspect")
def inspect_command(
    dataset: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    output: Annotated[Path, typer.Option(dir_okay=False)],
    episode: Annotated[int, typer.Option(min=0)] = 0,
) -> None:
    """Write an RGB, transport, boundary, and event composite outside the dataset."""

    try:
        result = create_inspection_image(dataset, episode, output)
    except Exception as error:
        typer.echo(f"Inspection failed: {error}", err=True)
        raise typer.Exit(code=1) from error
    typer.echo(f"Inspection image: {result}")


@app.command(name="appearance-audit")
def appearance_audit_command(
    registry: Annotated[Path, typer.Option(exists=True, dir_okay=False, readable=True)],
    seeds: Annotated[Path, typer.Option(exists=True, dir_okay=False, readable=True)],
    single_config: Annotated[Path, typer.Option(exists=True, dir_okay=False, readable=True)],
    corridor_config: Annotated[Path, typer.Option(exists=True, dir_okay=False, readable=True)],
    output: Annotated[Path, typer.Option(file_okay=False)],
) -> None:
    """Build and independently validate the complete appearance-candidate packet."""

    try:
        packet = create_appearance_audit(registry, seeds, single_config, corridor_config, output)
        validate_appearance_audit(output)
    except Exception as error:
        typer.echo(f"Appearance audit failed: {error}", err=True)
        raise typer.Exit(code=1) from error
    roots = packet["roots"]
    typer.echo(f"Candidate packet logical root: {packet['packet_logical_root_sha256']}")
    typer.echo(
        "Shared backend-independent roots: "
        f"registry={roots['appearance_registry_sha256']} "
        f"seeds={roots['seed_registry_sha256']} "
        f"procedural={roots['procedural_asset_root_sha256']} "
        f"assignment={roots['appearance_assignment_root_sha256']} "
        f"analytic={roots['portable_analytic_identity_root_sha256']} "
        f"outcome={roots['appearance_invariance_outcome_root_sha256']}"
    )
    typer.echo(
        "Renderer-local ecological-label root: "
        f"{roots['renderer_local_ecological_label_root_sha256']}"
    )
    typer.echo(f"Renderer-specific audit root: {roots['renderer_specific_audit_root_sha256']}")
    typer.echo(f"Candidate packet: {output}")


@app.command(name="appearance-failure-analysis")
def appearance_failure_analysis_command(
    packet: Annotated[Path, typer.Option(exists=True, file_okay=False, readable=True)],
    output: Annotated[Path, typer.Option(file_okay=False)],
) -> None:
    """Analyse the independently validated canonical Slice 5 negative evidence."""

    try:
        analysis = create_failure_analysis(packet, output)
        validate_failure_analysis(output, source_packet=packet)
    except Exception as error:
        typer.echo(f"Appearance failure analysis failed: {error}", err=True)
        raise typer.Exit(code=1) from error
    typer.echo(
        f"Baseline failure-analysis logical root: {analysis['baseline_failure_analysis_sha256']}"
    )
    typer.echo(f"Baseline failure analysis: {output}")


@app.command(name="appearance-revision-audit")
def appearance_revision_audit_command(
    baseline_registry: Annotated[Path, typer.Option(exists=True, dir_okay=False, readable=True)],
    design_seeds: Annotated[Path, typer.Option(exists=True, dir_okay=False, readable=True)],
    revision_registry: Annotated[Path, typer.Option(exists=True, dir_okay=False, readable=True)],
    qualification_seeds: Annotated[Path, typer.Option(exists=True, dir_okay=False, readable=True)],
    definition_lock: Annotated[Path, typer.Option(exists=True, dir_okay=False, readable=True)],
    single_config: Annotated[Path, typer.Option(exists=True, dir_okay=False, readable=True)],
    corridor_config: Annotated[Path, typer.Option(exists=True, dir_okay=False, readable=True)],
    output: Annotated[Path, typer.Option(file_okay=False)],
) -> None:
    """Build and validate the locked two-partition Revision 1 candidate packet."""

    try:
        packet = create_revision_audit(
            baseline_registry,
            design_seeds,
            revision_registry,
            qualification_seeds,
            definition_lock,
            single_config,
            corridor_config,
            output,
        )
        validate_revision_audit(output)
    except Exception as error:
        typer.echo(f"Appearance revision audit failed: {error}", err=True)
        raise typer.Exit(code=1) from error
    roots = packet["roots"]
    typer.echo(f"Revision packet root: {packet['complete_packet_root_sha256']}")
    typer.echo(
        f"Design outcomes: {packet['matrix_counts']['design']}; "
        f"qualification outcomes: {packet['matrix_counts']['qualification']}"
    )
    typer.echo(
        "Portable roots: "
        f"analytic={roots['portable_analytic_identity_root_sha256']} "
        f"invariance={roots['within_renderer_invariance_outcome_root_sha256']} "
        f"design={roots['design_partition_outcome_root_sha256']} "
        f"qualification={roots['qualification_partition_outcome_root_sha256']}"
    )
    typer.echo(
        "Renderer-local roots: "
        f"labels={roots['renderer_local_ecological_label_root_sha256']} "
        f"audit={roots['renderer_specific_audit_root_sha256']}"
    )
    typer.echo(f"Revision candidate packet: {output}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
