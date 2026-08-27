"""Typer CLI for generate, validate, and inspect operations."""

from collections import Counter
from pathlib import Path
from typing import Annotated

import typer

from epsbench.audit import create_appearance_audit, validate_appearance_audit
from epsbench.config import load_config
from epsbench.data import DatasetLoader, create_inspection_image, generate_dataset, validate_dataset
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
        f"ecological={roots['ecological_invariance_root_sha256']}"
    )
    typer.echo(f"Renderer-specific audit root: {roots['renderer_specific_audit_root_sha256']}")
    typer.echo(f"Candidate packet: {output}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
