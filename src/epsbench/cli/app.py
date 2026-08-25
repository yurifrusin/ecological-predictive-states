"""Typer CLI for generate, validate, and inspect operations."""

from pathlib import Path
from typing import Annotated

import typer

from epsbench.config import load_config
from epsbench.data import create_inspection_image, generate_dataset, validate_dataset

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="EPS-Bench Milestone 0 dataset utilities.",
)


@app.command()
def generate(
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False, readable=True)],
    output: Annotated[Path, typer.Option(file_okay=False)],
    episodes: Annotated[int, typer.Option(min=1)] = 1,
) -> None:
    """Generate a deterministic single-occluder dataset."""

    try:
        manifest = generate_dataset(load_config(config), episodes, output)
    except Exception as error:
        typer.echo(f"Generation failed: {error}", err=True)
        raise typer.Exit(code=1) from error
    typer.echo(
        f"Generated {len(manifest.episodes)} episode(s); "
        f"dataset logical hash {manifest.dataset_logical_sha256}"
    )


@app.command(name="validate")
def validate_command(
    dataset: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
) -> None:
    """Validate schemas, artifacts, alignment, remapping, and hashes."""

    try:
        manifest = validate_dataset(dataset)
    except Exception as error:
        typer.echo(f"Validation failed: {error}", err=True)
        raise typer.Exit(code=1) from error
    typer.echo(
        f"Valid dataset: {len(manifest.episodes)} episode(s); "
        f"logical hash {manifest.dataset_logical_sha256}"
    )


@app.command(name="inspect")
def inspect_command(
    dataset: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    output: Annotated[Path, typer.Option(dir_okay=False)],
    episode: Annotated[int, typer.Option(min=0)] = 0,
) -> None:
    """Write a labelled RGB/depth/segmentation composite outside the dataset."""

    try:
        result = create_inspection_image(dataset, episode, output)
    except Exception as error:
        typer.echo(f"Inspection failed: {error}", err=True)
        raise typer.Exit(code=1) from error
    typer.echo(f"Inspection image: {result}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
