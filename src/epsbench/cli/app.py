"""Typer CLI for generate, validate, and inspect operations."""

from collections import Counter
from pathlib import Path
from typing import Annotated

import typer

from epsbench.audit import create_appearance_audit, validate_appearance_audit
from epsbench.config import load_config
from epsbench.data import DatasetLoader, create_inspection_image, generate_dataset, validate_dataset
from epsbench.failure_analysis import create_failure_analysis, validate_failure_analysis
from epsbench.freeze import (
    create_freeze_audit,
    create_publication_record,
    create_renderer_receipt,
    validate_freeze_audit,
)
from epsbench.revision import create_revision_audit, validate_revision_audit
from epsbench.schema import ComponentTopologyAnnotation, DatasetManifest, ModalityPermissionSet
from epsbench.topology_qualification import qualify_topology, validate_topology_packet

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
        topology = events.annotation.capabilities.component_topology
        if isinstance(topology, ComponentTopologyAnnotation):
            typer.echo(
                f"Episode {episode.episode_index} component capability {topology.status}; "
                f"identity {topology.component_topology_sha256}; "
                f"portable graph {topology.portable_graph_sha256}; "
                f"events {dict(sorted(Counter(e.kind.value for e in topology.events).items()))}"
            )
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


@app.command(name="component-topology-audit")
def component_topology_audit(
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False, readable=True)],
    output: Annotated[Path, typer.Option(file_okay=False)],
    episodes: Annotated[int, typer.Option(min=1)] = 2,
) -> None:
    """Generate and independently validate complete component-topology evidence."""
    try:
        generate_dataset(load_config(config), episodes, output, component_topology=True)
        manifest = validate_dataset(output)
    except Exception as error:
        typer.echo(f"Component topology audit failed: {error}", err=True)
        raise typer.Exit(code=1) from error
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


@app.command(name="component-topology-qualify")
def component_topology_qualify(
    output: Annotated[Path, typer.Option(file_okay=False)],
    lock_commit: Annotated[str, typer.Option()],
) -> None:
    """Execute exactly 192 cells only after verifying the clean pushed prospective lock."""
    try:
        packet = qualify_topology(output, lock_commit)
    except Exception as error:
        typer.echo(f"Frozen topology qualification failed: {error}", err=True)
        raise typer.Exit(code=1) from error
    typer.echo(
        f"Complete packet {packet.packet_sha256}; qualification {packet.local_qualification}"
    )


@app.command(name="component-topology-validate")
def component_topology_validate(
    dataset: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
) -> None:
    """Reconstruct all 192 sources and exact membership, including retained failures."""
    try:
        packet = validate_topology_packet(dataset)
    except Exception as error:
        typer.echo(f"Topology packet validation failed: {error}", err=True)
        raise typer.Exit(code=1) from error
    typer.echo(
        f"Valid evidence packet {packet.packet_sha256}; qualification {packet.local_qualification}"
    )


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
        f"design-membership={roots['design_partition_membership_root_sha256']} "
        f"qualification-membership="
        f"{roots['qualification_partition_membership_root_sha256']} "
        f"profiles={roots['profile_admission_outcome_root_sha256']}"
    )
    typer.echo(
        "Renderer-local roots: "
        f"design-outcomes={roots['renderer_local_design_partition_outcome_root_sha256']} "
        f"qualification-outcomes="
        f"{roots['renderer_local_qualification_partition_outcome_root_sha256']} "
        f"labels={roots['renderer_local_ecological_label_root_sha256']} "
        f"audit={roots['renderer_specific_audit_root_sha256']}"
    )
    typer.echo(f"Revision candidate packet: {output}")


@app.command(name="appearance-freeze-audit")
def appearance_freeze_audit_command(
    benchmark_definition: Annotated[Path, typer.Option(exists=True, dir_okay=False, readable=True)],
    evaluation_seeds: Annotated[Path, typer.Option(exists=True, dir_okay=False, readable=True)],
    definition_lock: Annotated[Path, typer.Option(exists=True, dir_okay=False, readable=True)],
    revision_registry: Annotated[Path, typer.Option(exists=True, dir_okay=False, readable=True)],
    single_config: Annotated[Path, typer.Option(exists=True, dir_okay=False, readable=True)],
    corridor_config: Annotated[Path, typer.Option(exists=True, dir_okay=False, readable=True)],
    output: Annotated[Path, typer.Option(file_okay=False)],
    counterpart_evidence: Annotated[
        Path | None, typer.Option(exists=True, file_okay=False, readable=True)
    ] = None,
) -> None:
    """Build and independently validate the 192-cell freeze-candidate packet."""

    try:
        packet = create_freeze_audit(
            benchmark_definition,
            evaluation_seeds,
            definition_lock,
            revision_registry,
            single_config,
            corridor_config,
            output,
            counterpart_evidence,
        )
        validate_freeze_audit(output, counterpart_evidence)
    except Exception as error:
        typer.echo(f"Appearance freeze audit failed: {error}", err=True)
        raise typer.Exit(code=1) from error
    typer.echo(f"Freeze candidate packet root: {packet['complete_packet_root_sha256']}")
    typer.echo(
        f"Selected outcomes: {packet['selected_matrix_counts']}; "
        f"legacy control outcomes: {packet['control_matrix_counts']}"
    )
    typer.echo(
        f"Freeze candidate status: {packet['freeze_candidate_status']}; "
        f"seed disposition: {packet['seed_set_disposition']}"
    )
    typer.echo(f"Portable definition roots: {packet['portable_definition_roots']}")
    typer.echo(f"Portable apparatus roots: {packet['portable_apparatus_roots']}")
    typer.echo(
        f"Portable profile-readiness root: {packet['portable_profile_readiness_root_sha256']}"
    )
    typer.echo(f"Renderer-local roots: {packet['renderer_local_roots']}")
    typer.echo(f"Threshold margins: {packet['threshold_margin_summary']}")
    typer.echo(f"Freeze candidate packet: {output}")


@app.command(name="appearance-freeze-validate")
def appearance_freeze_validate_command(
    packet: Annotated[Path, typer.Argument(exists=True, file_okay=False, readable=True)],
    counterpart_evidence: Annotated[
        Path | None, typer.Option(exists=True, file_okay=False, readable=True)
    ] = None,
) -> None:
    """Independently validate a complete appearance freeze-candidate packet."""

    try:
        validated = validate_freeze_audit(packet, counterpart_evidence)
    except Exception as error:
        typer.echo(f"Appearance freeze validation failed: {error}", err=True)
        raise typer.Exit(code=1) from error
    typer.echo(
        f"Valid {validated['renderer_environment_id']} freeze packet: "
        f"{validated['complete_packet_root_sha256']}"
    )


@app.command(name="appearance-freeze-receipt")
def appearance_freeze_receipt_command(
    packet: Annotated[Path, typer.Argument(exists=True, file_okay=False, readable=True)],
    output: Annotated[Path, typer.Option(dir_okay=False)],
    counterpart_evidence: Annotated[
        Path | None, typer.Option(exists=True, file_okay=False, readable=True)
    ] = None,
) -> None:
    """Create a canonical one-renderer receipt from a validated freeze packet."""

    try:
        receipt = create_renderer_receipt(packet, output, counterpart_evidence)
    except Exception as error:
        typer.echo(f"Appearance freeze receipt failed: {error}", err=True)
        raise typer.Exit(code=1) from error
    typer.echo(f"Renderer qualification receipt: {receipt['receipt_sha256']}")


@app.command(name="appearance-freeze-publication-record")
def appearance_freeze_publication_record_command(
    packet: Annotated[Path, typer.Argument(exists=True, file_okay=False, readable=True)],
    artifact_archive: Annotated[Path, typer.Option(exists=True, dir_okay=False, readable=True)],
    artifact_metadata: Annotated[Path, typer.Option(exists=True, dir_okay=False, readable=True)],
    repository_metadata: Annotated[Path, typer.Option(exists=True, dir_okay=False, readable=True)],
    workflow_run_metadata: Annotated[
        Path, typer.Option(exists=True, dir_okay=False, readable=True)
    ],
    workflow_jobs_metadata: Annotated[
        Path, typer.Option(exists=True, dir_okay=False, readable=True)
    ],
    job_name: Annotated[str, typer.Option()],
    artifact_digest_sha256: Annotated[str, typer.Option()],
    artifact_url: Annotated[str, typer.Option()],
    output: Annotated[Path, typer.Option(dir_okay=False)],
    counterpart_evidence: Annotated[
        Path | None, typer.Option(exists=True, file_okay=False, readable=True)
    ] = None,
) -> None:
    """Record exact connected/public CI provenance for one validated packet."""

    try:
        record = create_publication_record(
            packet,
            artifact_archive,
            artifact_metadata,
            repository_metadata,
            workflow_run_metadata,
            workflow_jobs_metadata,
            job_name=job_name,
            artifact_digest_sha256=artifact_digest_sha256,
            artifact_url=artifact_url,
            output=output,
            counterpart_evidence_root=counterpart_evidence,
        )
    except Exception as error:
        typer.echo(f"Appearance freeze publication record failed: {error}", err=True)
        raise typer.Exit(code=1) from error
    typer.echo(f"CI packet publication record: {record['record_sha256']}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
