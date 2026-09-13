"""Prospective Slice 6 lock and exact frozen-membership evidence, without gate authority."""

from __future__ import annotations

import base64
import os
import subprocess
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import yaml
from pydantic import Field

from epsbench.annotations.component_topology import domain_hash
from epsbench.appearance import load_appearance_registry_any, profile_by_id
from epsbench.audit import _profile_config
from epsbench.config import load_config
from epsbench.data.generate import _renderer_provenance, generate_dataset
from epsbench.data.loader import DatasetLoader
from epsbench.data.paths import open_owned_regular_file
from epsbench.data.validate import validate_dataset
from epsbench.freeze import (
    _legacy_control_membership_domain,
    _renderer_environment_id,
    _selected_membership_domain,
    load_benchmark_definition,
    load_final_evaluation_seeds,
    validate_definition_lock,
)
from epsbench.schema import (
    ComponentEventKind,
    GitCommit,
    ModalityPermissionSet,
    RendererProvenance,
    Sha256,
    StrictModel,
)
from epsbench.utils.canonical import (
    canonical_json_bytes,
    sha256_bytes,
    write_canonical_json,
)

DEFINITION = Path("configs/component_topology_v0.yaml")
LOCK = Path("configs/component_topology_v0_lock.json")
FREEZE_DEFINITION = Path("configs/appearance_benchmark_freeze_v0.yaml")
FREEZE_LOCK = Path("configs/appearance_benchmark_freeze_v0_lock.json")
SEEDS = Path("configs/appearance_benchmark_v0_evaluation_episode_seeds.yaml")
REVISION = Path("configs/appearance_candidates_revision1.yaml")
SINGLE = Path("configs/benchmark_v0.yaml")
CORRIDOR = Path("configs/corridor_v0.yaml")
BASE = "0e9e0593cc7a0f445718575152eca5fc6651edc4"
BASE_TREE = "ec98e8bcc8ed62686ccc0f98a0fe8956690be698"
BRANCH = "codex/gate-0b-slice-6-component-topology"
FREEZE_ROOT = "28acd2c340b3ef1e2b8b2b7b31ca69356cd9baa35b94ff8a598e020099883435"
FREEZE_COMMIT = "d4072f912cc58bbc1ca41ceb2652e41783dbf3e1"


def definition_payload() -> dict[str, Any]:
    """Metadata only: constructing the definition never renders or inspects final roots."""
    return {
        "schema_version": "component_topology_definition_v1",
        "source_byte_convention": "utf8_source_CRLF_normalised_to_LF_matching_git_attributes",
        "base_commit": BASE,
        "base_tree": BASE_TREE,
        "source_schemas": {
            "transition": "0.1.0-dev.10",
            "datasets": ["0.1.0-dev.9", "0.1.0-dev.10"],
            "annotation": "component_topology_annotation_v1",
            "transport": "analytic_static_scene_transport_v3",
            "pixel_events": ["before_frame_fate_codes_v1", "after_frame_origin_codes_v1"],
        },
        "connectivity": "maximal_four_neighbour_equal_opaque_label_all_nonempty_components",
        "background": "zero_excluded_no_size_filter",
        "component_identity": "versioned_frame_surface_surface_rank_hash_not_persistent",
        "ordering": "surface_id_then_minimum_row_major_coordinate_ties_rejected",
        "support_ordering": "surface_id_then_before_component_id_then_after_component_id_unique",
        "event_ordering": "surface_id_then_sorted_before_ids_then_sorted_after_ids_unique",
        "coordinates": "zero_based_row_column_half_open_integer_bounds",
        "target_assignment": (
            "floor((1024_source_coordinate_plus_512_plus_signed_displacement)/1024)"
        ),
        "out_of_frame": "exclude_without_clamping",
        "support_samples": (
            "validity_one_and_reason_zero_only_boundary_occlusion_exit_samples_excluded"
        ),
        "support_edge": "forward_count_positive_OR_backward_count_positive_same_surface",
        "zero_support": "retain_every_same_surface_cartesian_pair_with_both_counts",
        "events": [kind.value for kind in ComponentEventKind],
        "isolated_components": (
            "lifecycle_only_if_opposite_surface_absent_or_every_component_"
            "pixel_code_one_or_two_otherwise_indeterminate"
        ),
        "complex_posture": "retain_and_allow_complete_many_to_many_evidence",
        "indeterminate_posture": (
            "retain_all_evidence_capability_indeterminate_qualification_fails_closed"
        ),
        "appearance_invariance": (
            "exact_full_topology_identity_per_matched_scene_root_all_six_profiles"
        ),
        "portable_renderer_equality": (
            "frame_surface_rank_component_ids_edge_existence_and_exact_event_records"
        ),
        "renderer_local_evidence": (
            "full_component_maps_masks_counts_extents_directional_counts_and_"
            "annotation_root_retained_not_equalised"
        ),
        "freeze_lock_commit": FREEZE_COMMIT,
        "freeze_lock_root": FREEZE_ROOT,
        "membership": "exact_frozen_160_selected_plus_32_legacy_controls_no_excluded_candidates",
        "required_scenes": ["single_occluder", "corridor"],
        "required_environments": ["windows_wgl_locked", "ubuntu_osmesa_locked"],
        "final_root_policy": (
            "no_rendering_before_pushed_readback_lock_no_adaptation_or_replacement_after_execution"
        ),
        "failure_policy": (
            "retain_every_failed_cell_and_partial_source_no_filter_no_"
            "replacement_any_failure_withholds_qualification"
        ),
        "source_change_policy": (
            "after_first_final_execution_scientific_source_change_OUT_OF_SCOPE_PENDING_OWNER"
        ),
        "authority": {
            "implementation_only": True,
            "full_gate_0b_complete": False,
            "gate_0c_authorised": False,
            "gate_0d_authorised": False,
            "model_protocol_frozen": False,
            "comparative_model_results_authorised": False,
            "scientific_result": None,
        },
    }


def _git(*args: str) -> str:
    environment = os.environ.copy()
    token = environment.get("GITHUB_TOKEN")
    if args and args[0] == "ls-remote" and token:
        # checkout removes persisted credentials; the read token stays in process environment.
        index = int(environment.get("GIT_CONFIG_COUNT", "0"))
        environment["GIT_CONFIG_COUNT"] = str(index + 1)
        environment[f"GIT_CONFIG_KEY_{index}"] = "http.https://github.com/.extraheader"
        encoded = base64.b64encode(f"x-access-token:{token}".encode()).decode()
        environment[f"GIT_CONFIG_VALUE_{index}"] = f"AUTHORIZATION: basic {encoded}"
    return subprocess.run(
        ["git", *args],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    ).stdout.strip()


def membership() -> list[dict[str, Any]]:
    definition = load_benchmark_definition(FREEZE_DEFINITION)
    seeds = load_final_evaluation_seeds(SEEDS)
    selected = _selected_membership_domain(definition, seeds)
    controls = _legacy_control_membership_domain(definition, seeds)
    rows = [{**r, "cell_class": "selected"} for r in selected] + [
        {**r, "cell_class": "control"} for r in controls
    ]
    if len(selected) != 160 or len(controls) != 32 or len({r["cell_id"] for r in rows}) != 192:
        raise ValueError("canonical frozen membership is not exactly 160 selected plus 32 controls")
    return sorted(rows, key=lambda r: r["cell_id"])


def scientific_paths() -> list[Path]:
    return sorted(
        {
            *Path("src").rglob("*.py"),
            *Path("tests").rglob("*.py"),
            *Path("scripts").glob("*.py"),
            *Path("configs").glob("*"),
            Path("uv.lock"),
            Path("pyproject.toml"),
            Path(".github/workflows/ci.yml"),
            Path("docs/GATE_0B_SLICE_6_COMPONENT_TOPOLOGY.md"),
        }
        - {LOCK}
    )


def create_topology_lock() -> dict[str, Any]:
    actual = yaml.safe_load(DEFINITION.read_text(encoding="utf-8"))
    if actual != definition_payload():
        raise ValueError("topology definition has unknown or changed scientific rules")
    freeze = validate_definition_lock(
        FREEZE_DEFINITION, SEEDS, FREEZE_LOCK, REVISION, SINGLE, CORRIDOR
    )
    if freeze["definition_lock_sha256"] != FREEZE_ROOT:
        raise ValueError("canonical appearance freeze dependency changed")
    files = {
        p.as_posix(): sha256_bytes(p.read_bytes().replace(b"\r\n", b"\n"))
        for p in scientific_paths()
        if p.is_file()
    }
    domain = {
        "schema_version": "component_topology_definition_lock_v1",
        "definition": actual,
        "definition_sha256": domain_hash("definition", actual),
        "scientific_source_sha256": files,
        "scientific_source_root": domain_hash("scientific_sources", files),
        "freeze_lock_sha256": FREEZE_ROOT,
        "qualification_membership": membership(),
        "qualification_membership_sha256": domain_hash("qualification_membership", membership()),
        "renderer_environments": load_benchmark_definition(
            FREEZE_DEFINITION
        ).renderer_policy.model_dump(mode="json")["supported_apparatus_environments"],
        "qualification_started": False,
    }
    return {**domain, "topology_lock_sha256": domain_hash("lock", domain)}


def validate_topology_lock() -> dict[str, Any]:
    expected = create_topology_lock()
    raw = LOCK.read_bytes()
    if raw != canonical_json_bytes(expected) + b"\n":
        raise ValueError(
            "topology lock differs from exact independent source/definition reconstruction"
        )
    return expected


def lock_preflight(lock_commit: str) -> tuple[dict[str, Any], str, str]:
    lock = validate_topology_lock()
    if _git("status", "--porcelain"):
        raise ValueError("frozen-root qualification requires a clean exact implementation head")
    head = _git("rev-parse", "HEAD")
    tree = _git("rev-parse", "HEAD^{tree}")
    _require_lock_ancestry(lock_commit, head)
    remote = _git("ls-remote", "origin", f"refs/heads/{BRANCH}").split()
    if remote != [head, f"refs/heads/{BRANCH}"]:
        raise ValueError("remote authorised branch readback differs from exact implementation head")
    return lock, head, tree


def _require_lock_ancestry(lock_commit: str, head: str) -> None:
    first = _git("log", "--diff-filter=A", "--format=%H", "--", LOCK.as_posix()).splitlines()
    if first != [lock_commit]:
        raise ValueError("prospective topology-lock commit is not the unique initial publication")
    _git("merge-base", "--is-ancestor", BASE, lock_commit)
    _git("merge-base", "--is-ancestor", lock_commit, head)
    if _git("show", f"{lock_commit}:{LOCK.as_posix()}") != LOCK.read_text(encoding="utf-8").strip():
        raise ValueError("prospective lock bytes differ from the original lock commit")


class TopologyCell(StrictModel):
    cell_id: str
    cell_class: Literal["selected", "control"]
    scene_family: Literal["single_occluder", "corridor"]
    profile_id: str
    seed_index: int
    evaluation_episode_root: int
    dataset_path: str
    validation_status: Literal["validated", "failed"]
    failure: str | None
    dataset_logical_sha256: Sha256 | None
    source_provenance_sha256: Sha256 | None
    component_topology_sha256: Sha256 | None
    portable_graph_sha256: Sha256 | None
    capability: Literal["available", "indeterminate"] | None
    component_counts: tuple[int, int] | None
    event_counts: dict[ComponentEventKind, int]
    forward_only_pairs: int
    backward_only_pairs: int
    zero_support_pairs: int


class TopologyPacket(StrictModel):
    schema_version: Literal["component_topology_qualification_packet_v1"]
    implementation_head: GitCommit
    implementation_tree: GitCommit
    lock_commit: GitCommit
    lock_tree: GitCommit
    lock_root: Sha256
    remote_readback: GitCommit
    environment_id: Literal["windows_wgl_locked", "ubuntu_osmesa_locked"]
    renderer: RendererProvenance
    cells: tuple[TopologyCell, ...] = Field(min_length=192, max_length=192)
    appearance_invariance: bool
    local_qualification: Literal["qualified", "failed_closed"]
    scene_profile_counts: dict[str, dict[ComponentEventKind, int]]
    portable_matrix_sha256: Sha256
    packet_sha256: Sha256
    scientific_result: None
    gate_advancement: Literal[False]


def _read_cell(
    root: Path, row: dict[str, Any], renderer: RendererProvenance, head: str
) -> TopologyCell:
    relative = "datasets/" + row["cell_id"]
    dataset = root / relative
    common = {
        k: row[k]
        for k in (
            "cell_id",
            "cell_class",
            "scene_family",
            "profile_id",
            "seed_index",
            "evaluation_episode_root",
        )
    }
    try:
        manifest = validate_dataset(dataset)
        expected_config = _profile_config(
            load_config(SINGLE if row["scene_family"] == "single_occluder" else CORRIDOR),
            profile_by_id(load_appearance_registry_any(REVISION), row["profile_id"]),
            row["evaluation_episode_root"],
            "appearance_candidate_registry_v2",
        )
        if (
            manifest.config_logical_sha256 != sha256_bytes(canonical_json_bytes(expected_config))
            or len(manifest.episodes) != 1
            or manifest.renderer_provenance != renderer
            or manifest.source_provenance.git_commit != head
            or manifest.source_provenance.git_dirty is not False
        ):
            raise ValueError(
                "source config, renderer, episode membership or exact clean head differs"
            )
        topology = (
            DatasetLoader(dataset, ModalityPermissionSet.ecological_only())
            .read_component_topology(0)
            .annotation
        )
        counts = Counter(e.kind for e in topology.events)
        return TopologyCell(
            **common,
            dataset_path=relative,
            validation_status="validated",
            failure=None,
            dataset_logical_sha256=manifest.dataset_logical_sha256,
            source_provenance_sha256=manifest.source_provenance_sha256,
            component_topology_sha256=topology.component_topology_sha256,
            portable_graph_sha256=topology.portable_graph_sha256,
            capability=topology.status,
            component_counts=(
                len(topology.frames[0].components),
                len(topology.frames[1].components),
            ),
            event_counts={kind: counts[kind] for kind in ComponentEventKind},
            forward_only_pairs=sum(
                s.forward_count > 0 and s.backward_count == 0 for s in topology.supports
            ),
            backward_only_pairs=sum(
                s.backward_count > 0 and s.forward_count == 0 for s in topology.supports
            ),
            zero_support_pairs=sum(not s.edge for s in topology.supports),
        )
    except Exception as error:
        failure = f"{type(error).__name__}: {error}".replace(
            str(root.resolve()), "<packet>"
        ).replace(str(root), "<packet>")
        return TopologyCell(
            **common,
            dataset_path=relative,
            validation_status="failed",
            failure=failure,
            dataset_logical_sha256=None,
            source_provenance_sha256=None,
            component_topology_sha256=None,
            portable_graph_sha256=None,
            capability=None,
            component_counts=None,
            event_counts={kind: 0 for kind in ComponentEventKind},
            forward_only_pairs=0,
            backward_only_pairs=0,
            zero_support_pairs=0,
        )


def packet_results(cells: tuple[TopologyCell, ...]) -> dict[str, Any]:
    matched: dict[tuple[str, int], list[TopologyCell]] = defaultdict(list)
    counts: dict[str, Counter[ComponentEventKind]] = defaultdict(Counter)
    for cell in cells:
        matched[(cell.scene_family, cell.evaluation_episode_root)].append(cell)
        counts[f"{cell.scene_family}/{cell.profile_id}"].update(cell.event_counts)
    invariant = (
        len(cells) == 192
        and len(matched) == 32
        and all(
            len(group) == 6
            and len({c.component_topology_sha256 for c in group}) == 1
            and all(c.validation_status == "validated" for c in group)
            for group in matched.values()
        )
    )
    qualified = invariant and all(c.capability == "available" for c in cells)
    portable = [
        {
            "cell_id": c.cell_id,
            "portable_graph_sha256": c.portable_graph_sha256,
            "event_counts": {k.value: v for k, v in c.event_counts.items()},
            "capability": c.capability,
        }
        for c in cells
    ]
    return {
        "appearance_invariance": invariant,
        "local_qualification": "qualified" if qualified else "failed_closed",
        "scene_profile_counts": {
            key: {kind: value[kind] for kind in ComponentEventKind}
            for key, value in sorted(counts.items())
        },
        "portable_matrix_sha256": domain_hash("portable_matrix", portable),
    }


def qualify_topology(output: Path, lock_commit: str) -> TopologyPacket:
    lock, head, tree = lock_preflight(lock_commit)  # No final rendering before this boundary.
    renderer = _renderer_provenance()
    environment = _renderer_environment_id(
        load_benchmark_definition(FREEZE_DEFINITION), renderer.model_dump(mode="json")
    )
    output.mkdir(parents=True, exist_ok=False)
    write_canonical_json(output / "topology_lock_snapshot.json", lock)
    write_canonical_json(
        output / "run_metadata.json",
        {"started_utc": datetime.now(UTC).isoformat(), "scientific_adaptation_permitted": False},
    )
    registry = load_appearance_registry_any(REVISION)
    seeds = load_final_evaluation_seeds(SEEDS)
    rows: list[TopologyCell] = []
    for index, row in enumerate(membership(), 1):
        destination = output / "datasets" / row["cell_id"]
        config = _profile_config(
            load_config(SINGLE if row["scene_family"] == "single_occluder" else CORRIDOR),
            profile_by_id(registry, row["profile_id"]),
            row["evaluation_episode_root"],
            registry.registry_version,
        )
        try:
            generate_dataset(
                config,
                1,
                destination,
                appearance_registry=registry,
                seed_registry=seeds,
                component_topology=True,
            )
        except Exception as error:
            failure_path = output / "generation_failures" / (row["cell_id"] + ".json")
            failure_path.parent.mkdir(exist_ok=True)
            write_canonical_json(
                failure_path,
                {
                    "cell_id": row["cell_id"],
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
        cell = _read_cell(output, row, renderer, head)
        rows.append(cell)
        print(
            f"{index}/192 {cell.cell_id}: {cell.validation_status}; {cell.capability}", flush=True
        )
    cells = tuple(rows)
    packet = TopologyPacket(
        schema_version="component_topology_qualification_packet_v1",
        implementation_head=head,
        implementation_tree=tree,
        lock_commit=lock_commit,
        lock_tree=_git("rev-parse", f"{lock_commit}^{{tree}}"),
        lock_root=lock["topology_lock_sha256"],
        remote_readback=head,
        environment_id=cast(Literal["windows_wgl_locked", "ubuntu_osmesa_locked"], environment),
        renderer=renderer,
        cells=cells,
        **packet_results(cells),
        packet_sha256="0" * 64,
        scientific_result=None,
        gate_advancement=False,
    )
    packet = packet.model_copy(
        update={
            "packet_sha256": domain_hash(
                "qualification_packet", packet.model_dump(mode="json", exclude={"packet_sha256"})
            )
        }
    )
    write_canonical_json(output / "packet.json", packet)
    validate_topology_packet(output)
    return packet


def validate_topology_packet(root: Path) -> TopologyPacket:
    lock = validate_topology_lock()
    with open_owned_regular_file(root, "packet.json") as owned:
        packet = TopologyPacket.model_validate_json(owned.payload)
        if owned.payload != canonical_json_bytes(packet) + b"\n":
            raise ValueError("topology qualification packet is not canonical")
    with open_owned_regular_file(root, "topology_lock_snapshot.json") as owned:
        if owned.payload != canonical_json_bytes(lock) + b"\n":
            raise ValueError("topology packet substitutes its prospective lock snapshot")
    if (
        packet.lock_root != lock["topology_lock_sha256"]
        or packet.remote_readback != packet.implementation_head
    ):
        raise ValueError("topology packet lock or readback binding differs")
    if (
        _git("rev-parse", f"{packet.implementation_head}^{{tree}}") != packet.implementation_tree
        or _git("rev-parse", f"{packet.lock_commit}^{{tree}}") != packet.lock_tree
    ):
        raise ValueError("topology packet source tree binding differs")
    _require_lock_ancestry(packet.lock_commit, packet.implementation_head)
    expected_environment = _renderer_environment_id(
        load_benchmark_definition(FREEZE_DEFINITION), packet.renderer.model_dump(mode="json")
    )
    if packet.environment_id != expected_environment:
        raise ValueError("topology packet renderer environment differs")
    expected = tuple(
        _read_cell(root, row, packet.renderer, packet.implementation_head) for row in membership()
    )
    if packet.cells != expected:
        raise ValueError(
            "topology packet cells differ from complete independent source reconstruction"
        )
    for key, value in packet_results(expected).items():
        if getattr(packet, key) != value:
            raise ValueError(f"topology packet independently reconstructed {key} differs")
    if packet.packet_sha256 != domain_hash(
        "qualification_packet", packet.model_dump(mode="json", exclude={"packet_sha256"})
    ):
        raise ValueError("topology qualification packet hash differs")
    return packet


def compare_topology_packets(first: Path, second: Path) -> dict[str, Any]:
    a, b = validate_topology_packet(first), validate_topology_packet(second)
    if (
        a.environment_id == b.environment_id
        or a.implementation_head != b.implementation_head
        or a.lock_root != b.lock_root
    ):
        raise ValueError(
            "cross-renderer comparison needs both locked environments at one exact head/lock"
        )
    differences = [
        x.cell_id
        for x, y in zip(a.cells, b.cells, strict=True)
        if (x.portable_graph_sha256, x.event_counts, x.capability)
        != (y.portable_graph_sha256, y.event_counts, y.capability)
        or x.validation_status != "validated"
        or y.validation_status != "validated"
    ]
    return {
        "schema_version": "component_topology_cross_renderer_comparison_v1",
        "implementation_head": a.implementation_head,
        "lock_root": a.lock_root,
        "packet_roots": [a.packet_sha256, b.packet_sha256],
        "compared_cells": 192,
        "portable_equality": not differences,
        "differing_cells": differences,
        "qualification": "qualified"
        if not differences and a.local_qualification == b.local_qualification == "qualified"
        else "failed_closed",
        "scientific_result": None,
        "gate_advancement": False,
    }
