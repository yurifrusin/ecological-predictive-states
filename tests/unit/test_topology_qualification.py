"""Prospective metadata and synthetic qualification tests; no final-root rendering."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import pytest

import epsbench.topology_qualification as qualification
from epsbench.annotations.component_topology import domain_hash
from epsbench.schema import ComponentEventKind as Kind
from epsbench.schema import RendererProvenance
from epsbench.topology_qualification import TopologyCell, TopologyPacket, packet_results
from epsbench.utils.canonical import canonical_json_bytes, write_canonical_json


def synthetic_cells() -> tuple[TopologyCell, ...]:
    # Artificial control metadata, unrelated to any final evaluation root or rendered source.
    scenes: tuple[Literal["single_occluder", "corridor"], ...] = ("single_occluder", "corridor")
    return tuple(
        TopologyCell(
            cell_id=f"synthetic-{scene}-{profile}-{root}",
            cell_class="control" if profile == 0 else "selected",
            scene_family=scene,
            profile_id=f"synthetic_profile_{profile}",
            seed_index=root,
            evaluation_episode_root=1000 + root,
            dataset_path=f"synthetic/{scene}-{profile}-{root}",
            validation_status="validated",
            failure=None,
            dataset_logical_sha256="1" * 64,
            source_provenance_sha256="2" * 64,
            component_topology_sha256="3" * 64,
            portable_graph_sha256="4" * 64,
            capability="available",
            component_counts=(1, 1),
            event_counts={kind: int(kind == Kind.CONTINUATION) for kind in Kind},
            forward_only_pairs=0,
            backward_only_pairs=0,
            zero_support_pairs=0,
        )
        for scene in scenes
        for root in range(16)
        for profile in range(6)
    )


def test_exact_frozen_membership_metadata_without_rendering() -> None:
    rows = qualification.membership()
    assert len(rows) == len({r["cell_id"] for r in rows}) == 192
    assert sum(r["cell_class"] == "selected" for r in rows) == 160
    assert sum(r["cell_class"] == "control" for r in rows) == 32
    assert len({r["evaluation_episode_root"] for r in rows}) == 16
    assert not any("stripes" in r["profile_id"] or "illumination" in r["profile_id"] for r in rows)


@pytest.mark.parametrize(
    "mutation", ["indeterminate", "failed_source", "appearance", "missing", "complex", "none"]
)
def test_prospective_acceptance_is_exact_and_retains_all_outcomes(mutation: str) -> None:
    rows = list(synthetic_cells())
    if mutation == "indeterminate":
        rows[0] = rows[0].model_copy(
            update={
                "capability": "indeterminate",
                "event_counts": {kind: int(kind == Kind.INDETERMINATE) for kind in Kind},
            }
        )
    elif mutation == "failed_source":
        rows[0] = rows[0].model_copy(update={"validation_status": "failed", "capability": None})
    elif mutation == "appearance":
        rows[0] = rows[0].model_copy(update={"component_topology_sha256": "5" * 64})
    elif mutation == "missing":
        rows.pop()
    elif mutation == "complex":
        rows = [
            row.model_copy(
                update={"event_counts": {kind: int(kind == Kind.COMPLEX) for kind in Kind}}
            )
            for row in rows
        ]
    result = packet_results(tuple(rows))
    assert result["local_qualification"] == (
        "qualified" if mutation in {"none", "complex"} else "failed_closed"
    )
    if mutation == "indeterminate":
        assert sum(c[Kind.INDETERMINATE] for c in result["scene_profile_counts"].values()) == 1
    if mutation == "complex":
        assert sum(c[Kind.COMPLEX] for c in result["scene_profile_counts"].values()) == 192


def test_preflight_blocks_dirty_source_before_output_or_generation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(qualification, "validate_topology_lock", lambda: {})
    monkeypatch.setattr(qualification, "_git", lambda *args: " M scientific_source.py")

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("lock preflight allowed final-root generation")

    monkeypatch.setattr(qualification, "generate_dataset", forbidden)
    with pytest.raises(ValueError, match="clean exact"):
        qualification.qualify_topology(tmp_path / "must-not-exist", "0" * 40)
    assert not (tmp_path / "must-not-exist").exists()


def test_unknown_definition_fields_fail_before_lock_or_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    definition = qualification.definition_payload()
    definition["unknown_rule"] = True
    path = tmp_path / "definition.yaml"
    path.write_bytes(canonical_json_bytes(definition))
    monkeypatch.setattr(qualification, "DEFINITION", path)
    with pytest.raises(ValueError, match="unknown or changed"):
        qualification.create_topology_lock()


@pytest.mark.parametrize("mutation", ["extra", "wrong_root", "changed_rule", "changed_source"])
def test_lock_rejects_complete_rehash_of_false_claims(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    expected = {
        "topology_lock_sha256": "1" * 64,
        "definition": {"a": 1},
        "sources": {"a": "2" * 64},
    }
    actual = json.loads(json.dumps(expected))
    if mutation == "extra":
        actual["extra"] = 1
    elif mutation == "changed_rule":
        actual["definition"]["a"] = 2
    elif mutation == "changed_source":
        actual["sources"]["a"] = "3" * 64
    actual["topology_lock_sha256"] = domain_hash(
        "lock", {k: v for k, v in actual.items() if k != "topology_lock_sha256"}
    )
    path = tmp_path / "lock.json"
    write_canonical_json(path, actual)
    monkeypatch.setattr(qualification, "LOCK", path)
    monkeypatch.setattr(qualification, "create_topology_lock", lambda: expected)
    with pytest.raises(ValueError, match="independent source"):
        qualification.validate_topology_lock()


def test_renderer_comparison_retains_mismatch_and_requires_same_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = synthetic_cells()
    # Synthetic packet isolates comparison logic without a rendered source claim.
    a = TopologyPacket(
        schema_version="component_topology_qualification_packet_v1",
        environment_id="windows_wgl_locked",
        implementation_head="a" * 40,
        implementation_tree="b" * 40,
        lock_commit="c" * 40,
        lock_tree="d" * 40,
        remote_readback="a" * 40,
        renderer=RendererProvenance(
            mujoco_version="3.12.0",
            numpy_version="2.4.6",
            renderer="mujoco.Renderer",
            backend="wgl-default",
            operating_system="Windows",
        ),
        lock_root="1" * 64,
        packet_sha256="2" * 64,
        cells=rows,
        **packet_results(rows),
        scientific_result=None,
        gate_advancement=False,
    )
    b = a.model_copy(update={"environment_id": "ubuntu_osmesa_locked", "packet_sha256": "3" * 64})
    monkeypatch.setattr(
        qualification, "validate_topology_packet", lambda p: a if p.name == "a" else b
    )
    assert (
        qualification.compare_topology_packets(Path("a"), Path("b"))["qualification"] == "qualified"
    )
    b = b.model_copy(
        update={
            "cells": (rows[0].model_copy(update={"portable_graph_sha256": "9" * 64}), *rows[1:])
        }
    )
    result = qualification.compare_topology_packets(Path("a"), Path("b"))
    assert result["qualification"] == "failed_closed"
    assert result["differing_cells"] == [rows[0].cell_id]
    b = b.model_copy(update={"implementation_head": "b" * 40})
    with pytest.raises(ValueError, match="one exact head"):
        qualification.compare_topology_packets(Path("a"), Path("b"))
