"""Independent small raster expectations; no native geometry or prediction claim."""

from __future__ import annotations

import ast
import hashlib
import json
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from epsbench.diagnostics.boundary_observation import VisibleRaster
from epsbench.diagnostics.neutral_observation_target import (
    CAUSE_STATUS,
    Candidate,
    SemanticTarget,
    evaluate,
)
from epsbench.diagnostics.visible_forecast_contract import REQUIRED, CausalView, Limits
from epsbench.schema import ModalityPermissionSet

A = "surface-0000000000000001"
B = "surface-0000000000000002"
C = "surface-0000000000000003"
D = "surface-0000000000000004"
ACCESS = ModalityPermissionSet(allowed=REQUIRED)
ZERO = (Fraction(0), Fraction(0), Fraction(0))


def mask(values: list[bool]) -> np.ndarray[Any, np.dtype[np.bool_]]:
    return np.array([values], dtype=np.bool_)


def raster(values: list[int], index: int, ids: dict[int, str]) -> VisibleRaster:
    return VisibleRaster(index, np.array([values], dtype=np.int32), tuple(ids.items()))


class Provider:
    def __init__(self, frames: dict[int, VisibleRaster]) -> None:
        self.frames = frames
        self.calls: list[int] = []

    def raster(self, index: int) -> VisibleRaster:
        self.calls.append(index)
        return self.frames[index]


def example(names: tuple[str, str] = (A, B)) -> tuple[Any, Provider]:
    a, b = names
    provider = Provider(
        {
            0: raster([1, 0, 0, 0], 0, {1: a}),
            1: raster([0, 2, 0, 0], 1, {2: b}),
            2: raster([1, 3, 4, 0], 2, {1: a, 3: C, 4: D}),
        }
    )
    source = CausalView(provider, ACCESS, 1, Limits(4, 4, 4)).materialize((ZERO,), ZERO)
    provider.calls.clear()
    return source, provider


def perfect(source: Any) -> Candidate:
    return Candidate(
        source,
        ((A, mask([True, False, False, False])), (B, mask([False] * 4))),
        mask([False, True, True, False]),
    )


def test_returning_known_new_union_and_neutral_replay() -> None:
    source, provider = example()
    saved = perfect(source).canonical_bytes()
    result = evaluate(source, saved, provider)
    assert provider.calls == [2]
    payload = json.loads(result.target.canonical_bytes())
    assert payload == {
        "version": "neutral-observation-target-v1:semantic",
        "target_index": 2,
        "shape": [1, 4],
        "known": {A: [[True, False, False, False]], B: [[False] * 4]},
        "new": [[False, True, True, False]],
    }
    report = result.report
    assert (report["N"], report["U"], report["C"], report["E"]) == (12, 0, 12, 0)
    assert report["observation_exact"] and report["complete_assertion"]
    assert report["target_occupied_pixels"] == 3
    assert report["first_observed_pixels"] == 2
    assert report["first_observed_fraction"] == Fraction(2, 3)
    labels = report["neutral_target_labels"]
    assert labels[A]["changes"] == (
        {"surface_id": A, "change": "region_appeared", "affected_image_pixels": 1},
    )
    assert labels[B]["changes"] == (
        {"surface_id": B, "change": "region_disappeared", "affected_image_pixels": 1},
    )
    assert labels[A]["visibility"]["before_visible_pixels"] == 0
    assert labels[A]["visibility"]["after_projected_image_fraction"] == 0.25
    assert report["neutral_candidate_labels"] == labels
    assert report["cause_status"] == CAUSE_STATUS
    assert "qualified" in report["cause_reason"]
    assert not any("cause_array" in key or "cause_score" in key for key in report)


def test_semantics_independent_of_candidate_partition_local_labels_and_receipt() -> None:
    source, provider = example()
    first = evaluate(source, perfect(source).canonical_bytes(), provider)
    different = Candidate(
        source, ((A, mask([False] * 4)), (B, mask([False, True, False, False]))), mask([False] * 4)
    )
    second = evaluate(source, different.canonical_bytes(), provider)
    assert first.target.canonical_bytes() == second.target.canonical_bytes()
    assert first.target.digest == second.target.digest
    assert first.report["E"] == 0 and second.report["E"] == 4
    assert first.receipt_bytes != second.receipt_bytes
    provider.frames[2] = raster([7, 9, 9, 0], 2, {7: A, 9: C})
    changed_partition = evaluate(source, perfect(source).canonical_bytes(), provider)
    assert first.target.canonical_bytes() == changed_partition.target.canonical_bytes()
    assert first.target.digest == changed_partition.target.digest
    assert first.report == changed_partition.report
    assert first.receipt_bytes != changed_partition.receipt_bytes
    payload = first.target.canonical_bytes().decode()
    assert C not in payload and D not in payload
    assert "sha256" not in payload and "identities" not in payload
    receipt = json.loads(changed_partition.receipt_bytes)
    assert receipt["semantic_target_sha256"] == first.target.digest
    assert set(receipt) == {
        "version",
        "input_sha256",
        "candidate_sha256",
        "semantic_target_sha256",
        "target_index",
        "shape",
        "observation_domain",
        "observation_sha256",
    }


def test_known_token_equivariance() -> None:
    source, provider = example()
    base = evaluate(source, perfect(source).canonical_bytes(), provider)
    renamed, target_provider = example((B, A))
    prediction = Candidate(
        renamed,
        ((B, mask([True, False, False, False])), (A, mask([False] * 4))),
        mask([False, True, True, False]),
    )
    other = evaluate(renamed, prediction.canonical_bytes(), target_provider)
    base_masks = dict(base.target.channels.known)
    other_masks = dict(other.target.channels.known)
    np.testing.assert_array_equal(base_masks[A], other_masks[B])
    np.testing.assert_array_equal(base_masks[B], other_masks[A])
    assert base.target.digest != other.target.digest
    assert base.report["E"] == other.report["E"] == 0
    assert base.report["first_observed_pixels"] == other.report["first_observed_pixels"] == 2


def test_unknown_is_not_empty_and_report_snapshot_is_immutable() -> None:
    source, provider = example()
    unknown = Candidate(source, ((A, None), (B, mask([False] * 4))), None)
    result = evaluate(source, unknown.canonical_bytes(), provider)
    report = result.report
    assert (report["N"], report["U"], report["C"], report["E"]) == (12, 8, 4, 0)
    assert report["conditional_error"] == 0
    assert report["channel_pixel_coverage"] == report["channel_coverage"] == Fraction(1, 3)
    assert report["ignorance_interval"] == (0, Fraction(2, 3))
    assert not report["observation_exact"] and not report["complete_assertion"]
    assert report["neutral_candidate_labels"][A] is None
    with pytest.raises(TypeError):
        report["E"] = 9  # type: ignore[index]
    with pytest.raises(TypeError):
        report["channels"][0]["error"] = 9
    target_bytes = result.target.canonical_bytes()
    view = result.target.channels.new
    assert view is not None
    with pytest.raises(ValueError):
        view.setflags(write=True)
    view.shape = (4, 1)
    assert result.target.canonical_bytes() == target_bytes
    provider.frames[2].segmentation.shape = (4, 1)
    assert result.target.canonical_bytes() == target_bytes
    all_unknown = evaluate(
        source, Candidate(source, ((A, None), (B, None)), None).canonical_bytes(), example()[1]
    )
    assert all_unknown.report["conditional_error"] is None
    assert all_unknown.report["ignorance_interval"] == (0, 1)
    assert all_unknown.report["channel_pixel_coverage"] == 0


def test_empty_inventory_still_scores_new_and_zero_occupied_denominator() -> None:
    provider = Provider({0: raster([0, 0], 0, {}), 1: raster([7, 0], 1, {7: C})})
    source = CausalView(provider, ACCESS, 0, Limits(2, 2, 2)).materialize((), ZERO)
    wrong = evaluate(
        source, Candidate(source, (), mask([False, False])).canonical_bytes(), provider
    )
    assert (wrong.report["N"], wrong.report["E"]) == (2, 1)
    assert wrong.report["first_observed_fraction"] == 1
    assert not wrong.report["observation_exact"]
    provider.frames[1] = raster([0, 0], 1, {})
    empty = evaluate(
        source, Candidate(source, (), mask([False, False])).canonical_bytes(), provider
    )
    assert empty.report["observation_exact"]
    assert empty.report["first_observed_fraction"] is None
    assert empty.report["neutral_target_labels"] == {}


@pytest.mark.parametrize(
    "change",
    [
        "extra",
        "missing-new",
        "future",
        "missing-known",
        "bool-index",
        "wrong-index",
        "digest",
        "shape",
        "nonbool",
        "overlap",
        "duplicate",
        "causes",
    ],
)
def test_malformed_envelopes_rejected_before_target_fetch(change: str) -> None:
    source, provider = example()
    data = perfect(source).canonical_bytes()
    payload = json.loads(data)
    if change == "extra":
        payload["seed"] = 8
    elif change == "missing-new":
        del payload["new"]
    elif change == "future":
        payload["known"][C] = [[False] * 4]
    elif change == "missing-known":
        del payload["known"][B]
    elif change == "bool-index":
        payload["target_index"] = True
    elif change == "wrong-index":
        payload["target_index"] = 3
    elif change == "digest":
        payload["input_sha256"] = "0" * 64
    elif change == "shape":
        payload["shape"] = [2, 2]
    elif change == "nonbool":
        payload["new"] = [[0, 1, 1, 0]]
    elif change == "overlap":
        payload["new"] = [[True, False, False, False]]
    elif change == "causes":
        payload["cause_codes"] = [[0] * 4]
    bad = (
        data.replace(b'"target_index":', b'"target_index":2,"target_index":')
        if change == "duplicate"
        else json.dumps(payload).encode()
    )
    with pytest.raises(ValueError):
        evaluate(source, bad, provider)
    assert provider.calls == []


def test_candidate_roundtrip_owned_views_and_target_validation() -> None:
    source, provider = example()
    array = mask([True, False, False, False])
    candidate = Candidate(
        source, ((A, array), (B, mask([False] * 4))), mask([False, True, True, False])
    )
    data = candidate.canonical_bytes()
    array[:] = False
    assert Candidate.from_bytes(data, source).canonical_bytes() == data
    assert candidate.canonical_bytes() == data
    with pytest.raises(ValueError):
        SemanticTarget(True, source.shape, (), mask([False] * 4))
    with pytest.raises(ValueError):
        SemanticTarget(2, source.shape, ((A, None),), mask([False] * 4))
    provider.frames[2] = raster([1, 0, 0, 0], 1, {1: A})
    with pytest.raises(ValueError, match="chronology"):
        evaluate(source, data, provider)
    assert provider.calls == [2]


def test_annotation_exports_preserved_and_native_modules_unloaded() -> None:
    import epsbench.annotations as annotations

    # Fixed receipt of the pre-change 61-name public export set, not a new export contract.
    assert len(annotations.__all__) == 61
    assert (
        hashlib.sha256(
            json.dumps(sorted(annotations.__all__), separators=(",", ":")).encode()
        ).hexdigest()
        == "9b55b039b04a26a363fa94958967bde8b2d6c714e309b15c211f9b3b3312f635"
    )
    assert annotations.derive_visibility.__module__ == "epsbench.annotations.derive"
    source = Path(annotations.__file__).read_text(encoding="utf-8")
    static = {
        node.module: tuple(alias.name for alias in node.names)
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ImportFrom) and node.module in annotations._NATIVE_EXPORTS
    }
    assert static == annotations._NATIVE_EXPORTS
    assert set(annotations.__all__) == {name for names in static.values() for name in names} | {
        "derive_visibility",
        "derive_boundary_structure",
        "classify_mask_changes",
    }
    assert "mujoco" not in sys.modules
    assert "epsbench.annotations.boundary_events" not in sys.modules
    assert "epsbench.annotations.optical_transport" not in sys.modules
    with pytest.raises(AttributeError):
        missing_name = "not_a_public_export"
        getattr(annotations, missing_name)


def test_exact_ci_route_excludes_legacy_native_jobs() -> None:
    root = Path(__file__).resolve().parents[1]
    jobs = yaml.safe_load((root / ".github/workflows/ci.yml").read_text())["jobs"]
    branch = "codex/neutral-observation-target-20261007"
    assert branch in jobs["neutral-target-source"]["if"]
    assert "codex/a1-integration-base-20261005" in jobs["neutral-target-source"]["if"]
    for name in ("quality", "qualify-wgl", "qualify-osmesa"):
        assert branch in jobs[name]["if"] and "!(" in jobs[name]["if"]
    for name in ("a1-source", "restricted-models-source", "corridor-aperture-source"):
        assert branch not in jobs[name]["if"]
    commands = "\n".join(step.get("run", "") for step in jobs["neutral-target-source"]["steps"])
    assert "check_neutral_target_source.py" in commands
    assert "--no-install-package mujoco" in commands
    assert "restricted-models" not in commands
    assert "epsbench generate" not in commands and "--boundary-observation-only" not in commands
