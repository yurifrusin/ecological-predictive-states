from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from epsbench import failure_analysis as failure_analysis_module
from epsbench.audit import _hash_json, _ValidatedAppearanceAudit
from epsbench.failure_analysis import (
    ANALYSIS_SCHEMA_VERSION,
    CANONICAL_BASE,
    CANONICAL_COUNTS,
    CANONICAL_ROOTS,
    _analysis_domains,
    _causal_diagnostic_answers,
    _markdown,
    _renderer_fingerprints,
    create_failure_analysis,
    validate_failure_analysis,
)
from epsbench.utils.canonical import canonical_json_bytes, write_canonical_json


@pytest.fixture(autouse=True)
def _accept_synthetic_source_packet(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "epsbench.failure_analysis._validate_appearance_audit_evidence",
        lambda packet_root, **kwargs: _synthetic_evidence(packet_root),
    )


def _synthetic_evidence(
    source: Path,
    packet: dict[str, Any] | None = None,
) -> _ValidatedAppearanceAudit:
    return _ValidatedAppearanceAudit(
        packet_payload=canonical_json_bytes(packet or {"packet_logical_root_sha256": "0" * 64})
        + b"\n",
        seed_matrix_payload=(source / "seed_matrix.json").read_bytes(),
    )


def _analysis_bundle(workspace: Path) -> tuple[Path, Path, dict[str, Any]]:
    root = workspace / "analysis"
    source = workspace / "source"
    root.mkdir()
    source.mkdir()
    cells = [
        {
            "generation_status": "failed",
            "cell_id": "test-cell",
            "profile_id": "test-profile",
            "scene_family": "corridor",
            "seed_index": 0,
            "candidate_seed": 1,
            "failure_type": "SyntheticFailure",
            "failure_message": "test-only source evidence",
        }
    ]
    profile, surface, thresholds, diagnostics = _analysis_domains(cells)
    children = {
        "profile_failure_matrix.json": profile,
        "surface_failure_matrix.json": surface,
        "threshold_margin_summary.json": thresholds,
    }
    portable = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "source_canonical_base": "0" * 40,
        "source_matrix_counts": {"admitted": 0, "rejected": 0},
        "source_profile_count": 0,
        "source_roots": {},
        "canonical_admission_thresholds": {},
        "analysis_method": "test-only",
    }
    renderer = {
        **portable,
        "baseline_failure_analysis_sha256": _hash_json(portable),
        "source_packet_logical_root_sha256": "0" * 64,
        "renderer_fingerprints": _renderer_fingerprints(cells),
        "profile_failure_matrix_sha256": _hash_json(profile),
        "surface_failure_matrix_sha256": _hash_json(surface),
        "threshold_margin_summary_sha256": _hash_json(thresholds),
        "diagnostic_summary": diagnostics,
        "causal_diagnostic_answers": _causal_diagnostic_answers(diagnostics),
    }
    analysis = {
        **renderer,
        "renderer_specific_failure_analysis_sha256": _hash_json(renderer),
    }
    write_canonical_json(source / "seed_matrix.json", {"cells": cells})
    for name, payload in children.items():
        write_canonical_json(root / name, payload)
    write_canonical_json(root / "baseline_failure_analysis.json", analysis)
    (root / "baseline_failure_analysis.md").write_bytes(_markdown(analysis).encode("utf-8"))
    return root, source, analysis


def _rehash_renderer_domain(analysis: dict[str, Any]) -> None:
    renderer = dict(analysis)
    renderer.pop("renderer_specific_failure_analysis_sha256")
    analysis["renderer_specific_failure_analysis_sha256"] = _hash_json(renderer)


def test_complete_failure_analysis_bundle_is_accepted(tmp_path: Path) -> None:
    root, source, expected = _analysis_bundle(tmp_path)
    assert validate_failure_analysis(root, source_packet=source) == expected


def test_failure_analysis_requires_source_evidence(tmp_path: Path) -> None:
    root, _, _ = _analysis_bundle(tmp_path)
    with pytest.raises(ValueError, match="source packet is required"):
        validate_failure_analysis(root)


def test_fully_rehashed_renderer_fingerprint_is_rejected(tmp_path: Path) -> None:
    root, source, analysis = _analysis_bundle(tmp_path)
    analysis["renderer_fingerprints"] = [
        canonical_json_bytes({"renderer": "forged"}).decode("utf-8")
    ]
    _rehash_renderer_domain(analysis)
    write_canonical_json(root / "baseline_failure_analysis.json", analysis)
    with pytest.raises(ValueError, match="renderer fingerprints differ from source evidence"):
        validate_failure_analysis(root, source_packet=source)


def test_fully_rehashed_causal_answer_is_rejected(tmp_path: Path) -> None:
    root, source, analysis = _analysis_bundle(tmp_path)
    analysis["causal_diagnostic_answers"][0]["answer"] = "forged causal answer"
    _rehash_renderer_domain(analysis)
    write_canonical_json(root / "baseline_failure_analysis.json", analysis)
    (root / "baseline_failure_analysis.md").write_bytes(_markdown(analysis).encode("utf-8"))
    with pytest.raises(ValueError, match="causal answers differ from reconstruction"):
        validate_failure_analysis(root, source_packet=source)


@pytest.mark.parametrize(
    "name",
    (
        "baseline_failure_analysis.json",
        "profile_failure_matrix.json",
        "surface_failure_matrix.json",
        "threshold_margin_summary.json",
    ),
)
def test_noncanonical_failure_analysis_json_is_rejected(tmp_path: Path, name: str) -> None:
    root, source, _ = _analysis_bundle(tmp_path)
    path = root / name
    payload = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON is not canonical"):
        validate_failure_analysis(root, source_packet=source)


def test_additional_failure_analysis_directory_is_rejected(tmp_path: Path) -> None:
    root, source, _ = _analysis_bundle(tmp_path)
    (root / "unexpected").mkdir()
    with pytest.raises(ValueError, match="missing or additional entries"):
        validate_failure_analysis(root, source_packet=source)


def test_undeclared_run_metadata_is_rejected(tmp_path: Path) -> None:
    root, source, _ = _analysis_bundle(tmp_path)
    write_canonical_json(root / "run.json", {"arbitrary": "volatile"})
    with pytest.raises(ValueError, match="missing or additional entries"):
        validate_failure_analysis(root, source_packet=source)


@pytest.mark.parametrize(
    "name",
    (
        "baseline_failure_analysis.json",
        "baseline_failure_analysis.md",
        "profile_failure_matrix.json",
        "surface_failure_matrix.json",
        "threshold_margin_summary.json",
    ),
)
def test_hardlinked_failure_analysis_artifact_is_rejected(
    tmp_path: Path,
    name: str,
) -> None:
    root, source, _ = _analysis_bundle(tmp_path)
    original = root / name
    alias_source = tmp_path / f"{name}.alias-source"
    alias_source.write_bytes(original.read_bytes())
    original.unlink()
    os.link(alias_source, original)
    with pytest.raises(ValueError, match="hard-link alias"):
        validate_failure_analysis(root, source_packet=source)


def test_failure_analysis_root_link_is_rejected(tmp_path: Path) -> None:
    root, source, _ = _analysis_bundle(tmp_path)
    alias = tmp_path / "alias"
    try:
        alias.symlink_to(root, target_is_directory=True)
    except OSError:
        pytest.skip("directory symbolic links are unavailable")
    with pytest.raises(ValueError, match="root must be a non-link directory"):
        validate_failure_analysis(alias, source_packet=source)


def test_seed_matrix_replacement_after_packet_validation_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, source, analysis = _analysis_bundle(tmp_path)
    original_matrix = json.loads((source / "seed_matrix.json").read_text(encoding="utf-8"))
    forged_matrix = json.loads(json.dumps(original_matrix))
    forged_matrix["cells"][0]["failure_message"] = "forged post-validation evidence"
    forged_profile, _, _, _ = _analysis_domains(forged_matrix["cells"])
    analysis["profile_failure_matrix_sha256"] = _hash_json(forged_profile)
    _rehash_renderer_domain(analysis)
    write_canonical_json(root / "profile_failure_matrix.json", forged_profile)
    write_canonical_json(root / "baseline_failure_analysis.json", analysis)

    def validate_then_replace(*args: Any, **kwargs: Any) -> _ValidatedAppearanceAudit:
        evidence = _synthetic_evidence(source)
        write_canonical_json(source / "seed_matrix.json", forged_matrix)
        return evidence

    monkeypatch.setattr(
        failure_analysis_module,
        "_validate_appearance_audit_evidence",
        validate_then_replace,
    )
    with pytest.raises(ValueError, match="reports differ from source evidence"):
        validate_failure_analysis(root, source_packet=source)


def test_failure_analysis_creation_uses_validated_seed_matrix_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "analysis"
    source.mkdir()
    original_cells = [
        {
            "generation_status": "failed",
            "cell_id": f"test-cell-{index}",
            "profile_id": "test-profile",
            "scene_family": "corridor",
            "seed_index": index,
            "candidate_seed": index,
            "failure_type": "SyntheticFailure",
            "failure_message": "validated source evidence",
        }
        for index in range(160)
    ]
    write_canonical_json(source / "seed_matrix.json", {"cells": original_cells})
    forged_matrix = {"cells": json.loads(json.dumps(original_cells))}
    forged_matrix["cells"][0]["failure_message"] = "forged post-validation evidence"
    packet = {
        "matrix_counts": CANONICAL_COUNTS,
        "roots": CANONICAL_ROOTS,
        "source_provenance": {"git_commit": CANONICAL_BASE},
        "profile_count": 10,
        "packet_logical_root_sha256": "0" * 64,
    }

    def validate_then_replace(*args: Any, **kwargs: Any) -> _ValidatedAppearanceAudit:
        evidence = _synthetic_evidence(source, packet)
        write_canonical_json(source / "seed_matrix.json", forged_matrix)
        return evidence

    monkeypatch.setattr(
        failure_analysis_module,
        "_validate_appearance_audit_evidence",
        validate_then_replace,
    )
    create_failure_analysis(source, output)
    profile_matrix = json.loads(
        (output / "profile_failure_matrix.json").read_text(encoding="utf-8")
    )
    assert profile_matrix["rows"][0]["failure_message"] == "validated source evidence"
    assert json.loads((source / "seed_matrix.json").read_text(encoding="utf-8")) == forged_matrix
