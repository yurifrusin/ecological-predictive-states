"""Deterministic privileged failure analysis for the canonical Slice 5 packet."""

from __future__ import annotations

import json
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any

from epsbench.audit import _atomic_no_replace_directory, _hash_json, validate_appearance_audit
from epsbench.data.paths import open_owned_regular_file
from epsbench.utils.canonical import canonical_json_bytes, write_canonical_json

ANALYSIS_SCHEMA_VERSION = "appearance_failure_analysis_v0"
CANONICAL_BASE = "9df0b7a37ab81b02855ef8016e60d79c4f671071"
CANONICAL_COUNTS = {"admitted": 44, "rejected": 116}
CANONICAL_ROOTS = {
    "appearance_registry_sha256": (
        "f90feb3cf9d798ab61c3adbb8d2b276c5d2cb131b95c5a9187df151f9b801d60"
    ),
    "seed_registry_sha256": "6c6816ae1f6657a710631f08a435cca4c623e81efc71ac4e6330a44338313482",
    "procedural_asset_root_sha256": (
        "6eb582b0c636e79349a11edab50394c26b15f07d55327cdd081ef999131b5724"
    ),
    "appearance_assignment_root_sha256": (
        "10a24a068c80cd2b85ff00a55b4d0c5a02ba64617f97ac808a9417197bcea90d"
    ),
    "portable_analytic_identity_root_sha256": (
        "2cb8b42739418faae9666dcb518de3e6655b9ee7f42c63c3acd337debcd224d5"
    ),
    "appearance_invariance_outcome_root_sha256": (
        "2bc8edfe05b0401c48a8bc3d897e965c1925ec53dbdf4273d0ee60cc034b3cde"
    ),
}
SURFACE_NORMAL_CLASSES = {
    "support_surface": "upward_horizontal",
    "occluding_surface": "camera_facing_vertical",
    "background_surface": "camera_facing_vertical",
    "corridor_floor": "upward_horizontal",
    "corridor_left_surface": "lateral_inward_positive_x",
    "corridor_right_surface": "lateral_inward_negative_x",
    "corridor_end_surface": "camera_facing_vertical",
}
THRESHOLDS = {
    "changed_controlled_pixel_fraction": 0.2,
    "normalized_controlled_rgb_mad": 0.025,
    "visible_surface_mean_luminance_lower": 0.05,
    "visible_surface_mean_luminance_upper": 0.95,
    "textured_surface_minimum_pixels": 100,
    "textured_surface_luminance_standard_deviation": 0.025,
}


class FailureAnalysisError(ValueError):
    """Raised when a baseline packet or derived analysis is not canonical."""


def _owned_json(root: Path, relative_path: str) -> dict[str, Any]:
    with open_owned_regular_file(root, relative_path) as owned:
        payload = json.loads(owned.payload.decode("utf-8"))
    if type(payload) is not dict:
        raise FailureAnalysisError(f"analysis source must be a JSON object: {relative_path}")
    return payload


def _margin_summary(values: list[float]) -> dict[str, Any]:
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "failure_count": sum(value < 0.0 for value in ordered),
        "near_threshold_count_abs_margin_le_0_005": sum(abs(value) <= 0.005 for value in ordered),
        "minimum_margin": float(ordered[0]) if ordered else None,
        "median_margin": float(median(ordered)) if ordered else None,
        "maximum_margin": float(ordered[-1]) if ordered else None,
    }


def _source_diagnostics(cell: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["semantic_surface_name"]: item for item in cell["source_texture_diagnostics"]}


def _analysis_domains(cells: list[dict[str, Any]]) -> tuple[dict[str, Any], ...]:
    profile_rows: list[dict[str, Any]] = []
    surface_rows: list[dict[str, Any]] = []
    margins: dict[str, list[float]] = defaultdict(list)
    failure_categories: Counter[str] = Counter()
    exposure_surface_failures: Counter[str] = Counter()
    exposure_slot_failures: Counter[str] = Counter()
    texture_surface_failures: Counter[str] = Counter()
    texture_family_failures: Counter[str] = Counter()
    source_variable_render_uniform = 0

    for cell in cells:
        if cell["generation_status"] != "success":
            failure_categories["generation_or_validation_failure"] += 1
            profile_rows.append(
                {
                    "cell_id": cell["cell_id"],
                    "profile_id": cell["profile_id"],
                    "scene_family": cell["scene_family"],
                    "seed_index": cell["seed_index"],
                    "candidate_seed": cell["candidate_seed"],
                    "generation_status": "failed",
                    "failure_type": cell["failure_type"],
                    "failure_message": cell["failure_message"],
                }
            )
            continue

        profile = cell["appearance_instance"]["profile"]
        family = profile["texture"]["family"]
        declared_frequency = profile["texture"]["cycles_per_tile"]
        declared_orientation = profile["texture"]["orientation"]
        illumination = profile["illumination"]
        renderer = cell["renderer_provenance"]
        semantic_by_label = {label: name for name, label in cell["semantic_surface_labels"].items()}
        assignment = cell["appearance_instance"]["style_assignment"]
        sources = _source_diagnostics(cell)
        frame_rows: list[dict[str, Any]] = []
        for frame_name in ("before", "after"):
            frame = cell["frame_metrics"][frame_name]
            changed_margin = (
                frame["changed_controlled_pixel_fraction"]
                - THRESHOLDS["changed_controlled_pixel_fraction"]
            )
            mad_margin = (
                frame["normalized_controlled_rgb_mad"] - THRESHOLDS["normalized_controlled_rgb_mad"]
            )
            margins["changed_controlled_pixel_fraction"].append(changed_margin)
            margins["normalized_controlled_rgb_mad"].append(mad_margin)
            if not frame["material_change_pass"]:
                failure_categories["material_rgb_change_failure"] += 1
            frame_rows.append(
                {
                    "frame": frame_name,
                    "changed_controlled_pixel_fraction_margin": changed_margin,
                    "normalized_controlled_rgb_mad_margin": mad_margin,
                    "material_rgb_change_pass": frame["material_change_pass"],
                }
            )
            for surface in frame["surface_diagnostics"]:
                semantic = semantic_by_label[surface["opaque_surface_label"]]
                style_slot = assignment[semantic]
                lower_margin = (
                    surface["luminance_mean"] - THRESHOLDS["visible_surface_mean_luminance_lower"]
                )
                upper_margin = (
                    THRESHOLDS["visible_surface_mean_luminance_upper"] - surface["luminance_mean"]
                )
                texture_applicable = (
                    family != "solid"
                    and surface["pixel_count"] >= THRESHOLDS["textured_surface_minimum_pixels"]
                )
                texture_margin = (
                    surface["luminance_standard_deviation"]
                    - THRESHOLDS["textured_surface_luminance_standard_deviation"]
                    if texture_applicable
                    else None
                )
                margins["visible_surface_mean_luminance_lower"].append(lower_margin)
                margins["visible_surface_mean_luminance_upper"].append(upper_margin)
                if texture_margin is not None:
                    margins["textured_surface_luminance_standard_deviation"].append(texture_margin)
                lower_failure = lower_margin < 0.0
                upper_failure = upper_margin < 0.0
                texture_failure = texture_margin is not None and texture_margin < 0.0
                if lower_failure:
                    failure_categories["lower_exposure_failure"] += 1
                    exposure_surface_failures[semantic] += 1
                    exposure_slot_failures[style_slot] += 1
                if upper_failure:
                    failure_categories["upper_exposure_failure"] += 1
                    exposure_surface_failures[semantic] += 1
                    exposure_slot_failures[style_slot] += 1
                if texture_failure:
                    failure_categories["texture_variation_failure"] += 1
                    texture_surface_failures[semantic] += 1
                    texture_family_failures[family] += 1
                source = sources.get(semantic)
                if (
                    source is not None
                    and source["source_luminance_standard_deviation"] >= 0.04
                    and texture_failure
                ):
                    source_variable_render_uniform += 1
                surface_rows.append(
                    {
                        "cell_id": cell["cell_id"],
                        "profile_id": cell["profile_id"],
                        "scene_family": cell["scene_family"],
                        "seed_index": cell["seed_index"],
                        "candidate_seed": cell["candidate_seed"],
                        "frame": frame_name,
                        "semantic_surface": semantic,
                        "opaque_surface_label": surface["opaque_surface_label"],
                        "surface_normal_class": SURFACE_NORMAL_CLASSES[semantic],
                        "style_slot": style_slot,
                        "visible_pixel_count": surface["pixel_count"],
                        "texture_family": family,
                        "declared_cycles_per_tile": declared_frequency,
                        "declared_orientation": declared_orientation,
                        "illumination": illumination[
                            "single_occluder"
                            if cell["scene_family"] == "single_occluder"
                            else "corridor"
                        ],
                        "ambient_rgb": illumination["ambient_rgb"],
                        "renderer_fingerprint": renderer,
                        "luminance_mean": surface["luminance_mean"],
                        "lower_exposure_margin": lower_margin,
                        "upper_exposure_margin": upper_margin,
                        "lower_exposure_failure": lower_failure,
                        "upper_exposure_failure": upper_failure,
                        "rendered_luminance_standard_deviation": surface[
                            "luminance_standard_deviation"
                        ],
                        "texture_variation_margin": texture_margin,
                        "texture_variation_failure": texture_failure,
                        "source_luminance_standard_deviation": (
                            source["source_luminance_standard_deviation"]
                            if source is not None
                            else None
                        ),
                        "rendered_to_source_luminance_std_ratio": surface.get(
                            "rendered_to_source_luminance_std_ratio"
                        ),
                    }
                )
        for criterion in (
            "structural_invariance",
            "portable_analytic_identity_equality",
            "ecological_label_equality",
            "depth_segmentation_invariance",
        ):
            if not cell["admission_checks"][criterion]:
                failure_categories["structural_or_ecological_invariance_failure"] += 1
        if not cell["admission_checks"]["determinism"]:
            failure_categories["determinism_failure"] += 1
        profile_rows.append(
            {
                "cell_id": cell["cell_id"],
                "profile_id": cell["profile_id"],
                "scene_family": cell["scene_family"],
                "seed_index": cell["seed_index"],
                "candidate_seed": cell["candidate_seed"],
                "generation_status": "success",
                "admission_status": cell["admission_status"],
                "rejection_reasons": cell["rejection_reasons"],
                "admission_checks": cell["admission_checks"],
                "frames": frame_rows,
            }
        )

    threshold_summary = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "canonical_admission_thresholds": THRESHOLDS,
        "criterion_margin_summaries": {
            name: _margin_summary(values) for name, values in sorted(margins.items())
        },
    }
    diagnostics = {
        "failure_category_occurrences": dict(sorted(failure_categories.items())),
        "exposure_failure_occurrences_by_semantic_surface": dict(
            sorted(exposure_surface_failures.items())
        ),
        "exposure_failure_occurrences_by_style_slot": dict(sorted(exposure_slot_failures.items())),
        "texture_failure_occurrences_by_semantic_surface": dict(
            sorted(texture_surface_failures.items())
        ),
        "texture_failure_occurrences_by_family": dict(sorted(texture_family_failures.items())),
        "high_source_variation_but_rendered_failure_occurrences": (source_variable_render_uniform),
    }
    profile_matrix = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "rows": profile_rows,
    }
    surface_matrix = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "privileged_analysis_only": True,
        "rows": surface_rows,
    }
    return profile_matrix, surface_matrix, threshold_summary, diagnostics


def _markdown(analysis: dict[str, Any]) -> str:
    diagnostics = analysis["diagnostic_summary"]
    categories = diagnostics["failure_category_occurrences"]
    lines = [
        "# Canonical Slice 5 baseline failure analysis",
        "",
        f"Logical analysis root: `{analysis['baseline_failure_analysis_sha256']}`.",
        "",
        "The independently validated baseline contains 160 successful cells, 44 admitted cells, "
        "116 rejected cells, and no profile admitted over all 16 cells.",
        "",
        "## Criterion findings",
        "",
    ]
    for name, count in sorted(categories.items()):
        lines.append(f"- {name}: {count} occurrence(s)")
    lines.extend(
        [
            "",
            "## Causal diagnostic answers",
            "",
        ]
    )
    for index, item in enumerate(analysis["causal_diagnostic_answers"], start=1):
        lines.append(f"{index}. **{item['question']}** {item['answer']}")
    lines.extend(
        [
            "",
            "This diagnosis is privileged apparatus analysis. It does not change the ecological "
            "learner interface, weaken admission, select a split, freeze a seed, or establish a "
            "scientific result.",
            "",
        ]
    )
    return "\n".join(lines)


def create_failure_analysis(packet_root: Path, output: Path) -> dict[str, Any]:
    """Validate the canonical packet, derive all margins, and publish atomically."""

    if output.exists():
        if not output.is_dir() or any(output.iterdir()):
            raise FileExistsError(f"analysis output is not an empty directory: {output}")
        output.rmdir()
    packet = validate_appearance_audit(
        packet_root,
        verify_current_source_provenance=False,
    )
    if packet["matrix_counts"] != CANONICAL_COUNTS or any(
        packet["roots"].get(name) != value for name, value in CANONICAL_ROOTS.items()
    ):
        raise FailureAnalysisError("source packet is not the protected canonical Slice 5 result")
    source_commit = packet["source_provenance"].get("git_commit")
    if source_commit != CANONICAL_BASE:
        raise FailureAnalysisError("source packet is not bound to the canonical base commit")
    matrix = _owned_json(packet_root, "seed_matrix.json")
    cells = matrix.get("cells")
    if type(cells) is not list or len(cells) != 160:
        raise FailureAnalysisError("canonical source matrix is incomplete")
    profile_matrix, surface_matrix, threshold_summary, diagnostics = _analysis_domains(cells)
    renderer_fingerprints = sorted(
        {
            canonical_json_bytes(cell["renderer_provenance"]).decode("utf-8")
            for cell in cells
            if cell["generation_status"] == "success"
        }
    )
    causal_answers = [
        {
            "question": "Are exposure failures concentrated on particular surfaces or normals?",
            "answer": (
                "Yes. Lower-exposure failures dominate both opposing corridor side-wall "
                "normal classes; the concentration persists across balanced style slots."
            ),
        },
        {
            "question": (
                "Are failures caused by directional-light asymmetry, palette luminance, or both?"
            ),
            "answer": (
                "The cross-profile side-wall failure and near-zero luminance identify "
                "zero-fill directional lighting as the dominant cause. The dim illumination "
                "profile also fails camera-facing surfaces, so intensity and palette can "
                "compound it."
            ),
        },
        {
            "question": (
                "Do failures follow semantic identity, style slot, or camera-facing orientation?"
            ),
            "answer": (
                "They follow surface normal/orientation more strongly than style slot: "
                "cyclic assignment moves every slot across surfaces, yet both lateral walls "
                "remain the dominant exposure failures."
            ),
        },
        {
            "question": "Where are texture-variation failures concentrated?",
            "answer": (
                "They occur at both low and high frequency in dark corridor walls, while the "
                "16-cycle high checker additionally fails every small projected "
                "single-occluder surface; darkness and minification are separate "
                "contributors."
            ),
        },
        {
            "question": "Can a variable source texture render nearly uniformly?",
            "answer": (
                f"Yes. {diagnostics['high_source_variation_but_rendered_failure_occurrences']} "
                "surface-frame observations retain source luminance standard deviation at "
                "or above 0.040 while failing the rendered 0.025 criterion."
            ),
        },
        {
            "question": "Are any margins numerically close to threshold?",
            "answer": (
                "Yes; the margin summary explicitly counts absolute margins at or below "
                "0.005. These are retained as sensitivity warnings, not reclassified or "
                "rounded into passes."
            ),
        },
        {
            "question": "Do WGL and OSMesa agree on category and margin sign?",
            "answer": (
                "The reviewed packets agree on the 44/116 outcome and profile rejection, "
                "but this single-packet analysis cannot independently compare item-level "
                "OSMesa margin signs. That renderer-level causal comparison remains "
                "uncertain."
            ),
        },
        {
            "question": "What is the smallest prospective response?",
            "answer": (
                "Bind a nonzero ambient fill to profile identity, use moderate-luminance "
                "balanced palettes, and reduce texture cycles to a 1-cycle low / 4-cycle "
                "high pair while preserving every canonical threshold."
            ),
        },
    ]
    portable_domain = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "source_canonical_base": CANONICAL_BASE,
        "source_matrix_counts": packet["matrix_counts"],
        "source_profile_count": packet["profile_count"],
        "source_roots": {name: packet["roots"][name] for name in sorted(CANONICAL_ROOTS)},
        "canonical_admission_thresholds": THRESHOLDS,
        "analysis_method": "privileged_surface_frame_threshold_margin_analysis_v0",
    }
    baseline_root = _hash_json(portable_domain)
    renderer_domain = {
        **portable_domain,
        "baseline_failure_analysis_sha256": baseline_root,
        "source_packet_logical_root_sha256": packet["packet_logical_root_sha256"],
        "renderer_fingerprints": renderer_fingerprints,
        "profile_failure_matrix_sha256": _hash_json(profile_matrix),
        "surface_failure_matrix_sha256": _hash_json(surface_matrix),
        "threshold_margin_summary_sha256": _hash_json(threshold_summary),
        "diagnostic_summary": diagnostics,
        "causal_diagnostic_answers": causal_answers,
    }
    analysis = {
        **renderer_domain,
        "renderer_specific_failure_analysis_sha256": _hash_json(renderer_domain),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        write_canonical_json(staging / "profile_failure_matrix.json", profile_matrix)
        write_canonical_json(staging / "surface_failure_matrix.json", surface_matrix)
        write_canonical_json(staging / "threshold_margin_summary.json", threshold_summary)
        write_canonical_json(staging / "baseline_failure_analysis.json", analysis)
        (staging / "baseline_failure_analysis.md").write_text(_markdown(analysis), encoding="utf-8")
        validate_failure_analysis(staging, source_packet=packet_root)
        _atomic_no_replace_directory(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return analysis


def validate_failure_analysis(
    output: Path,
    *,
    source_packet: Path | None = None,
) -> dict[str, Any]:
    """Independently validate every deterministic analysis identity."""

    expected = {
        "baseline_failure_analysis.json",
        "baseline_failure_analysis.md",
        "profile_failure_matrix.json",
        "surface_failure_matrix.json",
        "threshold_margin_summary.json",
    }
    actual = frozenset(path.name for path in output.iterdir())
    if actual not in {frozenset(expected), frozenset((*expected, "run.json"))}:
        raise FailureAnalysisError("failure-analysis artifact set is not exact")
    analysis = _owned_json(output, "baseline_failure_analysis.json")
    exact_fields = {
        "schema_version",
        "source_packet_logical_root_sha256",
        "source_canonical_base",
        "source_matrix_counts",
        "source_profile_count",
        "source_roots",
        "canonical_admission_thresholds",
        "analysis_method",
        "renderer_fingerprints",
        "profile_failure_matrix_sha256",
        "surface_failure_matrix_sha256",
        "threshold_margin_summary_sha256",
        "diagnostic_summary",
        "causal_diagnostic_answers",
        "baseline_failure_analysis_sha256",
        "renderer_specific_failure_analysis_sha256",
    }
    if set(analysis) != exact_fields or analysis["schema_version"] != ANALYSIS_SCHEMA_VERSION:
        raise FailureAnalysisError("failure-analysis schema is not strict")
    child_payloads: dict[str, dict[str, Any]] = {}
    for filename, field in (
        ("profile_failure_matrix.json", "profile_failure_matrix_sha256"),
        ("surface_failure_matrix.json", "surface_failure_matrix_sha256"),
        ("threshold_margin_summary.json", "threshold_margin_summary_sha256"),
    ):
        child_payloads[filename] = _owned_json(output, filename)
        if _hash_json(child_payloads[filename]) != analysis[field]:
            raise FailureAnalysisError(f"failure-analysis child identity mismatch: {filename}")
    portable_domain = {
        field: analysis[field]
        for field in (
            "schema_version",
            "source_canonical_base",
            "source_matrix_counts",
            "source_profile_count",
            "source_roots",
            "canonical_admission_thresholds",
            "analysis_method",
        )
    }
    if _hash_json(portable_domain) != analysis["baseline_failure_analysis_sha256"]:
        raise FailureAnalysisError("portable failure-analysis logical root mismatch")
    renderer_domain = dict(analysis)
    declared_renderer = renderer_domain.pop("renderer_specific_failure_analysis_sha256")
    if _hash_json(renderer_domain) != declared_renderer:
        raise FailureAnalysisError("renderer-specific failure-analysis logical root mismatch")
    markdown = (output / "baseline_failure_analysis.md").read_text(encoding="utf-8")
    if markdown != _markdown(analysis):
        raise FailureAnalysisError("failure-analysis Markdown differs from reconstruction")
    if source_packet is not None:
        packet = validate_appearance_audit(
            source_packet,
            verify_current_source_provenance=False,
        )
        matrix = _owned_json(source_packet, "seed_matrix.json")
        cells = matrix.get("cells")
        if type(cells) is not list:
            raise FailureAnalysisError("failure-analysis source matrix is invalid")
        expected_profile, expected_surface, expected_thresholds, expected_diagnostics = (
            _analysis_domains(cells)
        )
        expected_children = {
            "profile_failure_matrix.json": expected_profile,
            "surface_failure_matrix.json": expected_surface,
            "threshold_margin_summary.json": expected_thresholds,
        }
        if any(
            canonical_json_bytes(child_payloads[name]) != canonical_json_bytes(expected_payload)
            for name, expected_payload in expected_children.items()
        ):
            raise FailureAnalysisError("failure-analysis reports differ from source evidence")
        if analysis["diagnostic_summary"] != expected_diagnostics:
            raise FailureAnalysisError("failure-analysis diagnosis differs from source evidence")
        if analysis["source_packet_logical_root_sha256"] != packet["packet_logical_root_sha256"]:
            raise FailureAnalysisError("failure-analysis source packet root differs")
    return analysis
