"""Independent arbitrary-mask fixtures, not native geometry or empirical evidence."""

import importlib
import json
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from epsbench.diagnostics.boundary_observation import VisibleRaster
from epsbench.diagnostics.visible_forecast_contract import (
    REQUIRED,
    CausalInput,
    CausalView,
    Forecast,
    Limits,
    TokenFrame,
    evaluate,
    forecast,
    storage,
)
from epsbench.schema import Modality, ModalityPermissionSet

A = "surface-0000000000000001"
B = "surface-0000000000000002"
D = "surface-0000000000000003"
ACCESS = ModalityPermissionSet(allowed=REQUIRED)
LIMITS = Limits(5, 20, 4)
ZERO = (Fraction(0), Fraction(0), Fraction(0))
STEP = (Fraction(0), Fraction(1, 3), Fraction(0))


def raster(
    values: list[list[int]], index: int = 0, ids: dict[int, str] | None = None
) -> VisibleRaster:
    ids = ids or {1: A, 2: B, 3: D}
    a = np.array(values, dtype=np.int32)
    return VisibleRaster(index, a, tuple((int(v), ids[int(v)]) for v in np.unique(a) if v))


class Provider:
    def __init__(self, frames: tuple[VisibleRaster, ...]) -> None:
        self.frames = frames
        self.calls: list[int] = []

    def raster(self, index: int) -> VisibleRaster:
        self.calls.append(index)
        return self.frames[index]


def source(*frames: list[list[int]]) -> CausalInput:
    p = Provider(tuple(raster(v, i) for i, v in enumerate(frames)))
    return CausalView(p, ACCESS, len(frames) - 1, LIMITS).materialize(
        (STEP,) * (len(frames) - 1), STEP
    )


def score(
    s: CausalInput, values: list[list[int]], rule: str = "current-mask-persistence", k: int = 1
) -> dict[str, Any]:
    p = Provider((*(raster([[0]], i) for i in range(len(s.frames))), raster(values, len(s.frames))))
    result = evaluate(s, forecast(s, rule, k).canonical_bytes(), p)
    assert p.calls == [len(s.frames)]
    return result


def test_handwritten_occupancy_new_and_absent_denominators() -> None:
    s = source([[1, 0, 1], [0, 1, 0]], [[2, 0, 2], [0, 0, 0]])
    assert s.inventory == (A, B)
    result = score(s, [[1, 3, 0], [0, 1, 0]])
    assert (result["E"], result["U"], result["C"], result["N"]) == (4, 0, 12, 12)
    assert result["conditional_error"] == Fraction(1, 3)
    assert result["ignorance_interval"] == (Fraction(1, 3), Fraction(1, 3))
    assert (
        result["omitted_new_tokens"],
        result["omitted_new_pixels"],
        result["future_surface_pixels"],
        result["inventory_future_pixels"],
    ) == (1, 1, 3, 2)
    assert result["omitted_new_pixel_fraction"] == Fraction(1, 3)
    assert result["tokens"] == [
        {"token": A, "target_pixels": 2, "target_visible": True, "error": 2, "unknown": False},
        {"token": B, "target_pixels": 0, "target_visible": False, "error": 2, "unknown": False},
    ]
    assert result["strata"]["target_visible"] == result["tokens"][:1]
    assert result["strata"]["target_absent"] == result["tokens"][1:]
    assert not result["inventory_exact"]


def test_unknown_and_empty_are_not_success() -> None:
    s = source([[1, 0]], [[0, 1]])
    result = score(s, [[1, 0]], "k-frame-agreement", 2)
    assert (result["E"], result["U"], result["C"], result["N"]) == (0, 2, 0, 2)
    assert result["conditional_error"] is None
    assert result["ignorance_interval"] == (0, 1)
    assert result["pixel_coverage"] == result["channel_coverage"] == 0
    assert not result["whole_episode_coverage"] and not result["inventory_exact"]
    empty = score(source([[0], [0]]), [[0], [0]])
    assert empty["status"] == "NOT_APPLICABLE"
    for field in (
        "inventory_exact",
        "pixel_coverage",
        "channel_coverage",
        "ignorance_interval",
        "conditional_error",
        "whole_episode_coverage",
        "omitted_new_pixel_fraction",
    ):
        assert empty[field] is None
    absent = score(source([[1]]), [[0]])
    assert absent["E"] == 1 and absent["omitted_new_pixel_fraction"] is None
    new = score(source([[0]]), [[3]])
    assert new["status"] == "NOT_APPLICABLE" and new["omitted_new_pixel_fraction"] == 1


def test_agreement_budget_and_partial_unknown_accounting() -> None:
    s = source([[1, 2, 0]], [[1, 0, 2]])
    result = score(s, [[1, 2, 0]], "k-frame-agreement", 2)
    assert (result["E"], result["U"], result["C"], result["N"]) == (0, 3, 3, 6)
    assert result["pixel_coverage"] == Fraction(1, 2)
    assert result["ignorance_interval"] == (0, Fraction(1, 2))
    with pytest.raises(ValueError, match="unavailable"):
        forecast(source([[1]]), "k-frame-agreement", 2)
    record = forecast(s, "k-frame-agreement", 2).canonical_bytes()
    account = storage(s, record)
    assert account["materialized_input_mask_bytes"] == 12
    assert account["snapshot_mask_bytes"] == 15
    assert account["forecast_mask_bytes"] == 3
    assert account["window_referenced_input_mask_bytes"] == 12
    assert account["additional_retained_window_bytes"] == 0
    assert account["input_serialized_bytes"] == len(s.canonical_bytes())


def test_permissions_and_future_denied_before_fetch_or_parse() -> None:
    p = Provider((raster([[1]]),))
    for denied in (
        ModalityPermissionSet(allowed=frozenset()),
        ModalityPermissionSet(allowed=REQUIRED | {Modality.DEPTH}),
        ModalityPermissionSet(allowed=REQUIRED - {Modality.REGION_CORRESPONDENCE}),
        {"allowed": REQUIRED},
    ):
        with pytest.raises(PermissionError):
            CausalView(p, denied, 0, LIMITS)  # type: ignore[arg-type]
        with pytest.raises(PermissionError):
            CausalInput.from_bytes(b"not json", denied, LIMITS)  # type: ignore[arg-type]
        with pytest.raises(PermissionError):
            replace(source([[1]]), permissions=denied)  # type: ignore[arg-type]
    view = CausalView(p, ACCESS, 0, LIMITS)
    for future in (1, 100):
        with pytest.raises(PermissionError):
            view.raster(future)
    for invalid in (True, -1):
        with pytest.raises(ValueError):
            view.raster(invalid)
    with pytest.raises(ValueError):
        view.materialize((ZERO,), ZERO)
    assert p.calls == []


def test_limits_before_expansion(monkeypatch: pytest.MonkeyPatch) -> None:
    p = Provider((raster([[1]], 0), raster([[2]], 1)))

    def forbidden(*args: object) -> None:
        raise AssertionError("mask expansion occurred")

    monkeypatch.setattr("epsbench.diagnostics.visible_forecast_contract._project", forbidden)
    with pytest.raises(ValueError, match="token budget"):
        CausalView(p, ACCESS, 1, Limits(2, 1, 1)).materialize((ZERO,), ZERO)
    assert p.calls == [0, 1]
    for limits in (Limits(1, 2, 2), Limits(2, 1, 2)):
        q = Provider((raster([[1, 0]], 0), raster([[1, 0]], 1)))
        with pytest.raises(ValueError):
            CausalView(q, ACCESS, 1, limits).materialize((ZERO,), ZERO)
    with pytest.raises(ValueError):
        Limits(True, 1, 1)


def test_immutable_roundtrip_and_strict_serialized_ingress() -> None:
    s = source([[1, 0]], [[0, 1]])
    payload = s.canonical_bytes()
    assert CausalInput.from_bytes(payload, ACCESS, LIMITS).canonical_bytes() == payload
    f = forecast(s, "current-mask-persistence")
    saved = f.canonical_bytes()
    assert Forecast.from_bytes(saved, s).canonical_bytes() == saved
    with pytest.raises(ValueError):
        s.frames[0].masks[0][1].setflags(write=True)
    with pytest.raises(FrozenInstanceError):
        s.frames = ()  # type: ignore[misc]
    for field, value in (
        ("permissions", ["depth"]),
        ("shape", [True, 2]),
        ("limits", [True, 20, 4]),
        ("frames", [{"index": True, "masks": {A: [[True, False]]}}]),
        ("announced", ["0.0", "0", "0"]),
        ("hidden_pose", [1, 2, 3]),
    ):
        p = json.loads(payload)
        p[field] = value
        with pytest.raises((ValueError, PermissionError)):
            CausalInput.from_bytes(json.dumps(p).encode(), ACCESS, LIMITS)
    p = json.loads(payload)
    p["frames"][0]["masks"][A] = [[1, 0]]
    with pytest.raises(ValueError):
        CausalInput.from_bytes(json.dumps(p).encode(), ACCESS, LIMITS)
    with pytest.raises(ValueError, match="duplicate"):
        CausalInput.from_bytes(
            payload.replace(b'"version":', b'"version":"bad","version":'), ACCESS, LIMITS
        )
    forecast_cases: tuple[tuple[str, Any], ...] = (
        ("input_sha256", "0" * 64),
        ("masks", {}),
        ("decision_index", True),
        ("shape", [1, 1]),
        ("window", True),
        ("masks", {A: "UNKNOWN"}),
    )
    for field, value in forecast_cases:
        p = json.loads(saved)
        p[field] = value
        target = Provider(())
        with pytest.raises(ValueError):
            evaluate(s, json.dumps(p).encode(), target)
        assert target.calls == []


def test_provider_wrong_chronology_shape_and_direct_validation() -> None:
    with pytest.raises(ValueError, match="chronology"):
        CausalView(Provider((raster([[1]], 1),)), ACCESS, 0, LIMITS).materialize((), ZERO)
    with pytest.raises(ValueError, match="shape"):
        CausalView(
            Provider((raster([[1]], 0), raster([[1, 0]], 1))), ACCESS, 1, LIMITS
        ).materialize((ZERO,), ZERO)
    with pytest.raises(ValueError):
        TokenFrame(True, (1, 1), ())
    with pytest.raises(ValueError):
        TokenFrame(0, (1, 1), ((A, np.array([[1]], dtype=np.int32)),))
    with pytest.raises(ValueError):
        TokenFrame(0, (1, 1), ((A, np.array([[True]])), (B, np.array([[True]]))))
    with pytest.raises(ValueError):
        replace(source([[1]]), executed=(ZERO,))
    with pytest.raises(ValueError):
        replace(source([[1]]), announced=(0, 0, 0))  # type: ignore[arg-type]
    s = source([[1]])
    saved = forecast(s, "current-mask-persistence").canonical_bytes()
    with pytest.raises(ValueError, match="chronology"):
        evaluate(s, saved, Provider((raster([[0]]), raster([[1]], 0))))


def test_permutations_hidden_changes_and_visible_positive_control() -> None:
    old = source([[1, 2], [0, 1]])
    p = Provider((raster([[7, 3], [0, 7]], 0, {7: D, 3: A}),))
    renamed = CausalView(p, ACCESS, 0, LIMITS).materialize((), STEP)
    f = dict(forecast(old, "current-mask-persistence").masks)
    g = dict(forecast(renamed, "current-mask-persistence").masks)
    np.testing.assert_array_equal(f[A], g[D])
    np.testing.assert_array_equal(f[B], g[A])
    relabel = CausalView(
        Provider((raster([[7, 3], [0, 7]], 0, {7: A, 3: B}),)), ACCESS, 0, LIMITS
    ).materialize((), STEP)
    assert old.canonical_bytes() == relabel.canonical_bytes()
    assert score(old, [[1, 0], [2, 1]])["E"] == score(renamed, [[3, 0], [1, 3]])["E"]
    # Hidden evaluator metadata is never projected into the causal API.
    variants = [
        (old, {"pose": 1, "seed": 2, "catalogue": ("hidden-a",)}),
        (old, {"pose": 9, "seed": 8, "catalogue": ("hidden-b", "hidden-c")}),
    ]
    assert variants[0][1] != variants[1][1]
    assert variants[0][0].canonical_bytes() == variants[1][0].canonical_bytes()
    assert (
        forecast(variants[0][0], "current-mask-persistence").canonical_bytes()
        == forecast(variants[1][0], "current-mask-persistence").canonical_bytes()
    )
    assert source([[1, 2], [0, 0]]).canonical_bytes() != old.canonical_bytes()


def test_symbolic_ambiguity_and_distinct_finite_window_collision() -> None:
    # Same FULL causal input/actions, two unobserved continuations; contract domain only.
    s = source([[1, 0]], [[1, 0]])
    a, b = score(s, [[1, 0]]), score(s, [[0, 1]])
    assert a["inventory_exact"] and not b["inventory_exact"]
    assert b["E"] == 2
    # Different materialized input/digests, identical selected K=2 state/masks.
    left = source([[1, 0]], [[1, 0]], [[1, 0]])
    right = source([[0, 1]], [[1, 0]], [[1, 0]])
    assert left.digest != right.digest
    left_forecast, right_forecast = (
        forecast(left, "k-frame-agreement", 2),
        forecast(right, "k-frame-agreement", 2),
    )
    np.testing.assert_array_equal(left_forecast.masks[0][1], right_forecast.masks[0][1])
    assert left_forecast.input_sha256 != right_forecast.input_sha256
    assert all(
        np.array_equal(x.masks[0][1], y.masks[0][1])
        for x, y in zip(left.frames[-2:], right.frames[-2:], strict=True)
    )
    assert not score(left, [[0, 1]], "k-frame-agreement", 2)["inventory_exact"]


def test_guard_and_exact_ci_routes() -> None:
    # Existing guard installed by the bounded selector before this test is collected.
    for name in ("mujoco", "OpenGL", "glfw", "epsbench.data.generate", "epsbench.sim.renderer"):
        with pytest.raises(RuntimeError, match="forbidden"):
            importlib.import_module(name)
    ci = (Path(__file__).resolve().parents[1] / ".github/workflows/ci.yml").read_text()
    branch = "codex/causal-visible-forecast-contract-20261006"
    assert (
        ci.count("github.head_ref == '" + branch + "'") == 5
    )  # Four job predicates plus own selector.
    assert "github.head_ref != '" + branch + "'" in ci
    assert "scripts/check_visible_forecast_source.py" in ci


def test_holes_reappearance_missing_prefix_and_owned_snapshots() -> None:
    values = [[1, 1, 1], [1, 0, 1], [1, 1, 1]]
    s = source(values, [[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    result = score(s, values)
    assert result["N"] == 9 and result["E"] == 8
    assert result["tokens"][0]["target_visible"]
    assert not result["inventory_exact"]
    original = np.array([[True, False]], dtype=np.bool_)
    frame = TokenFrame(0, (1, 2), ((A, original),))
    original[:] = False
    assert frame.masks[0][1].tolist() == [[True, False]]
    with pytest.raises(ValueError):
        TokenFrame(0, (True, 2), ())
    with pytest.raises(ValueError):
        TokenFrame(0, (), ())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="prefix"):
        replace(source([[1]], [[1]]), frames=(source([[1]], [[1]]).frames[1],))
    data = json.loads(source([[1]], [[1]]).canonical_bytes())
    data["frames"][1]["index"] = 9
    with pytest.raises(ValueError, match="prefix"):
        CausalInput.from_bytes(json.dumps(data).encode(), ACCESS, LIMITS)
    # Declared actions affect input binding but these controls explicitly ignore them.
    before = source([[1, 0]])
    after = replace(before, announced=ZERO)
    assert before.digest != after.digest
    np.testing.assert_array_equal(
        forecast(before, "current-mask-persistence").masks[0][1],
        forecast(after, "current-mask-persistence").masks[0][1],
    )


@pytest.mark.parametrize("metadata", ["shape", "dtype"])
def test_input_and_forecast_metadata_mutation_is_detached(metadata: str) -> None:
    s = source([[1, 0, 0], [0, 1, 0]], [[1, 0, 0], [0, 1, 0]])
    before = s.canonical_bytes()
    digest = s.digest
    f = forecast(s, "k-frame-agreement", 2)
    saved = f.canonical_bytes()
    account = storage(s, saved)
    expected = score(s, [[1, 0, 0], [0, 1, 0]], "k-frame-agreement", 2)
    exposed = (s.frames[0].masks[0][1], s.frames[1].masks[0][1], f.masks[0][1])
    for mask in exposed:
        assert mask is not None
        assert not mask.flags.writeable
        with pytest.raises(ValueError):
            mask.setflags(write=True)
        with pytest.raises(ValueError):
            mask[0, 0] = False
        if metadata == "shape":
            mask.shape = (1, 6)
            assert mask.shape == (1, 6)
        else:
            mask.dtype = np.uint8  # type: ignore[misc, assignment]  # Runtime regression.
            assert mask.tolist() == [[1, 0, 0], [0, 1, 0]]
    # Exposed arrays may change their own metadata, never the admitted byte/shape state.
    assert s.canonical_bytes() == before and s.digest == digest
    assert f.canonical_bytes() == saved
    assert s.shape == f.shape == (2, 3)
    for mask in (s.frames[0].masks[0][1], s.frames[1].masks[0][1], f.masks[0][1]):
        assert mask is not None and mask.shape == (2, 3) and mask.dtype == np.bool_
        assert mask.tolist() == [[True, False, False], [False, True, False]]
    assert forecast(s, "k-frame-agreement", 2).canonical_bytes() == saved
    assert CausalInput.from_bytes(before, ACCESS, LIMITS).canonical_bytes() == before
    assert Forecast.from_bytes(saved, s).canonical_bytes() == saved
    assert storage(s, saved) == account
    p = Provider((raster([[0]], 0), raster([[0]], 1), raster([[1, 0, 0], [0, 1, 0]], 2)))
    assert evaluate(s, saved, p) == expected
    assert p.calls == [2]
