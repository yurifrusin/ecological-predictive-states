"""Synthetic causal chronology/rule/audit checks; no native or old studies."""

from dataclasses import fields, replace
from fractions import Fraction as F

import numpy as np
import pytest

from epsbench.diagnostics.causal_history_audit import (
    EvaluatorTarget,
    archive,
    audit_pair,
    score,
)
from epsbench.diagnostics.causal_history_core import (
    Command,
    CompletedFlow,
    MemorySequence,
    OpticalFrame,
    TemporalProjection,
    build_state,
    extrapolate,
    persistence,
)
from epsbench.schema import (
    BoundaryAxis,
    BoundaryKind,
    BoundaryOwnerSide,
    Modality,
    ModalityPermissionSet,
    OrientedBoundaryElement,
)

TOKEN = "surface-0000000000000001"
OTHER = "surface-0000000000000002"


def frame(i: int, column: int | None, token: str = TOKEN) -> OpticalFrame:
    a = np.zeros((2, 6), dtype=np.int32)
    if column is not None:
        a[0, column] = 1
    return OpticalFrame(i, a, ((1, token),) if column is not None else ())


def flow(i: int, command: F = F(1), dx: int = 1024, usable: bool = True) -> CompletedFlow:
    vectors = np.zeros((2, 6, 2), dtype=np.int32)
    vectors[..., 0] = dx
    validity = np.full((2, 6), int(usable), dtype=np.uint8)
    reasons = np.full((2, 6), 0 if usable else 3, dtype=np.uint8)
    return CompletedFlow(i, Command(command), vectors, validity, reasons)


def view(
    columns: tuple[int | None, ...],
    flows: tuple[CompletedFlow, ...] = (),
    token: str = TOKEN,
    colour: int = 0,
) -> TemporalProjection:
    frames = tuple(frame(i, c, token) for i, c in enumerate(columns))
    rgbs = tuple(np.full((2, 6, 3), colour, dtype=np.uint8) for _ in frames)
    return TemporalProjection(
        MemorySequence(frames, flows, rgbs), ModalityPermissionSet.all_modalities(), len(frames) - 1
    )


def test_temporal_and_permission_checks_precede_provider_access() -> None:
    class Bomb:
        calls = 0

        def frame(self, sequence_index: int) -> OpticalFrame:
            self.calls += 1
            raise AssertionError("provider accessed")

        def flow(self, source_index: int) -> CompletedFlow:
            self.calls += 1
            raise AssertionError("provider accessed")

        def rgb(self, sequence_index: int) -> np.ndarray:
            self.calls += 1
            raise AssertionError("provider accessed")

    provider = Bomb()
    p = TemporalProjection(provider, ModalityPermissionSet.ecological_only(), 2)
    for operation in (lambda: p.frame(3), lambda: p.flow(2), lambda: p.rgb(0)):
        with pytest.raises(PermissionError):
            operation()
    denied = TemporalProjection(provider, ModalityPermissionSet(allowed=frozenset()), 2)
    for operation in (lambda: denied.frame(0), lambda: denied.flow(0), lambda: denied.rgb(0)):
        with pytest.raises(PermissionError):
            operation()
    assert provider.calls == 0
    with pytest.raises(ValueError):
        p.flow(-1)
    with pytest.raises(ValueError):
        TemporalProjection(provider, frozenset(), 1)  # type: ignore[arg-type]


def test_snapshots_own_arrays_and_no_privileged_predictor_fields() -> None:
    a = np.zeros((2, 6), dtype=np.int32)
    a[0, 0] = 1
    f = OpticalFrame(0, a, ((1, TOKEN),))
    a[:] = 0
    assert f.mask(TOKEN).sum() == 1
    s = build_state(view((0,)), Command(F(0)))
    arrays = (
        f.segmentation,
        s.current.segmentation,
        s.tokens[0].last_mask,
        flow(0).vectors,
        flow(0).validity,
        flow(0).reasons,
        persistence(s)[0].mask,
    )
    for array in arrays:
        assert array is not None
        with pytest.raises(ValueError):
            array.setflags(write=True)
        with pytest.raises(ValueError):
            array.flat[0] = 99
    assert {v.name for v in fields(s)} == {"current", "tokens", "executed", "announced", "version"}
    assert {v.name for v in fields(s.current)} == {
        "sequence_index",
        "segmentation",
        "identities",
        "boundaries",
    }
    assert len(s.tokens) == 1
    assert s.current.mask(OTHER).sum() == 0


def test_independent_last_mask_motion_and_two_hidden_steps() -> None:
    p = view((0, 1, None, None), (flow(0), flow(1, usable=False), flow(2, usable=False)))
    s = build_state(p, Command(F(1)))
    token = s.tokens[0]
    assert not token.visible and token.last_index == 1
    assert token.last_mask[0, 1]
    assert token.motion is not None and token.motion.source_index == 0
    assert token.motion.sums == (1024, 0) and token.motion.count == 1
    assert sum(c for _, _, c in token.motion.reason_counts) == 1
    persisted = persistence(s)[0].mask
    assert persisted is not None and persisted.sum() == 0
    prediction = extrapolate(s)[0]
    assert prediction.shift == (3, 0) and prediction.mask is not None
    assert prediction.mask[0, 4]


@pytest.mark.parametrize(
    "dx,command,shift",
    [(512, F(1), 0), (1536, F(1), 2), (-512, F(1), 0), (-1536, F(1), -2), (1024, F(-2), -2)],
)
def test_signed_ties_to_even(dx: int, command: F, shift: int) -> None:
    s = build_state(view((2, 2), (flow(0, dx=dx),)), Command(command))
    assert extrapolate(s)[0].shift == (shift, 0)


def test_round_once_zero_reversal_clipping_and_unknowns() -> None:
    s = build_state(
        view((0, None, None), (flow(0, F(1, 2), 512), flow(1, F(1, 2), usable=False))),
        Command(F(1, 2)),
    )
    assert extrapolate(s)[0].shift == (2, 0)  # 3/2 ties to even, not per-step 0.
    cancelled = replace(s, announced=Command(F(-1)))
    assert extrapolate(cancelled)[0].shift == (0, 0)
    noflow = build_state(view((1, None), (flow(0, usable=False),)), Command(F(-1)))
    assert extrapolate(noflow)[0].mask is not None
    assert extrapolate(replace(noflow, announced=Command(F(1))))[0].status == "UNKNOWN_MOTION"
    unsupported = replace(s, announced=Command(F(0), forward=F(1)))
    assert extrapolate(unsupported)[0].status == "UNKNOWN"
    clipped = extrapolate(replace(s, announced=Command(F(10))))[0]
    assert clipped.mask is not None and clipped.mask.sum() == 0 and clipped.clipped_pixels == 1
    empty = build_state(view((None,)), Command(F(1)))
    assert persistence(empty) == extrapolate(empty) == ()


def test_identity_and_flow_validation() -> None:
    with pytest.raises(ValueError, match="identities"):
        OpticalFrame(0, frame(0, 0).segmentation, ())
    p = view((0, 1), (flow(0),))
    bad = MemorySequence((frame(0, 0), frame(1, 1, OTHER)), (flow(0),), ())
    with pytest.raises(ValueError, match="continuity"):
        build_state(replace(p, provider=bad), Command(F(1)))
    with pytest.raises(ValueError, match="validity"):
        replace(flow(0), reasons=np.ones((2, 6), dtype=np.uint8))
    with pytest.raises(ValueError):
        replace(flow(0), vectors=np.zeros((1, 6, 2), dtype=np.int32))
    with pytest.raises(ValueError, match="chronolog"):
        replace(build_state(p, Command(F(1))), executed=())


def test_local_boundary_mapping_is_explicit() -> None:
    boundary = OrientedBoundaryElement(
        frame_index=0,
        axis=BoundaryAxis.HORIZONTAL,
        row=0,
        column=0,
        negative_surface_id=TOKEN,
        positive_surface_id=None,
        kind=BoundaryKind.CONTROLLED_SILHOUETTE,
        owner_side=BoundaryOwnerSide.NEGATIVE_AXIS_SIDE,
        owner_surface_id=TOKEN,
    )
    payload = boundary.model_dump_json().encode()
    f = replace(frame(3, 0), boundaries=(payload,))
    assert f.sequence_index == 3
    with pytest.raises(ValueError, match="frame_index=0"):
        replace(
            f,
            boundaries=(boundary.model_copy(update={"frame_index": 1}).model_dump_json().encode(),),
        )
    with pytest.raises(ValueError, match="mapping"):
        replace(frame(3, 1), boundaries=(payload,))


def target(t: int, column: int | None, token: str = TOKEN) -> EvaluatorTarget:
    return EvaluatorTarget(t, token, frame(t, column, token).mask(token))


def test_overwrite_collision_with_distinct_earlier_histories() -> None:
    a = archive(view((0, 2, 3), (flow(0), flow(1)), colour=1), Command(F(1)))
    b = archive(view((1, 2, 3), (flow(0, dx=2048), flow(1)), OTHER, colour=2), Command(F(1)))
    result = audit_pair(a, b, target(3, 4), target(3, 5, OTHER), {TOKEN: OTHER}, 1)
    assert result.classification == "FINITE_E_SPECIFIC_LOSS"
    assert result.current_equal and result.state_equal and not result.oracle_equal
    assert result.rgb_action_equal is False and result.target_equal is False
    invalid = audit_pair(a, b, target(3, 4), target(3, 5, OTHER), {}, 1)
    assert invalid.classification == "INCONCLUSIVE"
    wrong_designation = audit_pair(
        a, b, target(3, 4, OTHER), target(3, 5, OTHER), {TOKEN: OTHER}, 1
    )
    assert wrong_designation.classification == "INCONCLUSIVE"


def test_shared_ambiguity_and_rule_failure_without_collision() -> None:
    a = archive(view((1, 2), (flow(0),)), Command(F(1)))
    b = archive(view((1, 2), (flow(0),), OTHER), Command(F(1)))
    shared = audit_pair(a, b, target(2, 3), target(2, 4, OTHER), {TOKEN: OTHER}, 3)
    assert shared.classification == "SHARED_AMBIGUITY" and shared.oracle_equal
    c = archive(view((1, 2, 2), (flow(0), flow(1, dx=2048)), OTHER, colour=1), Command(F(1)))
    a_long = archive(view((1, 2, 2), (flow(0), flow(1))), Command(F(1)))
    result = audit_pair(a_long, c, target(3, 4), target(3, 5, OTHER), {TOKEN: OTHER}, 1)
    assert result.classification == "NONCOLLISION" and not result.state_equal
    assert score(extrapolate(a.state)[0], target(2, 4)).exact is False
    assert (
        audit_pair(a, c, target(2, 4), target(2, 5, OTHER), {TOKEN: OTHER}, 3).classification
        == "INCONCLUSIVE"
    )


def test_one_alignment_across_whole_history_and_state_binding() -> None:
    a = archive(view((1, 2), (flow(0),)), Command(F(1)))
    bad = replace(a, frames=(frame(0, 1, OTHER), a.frames[1]))
    assert (
        audit_pair(a, bad, target(2, 3), target(2, 4), {TOKEN: TOKEN}, 1).classification
        == "INCONCLUSIVE"
    )
    altered = replace(a, state=replace(a.state, announced=Command(F(2))))
    assert (
        audit_pair(a, altered, target(2, 3), target(2, 4), {TOKEN: TOKEN}, 1).classification
        == "INCONCLUSIVE"
    )
    altered = replace(a, frames=(frame(0, 1), frame(1, 3)))
    assert (
        audit_pair(a, altered, target(2, 3), target(2, 4), {TOKEN: TOKEN}, 1).classification
        == "INCONCLUSIVE"
    )


def test_exact_mask_scores_and_unknown_separate_from_empty() -> None:
    s = build_state(view((1, None), (flow(0, usable=False),)), Command(F(1)))
    unknown = score(extrapolate(s)[0], target(2, None))
    assert unknown.status == "UNKNOWN_MOTION" and unknown.iou is None and unknown.exact is None
    empty = score(persistence(s)[0], target(2, None))
    assert empty.exact and empty.iou == 1 and empty.error_pixels == 0
    mismatch = score(persistence(s)[0], target(2, 1))
    assert mismatch.iou == 0 and mismatch.error_pixels == 1
    denied = replace(view((0,)), permissions=ModalityPermissionSet.ecological_only())
    with pytest.raises(PermissionError):
        archive(denied, Command(F(0)))
    assert Modality.RGB not in denied.permissions.allowed


def test_pair2_fixture_conditions_are_fail_closed() -> None:
    a = archive(
        view((0, 1, None, None), (flow(0), flow(1, usable=False), flow(2, usable=False)), colour=1),
        Command(F(1)),
    )
    b = archive(
        view(
            (0, 1, None, None),
            (flow(0), flow(1, usable=False), flow(2, usable=False)),
            OTHER,
            colour=2,
        ),
        Command(F(1)),
    )
    valid = audit_pair(a, b, target(4, 2), target(4, 3, OTHER), {TOKEN: OTHER}, 2)
    assert valid.fixture_valid and valid.classification == "FINITE_E_SPECIFIC_LOSS"
    c = archive(
        view(
            (0, 1, 1, None),
            (flow(0), flow(1, usable=False), flow(2, usable=False)),
            OTHER,
            colour=2,
        ),
        Command(F(1)),
    )
    invalid = audit_pair(a, c, target(4, 2), target(4, 3, OTHER), {TOKEN: OTHER}, 2)
    assert not invalid.fixture_valid and invalid.classification == "INCONCLUSIVE"
    assert invalid.mathematical_classification == "NONCOLLISION"


def test_nested_identity_input_is_owned() -> None:
    identities = [[1, TOKEN]]
    f = OpticalFrame(0, frame(0, 0).segmentation, identities)  # type: ignore[arg-type]
    identities[0][1] = OTHER
    assert f.identities == ((1, TOKEN),)


def test_arbitrarily_large_rational_shift_clips_without_overflow() -> None:
    s = build_state(view((1, 2), (flow(0),)), Command(F(10**100)))
    prediction = extrapolate(s)[0]
    assert prediction.mask is not None and prediction.mask.sum() == 0
    assert prediction.clipped_pixels == 1 and prediction.shift == (10**100, 0)


def test_rgb_archive_and_forecast_ownership() -> None:
    p = view((0,))
    original = p.provider.rgb(0)
    archived = archive(p, Command(F(0)))
    original[:] = 255
    assert np.all(archived.rgbs[0] == 0)
    with pytest.raises(ValueError):
        archived.rgbs[0].setflags(write=True)
    prediction = persistence(archived.state)[0]
    with pytest.raises(ValueError):
        replace(prediction, status="UNKNOWN")


def test_exact_region_mean_excludes_invalid_samples_and_translates_xy() -> None:
    source = np.zeros((2, 6), dtype=np.int32)
    source[0, :4] = 1
    vectors = np.zeros((2, 6, 2), dtype=np.int32)
    vectors[0, 0] = (1024, 1024)
    vectors[0, 3] = (100000, 100000)  # Invalid sample must never become zero/usable.
    validity = np.zeros((2, 6), dtype=np.uint8)
    validity[0, :3] = 1
    reasons = np.full((2, 6), 3, dtype=np.uint8)
    reasons[0, :3] = 0
    completed = CompletedFlow(0, Command(F(1)), vectors, validity, reasons)
    provider = MemorySequence(
        (OpticalFrame(0, source, ((1, TOKEN),)), frame(1, 1)), (completed,), ()
    )
    state = build_state(
        TemporalProjection(provider, ModalityPermissionSet.ecological_only(), 1), Command(F(3))
    )
    motion = state.tokens[0].motion
    assert motion is not None and motion.sums == (1024, 1024) and motion.count == 3
    assert motion.reason_counts[3] == (0, 3, 1)
    predicted = extrapolate(state)[0]
    assert predicted.displacement == (F(1), F(1)) and predicted.shift == (1, 1)
    assert predicted.mask is not None and predicted.mask[1, 2] and predicted.mask.sum() == 1
