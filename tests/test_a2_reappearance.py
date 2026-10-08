"""Handwritten public-mask checks for fixed A2 rules; no scene or outcome data."""

from __future__ import annotations

import json
from fractions import Fraction as Q
from typing import Any, cast

import numpy as np
import pytest

from epsbench.diagnostics import a2_reappearance as a2
from epsbench.diagnostics.bounded_prefix_memory import BoundedPrefixMemory
from epsbench.diagnostics.restricted_mask_projection import Trust
from epsbench.diagnostics.visible_forecast_contract import CausalInput, Limits, TokenFrame
from epsbench.schema import Modality, ModalityPermissionSet

ACCESS = ModalityPermissionSet(
    allowed=frozenset(
        {
            Modality.SURFACE_REGIONS,
            Modality.REGION_CORRESPONDENCE,
            Modality.EXECUTED_ACTION,
        }
    )
)
TRUST = Trust(True, True, True, True)
NAMES = tuple(f"surface-{i:016x}" for i in (1, 2, 3))


def mask(*columns: int, row: int = 1) -> np.ndarray:
    result = np.zeros((3, 8), dtype=np.bool_)
    for column in columns:
        result[row, column] = True
    return result


def make_memory(
    frames: tuple[dict[int, tuple[int, ...]], ...],
    commands: tuple[Q, Q] = (Q(1), Q(1)),
    announced: Q = Q(1),
    names: tuple[str, ...] = NAMES,
    rows: dict[int, int] | None = None,
) -> BoundedPrefixMemory:
    row_by_index = {} if rows is None else rows
    built = []
    for frame_index, visible in enumerate(frames):
        built.append(
            TokenFrame(
                frame_index,
                (3, 8),
                tuple(
                    (names[i], mask(*columns, row=row_by_index.get(i, 1)))
                    for i, columns in sorted(visible.items())
                ),
            )
        )
    source = CausalInput(
        tuple(built),
        tuple((Q(0), command, Q(0)) for command in commands),
        (Q(0), announced, Q(0)),
        ACCESS,
        Limits(3, 24, 3),
    )
    return BoundedPrefixMemory(source, ACCESS, TRUST, (False, False, False), "a" * 32, "b" * 40)


def payload(memory: BoundedPrefixMemory) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(a2.predict(memory)))


def test_exact_rules_controls_transport_and_binding() -> None:
    p = make_memory(
        (
            {0: (1,), 1: (0,), 2: (7,)},
            {0: (2,), 1: (4,), 2: (7,)},
            {1: (3, 4, 5), 2: (7,)},
        ),
    )
    out = payload(p)
    i = p.alignment.index(NAMES[0])
    row = out["rows"][i]
    assert out["methods"] == list(a2.METHODS)
    assert row["zero"]["q"] == "0"
    assert row["one"]["q"] == "1"
    assert row["half"]["q"] == "1/2"
    assert row["current_presence"]["q"] == "0"
    assert row["motion_only"]["q"] == "1"
    assert out["blockers"][i]["selected"] == [p.alignment.index(NAMES[1])]
    assert out["blockers"][i]["erased_support"] == [1]
    assert row["motion_overlap"]["q"] == "0"
    assert out["binding_sha256"] and out["feature_sha256"]
    assert all("token" not in json.dumps(item) for item in out["rows"])


@pytest.mark.parametrize(
    ("value", "expected"),
    ((Q(1, 2), 1), (Q(-1, 2), -1), (Q(3, 2), 2), (Q(-3, 2), -2)),
)
def test_half_ties_round_away_from_zero(value: Q, expected: int) -> None:
    assert a2._round_away(value) == expected


def test_mirror_and_row_permutation_preserve_rule_values() -> None:
    original = make_memory(
        ({0: (1,), 1: (5,), 2: (7,)}, {0: (2,), 1: (5,), 2: (7,)}, {1: (5,), 2: (7,)})
    )
    renamed = tuple(f"surface-{i:016x}" for i in (3, 1, 2))
    permuted = make_memory(
        ({0: (1,), 1: (5,), 2: (7,)}, {0: (2,), 1: (5,), 2: (7,)}, {1: (5,), 2: (7,)}),
        names=renamed,
    )
    a, b = payload(original), payload(permuted)
    for physical, old_name in enumerate(NAMES):
        oi = original.alignment.index(old_name)
        pi = permuted.alignment.index(renamed[physical])
        assert a["rows"][oi] == b["rows"][pi]
    mirrored = make_memory(
        (
            {0: (6,), 1: (2,)},
            {0: (5,), 1: (2,)},
            {1: (2,)},
        ),
        (Q(-1), Q(-1)),
        Q(-1),
    )
    reflected = payload(mirrored)
    for name in NAMES[:2]:
        assert (
            a["rows"][original.alignment.index(name)]
            == reflected["rows"][mirrored.alignment.index(name)]
        )


def test_zero_denominator_and_insufficient_history_fallbacks() -> None:
    zero = make_memory(
        ({0: (1,), 1: (6,)}, {1: (6,)}, {0: (2,), 1: (6,)}),
        (Q(1), Q(-1)),
        Q(1),
    )
    out = payload(zero)
    idx = zero.alignment.index(NAMES[0])
    assert out["rows"][idx]["motion_overlap"] == {
        "q": "1/2",
        "fallback": True,
        "reason": "ZERO_COMMAND_DENOMINATOR",
    }
    short = make_memory(({0: (1,), 1: (6,)}, {1: (6,)}, {1: (6,)}))
    out = payload(short)
    idx = short.alignment.index(NAMES[0])
    assert out["rows"][idx]["motion_only"]["reason"] == "INSUFFICIENT_HISTORY"
    assert out["rows"][idx]["motion_overlap"]["q"] == "1/2"


def test_selected_blocker_without_history_or_with_zero_denominator_falls_back() -> None:
    insufficient = make_memory(({0: (1,)}, {0: (2,)}, {1: (3,)}))
    out = payload(insufficient)
    query = insufficient.alignment.index(NAMES[0])
    blocker = insufficient.alignment.index(NAMES[1])
    assert out["blockers"][query]["selected"] == [blocker]
    assert out["blockers"][query]["erased_support"] == [None]
    assert out["rows"][query]["motion_overlap"] == {
        "q": "1/2",
        "fallback": True,
        "reason": "BLOCKER_INSUFFICIENT_HISTORY",
    }

    zero_denominator = make_memory(
        ({0: (2,), 1: (6,)}, {0: (3,)}, {1: (2,)}),
        (Q(1), Q(-1)),
        Q(1),
    )
    out = payload(zero_denominator)
    query = zero_denominator.alignment.index(NAMES[0])
    blocker = zero_denominator.alignment.index(NAMES[1])
    assert out["blockers"][query]["selected"] == [blocker]
    assert out["rows"][query]["motion_overlap"] == {
        "q": "1/2",
        "fallback": True,
        "reason": "BLOCKER_ZERO_COMMAND_DENOMINATOR",
    }


def test_future_transport_uses_original_mask_after_current_crop() -> None:
    p = make_memory(
        ({0: (4,), 1: (0,)}, {0: (6,), 1: (0,)}, {1: (0,)}),
        (Q(1), Q(2)),
        Q(-2),
    )
    out = payload(p)
    idx = p.alignment.index(NAMES[0])
    assert out["rows"][idx]["motion_only"]["q"] == "1"
    assert out["rows"][idx]["motion_overlap"]["q"] == "1"


def test_exact_retrace_and_equal_distance_control() -> None:
    p = make_memory(
        ({0: (2,), 1: (6,)}, {0: (3,), 1: (6,)}, {1: (6,)}),
        (Q(1), Q(1)),
        Q(-2),
    )
    out = payload(p)
    idx = p.alignment.index(NAMES[0])
    assert out["rows"][idx]["exact_retrace"]["q"] == "1"
    assert out["rows"][idx]["toward_last_view"]["q"] == "1/2"


def test_all_inventory_rows_include_currently_hidden_query() -> None:
    p = make_memory(({0: (1,), 1: (6,)}, {1: (6,)}, {1: (6,)}))
    out = payload(p)
    assert out["alignment_count"] == len(p.alignment) == len(out["rows"])
    idx = p.alignment.index(NAMES[0])
    assert out["rows"][idx]["current_presence"]["q"] == "0"
    for row_index, row in enumerate(out["rows"]):
        assert set(row) == set(a2.METHODS)
        assert all(method["q"] in {"0", "1", "1/2"} for method in row.values())
        assert all(
            0 <= selected < len(p.alignment) for selected in out["blockers"][row_index]["selected"]
        )


def test_binding_changes_provenance_without_changing_predictions() -> None:
    source = make_memory(({0: (1,), 1: (6,)}, {0: (2,), 1: (6,)}, {1: (6,)}))
    other = BoundedPrefixMemory(
        source.revalidate(), ACCESS, TRUST, (False, False, False), "c" * 32, "d" * 40
    )
    first, second = payload(source), payload(other)
    assert first["feature_sha256"] == second["feature_sha256"]
    assert first["binding_sha256"] != second["binding_sha256"]
    assert first["rows"] == second["rows"]
    assert first["blockers"] == second["blockers"]


def test_permission_denial_precedes_feature_and_binding_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    p = make_memory(({0: (1,), 1: (6,)}, {0: (2,), 1: (6,)}, {1: (6,)}))
    denied = ModalityPermissionSet(allowed=frozenset({Modality.SURFACE_REGIONS}))
    object.__setattr__(p, "access", denied)

    def blocked(*args: Any, **kwargs: Any) -> bytes:
        raise AssertionError("feature or binding accessed before permission check")

    monkeypatch.setattr(BoundedPrefixMemory, "feature_bytes", blocked)
    monkeypatch.setattr(BoundedPrefixMemory, "binding_bytes", blocked)
    with pytest.raises(PermissionError):
        a2.predict(p)


def test_nonlateral_commands_rejected() -> None:
    p = make_memory(({0: (1,), 1: (6,)}, {0: (2,), 1: (6,)}, {1: (6,)}))
    source = p.revalidate()
    bad = CausalInput(
        source.frames,
        ((Q(1), Q(1), Q(0)), source.executed[1]),
        source.announced,
        ACCESS,
        source.limits,
    )
    invalid = BoundedPrefixMemory(bad, ACCESS, TRUST, (False, False, False), "a" * 32, "b" * 40)
    with pytest.raises(ValueError, match="pure lateral"):
        a2.predict(invalid)
