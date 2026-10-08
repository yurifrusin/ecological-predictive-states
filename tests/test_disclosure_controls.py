"""Public Boolean/numeric fixtures; no private study table or physical data."""

from __future__ import annotations

import json
from fractions import Fraction as Q
from typing import Any, cast

import numpy as np
import pytest

from epsbench.diagnostics import disclosure_controls as controls
from epsbench.diagnostics.bounded_prefix_memory import BoundedPrefixMemory
from epsbench.diagnostics.disclosure_report import Group, summarize
from epsbench.diagnostics.restricted_mask_projection import Trust
from epsbench.diagnostics.visible_forecast_contract import CausalInput, Limits, TokenFrame
from epsbench.schema import Modality, ModalityPermissionSet
from epsbench.utils.canonical import canonical_json_bytes

ACCESS = ModalityPermissionSet(
    allowed=frozenset(
        {Modality.SURFACE_REGIONS, Modality.REGION_CORRESPONDENCE, Modality.EXECUTED_ACTION}
    )
)
TRUST = Trust(True, True, True, True)
NAMES = ("surface-0000000000000001", "surface-0000000000000002")
SHAPE = (4, 20)


def mask(columns: range, height: int = 2) -> np.ndarray:
    result = np.zeros(SHAPE, dtype=np.bool_)
    result[:height, list(columns)] = True
    return result


def memory(
    announced: Q,
    names: tuple[str, ...] = NAMES,
    frames: tuple[TokenFrame, ...] | None = None,
    contacts: bool = True,
) -> BoundedPrefixMemory:
    if frames is None:
        frames = (
            TokenFrame(0, SHAPE, ((names[0], mask(range(0, 8))), (names[1], mask(range(12, 16))))),
            TokenFrame(1, SHAPE, ((names[0], mask(range(4, 12))), (names[1], mask(range(12, 16))))),
            TokenFrame(2, SHAPE, ((names[1], mask(range(12, 16))),)),
        )
    source = CausalInput(
        frames, ((Q(0), Q(1), Q(0)),) * 2, (Q(0), announced, Q(0)), ACCESS, Limits(3, 80, 16)
    )
    return BoundedPrefixMemory(source, ACCESS, TRUST, (contacts,) * 3, "a" * 32, "b" * 40)


def test_threshold_transport_overlap_and_first_prefix_lookup() -> None:
    out = json.loads(controls.predict(memory(Q(1))))
    row = out["rows"][0]
    assert row["methods"]["motion_only"]["support"] == 16
    assert row["methods"]["motion_only"]["p"] == "1"
    assert row["methods"]["motion_overlap"]["support"] == 8
    assert row["methods"]["motion_overlap"]["p"] == "0"
    assert row["selected"] == [1]
    assert row["methods"]["current_support"]["p"] == "0"
    retrace = json.loads(controls.predict(memory(Q(-1))))
    assert retrace["rows"][0]["methods"]["exact_retrace"]["support"] == 16
    assert retrace["rows"][0]["methods"]["exact_retrace"]["p"] == "1"


@pytest.mark.parametrize(
    ("a", "b", "choice"),
    ((Q(3, 4), Q(0), None), (Q(1), Q(1), 0), (Q(0), Q(1), 1), (Q(1, 2), Q(1, 2), None)),
)
def test_exact_policy_and_tie_priority(a: Q, b: Q, choice: int | None) -> None:
    assert controls.policy(a, b) == choice


def test_all_rows_fallback_insufficient_blocker_and_cropping() -> None:
    frames = (
        TokenFrame(0, SHAPE, ((NAMES[0], mask(range(0, 8))),)),
        TokenFrame(1, SHAPE, ((NAMES[0], mask(range(4, 12))),)),
        TokenFrame(2, SHAPE, ((NAMES[1], mask(range(12, 16))),)),
    )
    out = json.loads(controls.predict(memory(Q(1), frames=frames)))
    assert len(out["rows"]) == 2
    assert out["rows"][0]["methods"]["motion_overlap"]["p"] == "1/2"
    assert out["rows"][0]["methods"]["motion_overlap"]["reason"] == "BLOCKER_INSUFFICIENT_HISTORY"
    assert out["rows"][1]["methods"]["motion_only"]["fallback"]
    cropped = json.loads(controls.predict(memory(Q(4))))
    assert cropped["rows"][0]["methods"]["motion_only"]["support"] == 0


def test_permutation_diversity_excludes_commands_and_preserves_contacts() -> None:
    a = memory(Q(1))
    b = memory(Q(-1), names=NAMES[::-1])
    assert controls.optical_digest(a) == controls.optical_digest(b)
    assert controls.optical_digest(a) != controls.optical_digest(memory(Q(1), contacts=False))
    old, renamed = (
        json.loads(controls.predict(a)),
        json.loads(controls.predict(memory(Q(1), names=NAMES[::-1]))),
    )
    assert old["rows"][0]["methods"] == renamed["rows"][1]["methods"]
    assert old["rows"][1]["methods"] == renamed["rows"][0]["methods"]


def make_score(empty: bool = False) -> bytes:
    if empty:
        frames = tuple(TokenFrame(i, SHAPE, ()) for i in range(3))
        a, b = memory(Q(-1), frames=frames), memory(Q(1), frames=frames)
        futures = (TokenFrame(3, SHAPE, ()),) * 2
    else:
        a, b = memory(Q(-1)), memory(Q(1))
        futures = (TokenFrame(3, SHAPE, ((NAMES[0], mask(range(0, 8))),)), TokenFrame(3, SHAPE, ()))
    return controls.score(a, b, (controls.predict(a), controls.predict(b)), futures, TRUST)


def test_score_costs_forecast_binding_and_before_future_validation() -> None:
    out = json.loads(make_score())
    assert out["rows"][0]["oracle_cost"] == 1
    assert out["rows"][0]["methods"]["one"] == {"choice": 0, "cost": 1, "regret": 0}
    assert out["rows"][0]["methods"]["half"] == {"choice": None, "cost": 2, "regret": 1}
    assert len(out["future_sha256"]) == 2
    a, b = memory(Q(-1)), memory(Q(1))
    with pytest.raises(ValueError, match="forecast"):
        controls.score(a, b, (controls.predict(b), controls.predict(a)), cast(Any, object()), TRUST)
    with pytest.raises(ValueError):
        controls.validate(a, controls.predict(a) + b"\n")


def test_qualified_empty_population_fail_missing_evidence_inconclusive_and_precedence() -> None:
    empty = make_score(True)
    groups = tuple(
        Group("slab" if i < 12 else "doorway", f"{i:064x}", True, True, True, False, empty, ())
        for i in range(24)
    )
    report = json.loads(summarize(groups))
    assert report["status"] == "FAIL" and report["reason"] == "TASK_ADEQUACY"
    assert report["families"]["slab"]["null_slots"] == 12
    assert report["families"]["slab"]["oracle_cost"] is None
    missing = Group("slab", "f" * 64, False, False, True, False, None, None)
    assert json.loads(summarize((missing,)))["status"] == "INCONCLUSIVE"
    contradiction = Group("slab", "f" * 64, False, False, True, True, None, None)
    assert json.loads(summarize((contradiction,)))["status"] == "FAIL"
    parity = Group("slab", "f" * 64, False, False, False, True, None, None)
    assert json.loads(summarize((parity,)))["status"] == "STOP"


def test_ancestry_grouping_and_conditional_mean_no_null_imputation() -> None:
    present, empty = make_score(), make_score(True)
    groups = (
        Group("slab", "1" * 64, True, True, True, False, present, ("UNKNOWN", "VISIBLE")),
        Group("slab", "1" * 64, True, True, True, False, empty, ()),
        Group("slab", "2" * 64, True, True, True, False, empty, ()),
    )
    report = json.loads(summarize(groups))
    family = report["families"]["slab"]
    assert family["slots"] == 3 and family["ancestries"] == 2 and family["eligible"] == 1
    assert family["oracle_cost"] == "1"
    assert family["methods"]["half"] == {"cost": "2", "regret": "1"}


def test_malformed_costs_and_boolean_integer_rejected() -> None:
    p = json.loads(make_score())
    p["rows"][0]["methods"]["one"]["cost"] = True
    with pytest.raises(ValueError):
        Group(
            "slab",
            "0" * 64,
            True,
            True,
            True,
            False,
            canonical_json_bytes(p),
            ("UNKNOWN", "VISIBLE"),
        )
    with pytest.raises(ValueError):
        controls.policy(cast(Any, True), Q(1))


def test_strict_threshold_new_units_and_future_trust() -> None:
    fifteen = mask(range(0, 8))
    fifteen[1, 7] = False
    frames = tuple(TokenFrame(i, SHAPE, ((NAMES[0], fifteen),)) for i in range(3))
    a, b = memory(Q(-1), frames=frames), memory(Q(1), frames=frames)
    assert json.loads(controls.predict(a))["rows"][0]["methods"]["current_support"]["p"] == "0"
    new_name = "surface-0000000000000003"
    future = TokenFrame(3, SHAPE, ((NAMES[0], mask(range(0, 8))), (new_name, mask(range(12, 16)))))
    out = json.loads(
        controls.score(
            a,
            b,
            (controls.predict(a), controls.predict(b)),
            (future, TokenFrame(3, SHAPE, ())),
            TRUST,
        )
    )
    assert len(out["rows"]) == 1 and out["rows"][0]["success"] == [True, False]
    assert out["new_counts"] == [1, 0] and out["new_pixels"] == [8, 0]
    with pytest.raises(ValueError):
        controls.score(
            a,
            b,
            (controls.predict(a), controls.predict(b)),
            cast(Any, object()),
            Trust(False, True, True, True),
        )


def test_denied_memory_before_feature_or_binding_reads() -> None:
    m = memory(Q(1))
    object.__setattr__(m, "access", ModalityPermissionSet(allowed=frozenset({Modality.DEPTH})))
    object.__setattr__(m, "_source", b"unreadable")
    for function in (controls.predict, controls.optical_digest):
        with pytest.raises(PermissionError):
            function(m)
    with pytest.raises(PermissionError):
        controls.validate(m, cast(Any, object()))


def numeric_groups() -> tuple[Group, ...]:
    """Artificial aggregate arithmetic records, not generated geometry or efficacy evidence."""
    groups = []
    for i in range(24):
        success = [i % 2 == 0, i % 2 == 1]
        methods = {
            m: {"choice": None, "cost": 2, "regret": 1}
            for m in (
                "zero",
                "half",
                "current_support",
                "exact_retrace",
                "motion_only",
                "motion_overlap",
                "always_abstain",
            )
        }
        for m, choice in (("one", 0), ("constant_0", 0), ("constant_1", 1)):
            cost = 1 if success[choice] else 5
            methods[m] = {"choice": choice, "cost": cost, "regret": cost - 1}
        data = canonical_json_bytes(
            {
                "version": controls.VERSION + ":score",
                "forecast_sha256": ["0" * 64] * 2,
                "future_sha256": ["1" * 64] * 2,
                "optical_sha256": f"{i + 1:064x}",
                "new_counts": [0, 0],
                "new_pixels": [0, 0],
                "rows": [
                    {
                        "row": 0,
                        "remembered": True,
                        "support": [16 if s else 0 for s in success],
                        "success": success,
                        "retrace": [False, False],
                        "oracle_cost": 1,
                        "methods": methods,
                    }
                ],
            }
        )
        groups.append(
            Group(
                "slab" if i < 12 else "doorway",
                f"{i + 1:064x}",
                True,
                True,
                True,
                False,
                data,
                ("COMPLETE_OCCLUSION",),
            )
        )
    return tuple(groups)


def test_aggregate_pass_stop_and_duplicate_ancestry_not_extra_n() -> None:
    groups = numeric_groups()
    report = json.loads(summarize(groups))
    assert report["status"] == "PASS"
    assert report["overall_oracle_cost"] == "1"
    assert report["families"]["slab"]["methods"]["constant_0"] == {"cost": "3", "regret": "2"}
    changed = []
    duplicated = []
    for i, group in enumerate(groups):
        assert group.score is not None
        p = json.loads(group.score)
        if group.family == "slab":
            winner = 0 if p["rows"][0]["success"][0] else 1
            p["rows"][0]["methods"]["motion_only"] = {"choice": winner, "cost": 1, "regret": 0}
        changed.append(
            Group(
                group.family,
                group.ancestry,
                True,
                True,
                True,
                False,
                canonical_json_bytes(p),
                group.causes,
            )
        )
        duplicated.append(
            Group(
                group.family,
                f"{1 + i // 2:064x}",
                True,
                True,
                True,
                False,
                group.score,
                group.causes,
            )
        )
    assert json.loads(summarize(tuple(changed)))["status"] == "STOP"
    duplicate_report = json.loads(summarize(tuple(duplicated)))
    assert duplicate_report["distinct_ancestries"] == 12
    assert duplicate_report["status"] == "FAIL"


def test_software_smoke_validation_and_inspection() -> None:
    a = memory(Q(-1))
    saved = controls.predict(a)
    out = controls.validate(a, saved)
    assert len(out["rows"]) == len(a.alignment) == 2
    assert controls.predict(a) == saved
    assert json.loads(summarize(numeric_groups()))["status"] == "PASS"
    print(
        "Synthetic software inspection:",
        controls.VERSION,
        "threshold",
        out["threshold"],
        "rows",
        len(out["rows"]),
        "methods",
        ",".join(out["methods"]),
    )
