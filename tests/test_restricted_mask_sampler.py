"""Synthetic source checks for the prospective sampler; no membership is drawn."""

from __future__ import annotations

import json
from fractions import Fraction
from typing import Any

import numpy as np
import pytest

from epsbench.diagnostics import restricted_mask_projection as projection_api
from epsbench.diagnostics import restricted_mask_sampler as sampler
from epsbench.diagnostics.boundary_observation import VisibleRaster
from epsbench.diagnostics.causal_region_lifecycle import TrustedObservation, advance
from epsbench.diagnostics.visible_forecast_contract import REQUIRED, CausalView, Limits
from epsbench.schema import Action, ModalityPermissionSet
from epsbench.utils.canonical import canonical_json_bytes, sha256_bytes

ACCESS = ModalityPermissionSet(allowed=REQUIRED)
TRUST = projection_api.Trust(True, True, True, True)


def unit_bytes() -> bytes:
    a = sampler.Box(
        (Fraction(0), Fraction(1, 2), Fraction(1, 4)),
        (Fraction(1), Fraction(1), Fraction(1)),
    )
    b = sampler.Box(
        (Fraction(-1), Fraction(2), Fraction(1, 4)),
        (Fraction(0), Fraction(5, 2), Fraction(1)),
    )
    return sampler._unit_payload(
        "train",
        0,
        a,
        b,
        Fraction(0),
        Fraction(3, 4),
        Fraction(3, 4),
        (Fraction(1, 4), Fraction(5, 4)),
    )


def raw_audit() -> tuple[dict[str, Any], dict[str, Any]]:
    labels = [[0] * 32 for _ in range(32)]
    raw = {"labels": labels, "ties": []}
    audit = {
        "labels": labels,
        "ties": [],
        "boundaries": [],
        "supports": [[4, 0, False], [4, 0, False]],
        "footprints": [
            {
                "polygon": [["0", "0"], ["1", "0"], ["1", "1"]],
                "depths": ["1", "2"],
                "domain": True,
            },
            {
                "polygon": [["0", "0"], ["1", "0"], ["1", "1"]],
                "depths": ["1", "2"],
                "domain": True,
            },
        ],
    }
    return raw, audit


class EmptyPrefixProvider:
    def raster(self, index: int) -> VisibleRaster:
        return VisibleRaster(index, np.zeros((32, 32), dtype=np.int32), ())

    def observation(self, index: int) -> TrustedObservation:
        return TrustedObservation(self.raster(index), True, True, True)


def target_projection(episode: str, head: str, announced: Action) -> projection_api.Projection:
    provider = EmptyPrefixProvider()
    executed = Action(name="lateral_right", delta_forward=0.0, delta_lateral=0.75, delta_yaw=0.0)
    command = projection_api.numeric(executed)
    announced_command = projection_api.numeric(announced)
    source = CausalView(provider, ACCESS, 1, Limits(2, 1024, 1)).materialize(
        (command,), announced_command
    )
    state = None
    for index in range(2):
        update = advance(
            state,
            provider,
            ACCESS,
            index,
            1,
            None if index == 0 else executed,
        )
        assert update.state is not None and update.unresolved is None
        state = update.state
    assert state is not None
    return projection_api.Projection(
        source,
        state,
        ACCESS,
        TRUST,
        (False, False),
        episode,
        head,
        1,
        announced,
    )


def declaration(payload: bytes, **updates: object) -> dict[str, object]:
    unit = sampler.validate_unit(payload)
    value: dict[str, object] = {
        "unit": payload,
        "unit_fingerprint": unit.fingerprint,
        "phase": "prefix",
        "branch": 0,
        "episode": "a" * 32,
        "source_head": "b" * 40,
        "source_tree": "d" * 40,
        "context_sha256": "c" * 64,
        "frame_index": 0,
        "lateral": "0",
        "raw_lateral": "0",
        "audit_lateral": "0",
        "action": None,
    }
    value.update(updates)
    return value


def test_handwritten_unit_parses_exact_rational_action_binding() -> None:
    unit = sampler.validate_unit(unit_bytes())
    assert unit.split == "train"
    assert unit.prefix1_lateral == Fraction(3, 4)
    assert unit.target_laterals == (Fraction(1, 4), Fraction(5, 4))
    assert unit.fingerprint == sampler._fingerprint(unit.box_a, unit.box_b)


@pytest.mark.parametrize(
    "change",
    [
        lambda p: p[:-1],
        lambda p: canonical_json_bytes({**json.loads(p), "ordinal": 64}),
        lambda p: canonical_json_bytes({**json.loads(p), "fingerprint": "0" * 64}),
        lambda p: canonical_json_bytes({**json.loads(p), "executed_lateral": "1/2"}),
        lambda p: canonical_json_bytes(
            {
                **json.loads(p),
                "boxes": {
                    **json.loads(p)["boxes"],
                    "a": {
                        **json.loads(p)["boxes"]["a"],
                        "lower": ["1/3", "1/2", "1/4"],
                    },
                },
            }
        ),
    ],
)
def test_unit_rejects_noncanonical_or_out_of_contract_bytes(change: object) -> None:
    with pytest.raises(ValueError):
        sampler.validate_unit(change(unit_bytes()))  # type: ignore[operator]


def test_local_raw_audit_receipt_is_bound_to_unit_and_prefix_context() -> None:
    payload = unit_bytes()
    raw, audit = raw_audit()
    result = sampler.verify_qualification(declaration(payload), raw, audit, None)
    assert result.unit_fingerprint == sampler.validate_unit(payload).fingerprint
    assert result.phase == "prefix" and result.frame_index == 0
    assert result.labels[0] == (0,) * 32


def test_second_prefix_receipt_requires_the_genuine_executed_action() -> None:
    payload = unit_bytes()
    raw, audit = raw_audit()
    decl = declaration(
        payload,
        branch=1,
        frame_index=1,
        lateral="3/4",
        raw_lateral="3/4",
        audit_lateral="3/4",
        action={
            "name": "lateral_right",
            "delta_forward": 0.0,
            "delta_lateral": 0.75,
            "delta_yaw": 0.0,
        },
    )
    result = sampler.verify_qualification(decl, raw, audit, None)
    assert result.frame_index == 1
    decl["action"] = {
        "name": "lateral_left",
        "delta_forward": 0.0,
        "delta_lateral": -0.75,
        "delta_yaw": 0.0,
    }
    with pytest.raises(ValueError):
        sampler.verify_qualification(decl, raw, audit, None)


def test_fingerprint_is_geometry_only_across_names_and_split_labels() -> None:
    payload = unit_bytes()
    unit = sampler.validate_unit(payload)
    renamed = sampler._unit_payload(
        "development",
        0,
        unit.box_a,
        unit.box_b,
        unit.prefix0_lateral,
        unit.executed_lateral,
        unit.prefix1_lateral,
        unit.target_laterals,
    )
    assert sampler.validate_unit(renamed).fingerprint == unit.fingerprint
    assert renamed != payload


def test_raw_audit_mismatch_and_ties_do_not_qualify() -> None:
    payload = unit_bytes()
    raw, audit = raw_audit()
    audit["labels"] = [[1] + [0] * 31, *audit["labels"][1:]]
    with pytest.raises(ValueError, match="disagreement"):
        sampler.verify_qualification(declaration(payload), raw, audit, None)
    raw, audit = raw_audit()
    raw["ties"] = [1]
    with pytest.raises(ValueError, match="tie"):
        sampler.verify_qualification(declaration(payload), raw, audit, None)


def test_target_receipt_requires_typed_projection_before_parsing_labels() -> None:
    payload = unit_bytes()
    raw, audit = raw_audit()
    decl = declaration(
        payload,
        phase="target",
        branch=0,
        frame_index=2,
        lateral="1/4",
        raw_lateral="1/4",
        audit_lateral="1/4",
        action={
            "name": "lateral_left",
            "delta_forward": 0.0,
            "delta_lateral": -0.5,
            "delta_yaw": 0.0,
        },
    )
    with pytest.raises(ValueError, match="typed before-only projection"):
        sampler.verify_qualification(decl, raw, audit, None)


def test_target_receipt_binds_announced_action_to_typed_before_projection() -> None:
    payload = unit_bytes()
    raw, audit = raw_audit()
    action = Action(name="lateral_left", delta_forward=0.0, delta_lateral=-0.5, delta_yaw=0.0)
    episode, head = "a" * 32, "b" * 40
    projection = target_projection(episode, head, action)
    decl = declaration(
        payload,
        phase="target",
        branch=0,
        episode=episode,
        source_head=head,
        context_sha256=sha256_bytes(projection.binding_bytes()),
        frame_index=2,
        lateral="1/4",
        raw_lateral="1/4",
        audit_lateral="1/4",
        action=action.model_dump(mode="json"),
    )
    result = sampler.verify_qualification(decl, raw, audit, projection)
    assert result.phase == "target" and result.frame_index == 2
    assert result.unit_payload_sha256 == sha256_bytes(payload)
