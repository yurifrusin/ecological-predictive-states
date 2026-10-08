"""Public handwritten prefixes only; no physical scene or retained outcome fixtures."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from fractions import Fraction as Q
from typing import Any

import numpy as np
import pytest

from epsbench.diagnostics import bounded_prefix_memory as m
from epsbench.diagnostics.boundary_observation import BoundaryObservationView
from epsbench.diagnostics.restricted_mask_projection import Projection, Trust, lifecycle
from epsbench.diagnostics.visible_forecast_contract import (
    REQUIRED,
    CausalInput,
    Limits,
    TokenFrame,
)
from epsbench.schema import Action, Modality, ModalityPermissionSet
from epsbench.utils.canonical import sha256_bytes

NAMES = tuple(f"surface-{i:016x}" for i in range(1, 4))
ACCESS = ModalityPermissionSet(allowed=REQUIRED)
TRUST = Trust(True, True, True, True)
COMMAND = (Q(0), Q(1, 2), Q(0))
ACTION = Action(name="lateral_right", delta_forward=0.0, delta_lateral=0.5, delta_yaw=0.0)


def prefix(right: bool = False, names: tuple[str, ...] = NAMES, count: int = 3) -> CausalInput:
    labels = [1, 2, 0, 3] if right else [1, 2, 3, 0]
    images = [labels, [1, 2, 0, 0], [1, 2, 0, 0], [0, 0, 0, 0]][:count]
    frames = tuple(
        TokenFrame(
            i,
            (1, 4),
            tuple((names[label - 1], np.array([row]) == label) for label in sorted(set(row) - {0})),
        )
        for i, row in enumerate(images)
    )
    return CausalInput(frames, (COMMAND,) * (count - 1), COMMAND, ACCESS, Limits(4, 4, 3))


def memory(
    source: CausalInput, available: tuple[bool, bool, bool] = (True, True, True)
) -> m.BoundedPrefixMemory:
    return m.BoundedPrefixMemory(source, ACCESS, TRUST, available, "a" * 32, "b" * 40)


def test_lossless_handwritten_recovery_and_binding() -> None:
    source = prefix()
    p = memory(source)
    features = json.loads(p.feature_bytes())
    assert set(features) == {"shape", "frames", "executed", "announced"}
    assert features["shape"] == [1, 4]
    assert features["executed"] == [["0", "1/2", "0"]] * 2
    assert features["announced"] == ["0", "1/2", "0"]
    assert [f["index"] for f in features["frames"]] == [0, 1, 2]
    for frame, payload in zip(source.frames, features["frames"], strict=True):
        observed = dict(frame.masks)
        for i, name in enumerate(p.alignment):
            mask = observed.get(name, np.zeros(source.shape, dtype=np.bool_))
            assert payload["masks"][i] == mask.tolist()
            assert payload["observed"][i] is (name in observed)
    target = p.alignment.index(NAMES[2])
    assert features["frames"][0]["masks"][target] == [[False, False, True, False]]
    assert all(not f["observed"][target] for f in features["frames"][1:])
    assert features["frames"][0]["contacts"]["pairs"] == [[0, 1], [1, 2]]
    assert p.revalidate().canonical_bytes() == source.canonical_bytes()
    binding = json.loads(p.binding_bytes())
    assert binding["input_sha256"] == source.digest
    assert binding["feature_sha256"] == sha256_bytes(p.feature_bytes())
    assert binding["alignment"] == list(p.alignment)
    assert binding["target_index"] == 3
    assert binding["version"] == m.VERSION
    assert all(name not in p.feature_bytes().decode() for name in NAMES)
    assert "source_head" not in features and "episode" not in features
    assert p.storage()["numeric_bytes"] == len(p.feature_bytes())


def test_old_alias_new_retains_older_difference() -> None:
    sources = (prefix(), prefix(True))
    old = [
        Projection(
            s,
            lifecycle(s, TRUST, (ACTION.model_dump_json().encode(),) * 2),
            ACCESS,
            TRUST,
            (True, True),
            "a" * 32,
            "b" * 40,
            2,
            ACTION,
        )
        for s in sources
    ]
    new = [memory(s) for s in sources]
    assert old[0].feature_bytes() == old[1].feature_bytes()
    assert new[0].feature_bytes() != new[1].feature_bytes()
    assert new[0].binding_bytes() != new[1].binding_bytes()
    assert sources[0].frames[1:] == sources[1].frames[1:]


def test_joint_opaque_permutation_changes_only_aligned_rows_and_pairs() -> None:
    old, new = memory(prefix()), memory(prefix(names=NAMES[::-1]))
    left, right = json.loads(old.feature_bytes()), json.loads(new.feature_bytes())
    row_map = [new.alignment.index(NAMES[::-1][i]) for i in range(3)]
    for before, after in zip(left["frames"], right["frames"], strict=True):
        assert [after["masks"][i] for i in row_map] == before["masks"]
        assert [after["observed"][i] for i in row_map] == before["observed"]
        transformed = sorted(
            sorted((row_map[a], row_map[b])) for a, b in before["contacts"]["pairs"]
        )
        assert after["contacts"]["pairs"] == transformed
    assert left["executed"] == right["executed"] and left["announced"] == right["announced"]


def test_missing_images_rejected_not_zero_support() -> None:
    for count in (1, 2, 4):
        with pytest.raises(ValueError, match="exactly three"):
            memory(prefix(count=count))
    source = prefix()
    object.__setattr__(source, "frames", (source.frames[0], None, source.frames[2]))
    with pytest.raises(ValueError, match="missing images"):
        memory(source)
    source = prefix()
    object.__setattr__(source.frames[1], "index", 5)
    with pytest.raises(ValueError, match="complete consecutive"):
        memory(source)


def test_zero_observation_and_not_yet_seen_are_recoverable() -> None:
    source = prefix()
    frames: tuple[TokenFrame, ...] = (TokenFrame(0, (1, 4), ()), source.frames[1], source.frames[0])
    frames = tuple(TokenFrame(i, f.shape, f.masks) for i, f in enumerate(frames))
    source = CausalInput(frames, (COMMAND,) * 2, COMMAND, ACCESS, Limits(3, 4, 3))
    features = json.loads(memory(source).feature_bytes())
    assert features["frames"][0]["masks"] == [[[False] * 4]] * 3
    assert features["frames"][0]["observed"] == [False] * 3
    assert features["frames"][0]["contacts"] == {"available": True, "pairs": []}


def test_unavailable_contacts_skip_extraction(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("unavailable relation endpoint fetched")

    monkeypatch.setattr(BoundaryObservationView, "observe", forbidden)
    features = json.loads(memory(prefix(), (False, False, False)).feature_bytes())
    assert all(f["contacts"] == {"available": False, "pairs": None} for f in features["frames"])
    assert features["frames"][0]["masks"][2] == [[False, False, True, False]]


@pytest.mark.parametrize("available", [(True, False, True), (False, True, False)])
def test_availability_does_not_change_masks(available: tuple[bool, bool, bool]) -> None:
    full = json.loads(memory(prefix()).feature_bytes())
    partial = json.loads(memory(prefix(), available).feature_bytes())
    for a, b, flag in zip(full["frames"], partial["frames"], available, strict=True):
        assert a["masks"] == b["masks"] and a["observed"] == b["observed"]
        assert b["contacts"]["available"] is flag
        assert b["contacts"]["pairs"] == (a["contacts"]["pairs"] if flag else None)


def test_typed_denial_first(monkeypatch: pytest.MonkeyPatch) -> None:
    source = prefix()

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("prefix serialized before access denial")

    monkeypatch.setattr(CausalInput, "canonical_bytes", forbidden)
    strings: Any = frozenset(modality.value for modality in REQUIRED)
    mutable: Any = set(REQUIRED)
    for denied in (
        ModalityPermissionSet(allowed=frozenset()),
        ModalityPermissionSet(allowed=frozenset({Modality.PRIVILEGED_GENERATION_RECORDS})),
        ModalityPermissionSet.model_construct(allowed=strings),
        ModalityPermissionSet.model_construct(allowed=mutable),
    ):
        with pytest.raises(PermissionError):
            m.BoundedPrefixMemory(source, denied, TRUST, (True, True, True), "a" * 32, "b" * 40)
        with pytest.raises(PermissionError):
            m.BoundedPrefixMemory(None, denied, TRUST, (), "bad", "bad")  # type: ignore[arg-type]


def test_owned_bytes_and_arrays_are_immutable() -> None:
    source = prefix()
    p = memory(source)
    before = p.feature_bytes(), p.binding_bytes(), p.revalidate().canonical_bytes()
    payload = json.loads(p.feature_bytes())
    payload["frames"][0]["masks"][2][0][2] = False
    with pytest.raises(ValueError):
        source.frames[0].masks[0][1][:] = False
    object.__setattr__(source, "frames", ())
    assert before == (p.feature_bytes(), p.binding_bytes(), p.revalidate().canonical_bytes())
    with pytest.raises(FrozenInstanceError):
        p.alignment = ()  # type: ignore[misc]


@pytest.mark.parametrize("shape,nodes", [((32, 32), 16), ((1, 1025), 1), ((1, 17), 17)])
def test_exact_caps(shape: tuple[int, int], nodes: int) -> None:
    frames = []
    for i in range(3):
        masks = []
        for n in range(nodes):
            a = np.zeros(shape, dtype=np.bool_)
            a.flat[n] = True
            masks.append((f"surface-{n:016x}", a))
        frames.append(TokenFrame(i, shape, tuple(masks)))
    source = CausalInput(tuple(frames), (COMMAND,) * 2, COMMAND, ACCESS, Limits(3, 2048, 17))
    if shape[0] * shape[1] <= 1024 and nodes <= 16:
        assert len(memory(source).alignment) == 16
    else:
        with pytest.raises(ValueError, match="budget exceeded"):
            memory(source)


def test_trust_query_and_source_binding() -> None:
    source = prefix()
    original = memory(source)
    changed = CausalInput(
        source.frames, (COMMAND,) * 2, (Q(0), Q(-1, 2), Q(0)), ACCESS, source.limits
    )
    assert memory(changed).feature_bytes() != original.feature_bytes()
    other = m.BoundedPrefixMemory(source, ACCESS, TRUST, (True, True, True), "c" * 32, "d" * 40)
    assert other.feature_bytes() == original.feature_bytes()
    assert other.binding_bytes() != original.binding_bytes()
    with pytest.raises(ValueError, match="complete trusted"):
        m.BoundedPrefixMemory(
            source, ACCESS, Trust(True, False, True, True), (True, True, True), "a" * 32, "b" * 40
        )
    with pytest.raises(ValueError, match="source identity"):
        m.BoundedPrefixMemory(source, ACCESS, TRUST, (True, True, True), "a" * 32, "not-a-head")


def test_guarded_smoke_validation_and_inspection() -> None:
    p = memory(prefix())
    recovered = p.revalidate()
    assert memory(recovered).feature_bytes() == p.feature_bytes()
    print(
        json.dumps(
            {
                "version": m.VERSION,
                "shape": recovered.shape,
                "frames": 3,
                "known_nodes": len(p.alignment),
                "storage": p.storage(),
                "input_sha256": recovered.digest,
                "synthetic_only": True,
            },
            sort_keys=True,
        )
    )
