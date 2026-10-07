"""Pure chronological oracle REGION bookkeeping; no predictive readout or native adapter."""

from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import numpy as np

from epsbench.annotations.derive import classify_mask_changes
from epsbench.diagnostics.boundary_observation import VisibleRaster
from epsbench.schema import (
    Action,
    MaskChangeKind,
    Modality,
    ModalityPermissionSet,
    RegionCorrespondence,
    RegionMaskChange,
    VisibilityState,
)
from epsbench.utils.canonical import canonical_json_bytes, sha256_bytes

VERSION = "causal-oracle-region-lifecycle-v1"
IDENTITY = "ORACLE_REGION_UNIT"
REQUIRED = frozenset(
    {Modality.SURFACE_REGIONS, Modality.REGION_CORRESPONDENCE, Modality.EXECUTED_ACTION}
)


def integer(value: object) -> int:
    if type(value) is not int or value < 0:
        raise ValueError("nonnegative exact integer required")
    return value


def action_bytes(action: Action) -> bytes:
    if type(action) is not Action:
        raise ValueError("exact Action required")
    # Revalidate even a model_construct instance, and normalize to the existing JSON schema.
    checked = Action.model_validate_json(canonical_json_bytes(action))
    return canonical_json_bytes(checked)


def check_action(payload: bytes) -> None:
    if type(payload) is not bytes or action_bytes(Action.model_validate_json(payload)) != payload:
        raise ValueError("canonical validated Action bytes required")


def mask_bytes(raster: VisibleRaster, token: str) -> bytes:
    label = dict((token, label) for label, token in raster.identities).get(token)
    bits = (
        raster.segmentation == label
        if label is not None
        else np.zeros_like(raster.segmentation, dtype=np.bool_)
    )
    return np.packbits(bits, bitorder="big").tobytes()


def check_mask(mask: bytes, area: int) -> None:
    if type(mask) is not bytes or len(mask) != (area + 7) // 8:
        raise ValueError("exact packed mask length required")
    unused = (-area) % 8
    if unused and mask[-1] & ((1 << unused) - 1):
        raise ValueError("mask padding must be zero")


@dataclass(frozen=True)
class Node:
    token: str
    current_mask: bytes
    previous_mask: bytes | None
    first_seen: int
    last_seen: int
    status: Literal["VISIBLE", "REMEMBERED_ABSENT"]


@dataclass(frozen=True)
class State:
    decision_index: int
    shape: tuple[int, int]
    executed_commands: tuple[bytes, ...]
    nodes: tuple[Node, ...]

    def __post_init__(self) -> None:
        t = integer(self.decision_index)
        if type(self.shape) is not tuple or len(self.shape) != 2:
            raise ValueError("exact image shape required")
        h, w = (integer(v) for v in self.shape)
        if not h or not w:
            raise ValueError("positive image area required")
        if type(self.executed_commands) is not tuple or len(self.executed_commands) != t:
            raise ValueError("complete executed-command log required")
        for command in self.executed_commands:
            check_action(command)
        if type(self.nodes) is not tuple or any(type(n) is not Node for n in self.nodes):
            raise ValueError("immutable exact nodes required")
        tokens = tuple(n.token for n in self.nodes)
        if any(
            type(k) is not str or re.fullmatch(r"surface-[0-9a-f]{16}", k) is None for k in tokens
        ):
            raise ValueError("opaque region token required")
        if tokens != tuple(sorted(set(tokens))):
            raise ValueError("sorted unique observed-ever inventory required")
        used_current = used_previous = 0
        for node in self.nodes:
            check_mask(node.current_mask, h * w)
            first, last = integer(node.first_seen), integer(node.last_seen)
            visible = any(node.current_mask)
            if not first <= last <= t or (last == t) != visible:
                raise ValueError("first/last observation indices differ from support")
            if node.status != ("VISIBLE" if visible else "REMEMBERED_ABSENT"):
                raise ValueError("status differs from current raster support")
            current = int.from_bytes(node.current_mask, "big")
            if used_current & current:
                raise ValueError("current region masks overlap")
            used_current |= current
            if t == 0:
                if node.previous_mask is not None or first != 0 or not visible:
                    raise ValueError("initial nodes must be first observed with null predecessor")
            else:
                if node.previous_mask is None:
                    raise ValueError("completed predecessor mask required")
                check_mask(node.previous_mask, h * w)
                previous = int.from_bytes(node.previous_mask, "big")
                if used_previous & previous:
                    raise ValueError("previous region masks overlap")
                used_previous |= previous
                if first == t - 1 and not previous:
                    raise ValueError("first-seen predecessor must have visible support")
                if first == t and previous:
                    raise ValueError("new node cannot have prior support")
                if previous and (first > t - 1 or (not visible and last != t - 1)):
                    raise ValueError("predecessor support differs from last-seen index")
                if not visible and not previous and last >= t - 1:
                    raise ValueError("remembered support differs from last-seen index")

    @property
    def inventory(self) -> tuple[str, ...]:
        return tuple(n.token for n in self.nodes)

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(
            {
                "version": VERSION,
                "decision_index": self.decision_index,
                "shape": list(self.shape),
                "identity_kind": IDENTITY,
                "executed_commands": [json.loads(c) for c in self.executed_commands],
                "nodes": [
                    {
                        "token": n.token,
                        "current_mask": base64.b64encode(n.current_mask).decode("ascii"),
                        "previous_mask": None
                        if n.previous_mask is None
                        else base64.b64encode(n.previous_mask).decode("ascii"),
                        "first_seen": n.first_seen,
                        "last_seen": n.last_seen,
                        "status": n.status,
                    }
                    for n in self.nodes
                ],
            }
        )

    def digest(self) -> str:
        return sha256_bytes(self.canonical_bytes())

    def storage(self) -> dict[str, int]:
        """Logical payload/serialized costs, not interpreter RSS or transactional peak."""
        return {
            "inventory_nodes": len(self.nodes),
            "mask_payload_bytes": sum(
                len(n.current_mask) + len(n.previous_mask or b"") for n in self.nodes
            ),
            "command_payload_bytes": sum(map(len, self.executed_commands)),
            "serialized_bytes": len(self.canonical_bytes()),
        }


def keys(value: Any, expected: set[str]) -> dict[str, Any]:
    if type(value) is not dict or set(value) != expected:
        raise ValueError("closed JSON keys required")
    return value


def decode_mask(value: Any) -> bytes:
    if type(value) is not str:
        raise ValueError("canonical base64 string required")
    result = base64.b64decode(value.encode("ascii"), validate=True)
    if base64.b64encode(result).decode("ascii") != value:
        raise ValueError("noncanonical base64")
    return result


def decode(payload: bytes) -> State:
    if type(payload) is not bytes:
        raise ValueError("exact canonical bytes required")
    root = keys(
        json.loads(payload),
        {"version", "decision_index", "shape", "identity_kind", "executed_commands", "nodes"},
    )
    if root["version"] != VERSION or root["identity_kind"] != IDENTITY:
        raise ValueError("versioned oracle region identity required")
    if type(root["shape"]) is not list or len(root["shape"]) != 2:
        raise ValueError("two dimensions required")
    if type(root["nodes"]) is not list or type(root["executed_commands"]) is not list:
        raise ValueError("node and command arrays required")
    nodes = []
    for value in root["nodes"]:
        n = keys(
            value, {"token", "current_mask", "previous_mask", "first_seen", "last_seen", "status"}
        )
        nodes.append(
            Node(
                n["token"],
                decode_mask(n["current_mask"]),
                None if n["previous_mask"] is None else decode_mask(n["previous_mask"]),
                n["first_seen"],
                n["last_seen"],
                n["status"],
            )
        )
    commands = []
    for value in root["executed_commands"]:
        keys(value, {"name", "delta_forward", "delta_lateral", "delta_yaw"})
        commands.append(action_bytes(Action.model_validate_json(canonical_json_bytes(value))))
    state = State(root["decision_index"], tuple(root["shape"]), tuple(commands), tuple(nodes))
    if state.canonical_bytes() != payload:
        raise ValueError("JSON must equal its canonical decode/re-encode, without newline")
    return state


@dataclass(frozen=True)
class TrustedObservation:
    """Adapter assertions, not inferred qualification; no hidden membership is carried."""

    raster: VisibleRaster | None
    complete_image: bool
    complete_association: bool
    stable_identity: bool
    unresolved_labels: int | None = None
    unresolved_pixels: int | None = None


class Provider(Protocol):
    def observation(self, index: int) -> TrustedObservation: ...


@dataclass(frozen=True)
class Unresolved:
    attempted_index: int
    reason: str
    known_inventory: int
    excluded_inventory: int
    unresolved_labels: int | None
    unresolved_pixels: int | None


@dataclass(frozen=True)
class Update:
    state: State | None
    unresolved: Unresolved | None


def advance(
    state: State | None,
    provider: Provider,
    permissions: ModalityPermissionSet,
    observation_index: int,
    decision_index: int,
    executed_action: Action | None = None,
) -> Update:
    """Atomic update; a rejection keeps the exact prior state and has no labels."""
    if (
        type(permissions) is not ModalityPermissionSet
        or type(permissions.allowed) is not frozenset
        or any(type(m) is not Modality for m in permissions.allowed)
        or permissions.allowed != REQUIRED
    ):
        raise PermissionError("exact typed causal modalities required before access")
    i, cutoff = integer(observation_index), integer(decision_index)
    if i > cutoff:
        raise PermissionError("future observation denied before provider access")
    if state is not None and type(state) is not State:
        raise ValueError("exact previous state required")
    expected = 0 if state is None else state.decision_index + 1
    inventory = 0 if state is None else len(state.nodes)

    def reject(reason: str, receipt: TrustedObservation | None = None) -> Update:
        def count(value: int | None) -> int | None:
            return value if type(value) is int and value >= 0 else None

        return Update(
            state,
            Unresolved(
                i,
                reason,
                inventory,
                inventory,
                count(receipt.unresolved_labels) if receipt else None,
                count(receipt.unresolved_pixels) if receipt else None,
            ),
        )

    if i != expected:
        return reject("nonconsecutive observation")
    try:
        if state is None:
            if executed_action is not None:
                return reject("initial observation cannot have an executed predecessor command")
            commands: tuple[bytes, ...] = ()
        else:
            if executed_action is None:
                return reject("executed predecessor command missing")
            commands = (*state.executed_commands, action_bytes(executed_action))
    except ValueError:
        return reject("invalid executed Action")
    receipt = provider.observation(i)
    if type(receipt) is not TrustedObservation:
        return reject("trusted observation receipt missing")
    if any(
        v is not True
        for v in (receipt.complete_image, receipt.complete_association, receipt.stable_identity)
    ):
        return reject("incomplete observation or unresolved oracle association", receipt)
    if any(
        v is not None and (type(v) is not int or v < 0)
        for v in (receipt.unresolved_labels, receipt.unresolved_pixels)
    ):
        return reject("invalid unresolved support counts", receipt)
    if receipt.unresolved_labels not in (None, 0) or receipt.unresolved_pixels not in (None, 0):
        return reject("unresolved support despite completeness assertion", receipt)
    raster = receipt.raster
    if type(raster) is not VisibleRaster or raster.sequence_index != i:
        return reject("typed raster chronology differs", receipt)
    try:
        # Snapshot/revalidate trusted visible shape even if a caller forged/mutated its instance.
        raster = VisibleRaster(raster.sequence_index, raster.segmentation, raster.identities)
        shape = (int(raster.segmentation.shape[0]), int(raster.segmentation.shape[1]))
        if state is not None and shape != state.shape:
            return reject("image shape changed", receipt)
        previous = {} if state is None else {n.token: n for n in state.nodes}
        tokens = sorted(set(previous) | {k for _, k in raster.identities})
        empty = bytes((shape[0] * shape[1] + 7) // 8)
        nodes = []
        for token in tokens:
            old = previous.get(token)
            current = mask_bytes(raster, token)
            visible = any(current)
            nodes.append(
                Node(
                    token,
                    current,
                    None if state is None else old.current_mask if old else empty,
                    old.first_seen if old else i,
                    i if visible else old.last_seen if old else i,
                    "VISIBLE" if visible else "REMEMBERED_ABSENT",
                )
            )
        candidate = State(i, shape, commands, tuple(nodes))
    except (ValueError, TypeError):
        return reject("invalid trusted raster or state invariants", receipt)
    return Update(candidate, None)


@dataclass(frozen=True)
class CompletedLabels:
    decision_index: int
    inventory: tuple[str, ...]
    visibility: tuple[VisibilityState, ...]
    correspondence: tuple[RegionCorrespondence, ...]
    changes: tuple[RegionMaskChange, ...]

    def coverage(self) -> dict[str, int | str]:
        return {
            "inventory_denominator": len(self.inventory),
            "reconstructed_nodes": len(self.visibility),
            "status": "COMPLETE_SELECTED" if self.inventory else "NOT_APPLICABLE",
            "never_observed_units": "UNKNOWN_OMITTED",
            "unselected_labels": "OMITTED",
        }


def reconstruct(state: State) -> CompletedLabels:
    """Completed t-1 -> t neutral labels, never a forecast made at t-1."""
    if type(state) is not State or state.decision_index == 0:
        raise ValueError("completed exact state required; initialization has no transition")
    visibility, correspondence, changes = [], [], []
    area = state.shape[0] * state.shape[1]
    for n in state.nodes:
        if n.previous_mask is None:
            raise ValueError("completed previous support required")
        before, after = (
            int.from_bytes(n.previous_mask, "big"),
            int.from_bytes(n.current_mask, "big"),
        )
        b, a, overlap = before.bit_count(), after.bit_count(), (before & after).bit_count()
        gained, lost = (after & ~before).bit_count(), (before & ~after).bit_count()
        visibility.append(
            VisibilityState(
                surface_id=n.token,
                before_visible_pixels=b,
                after_visible_pixels=a,
                before_projected_image_fraction=b / area,
                after_projected_image_fraction=a / area,
            )
        )
        correspondence.append(
            RegionCorrespondence(
                surface_id=n.token,
                before_visible_pixels=b,
                after_visible_pixels=a,
                same_image_coordinate_overlap_pixels=overlap,
            )
        )
        for kind in classify_mask_changes(b, a, gained, lost):
            count = (
                gained
                if kind in (MaskChangeKind.REGION_APPEARED, MaskChangeKind.GAINED_IMAGE_PIXELS)
                else (
                    lost
                    if kind in (MaskChangeKind.REGION_DISAPPEARED, MaskChangeKind.LOST_IMAGE_PIXELS)
                    else 0
                )
            )
            changes.append(
                RegionMaskChange(surface_id=n.token, change=kind, affected_image_pixels=count)
            )
    return CompletedLabels(
        state.decision_index,
        state.inventory,
        tuple(visibility),
        tuple(correspondence),
        tuple(changes),
    )
