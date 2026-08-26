"""Derive the minimal segmentation-based ecological annotations."""

from collections import Counter

import numpy as np

from epsbench.schema import (
    BoundaryContact,
    BoundaryStructure,
    MaskChangeKind,
    RegionCorrespondence,
    RegionMaskChange,
    SurfaceReference,
    VisibilityState,
)


def classify_mask_changes(
    before_pixels: int,
    after_pixels: int,
    gained_image_pixels: int,
    lost_image_pixels: int,
) -> tuple[MaskChangeKind, ...]:
    """Classify neutral same-coordinate mask changes without assigning an optical cause."""

    if before_pixels == 0 and after_pixels > 0:
        return (MaskChangeKind.REGION_APPEARED,)
    if before_pixels > 0 and after_pixels == 0:
        return (MaskChangeKind.REGION_DISAPPEARED,)
    changes: list[MaskChangeKind] = []
    if gained_image_pixels:
        changes.append(MaskChangeKind.GAINED_IMAGE_PIXELS)
    if lost_image_pixels:
        changes.append(MaskChangeKind.LOST_IMAGE_PIXELS)
    if not changes:
        changes.append(MaskChangeKind.MASK_UNCHANGED)
    return tuple(changes)


def derive_visibility(
    before: np.ndarray,
    after: np.ndarray,
    surfaces: tuple[SurfaceReference, ...],
) -> tuple[
    tuple[VisibilityState, ...],
    tuple[RegionCorrespondence, ...],
    tuple[RegionMaskChange, ...],
]:
    if before.shape != after.shape or before.ndim != 2:
        raise ValueError("before and after segmentation arrays must be aligned 2D images")
    pixel_total = before.size
    states: list[VisibilityState] = []
    correspondence: list[RegionCorrespondence] = []
    mask_changes: list[RegionMaskChange] = []
    for surface in surfaces:
        before_mask = before == surface.segmentation_label
        after_mask = after == surface.segmentation_label
        before_count = int(np.count_nonzero(before_mask))
        after_count = int(np.count_nonzero(after_mask))
        overlap_count = int(np.count_nonzero(before_mask & after_mask))
        states.append(
            VisibilityState(
                surface_id=surface.surface_id,
                before_visible_pixels=before_count,
                after_visible_pixels=after_count,
                before_projected_image_fraction=before_count / pixel_total,
                after_projected_image_fraction=after_count / pixel_total,
            )
        )
        correspondence.append(
            RegionCorrespondence(
                surface_id=surface.surface_id,
                before_visible_pixels=before_count,
                after_visible_pixels=after_count,
                same_image_coordinate_overlap_pixels=overlap_count,
            )
        )
        added_count = int(np.count_nonzero(after_mask & ~before_mask))
        removed_count = int(np.count_nonzero(before_mask & ~after_mask))
        change_kinds = classify_mask_changes(
            before_count,
            after_count,
            added_count,
            removed_count,
        )
        for change_kind in change_kinds:
            if change_kind in {
                MaskChangeKind.GAINED_IMAGE_PIXELS,
                MaskChangeKind.REGION_APPEARED,
            }:
                affected_image_pixels = added_count
            elif change_kind in {
                MaskChangeKind.LOST_IMAGE_PIXELS,
                MaskChangeKind.REGION_DISAPPEARED,
            }:
                affected_image_pixels = removed_count
            else:
                affected_image_pixels = 0
            mask_changes.append(
                RegionMaskChange(
                    surface_id=surface.surface_id,
                    change=change_kind,
                    affected_image_pixels=affected_image_pixels,
                )
            )
    return tuple(states), tuple(correspondence), tuple(mask_changes)


def derive_boundary_structure(
    segmentation: np.ndarray,
    surfaces: tuple[SurfaceReference, ...],
    frame_index: int,
) -> BoundaryStructure:
    """Count four-neighbour label transitions and known-surface contacts."""

    if segmentation.ndim != 2:
        raise ValueError("segmentation must be a 2D image")
    label_to_id = {surface.segmentation_label: surface.surface_id for surface in surfaces}
    contacts: Counter[tuple[str, str]] = Counter()
    total = 0
    for left, right in (
        (segmentation[:, :-1], segmentation[:, 1:]),
        (segmentation[:-1, :], segmentation[1:, :]),
    ):
        changed = left != right
        known_edge = np.isin(left, tuple(label_to_id)) | np.isin(right, tuple(label_to_id))
        edge_mask = changed & known_edge
        total += int(np.count_nonzero(edge_mask))
        first_labels = left[edge_mask]
        second_labels = right[edge_mask]
        for first_label, second_label in zip(
            first_labels.tolist(), second_labels.tolist(), strict=True
        ):
            if first_label not in label_to_id or second_label not in label_to_id:
                continue
            first_id, second_id = sorted((label_to_id[first_label], label_to_id[second_label]))
            contacts[(first_id, second_id)] += 1
    records = tuple(
        BoundaryContact(
            first_surface_id=first,
            second_surface_id=second,
            pixel_count=count,
        )
        for (first, second), count in sorted(contacts.items())
    )
    return BoundaryStructure(
        frame_index=frame_index,  # type: ignore[arg-type]
        total_boundary_pixels=total,
        contacts=records,
    )
