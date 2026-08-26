import numpy as np

from epsbench.annotations import classify_mask_changes, derive_visibility
from epsbench.schema import MaskChangeKind, SurfaceReference


def test_mask_change_classification_is_neutral_about_optical_cause() -> None:
    assert classify_mask_changes(0, 5, 5, 0) == (MaskChangeKind.REGION_APPEARED,)
    assert classify_mask_changes(5, 0, 0, 5) == (MaskChangeKind.REGION_DISAPPEARED,)
    assert classify_mask_changes(5, 7, 3, 1) == (
        MaskChangeKind.GAINED_IMAGE_PIXELS,
        MaskChangeKind.LOST_IMAGE_PIXELS,
    )
    assert classify_mask_changes(5, 5, 0, 0) == (MaskChangeKind.MASK_UNCHANGED,)


def test_shifted_region_records_neutral_gain_and_loss_not_accretion_or_deletion() -> None:
    surface = SurfaceReference(surface_id="surface-0123456789abcdef", segmentation_label=41)
    before = np.array([[41, 41, 0]], dtype=np.int32)
    after = np.array([[0, 41, 41]], dtype=np.int32)
    _, correspondence, changes = derive_visibility(before, after, (surface,))
    assert correspondence[0].same_image_coordinate_overlap_pixels == 1
    assert {(change.change, change.affected_image_pixels) for change in changes} == {
        (MaskChangeKind.GAINED_IMAGE_PIXELS, 1),
        (MaskChangeKind.LOST_IMAGE_PIXELS, 1),
    }
    encoded = " ".join(change.model_dump_json() for change in changes)
    assert "accret" not in encoded
    assert "delet" not in encoded


def test_expanding_mask_records_image_pixel_gain_without_causal_visibility_event() -> None:
    surface = SurfaceReference(surface_id="surface-0123456789abcdef", segmentation_label=41)
    before = np.array([[41, 41, 0]], dtype=np.int32)
    after = np.array([[41, 41, 41]], dtype=np.int32)
    _, _, changes = derive_visibility(before, after, (surface,))
    assert len(changes) == 1
    assert changes[0].change is MaskChangeKind.GAINED_IMAGE_PIXELS
    assert changes[0].affected_image_pixels == 1
