import numpy as np

from epsbench.annotations import classify_visibility_event, derive_visibility
from epsbench.schema import SurfaceReference, VisibilityEventKind


def test_visibility_count_classification_is_analytic() -> None:
    assert classify_visibility_event(0, 5) is VisibilityEventKind.APPEARING
    assert classify_visibility_event(5, 0) is VisibilityEventKind.DISAPPEARING
    assert classify_visibility_event(5, 7) is VisibilityEventKind.ACCRETING
    assert classify_visibility_event(7, 5) is VisibilityEventKind.DELETING
    assert classify_visibility_event(5, 5) is VisibilityEventKind.STABLE


def test_shifted_region_records_accretion_and_deletion() -> None:
    surface = SurfaceReference(surface_id="surface-0123456789abcdef", segmentation_label=41)
    before = np.array([[41, 41, 0]], dtype=np.int32)
    after = np.array([[0, 41, 41]], dtype=np.int32)
    _, _, events = derive_visibility(before, after, (surface,))
    assert {(event.event, event.affected_pixels) for event in events} == {
        (VisibilityEventKind.ACCRETING, 1),
        (VisibilityEventKind.DELETING, 1),
    }
