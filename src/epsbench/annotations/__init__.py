"""Ecological-oracle annotation derivation."""

from epsbench.annotations.derive import (
    classify_mask_changes,
    derive_boundary_structure,
    derive_visibility,
)
from epsbench.annotations.optical_transport import (
    ANALYTIC_BOUNDARY_RULE,
    ANALYTIC_BOUNDARY_WIDTH_PIXELS,
    ANALYTIC_TRANSPORT_METHOD,
    FLOW_FIXED_POINT_SCALE,
    FLOW_QUANTISATION_ROUNDING,
    AnalyticCamera,
    AnalyticTransportArrays,
    DirectionalTransportArrays,
    TransportReasonCode,
    analytic_boundary_ambiguity,
    compute_analytic_transport,
    focal_scales_from_vertical_fov,
    pixel_rays_world,
    project_world_points,
)

__all__ = [
    "ANALYTIC_BOUNDARY_RULE",
    "ANALYTIC_BOUNDARY_WIDTH_PIXELS",
    "ANALYTIC_TRANSPORT_METHOD",
    "FLOW_FIXED_POINT_SCALE",
    "FLOW_QUANTISATION_ROUNDING",
    "AnalyticCamera",
    "AnalyticTransportArrays",
    "DirectionalTransportArrays",
    "TransportReasonCode",
    "analytic_boundary_ambiguity",
    "classify_mask_changes",
    "compute_analytic_transport",
    "derive_boundary_structure",
    "derive_visibility",
    "focal_scales_from_vertical_fov",
    "pixel_rays_world",
    "project_world_points",
]
