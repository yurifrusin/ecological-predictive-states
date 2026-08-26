"""Deterministic MuJoCo instrumentation."""

from epsbench.sim.corridor import (
    CORRIDOR_SURFACE_NAMES,
    compile_corridor_scene_contract,
    compute_corridor_analytic_transport,
    corridor_generation_seeds,
    render_corridor_transition,
    sample_corridor_geometry,
)
from epsbench.sim.single_occluder import (
    SINGLE_OCCLUDER_SURFACE_NAMES,
    compile_single_occluder_scene_contract,
    compute_single_occluder_analytic_transport,
)
from epsbench.sim.single_occluder import render_transition as render_single_occluder_transition

__all__ = [
    "CORRIDOR_SURFACE_NAMES",
    "SINGLE_OCCLUDER_SURFACE_NAMES",
    "compile_corridor_scene_contract",
    "compile_single_occluder_scene_contract",
    "compute_corridor_analytic_transport",
    "compute_single_occluder_analytic_transport",
    "corridor_generation_seeds",
    "render_corridor_transition",
    "render_single_occluder_transition",
    "sample_corridor_geometry",
]
