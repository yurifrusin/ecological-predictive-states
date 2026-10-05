"""Lazy, bounded six-member native sequence producer; no CLI or launch route.

Importing this module and building XML are native-free. Actual production remains
held for a later independently reviewed source/image-bound execution controller.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any, Protocol
from xml.etree.ElementTree import Element, SubElement, tostring

import numpy as np

from epsbench.diagnostics.causal_history_core import Array, CompletedFlow, OpticalFrame
from epsbench.diagnostics.causal_history_sequence import (
    HEIGHT,
    WIDTH,
    FlowEvidence,
    FrameEvidence,
    RetainedFlow,
    RetainedFrame,
    SequenceEvidence,
    candidate,
    canonical,
    commands,
    digest,
)
from epsbench.schema import (
    BoundaryAxis,
    BoundaryKind,
    BoundaryOwnerSide,
    OrientedBoundaryElement,
)
from epsbench.utils.seeding import rng_for

SURFACES = ("support_surface", "occluding_surface", "background_surface")
ATTACHMENTS = ((SURFACES[0], SURFACES[1]), (SURFACES[0], SURFACES[2]))
REMAPPING_NAMESPACE = "causal-history-sequence-v1:opaque-surface-remapping"
RUNTIME = {
    "meanmass": 1,
    "meaninertia": 1,
    "meansize": 1,
    "extent": 8,
    "center": (0, 0, 0),
    "znear": 0.01,
    "zfar": 20,
    "offsamples": 0,
}


def _numbers(values: Any) -> str:
    return " ".join(str(v) for v in values)


def build_sequence_xml(payload: bytes, member: str) -> str:
    config, pair, ordinal = candidate(payload, member)
    cam, app = config["camera"], config["appearance"]
    root = Element("mujoco", model="epsbench_causal_history_sequence_v1")
    SubElement(root, "compiler", angle="radian")
    SubElement(root, "option", gravity="0 0 -9.81", timestep="0.01")
    SubElement(
        root,
        "statistic",
        attrib={
            k: _numbers(v) if isinstance(v, tuple) else str(v)
            for k, v in RUNTIME.items()
            if k in ("meanmass", "meaninertia", "meansize", "extent", "center")
        },
    )
    visual = SubElement(root, "visual")
    SubElement(visual, "global", offwidth=str(cam["width"]), offheight=str(cam["height_pixels"]))
    SubElement(visual, "quality", shadowsize="0", offsamples="0")
    SubElement(visual, "map", znear="0.01", zfar="20", fogstart="100", fogend="101", haze="0")
    SubElement(visual, "rgba", haze="0 0 0 0")
    SubElement(visual, "headlight", active="0", ambient="0 0 0", diffuse="0 0 0", specular="0 0 0")
    asset = SubElement(root, "asset")
    SubElement(asset, "material", name="flat", specular="0", shininess="0")
    body = SubElement(root, "worldbody")
    if app["shadows"] is not False or app["textures"] is not False:
        raise ValueError("fixed disabled shadows/textures required")
    SubElement(
        body,
        "light",
        name="key",
        directional="true",
        castshadow="false",
        pos="-1 -2 5",
        dir=_numbers(app["light_direction"]),
        ambient=_numbers(app["ambient"]),
        diffuse=_numbers(app["diffuse"]),
        specular=_numbers(app["specular"]),
    )
    SubElement(
        body,
        "geom",
        name=SURFACES[0],
        type="plane",
        pos=_numbers((0, 0, config["support"]["z"])),
        size=_numbers((*config["support"]["half_extent"], 0.1)),
        rgba=_numbers(app["rgba_support"]),
        material="flat",
    )
    lower = config["occluder"]["lower"]
    upper = config["occluder"]["upper"]
    bgx = pair["background_x"][ordinal]
    for name, lo, hi, color in (
        (SURFACES[1], lower, upper, "rgba_occluder"),
        (
            SURFACES[2],
            (bgx[0], *config["background_yz"]["lower"]),
            (bgx[1], *config["background_yz"]["upper"]),
            "rgba_background",
        ),
    ):
        SubElement(
            body,
            "geom",
            name=name,
            type="box",
            pos=_numbers([(a + b) / 2 for a, b in zip(lo, hi, strict=True)]),
            size=_numbers([(b - a) / 2 for a, b in zip(lo, hi, strict=True)]),
            rgba=_numbers(app[color]),
            material="flat",
        )
    SubElement(
        body,
        "camera",
        name="monocular_camera",
        pos=_numbers((pair["poses"][0], cam["forward"], cam["height"])),
        xyaxes=_numbers((1, 0, 0, 0, cam["up_y"], 1)),
        fovy=str(cam["fovy_degrees"]),
    )
    return tostring(root, encoding="unicode")


def opaque_mapping(seed: int, raw_ids: tuple[int, ...]) -> tuple[tuple[int, int, str], ...]:
    if (
        len(raw_ids) != 3
        or len(set(raw_ids)) != 3
        or any(type(v) is not int or v < 0 for v in raw_ids)
    ):
        raise ValueError("complete three-surface raw inventory required")
    rng = rng_for(seed, REMAPPING_NAMESPACE)
    labels = rng.permutation(np.arange(1, 4)).tolist()
    tokens = [f"surface-{int(v):016x}" for v in rng.integers(0, 2**64, 3, dtype=np.uint64)]
    if len(set(tokens)) != 3:
        raise ValueError("opaque token collision")
    return tuple(
        (raw, int(label), token) for raw, label, token in zip(raw_ids, labels, tokens, strict=True)
    )


@dataclass(frozen=True)
class Capture:
    rgb: Array
    raw: Array
    camera: Any
    evidence: FrameEvidence
    arrays: tuple[Array, ...]
    boundaries: tuple[dict[str, Any], ...]
    operational: dict[str, Any]


class Backend(Protocol):
    @property
    def raw_ids(self) -> tuple[int, ...]: ...

    @property
    def compiled(self) -> dict[str, Any]: ...

    def capture(self, position: float) -> Capture: ...
    def transport(self, before: Capture, after: Capture) -> tuple[Any, Array]: ...
    def close(self) -> None: ...


@dataclass(frozen=True)
class ProducedSequence:
    sequence: SequenceEvidence
    frames: tuple[RetainedFrame, ...]
    flows: tuple[RetainedFlow, ...]
    operational: tuple[dict[str, Any], ...]


def admit_flow(
    validity: Array,
    reasons: Array,
    analytic_source: Array,
    actual_source: Array,
    actual_target: Array,
    projected_samples: Array,
) -> tuple[FlowEvidence, Array, Array]:
    """Original validity is unchanged; mismatches deny ecological flow admission."""
    shape = (HEIGHT, WIDTH)
    if (
        any(
            a.shape != shape or a.dtype != np.int32
            for a in (analytic_source, actual_source, actual_target)
        )
        or projected_samples.shape != (*shape, 2)
        or projected_samples.dtype != np.int32
        or validity.shape != shape
        or validity.dtype != np.uint8
        or reasons.shape != shape
        or reasons.dtype != np.uint8
        or np.any(validity > 1)
        or np.any(reasons > 4)
        or not np.array_equal(validity == 1, reasons == 0)
    ):
        raise ValueError("native/analytic admission array contract differs")
    valid = validity == 1
    source_mismatch = valid & ((analytic_source < 0) | (analytic_source != actual_source))
    rows, columns = projected_samples[..., 1], projected_samples[..., 0]
    inside = (rows >= 0) & (rows < HEIGHT) & (columns >= 0) & (columns < WIDTH)
    sampled = actual_target[np.clip(rows, 0, HEIGHT - 1), np.clip(columns, 0, WIDTH - 1)]
    target_mismatch = valid & (~inside | (sampled != analytic_source))
    source_count, target_count = int(source_mismatch.sum()), int(target_mismatch.sum())
    evidence = FlowEvidence(
        admitted=source_count == target_count == 0,
        source_mismatch_count=source_count,
        target_mismatch_count=target_count,
        reason_counts=tuple(int(np.count_nonzero(reasons == i)) for i in range(5)),
    )
    return evidence, source_mismatch, target_mismatch


def produce_sequence(
    payload: bytes, member: str, *, factory: Callable[[str], Backend]
) -> ProducedSequence:
    """Explicit backend injection; no default native route is enabled here."""
    config, pair, ordinal = candidate(payload, member)
    xml = build_sequence_xml(payload, member)
    backend = factory(xml)  # Exactly one compilation/resource owner for one sequence.
    try:
        mapping = opaque_mapping(pair["seeds"][ordinal], backend.raw_ids)
        by_raw = {raw: (label, token) for raw, label, token in mapping}
        captures = []
        frames = []
        for i, position in enumerate(pair["poses"]):
            captured = backend.capture(
                position
            )  # One RGB + one paired draw, never a counterfactual draw.
            raw = captured.raw
            if (
                raw.shape != (HEIGHT, WIDTH)
                or raw.dtype != np.int32
                or set(int(v) for v in np.unique(raw)) - {-1} - set(by_raw)
            ):
                raise ValueError("paired raw segmentation/inventory differs")
            seg = np.zeros(raw.shape, dtype=np.int32)
            for raw_id, (label, _) in by_raw.items():
                seg[raw == raw_id] = label
            visible = tuple(
                (label, token) for raw_id, (label, token) in by_raw.items() if np.any(raw == raw_id)
            )

            def token(raw_id: int | None) -> str | None:
                return None if raw_id is None else by_raw[raw_id][1]

            boundaries = tuple(
                OrientedBoundaryElement(
                    frame_index=0,
                    axis=BoundaryAxis(b["axis"]),
                    row=b["row"],
                    column=b["column"],
                    negative_surface_id=token(b["negative_raw_geom_id"]),
                    positive_surface_id=token(b["positive_raw_geom_id"]),
                    kind=BoundaryKind(b["kind"]),
                    owner_side=BoundaryOwnerSide(b["owner_side"]),
                    owner_surface_id=token(b["owner_raw_geom_id"]),
                )
                .model_dump_json()
                .encode()
                for b in captured.boundaries
            )
            frames.append(
                RetainedFrame(
                    OpticalFrame(i, seg, visible, boundaries),
                    captured.rgb,
                    captured.evidence,
                    captured.arrays,
                )
            )
            captures.append(captured)
        flows = []
        for i, command in enumerate(commands(config, member)):
            transport, samples = backend.transport(captures[i], captures[i + 1])
            forward, backward = transport.forward, transport.backward
            evidence, source_mismatch, target_mismatch = admit_flow(
                forward.validity,
                forward.reasons,
                transport.before_surface_assignment,
                captures[i].raw,
                captures[i + 1].raw,
                samples,
            )
            optical = CompletedFlow(
                i, command, forward.vectors_fixed, forward.validity, forward.reasons
            )
            arrays = (
                transport.before_surface_assignment,
                transport.after_surface_assignment,
                transport.before_boundary_ambiguous,
                transport.after_boundary_ambiguous,
                forward.target_hit_assignment,
                backward.vectors_fixed,
                backward.validity,
                backward.reasons,
                backward.target_hit_assignment,
                samples,
                source_mismatch,
                target_mismatch,
            )
            flows.append(RetainedFlow(optical, evidence, arrays))
        return ProducedSequence(
            SequenceEvidence(
                mapping=mapping, compiled=backend.compiled, xml_sha256=digest(xml.encode())
            ),
            tuple(frames),
            tuple(flows),
            tuple(c.operational for c in captures),
        )
    finally:
        backend.close()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        values = [_jsonable(v) for v in value]
        return sorted(values, key=canonical) if isinstance(value, (set, frozenset)) else values
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


class NativeBackend:
    """Lazy source implementation only; invocation is held pending execution review."""

    def __init__(self, xml: str):
        import mujoco

        from epsbench.annotations.boundary_events import verify_attachment_contract
        from epsbench.sim.canonical_paired import CanonicalPairedRenderer
        from epsbench.sim.compiled import extract_compiled_scene_contract

        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.camera_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, "monocular_camera"
        )
        compiled = extract_compiled_scene_contract(
            self.model, self.data, SURFACES, "monocular_camera"
        )
        self.raw_ids = tuple(compiled.raw_geom_ids[n] for n in SURFACES)
        self.attachment = verify_attachment_contract(
            self.model, self.data, compiled.raw_geom_ids, ATTACHMENTS
        )
        self.compiled = _jsonable(asdict(compiled))
        self.compiled["statistics"] = {
            name: float(getattr(self.model.stat, name))
            for name in ("meanmass", "meaninertia", "meansize", "extent")
        }
        self.compiled["statistics"]["center"] = self.model.stat.center.tolist()
        self.compiled["visual"] = {
            "znear": float(self.model.vis.map.znear),
            "zfar": float(self.model.vis.map.zfar),
            "offsamples": int(self.model.vis.quality.offsamples),
        }
        if (
            self.compiled["statistics"]
            != {
                "meanmass": 1.0,
                "meaninertia": 1.0,
                "meansize": 1.0,
                "extent": 8.0,
                "center": [0.0, 0.0, 0.0],
            }
            or self.model.vis.quality.offsamples != 0
        ):
            raise ValueError("compiled fixed statistics differ")
        self.renderer = mujoco.Renderer(self.model, height=HEIGHT, width=WIDTH)
        try:
            self.paired = CanonicalPairedRenderer(self.renderer)
        except BaseException:
            self.renderer.close()
            raise

    def capture(self, position: float) -> Capture:
        from epsbench.annotations.boundary_events import (
            classify_oriented_boundary_lattice,
            projected_attachment_locus_edges,
        )
        from epsbench.annotations.optical_transport import (
            AnalyticCamera,
            counterfactual_surface_assignments,
        )

        self.model.cam_pos[self.camera_id, 0] = position
        self.mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, camera=self.camera_id)
        self.renderer.scene.flags[self.mujoco.mjtRndFlag.mjRND_SHADOW] = False
        self.renderer.scene.flags[self.mujoco.mjtRndFlag.mjRND_FOG] = False
        self.renderer.scene.flags[self.mujoco.mjtRndFlag.mjRND_HAZE] = False
        rgb = np.asarray(self.renderer.render(), dtype=np.uint8).copy()
        # The same scene/camera remains installed; paired owns its one draw/read.
        paired = self.paired.capture()
        expected_near = float(self.model.vis.map.znear * self.model.stat.extent)
        expected_far = float(self.model.vis.map.zfar * self.model.stat.extent)
        if (paired.near, paired.far) != (expected_near, expected_far):
            raise ValueError("paired clipping differs from actual compiled map/extent")
        camera = AnalyticCamera(
            (
                float(self.data.cam_xpos[self.camera_id, 0]),
                float(self.data.cam_xpos[self.camera_id, 1]),
                float(self.data.cam_xpos[self.camera_id, 2]),
            ),
            tuple(float(v) for v in self.data.cam_xmat[self.camera_id].reshape(-1)),
            float(self.model.cam_fovy[self.camera_id]),
        )
        counterfactual = counterfactual_surface_assignments(
            self.model, self.data, self.raw_ids, WIDTH, HEIGHT, camera
        )
        attachment_edges = projected_attachment_locus_edges(
            paired.raw_geom_segmentation, self.attachment, camera, WIDTH, HEIGHT
        )
        boundaries = classify_oriented_boundary_lattice(
            0, paired.raw_geom_segmentation, counterfactual, attachment_edges, strict=False
        )
        evidence = FrameEvidence(
            camera=_jsonable(asdict(camera)),
            compiled=self.compiled,
            paired_stable=dict(paired.stable_state),
            scene_map=tuple(asdict(v) for v in paired.scene_map),
            near=paired.near,
            far=paired.far,
            orientation=paired.orientation,
            raw_boundaries=tuple(asdict(v) for v in boundaries),
            attachment=_jsonable(asdict(self.attachment)),
        )
        return Capture(
            rgb,
            paired.raw_geom_segmentation,
            camera,
            evidence,
            (
                paired.raw_geom_segmentation,
                paired.depth,
                paired.native_id_rgb,
                paired.native_depth_pre_metric,
                *(counterfactual[r] for r in self.raw_ids),
            ),
            evidence.raw_boundaries,
            dict(paired.operational_state),
        )

    def transport(self, before: Capture, after: Capture) -> tuple[Any, Array]:
        from epsbench.annotations.optical_transport import (
            _nearest_controlled_intersections,
            compute_analytic_transport,
            pixel_rays_world,
            project_world_points,
        )

        transport = compute_analytic_transport(
            self.model, self.data, self.raw_ids, WIDTH, HEIGHT, before.camera, after.camera
        )
        rays = pixel_rays_world(WIDTH, HEIGHT, before.camera)
        origin = np.asarray(before.camera.world_position, dtype=np.float64)
        distance, _ = _nearest_controlled_intersections(
            self.model, self.data, self.raw_ids, origin, rays
        )
        points = origin + rays * np.where(np.isfinite(distance), distance, 1.0)[..., None]
        x, y, inside = project_world_points(points, WIDTH, HEIGHT, after.camera)
        # Unquantized projection chooses the actual paired target sample. Never
        # derive this admission check from rounded fixed-point flow.
        samples = np.full((HEIGHT, WIDTH, 2), -1, dtype=np.int32)
        samples[..., 0][inside] = np.floor(x[inside]).astype(np.int32)
        samples[..., 1][inside] = np.floor(y[inside]).astype(np.int32)
        return transport, samples

    def close(self) -> None:
        self.renderer.close()
