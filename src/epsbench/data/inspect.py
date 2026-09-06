"""Human-readable composite inspection image generation."""

from __future__ import annotations

import hashlib
import io
import os
import tempfile
import textwrap
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from epsbench.data.loader import DatasetLoader
from epsbench.data.validate import validate_dataset
from epsbench.schema import ComponentTopologyAnnotation, ModalityPermissionSet


def _segmentation_image(segmentation: np.ndarray) -> Image.Image:
    output = np.zeros((*segmentation.shape, 3), dtype=np.uint8)
    for label in np.unique(segmentation):
        if label == 0:
            continue
        colour = hashlib.sha256(str(int(label)).encode()).digest()[:3]
        output[segmentation == label] = np.frombuffer(colour, dtype=np.uint8)
    return Image.fromarray(output, mode="RGB")


def _flow_image(
    flow_pixels: np.ndarray,
    validity: np.ndarray,
) -> Image.Image:
    valid = validity == 1
    horizontal = flow_pixels[..., 0]
    vertical = flow_pixels[..., 1]
    magnitude = np.hypot(horizontal, vertical)
    if np.any(valid):
        scale = float(np.percentile(magnitude[valid], 98))
    else:
        scale = 1.0
    if scale <= 0.0:
        scale = 1.0
    angle = np.mod(np.arctan2(vertical, horizontal) / (2.0 * np.pi), 1.0)
    hsv = np.zeros((*valid.shape, 3), dtype=np.uint8)
    hsv[..., 0] = np.asarray(angle * 255.0, dtype=np.uint8)
    hsv[..., 1] = np.asarray(np.clip(magnitude / scale, 0.0, 1.0) * 255.0, dtype=np.uint8)
    hsv[..., 2] = np.where(valid, 255, 30).astype(np.uint8)
    return Image.fromarray(hsv, mode="HSV").convert("RGB")


def _reason_image(reasons: np.ndarray) -> Image.Image:
    palette = np.asarray(
        (
            (58, 190, 92),
            (20, 20, 20),
            (245, 190, 55),
            (214, 62, 67),
            (140, 78, 190),
        ),
        dtype=np.uint8,
    )
    return Image.fromarray(palette[reasons], mode="RGB")


def _boundary_overlay(segmentation: Image.Image, boundary: object, frame_index: int) -> Image.Image:
    from epsbench.schema import AvailableOrientedBoundaryOwnership

    if not isinstance(boundary, AvailableOrientedBoundaryOwnership):
        raise TypeError("inspection requires available oriented boundary ownership")
    output = segmentation.copy()
    draw = ImageDraw.Draw(output)
    colours = {
        "occluding_contour": (255, 40, 40),
        "attached_junction": (40, 220, 255),
        "controlled_silhouette": (255, 210, 35),
        "multi_surface_junction_ambiguous": (190, 70, 255),
        "unresolved_boundary": (255, 255, 255),
    }
    for element in boundary.elements:
        if element.frame_index != frame_index:
            continue
        colour = colours[element.kind.value]
        if element.axis.value == "horizontal":
            points = ((element.column, element.row), (element.column + 1, element.row))
            owner_points = {
                "negative_axis_side": (element.column, element.row),
                "positive_axis_side": (element.column + 1, element.row),
            }
        else:
            points = ((element.column, element.row), (element.column, element.row + 1))
            owner_points = {
                "negative_axis_side": (element.column, element.row),
                "positive_axis_side": (element.column, element.row + 1),
            }
        draw.line(points, fill=colour, width=1)
        owner_point = owner_points.get(element.owner_side.value)
        if owner_point is not None:
            draw.point(owner_point, fill=(0, 0, 0))
    return output


def _event_image(codes: np.ndarray) -> Image.Image:
    palette = np.asarray(
        (
            (58, 190, 92),
            (220, 55, 55),
            (245, 190, 55),
            (140, 78, 190),
            (20, 20, 20),
            (255, 255, 255),
        ),
        dtype=np.uint8,
    )
    return Image.fromarray(palette[codes], mode="RGB")


def _write_png_atomically_no_clobber(image: Image.Image, output: Path) -> None:
    """Publish a complete PNG atomically without replacing any directory entry."""

    output.parent.mkdir(parents=True, exist_ok=True)
    if not output.parent.is_dir():
        raise ValueError("inspection output parent must be a directory")
    if os.path.lexists(output):
        raise FileExistsError(f"inspection output already exists: {output}")

    encoded = io.BytesIO()
    image.save(encoded, format="PNG", compress_level=9, optimize=False)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as temporary:
            temporary.write(encoded.getbuffer())
            temporary.flush()
            os.fsync(temporary.fileno())
        try:
            os.link(temporary_path, output)
        except FileExistsError as error:
            raise FileExistsError(f"inspection output already exists: {output}") from error
    finally:
        temporary_path.unlink(missing_ok=True)


def create_inspection_image(dataset: Path, episode_index: int, output: Path) -> Path:
    root = dataset.resolve()
    resolved_output = output.resolve()
    if resolved_output.is_relative_to(root):
        raise ValueError("inspection output must be outside the immutable dataset directory")
    validate_dataset(root)
    loader = DatasetLoader(root, ModalityPermissionSet.all_modalities())
    before_rgb = Image.fromarray(loader.read_rgb(episode_index, 0), mode="RGB")
    after_rgb = Image.fromarray(loader.read_rgb(episode_index, 1), mode="RGB")
    before_segmentation = _segmentation_image(loader.read_segmentation(episode_index, 0))
    after_segmentation = _segmentation_image(loader.read_segmentation(episode_index, 1))
    transport = loader.read_analytic_optical_transport(episode_index)
    forward_flow = _flow_image(
        transport.forward_flow_pixels,
        transport.forward_validity,
    )
    backward_flow = _flow_image(
        transport.backward_flow_pixels,
        transport.backward_validity,
    )
    forward_reasons = _reason_image(transport.forward_reasons)
    backward_reasons = _reason_image(transport.backward_reasons)
    boundary = loader.read_oriented_boundaries(episode_index)
    visibility_events = loader.read_ecological_visibility_events(episode_index)
    before_boundary = _boundary_overlay(before_segmentation, boundary, 0)
    after_boundary = _boundary_overlay(after_segmentation, boundary, 1)
    before_events = _event_image(visibility_events.before_fate_codes)
    after_events = _event_image(visibility_events.after_origin_codes)
    ecological = loader.read_ecological_transition(episode_index)
    panels: tuple[tuple[str, Image.Image], ...] = (
        ("RGB - before", before_rgb),
        ("RGB - after", after_rgb),
        ("Opaque segmentation - before", before_segmentation),
        ("Opaque segmentation - after", after_segmentation),
        ("Analytic transport - forward", forward_flow),
        ("Analytic transport - backward", backward_flow),
        ("Transport reasons - forward", forward_reasons),
        ("Transport reasons - backward", backward_reasons),
        ("Oriented boundaries - before", before_boundary),
        ("Oriented boundaries - after", after_boundary),
        ("Event fate - before frame", before_events),
        ("Event origin - after frame", after_events),
    )
    topology = visibility_events.annotation.capabilities.component_topology
    topology_lines: list[str] = []
    if isinstance(topology, ComponentTopologyAnnotation):
        loaded = loader.read_component_topology(episode_index)
        panels += (
            (
                "Four-neighbour components - before",
                _segmentation_image(loaded.before_component_labels),
            ),
            (
                "Four-neighbour components - after",
                _segmentation_image(loaded.after_component_labels),
            ),
        )
        topology_lines = [
            f"Component capability: {topology.status}; four-neighbour; background excluded",
            f"Component topology SHA-256: {topology.component_topology_sha256}",
            f"Portable graph SHA-256: {topology.portable_graph_sha256}",
            f"Events: {dict(sorted(Counter(e.kind.value for e in topology.events).items()))}",
        ]
        aliases = {
            c.component_id: f"F{f.frame_index}:L{c.map_label}"
            for f in topology.frames
            for c in f.components
        }
        for frame in topology.frames:
            for component in frame.components:
                topology_lines.extend(
                    [
                        f"{aliases[component.component_id]} surface {component.surface_id}; "
                        f"pixels {component.pixel_count}; "
                        f"min(row,col) {component.minimum_row_column}; "
                        f"bounds {component.bounds_top_left_bottom_right_exclusive}",
                        f"  {component.component_id}",
                    ]
                )
        topology_lines.append(
            f"Support pairs {len(topology.supports)}; zero support "
            f"{sum(not s.edge for s in topology.supports)}; one direction only "
            f"{sum((s.forward_count == 0) != (s.backward_count == 0) for s in topology.supports)}. "
            "All pairs retained in source; supported edges below."
        )
        topology_lines.extend(
            f"{aliases[s.before_component_id]} -> {aliases[s.after_component_id]} "
            f"forward={s.forward_count} backward={s.backward_count} edge={s.edge}"
            for s in topology.supports
            if s.edge
        )
        topology_lines.extend(
            f"{e.kind.value}: {[aliases[i] for i in e.before_component_ids]} -> "
            f"{[aliases[i] for i in e.after_component_ids]}"
            for e in topology.events
        )
    width, height = before_rgb.size
    panel_width = max(width, 360 if topology_lines else 240)
    panel_height = round(height * panel_width / width)
    label_height = 24
    header_height = 224
    detail_lines = [line for text in topology_lines for line in textwrap.wrap(text, width=110)]
    panel_rows = len(panels) // 2
    details_y = header_height + panel_rows * (panel_height + label_height)
    canvas = Image.new(
        "RGB",
        (2 * panel_width, details_y + (len(detail_lines) * 18 + 12 if detail_lines else 0)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 4), f"Analytic transport: {transport.method}", fill="black")
    draw.text((6, 20), f"Identity: {transport.analytic_transport_sha256[:32]}", fill="black")
    draw.text((6, 36), f"          {transport.analytic_transport_sha256[32:]}", fill="black")
    draw.text(
        (6, 52),
        f"Boundary: {boundary.oriented_boundary_sha256[:32]}",
        fill="black",
    )
    draw.text(
        (6, 68),
        f"          {boundary.oriented_boundary_sha256[32:]}",
        fill="black",
    )
    draw.text(
        (6, 84),
        f"Events:   {visibility_events.annotation.visibility_event_sha256[:32]}",
        fill="black",
    )
    draw.text(
        (6, 100),
        f"          {visibility_events.annotation.visibility_event_sha256[32:]}",
        fill="black",
    )
    draw.text((6, 116), "Boundary: red occluding; cyan attached; yellow silhouette", fill="black")
    draw.text((6, 132), "purple ambiguous; black dot marks declared owner side", fill="black")
    summaries = visibility_events.annotation.occluding_event_summaries
    draw.text(
        (6, 148),
        "Events: green stable; red causal; yellow frame; purple ambiguous; black none",
        fill="black",
    )
    summary_lines = [
        f"{item.kind} affected {item.affected_surface_id[8:16]} "
        f"owner {item.owner_surface_id[8:16]} pixels {item.pixel_count}"
        for item in summaries
    ] or ["none"]
    draw.text((6, 164), f"Accretion/deletion: {summary_lines[0]}", fill="black")
    if len(summary_lines) > 1:
        draw.text((6, 180), f"                    {summary_lines[1]}", fill="black")
    draw.text(
        (6, 196),
        f"Occlusion: {ecological.occlusion.status}; component topology: {topology.status}",
        fill="black",
    )
    for panel_index, (label, panel) in enumerate(panels):
        column = panel_index % 2
        row = panel_index // 2
        x = column * panel_width
        y = header_height + row * (panel_height + label_height)
        draw.text((x + 6, y + 5), label, fill="black")
        resized = panel.resize((panel_width, panel_height), resample=Image.Resampling.NEAREST)
        canvas.paste(resized, (x, y + label_height))
        if isinstance(topology, ComponentTopologyAnnotation) and panel_index >= 12:
            for component in topology.frames[panel_index - 12].components:
                r, c = component.minimum_row_column
                draw.text(
                    (x + c * panel_width // width, y + label_height + r * panel_height // height),
                    str(component.map_label),
                    fill="white",
                    stroke_width=1,
                    stroke_fill="black",
                )
    for index, line in enumerate(detail_lines):
        draw.text((6, details_y + index * 18), line, fill="black")
    _write_png_atomically_no_clobber(canvas, output)
    return output
