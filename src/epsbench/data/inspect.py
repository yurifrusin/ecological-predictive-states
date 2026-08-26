"""Human-readable composite inspection image generation."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from epsbench.data.loader import DatasetLoader
from epsbench.data.validate import validate_dataset
from epsbench.schema import ModalityPermissionSet


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
    panels = (
        ("RGB before", before_rgb),
        ("RGB after", after_rgb),
        ("Opaque surface labels before", before_segmentation),
        ("Opaque surface labels after", after_segmentation),
        ("Forward transport (valid pixels)", forward_flow),
        ("Backward transport (valid pixels)", backward_flow),
        ("Forward validity/reason codes", forward_reasons),
        ("Backward validity/reason codes", backward_reasons),
    )
    width, height = before_rgb.size
    label_height = 24
    header_height = 58
    canvas = Image.new(
        "RGB",
        (2 * width, header_height + 4 * (height + label_height)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 4), f"Analytic transport: {transport.method}", fill="black")
    draw.text((6, 20), f"Identity: {transport.analytic_transport_sha256[:32]}", fill="black")
    draw.text((6, 36), f"          {transport.analytic_transport_sha256[32:]}", fill="black")
    for panel_index, (label, panel) in enumerate(panels):
        column = panel_index % 2
        row = panel_index // 2
        x = column * width
        y = header_height + row * (height + label_height)
        draw.text((x + 6, y + 5), label, fill="black")
        canvas.paste(panel, (x, y + label_height))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, compress_level=9, optimize=False)
    return output
