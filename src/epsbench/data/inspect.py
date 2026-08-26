"""Human-readable composite inspection image generation."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from epsbench.data.loader import DatasetLoader
from epsbench.data.validate import validate_dataset
from epsbench.schema import ModalityPermissionSet


def _depth_image(depth: np.ndarray) -> Image.Image:
    finite = np.isfinite(depth)
    if not np.any(finite):
        scaled = np.zeros(depth.shape, dtype=np.uint8)
    else:
        lower, upper = np.percentile(depth[finite], (2, 98))
        if upper <= lower:
            upper = lower + 1.0
        normalised = np.clip((depth - lower) / (upper - lower), 0.0, 1.0)
        scaled = np.asarray((1.0 - normalised) * 255.0, dtype=np.uint8)
    return Image.fromarray(scaled, mode="L").convert("RGB")


def _segmentation_image(segmentation: np.ndarray) -> Image.Image:
    output = np.zeros((*segmentation.shape, 3), dtype=np.uint8)
    for label in np.unique(segmentation):
        if label == 0:
            continue
        colour = hashlib.sha256(str(int(label)).encode()).digest()[:3]
        output[segmentation == label] = np.frombuffer(colour, dtype=np.uint8)
    return Image.fromarray(output, mode="RGB")


def create_inspection_image(dataset: Path, episode_index: int, output: Path) -> Path:
    root = dataset.resolve()
    resolved_output = output.resolve()
    if resolved_output.is_relative_to(root):
        raise ValueError("inspection output must be outside the immutable dataset directory")
    validate_dataset(root)
    loader = DatasetLoader(root, ModalityPermissionSet.all_modalities())
    before_rgb = Image.fromarray(loader.read_rgb(episode_index, 0), mode="RGB")
    after_rgb = Image.fromarray(loader.read_rgb(episode_index, 1), mode="RGB")
    before_depth = _depth_image(loader.read_depth(episode_index, 0))
    after_depth = _depth_image(loader.read_depth(episode_index, 1))
    before_segmentation = _segmentation_image(loader.read_segmentation(episode_index, 0))
    after_segmentation = _segmentation_image(loader.read_segmentation(episode_index, 1))
    panels = (
        ("RGB before", before_rgb),
        ("RGB after", after_rgb),
        ("Depth before (near=white)", before_depth),
        ("Depth after (near=white)", after_depth),
        ("Opaque surface labels before", before_segmentation),
        ("Opaque surface labels after", after_segmentation),
    )
    width, height = before_rgb.size
    label_height = 24
    canvas = Image.new("RGB", (2 * width, 3 * (height + label_height)), "white")
    draw = ImageDraw.Draw(canvas)
    for panel_index, (label, panel) in enumerate(panels):
        column = panel_index % 2
        row = panel_index // 2
        x = column * width
        y = row * (height + label_height)
        draw.text((x + 6, y + 5), label, fill="black")
        canvas.paste(panel, (x, y + label_height))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, compress_level=9, optimize=False)
    return output
