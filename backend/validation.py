"""
Heuristic leaf validation (no ML).
Goal: reject smooth round objects (e.g. fruit) while keeping yellow or textured real leaves.
"""

from __future__ import annotations

import numpy as np


def validate_leaf(img_batch: np.ndarray) -> tuple[bool, str]:
    """
    Returns (is_valid_leaf, error_message).
    error_message is empty when valid.
    """
    img = img_batch[0]
    variance = float(img.std())

    # Mean absolute gradient (edges / vein texture)
    dx = np.abs(np.diff(img, axis=1))
    dy = np.abs(np.diff(img, axis=0))
    edge_energy = float((dx.mean() + dy.mean()) / 2.0)

    # Smooth blobs: low variance AND weak edges (typical of fruit/skin, not lamina)
    if variance < 0.045 and edge_energy < 0.028:
        return (
            False,
            "This does not look like an apple leaf (it may be fruit or another smooth object). "
            "Please upload a close-up of a leaf.",
        )

    # Empty or nearly uniform image
    if variance < 0.018:
        return False, "Image has almost no usable detail. Please try a sharper, well-lit leaf photo."

    return True, ""
