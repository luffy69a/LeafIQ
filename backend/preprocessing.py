"""Image tensor prep (must match training: RGB, 224×224, values in [0, 1])."""

from __future__ import annotations

import numpy as np
from PIL import Image

from config import IMG_SIZE


def preprocess_image(pil_image: Image.Image) -> np.ndarray:
    """Batch shape (1, 224, 224, 3) float32, scaled 0–1."""
    pil_image = pil_image.convert("RGB")
    pil_image = pil_image.resize(IMG_SIZE, Image.Resampling.LANCZOS)
    arr = np.asarray(pil_image, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def preprocess_for_fallback(pil_image: Image.Image) -> np.ndarray:
    """Same scaling as preprocess_image (used when model is unavailable)."""
    return preprocess_image(pil_image)
