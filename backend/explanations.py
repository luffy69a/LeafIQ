"""Human-readable explanations, treatment lookup, and simple severity estimate."""

from __future__ import annotations

import numpy as np

from config import TREATMENTS


def get_explanation(label: str) -> str:
    """Short reason shown to the user (not pixel-level truth, demo-friendly)."""
    text = {
        "Apple Scab": "Dark circular spots detected.",
        "Rust": "Orange/yellow patches detected.",
        "Black Rot": "Dark lesions or browning consistent with decay.",
        "Healthy": "No visible disease patterns.",
    }
    return text.get(label, "Pattern is consistent with the predicted class.")


def get_treatment(label: str) -> str:
    return TREATMENTS.get(
        label,
        "Consult your local agriculture extension for site-specific advice.",
    )


def estimate_severity(label: str, img_array: np.ndarray) -> str:
    """
    Rough Mild / Moderate / Severe from luminance spread and dark area fraction.
    Healthy → no severity.
    """
    if label == "Healthy":
        return "—"

    img = img_array[0]
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    dark_frac = float((lum < 0.4).mean())
    spread = float(lum.std())
    score = dark_frac * 1.8 + spread

    if score < 0.28:
        return "Mild"
    if score < 0.48:
        return "Moderate"
    return "Severe"
