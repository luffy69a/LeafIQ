"""App constants: paths, classes, thresholds, treatment text."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent / "model.h5"

# Same order as train.py CLASS_ORDER / dataset folders (model output index)
CLASS_NAMES = ["Apple Scab", "Black Rot", "Rust", "Healthy"]
HEALTHY_INDEX = 3
HEALTH_SCORE_THRESHOLD = 0.50

IMG_SIZE = (224, 224)

# Model calibration (softmax temperature)
TEMPERATURE = 1.5

# Below this top-class confidence (%) → uncertain response (model path emphasis)
CONFIDENCE_THRESHOLD_PCT = 70.0

TREATMENTS = {
    "Apple Scab": (
        "Remove infected leaves and fruit; improve air flow (prune). "
        "Apply a fungicide labeled for apple scab per local guidance; avoid overhead watering."
    ),
    "Black Rot": (
        "Prune dead or cankered wood; remove mummified fruit. "
        "Use labeled fungicides in season; improve drainage and spacing so leaves dry faster."
    ),
    "Rust": (
        "Rake and remove fallen leaves that may carry spores. "
        "Apply rust-appropriate fungicide if severe; avoid long wet periods on foliage."
    ),
    "Healthy": (
        "No disease signs detected. Keep monitoring, good spacing, balanced watering, and seasonal pruning."
    ),
}
