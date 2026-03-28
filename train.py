"""
Train apple disease classifier (MobileNetV2 + augmented fine-tuning).
Run from project root:  python train.py
Saves: model.h5 in project root (same folder as this file).

Uses rescale=1/255 for images — match inference (scale pixels to 0–1, not mobilenet preprocess).
"""

from pathlib import Path

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "dataset"
MODEL_OUT = ROOT / "model.h5"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 8
FINE_TUNE_LAST_N = 20
SEED = 42

# Folder names under dataset/ — must match backend CLASS_NAMES (indices 0..3)
CLASS_ORDER = ["Apple Scab", "Black Rot", "Rust", "Healthy"]
NUM_CLASSES = 4


def verify_dataset() -> dict[str, int]:
    """Check dataset path, 4 classes, count images per folder."""
    if not DATASET_DIR.is_dir():
        raise SystemExit(f"Dataset folder not found: {DATASET_DIR}")

    counts: dict[str, int] = {}
    for name in CLASS_ORDER:
        folder = DATASET_DIR / name
        if not folder.is_dir():
            raise SystemExit(f"Missing class folder (need exactly 4): {folder}")
        n = sum(1 for f in folder.iterdir() if f.is_file() and not f.name.startswith("."))
        counts[name] = n

    total = sum(counts.values())
    print(f"Dataset path: {DATASET_DIR.resolve()}")
    print(f"Classes ({NUM_CLASSES}): {CLASS_ORDER}")
    print("Images per class:", counts)
    print("Total images on disk:", total)
    if any(n == 0 for n in counts.values()):
        raise SystemExit("Each class folder must contain at least one image file.")

    return counts


def main() -> None:
    verify_dataset()

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        shear_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    val_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2,
    )

    train_gen = train_datagen.flow_from_directory(
        str(DATASET_DIR),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
        classes=CLASS_ORDER,
        seed=SEED,
    )

    val_gen = val_datagen.flow_from_directory(
        str(DATASET_DIR),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        classes=CLASS_ORDER,
        seed=SEED,
    )

    if len(train_gen.class_indices) != NUM_CLASSES:
        raise SystemExit(f"Expected {NUM_CLASSES} classes, got {len(train_gen.class_indices)}")

    print("Keras class_indices:", train_gen.class_indices)
    print("Images loaded for training:", train_gen.samples)
    print("Images loaded for validation:", val_gen.samples)

    base_model = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = True
    for layer in base_model.layers[:-FINE_TUNE_LAST_N]:
        layer.trainable = False

    model = keras.Sequential(
        [
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.4),
            layers.Dense(NUM_CLASSES, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=1,
        ),
    ]

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    model.save(str(MODEL_OUT))
    print(f"Saved model to {MODEL_OUT}")


if __name__ == "__main__":
    main()
