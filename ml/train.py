"""Train a compact CNN on the prepared Quick, Draw! bitmap splits."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "data" / "quickdraw" / "processed"
DEFAULT_OUTPUT_DIR = ROOT / "backend" / "model"
AUGMENTATION_PREFIXES = ("augment_", "finetune_")
SKIPPED_ON_RESUME = {"bitmap", "normalize_0_1"}


def load_split(path: Path, expected_classes: int) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Prepared split not found: {path}. Run ml/prepare_data.py first.")
    with np.load(path, allow_pickle=False) as dataset:
        if set(dataset.files) != {"images", "labels"}:
            raise ValueError(f"{path} must contain exactly 'images' and 'labels'.")
        images = dataset["images"]
        labels = dataset["labels"]
    if images.ndim != 3 or images.shape[1:] != (28, 28) or images.dtype != np.uint8:
        raise ValueError(f"{path}: images must be uint8 with shape (N, 28, 28); got {images.shape}/{images.dtype}.")
    if labels.ndim != 1 or len(labels) != len(images) or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(f"{path}: labels must be an integer vector matching the number of images.")
    if len(labels) == 0 or labels.min() < 0 or labels.max() >= expected_classes:
        raise ValueError(f"{path}: labels are empty or outside the configured class range.")
    return images[..., np.newaxis], labels.astype(np.int32, copy=False)


def build_model(num_classes: int, seed: int, initial_model: Path | None = None, learning_rate: float = 0.001, augment_finetuning: bool = False):
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    if initial_model is not None:
        base = tf.keras.models.load_model(initial_model, compile=False)
        if base.output_shape[-1] != num_classes:
            raise ValueError("Initial model output width does not match the configured class count.")
        inputs = base.input
        x = base.get_layer("normalize_0_1")(inputs)
        if augment_finetuning:
            x = tf.keras.layers.RandomTranslation(0.08, 0.08, fill_mode="constant", fill_value=0.0, name="finetune_translation")(x)
            x = tf.keras.layers.RandomRotation(0.045, fill_mode="constant", fill_value=0.0, name="finetune_rotation")(x)
            x = tf.keras.layers.RandomZoom((-0.08, 0.08), (-0.08, 0.08), fill_mode="constant", fill_value=0.0, name="finetune_scale")(x)
        # Reuse only the learned layers. The base model's own augmentation layers
        # are skipped, so "no augmentation" fine-tuning really has none and
        # --augment-finetuning applies augmentation once, not twice.
        for layer in base.layers:
            if (
                isinstance(layer, tf.keras.layers.InputLayer)
                or layer.name in SKIPPED_ON_RESUME
                or layer.name.startswith(AUGMENTATION_PREFIXES)
            ):
                continue
            x = layer(x)
        model = tf.keras.Model(inputs=inputs, outputs=x, name="quickdraw_cnn_finetuned")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
        )
        return model
    inputs = tf.keras.Input(shape=(28, 28, 1), dtype=tf.uint8, name="bitmap")
    x = tf.keras.layers.Rescaling(1.0 / 255.0, name="normalize_0_1")(inputs)
    # Quick Draw examples are clean bitmap strokes; light geometry jitter helps
    # the model handle varied scale, placement, and slant from the game canvas.
    x = tf.keras.layers.RandomTranslation(0.08, 0.08, fill_mode="constant", fill_value=0.0, name="augment_translation")(x)
    x = tf.keras.layers.RandomRotation(0.045, fill_mode="constant", fill_value=0.0, name="augment_rotation")(x)
    x = tf.keras.layers.RandomZoom((-0.08, 0.08), (-0.08, 0.08), fill_mode="constant", fill_value=0.0, name="augment_scale")(x)
    # Block 1: 28x28x1 -> 28x28x32 -> 14x14x32 (edges, stroke fragments)
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu", name="block1_conv")(x)
    x = tf.keras.layers.BatchNormalization(name="block1_bn")(x)
    x = tf.keras.layers.MaxPooling2D(name="block1_pool")(x)
    # Block 2: 14x14x32 -> 14x14x64 -> 7x7x64 (corners, curves, simple parts)
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu", name="block2_conv")(x)
    x = tf.keras.layers.BatchNormalization(name="block2_bn")(x)
    x = tf.keras.layers.MaxPooling2D(name="block2_pool")(x)
    # Block 3: 7x7x64 -> 7x7x96 -> 96 (object parts, averaged over position)
    x = tf.keras.layers.Conv2D(96, 3, padding="same", activation="relu", name="block3_conv")(x)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = tf.keras.layers.Dropout(0.3, name="dropout_features")(x)
    x = tf.keras.layers.Dense(128, activation="relu", name="embedding")(x)
    x = tf.keras.layers.Dropout(0.25, name="dropout_embedding")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="class_probabilities")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="quickdraw_cnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    return model


def augmentation_layer_names(model) -> list[str]:
    """Names of the random-augmentation layers actually present in a model."""
    return [layer.name for layer in model.layers if layer.name.startswith(AUGMENTATION_PREFIXES)]


def write_history(path: Path, history: dict[str, list[float]]) -> None:
    path.write_text(json.dumps({key: [float(v) for v in values] for key, values in history.items()}, indent=2) + "\n", encoding="utf-8")
    csv_path = path.with_suffix(".csv")
    keys = list(history)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", *keys])
        for index in range(max((len(values) for values in history.values()), default=0)):
            writer.writerow([index + 1, *(history[key][index] if index < len(history[key]) else "" for key in keys)])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Directory containing train.npz and validation.npz.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for model and training metadata.")
    parser.add_argument("--epochs", type=int, default=30, help="Maximum epochs; early stopping may stop sooner.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=5, help="Epochs without validation-accuracy improvement before stopping.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--initial-model", type=Path, help="Continue training from a compatible saved model.")
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--augment-finetuning", action="store_true", help="Add light random geometry augmentation when resuming a model.")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1 or args.learning_rate <= 0:
        parser.error("epochs, batch-size, patience, and learning-rate must all be positive.")

    try:
        metadata_path = args.data_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Prepared metadata not found: {metadata_path}. Run ml/prepare_data.py --download first.")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        class_names = metadata.get("class_names")
        if not isinstance(class_names, list) or len(class_names) < 2 or len(set(class_names)) != len(class_names):
            raise ValueError("metadata.json must contain unique class_names for at least two classes.")
        train_images, train_labels = load_split(args.data_dir / "train.npz", len(class_names))
        validation_images, validation_labels = load_split(args.data_dir / "validation.npz", len(class_names))

        random.seed(args.seed)
        np.random.seed(args.seed)
        import tensorflow as tf

        tf.keras.utils.set_random_seed(args.seed)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        model_path = args.output_dir / "doodle_classifier.keras"
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=model_path,
                monitor="val_accuracy",
                mode="max",
                save_best_only=True,
                verbose=1,
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                mode="max",
                patience=args.patience,
                restore_best_weights=True,
                verbose=1,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=max(2, args.patience // 2),
                min_lr=1e-6,
                verbose=1,
            ),
        ]
        model = build_model(len(class_names), args.seed, args.initial_model, args.learning_rate, args.augment_finetuning)
        model.summary()
        history = model.fit(
            train_images,
            train_labels,
            validation_data=(validation_images, validation_labels),
            epochs=args.epochs,
            batch_size=args.batch_size,
            shuffle=True,
            callbacks=callbacks,
            verbose=2,
        )
        # Save restored best weights/model even if early stopping restored an earlier epoch.
        model.save(model_path)
        write_history(args.output_dir / "training_history.json", history.history)
        (args.output_dir / "class_names.json").write_text(json.dumps(class_names, indent=2) + "\n", encoding="utf-8")
        training_record = {
            "architecture": "3-block convolutional neural network with training-only spatial augmentation",
            "parameters": int(model.count_params()),
            "augmentation_layers": augmentation_layer_names(model),
            "input": {"shape": [28, 28, 1], "dtype": "uint8", "normalization": "Rescaling(1/255) inside the saved model"},
            "num_classes": len(class_names),
            "class_names": class_names,
            "seed": args.seed,
            "initial_model": str(args.initial_model) if args.initial_model else None,
            "learning_rate": args.learning_rate,
            "finetune_augmentation": args.augment_finetuning,
            "requested_epochs": args.epochs,
            "completed_epochs": len(history.epoch),
            "best_validation_accuracy": float(max(history.history.get("val_accuracy", [0.0]))),
            "model_file": model_path.name,
            "note": "Held-out test metrics are produced separately by ml/evaluate.py; these validation metrics are not test accuracy.",
        }
        (args.output_dir / "training_info.json").write_text(json.dumps(training_record, indent=2) + "\n", encoding="utf-8")
        print(f"Saved best validation model to {model_path}")
        print(f"Best validation accuracy: {training_record['best_validation_accuracy']:.4f}; this is not held-out test accuracy.")
        print("Next, run: python ml/evaluate.py")
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Training failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
