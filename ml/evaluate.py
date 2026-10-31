"""Evaluate a trained Quick, Draw! CNN.

Typical use, once per final model:

    python ml/evaluate.py --fit-temperature

That (1) fits a temperature on the VALIDATION split for confidence calibration,
(2) scores the held-out TEST split once, and (3) writes metrics, the confusion
matrix, saved probabilities for plotting, and the per-class difficulty tiers the
game uses.

To compare candidate models, use the validation split so the test set is never
used for model selection:

    python ml/evaluate.py --model-dir data/quickdraw/candidate --split validation
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from metrics import (  # noqa: E402
    apply_temperature,
    calibration_bins,
    confusion_matrix,
    difficulty_tiers,
    fit_temperature,
    negative_log_likelihood,
    per_class_scores,
    top_confusions,
    top_k_accuracy,
)
from train import DEFAULT_DATA_DIR, DEFAULT_OUTPUT_DIR, load_split  # noqa: E402


def predict_split(model, data_dir: Path, split: str, num_classes: int, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    images, labels = load_split(data_dir / f"{split}.npz", num_classes)
    probabilities = model.predict(images, batch_size=batch_size, verbose=1)
    if probabilities.shape != (len(labels), num_classes) or not np.all(np.isfinite(probabilities)):
        raise ValueError("Model returned malformed or non-finite prediction probabilities.")
    if np.any(probabilities < 0) or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-3):
        raise ValueError("Model output is not a valid softmax probability matrix.")
    return probabilities, labels


def summarise(probabilities: np.ndarray, labels: np.ndarray, names: list[str], split: str, temperature: float) -> tuple[dict, np.ndarray]:
    predicted = probabilities.argmax(axis=1)
    confusion = confusion_matrix(labels, predicted, len(names))
    per_class = per_class_scores(confusion, names)
    confidence = probabilities.max(axis=1)
    correct = predicted == labels
    raw_ece, raw_bins = calibration_bins(probabilities, labels)
    calibrated = apply_temperature(probabilities, temperature)
    calibrated_ece, calibrated_bins = calibration_bins(calibrated, labels)
    metrics = {
        "evaluation_split": "held-out test" if split == "test" else split,
        "examples": int(len(labels)),
        "top_1_accuracy": float(correct.mean()),
        "top_3_accuracy": top_k_accuracy(probabilities, labels, 3),
        "macro_precision": float(np.mean([row["precision"] for row in per_class])),
        "macro_recall": float(np.mean([row["recall"] for row in per_class])),
        "macro_f1": float(np.mean([row["f1"] for row in per_class])),
        "class_names": names,
        "per_class": per_class,
        "top_confusions": top_confusions(confusion, names, limit=12),
        "confidence": {
            "mean_when_correct": float(confidence[correct].mean()) if correct.any() else None,
            "mean_when_wrong": float(confidence[~correct].mean()) if (~correct).any() else None,
        },
        "calibration": {
            "temperature": temperature,
            "ece_raw": raw_ece,
            "ece_temperature_scaled": calibrated_ece,
            "nll_raw": negative_log_likelihood(probabilities, labels),
            "nll_temperature_scaled": negative_log_likelihood(calibrated, labels),
            "reliability_raw": raw_bins,
            "reliability_temperature_scaled": calibrated_bins,
        },
        "confidence_interpretation": "Raw softmax outputs are not calibrated probabilities; see calibration for ECE before and after temperature scaling.",
    }
    return metrics, confusion


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--split", choices=["test", "validation"], default="test",
                        help="Use 'validation' to compare candidate models; 'test' only for the final model.")
    parser.add_argument("--fit-temperature", action="store_true",
                        help="Fit a calibration temperature on the validation split and save calibration.json.")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive.")

    try:
        model_path = args.model_dir / "doodle_classifier.keras"
        names_path = args.model_dir / "class_names.json"
        if not model_path.exists():
            raise FileNotFoundError("Trained model missing. Run ml/train.py first.")
        if names_path.exists():
            names = json.loads(names_path.read_text(encoding="utf-8"))
        else:
            metadata_path = args.data_dir / "metadata.json"
            if not metadata_path.exists():
                raise FileNotFoundError("Class mapping and prepared dataset metadata are both missing.")
            names = json.loads(metadata_path.read_text(encoding="utf-8")).get("class_names")
        if not isinstance(names, list) or len(names) < 2:
            raise ValueError("class_names.json must contain at least two labels.")

        import tensorflow as tf

        model = tf.keras.models.load_model(model_path, compile=False)
        if model.output_shape[-1] != len(names):
            raise ValueError("Model output width does not match class_names.json.")

        calibration_path = args.model_dir / "calibration.json"
        if args.fit_temperature:
            validation_probabilities, validation_labels = predict_split(model, args.data_dir, "validation", len(names), args.batch_size)
            temperature = fit_temperature(validation_probabilities, validation_labels)
            calibration = {
                "temperature": temperature,
                "fitted_on": "validation",
                "validation_nll_raw": negative_log_likelihood(validation_probabilities, validation_labels),
                "validation_nll_scaled": negative_log_likelihood(apply_temperature(validation_probabilities, temperature), validation_labels),
            }
            calibration_path.write_text(json.dumps(calibration, indent=2) + "\n", encoding="utf-8")
            print(f"Fitted temperature on validation: T = {temperature}")
        elif calibration_path.exists():
            temperature = float(json.loads(calibration_path.read_text(encoding="utf-8"))["temperature"])
        else:
            temperature = 1.0

        probabilities, labels = predict_split(model, args.data_dir, args.split, len(names), args.batch_size)
        metrics, confusion = summarise(probabilities, labels, names, args.split, temperature)

        suffix = "" if args.split == "test" else f"_{args.split}"
        args.model_dir.mkdir(parents=True, exist_ok=True)
        (args.model_dir / f"test_metrics{suffix}.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        with (args.model_dir / f"confusion_matrix{suffix}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["actual\\predicted", *names])
            for name, row in zip(names, confusion):
                writer.writerow([name, *row.tolist()])
        np.savez_compressed(args.model_dir / f"predictions{suffix}.npz",
                            probabilities=probabilities.astype(np.float16), labels=labels.astype(np.int16))
        if args.split == "test":
            tiers = difficulty_tiers(metrics["per_class"], key="f1")
            (args.model_dir / "class_difficulty.json").write_text(json.dumps({
                "source": "held-out test per-class F1, split into thirds",
                "tiers": tiers,
                "f1": {row["class"]: round(row["f1"], 4) for row in metrics["per_class"]},
            }, indent=2) + "\n", encoding="utf-8")

        calibration_summary = metrics["calibration"]
        print(f"{metrics['evaluation_split']} examples: {len(labels):,}")
        print(f"Top-1 accuracy: {metrics['top_1_accuracy']:.4f}   Top-3 accuracy: {metrics['top_3_accuracy']:.4f}")
        print(f"Macro precision/recall/F1: {metrics['macro_precision']:.4f}/{metrics['macro_recall']:.4f}/{metrics['macro_f1']:.4f}")
        print(f"ECE raw: {calibration_summary['ece_raw']:.4f}   ECE after T={temperature}: {calibration_summary['ece_temperature_scaled']:.4f}")
        print("Most common confusions:")
        for row in metrics["top_confusions"][:5]:
            print(f"  {row['true']} -> {row['predicted']}: {row['count']}")
        print(f"Saved outputs in {args.model_dir}")
        return 0
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
