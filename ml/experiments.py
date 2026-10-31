"""Controlled architecture comparison on the prepared Quick, Draw! splits.

    python ml/experiments.py                                    # all candidates, default protocol
    python ml/experiments.py --runs baseline_cnn compact_bn_cnn compact_bn_cnn:noaug compact_bn_cnn:stroke
    python ml/experiments.py --smoke                            # 2 epochs on 5% of the data (pipeline check)

Protocol (identical for every run, so differences come from the architecture):
* the same train / validation split, seed and batch size;
* AdamW (lr 1e-3, weight decay 1e-4), ReduceLROnPlateau on validation loss;
* early stopping on validation accuracy with best-weight restore and checkpointing;
* the same training-only augmentation (translation, rotation, zoom), unless the run
  is an explicit ``:noaug`` ablation; ``:stroke`` adds random stroke-width change
  (motivated by reports/robustness.json).

Model selection uses VALIDATION metrics only. The test split is never read here;
`ml/promote.py` scores the selected candidate on test exactly once.

Each run writes to experiments/<run>/: model.keras, history.csv, exported NumPy
weights, and summary.json. experiments/comparison.json and comparison.md
collect every run (including the existing checkpoint, evaluated on validation).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from metrics import (  # noqa: E402
    apply_temperature, calibration_bins, confusion_matrix, fit_temperature,
    negative_log_likelihood, per_class_scores, top_confusions, top_k_accuracy,
)
from train import DEFAULT_DATA_DIR, load_split  # noqa: E402

EXPERIMENT_DIR = ROOT / "experiments"
EXISTING_MODEL = ROOT / "backend" / "model" / "doodle_classifier.keras"


def validation_report(probabilities: np.ndarray, labels: np.ndarray, names: list[str]) -> dict:
    predicted = probabilities.argmax(axis=1)
    confusion = confusion_matrix(labels, predicted, len(names))
    per_class = per_class_scores(confusion, names)
    temperature = fit_temperature(probabilities, labels)
    ece_raw, _ = calibration_bins(probabilities, labels)
    ece_scaled, _ = calibration_bins(apply_temperature(probabilities, temperature), labels)
    return {
        "split": "validation",
        "examples": int(len(labels)),
        "top_1_accuracy": float((predicted == labels).mean()),
        "top_3_accuracy": top_k_accuracy(probabilities, labels, 3),
        "macro_f1": float(np.mean([row["f1"] for row in per_class])),
        "nll": negative_log_likelihood(probabilities, labels),
        "ece_raw": ece_raw,
        "temperature_fit_on_validation": temperature,
        "ece_temperature_scaled": ece_scaled,
        "worst_classes": sorted(({"class": r["class"], "f1": round(r["f1"], 4)} for r in per_class), key=lambda r: r["f1"])[:6],
        "per_class_f1": {r["class"]: round(r["f1"], 4) for r in per_class},
        "top_confusions": top_confusions(confusion, names, limit=6),
    }


def keras_latency_ms(model, sample: np.ndarray, repeats: int = 100) -> float:
    model(sample[:1], training=False)  # warm-up / trace
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        model(sample[:1], training=False)
        times.append((time.perf_counter() - start) * 1000)
    return float(np.median(times))


def numpy_latency_ms(prefix: Path, sample: np.ndarray, repeats: int = 100) -> float | None:
    try:
        from backend.ml.engine import NumpyCNN

        engine = NumpyCNN.load(Path(str(prefix) + ".weights.npz"), Path(str(prefix) + ".arch.json"))
        engine.run(sample[:1])
        times = []
        for _ in range(repeats):
            start = time.perf_counter()
            engine.run(sample[:1])
            times.append((time.perf_counter() - start) * 1000)
        return float(np.median(times))
    except Exception as exc:  # report, don't hide
        print(f"  NumPy engine latency unavailable: {exc}")
        return None


def numpy_parity(prefix: Path, model, sample: np.ndarray) -> float | None:
    """Max |p_keras - p_numpy| on a validation sample: proves the exported engine matches."""
    try:
        from backend.ml.engine import NumpyCNN

        engine = NumpyCNN.load(Path(str(prefix) + ".weights.npz"), Path(str(prefix) + ".arch.json"))
        return float(np.abs(engine.predict(sample) - model.predict(sample, verbose=0)).max())
    except Exception as exc:
        print(f"  NumPy parity check unavailable: {exc}")
        return None


def run_one(tf, spec: str, args, data, names) -> dict:
    from architectures import CANDIDATES
    from export_weights import export

    arch_name, _, flag = spec.partition(":")
    if arch_name not in CANDIDATES:
        raise ValueError(f"Unknown candidate {arch_name}; choose from {sorted(CANDIDATES)}")
    if flag not in ("", "noaug", "stroke"):
        raise ValueError(f"Unknown run flag :{flag} (use :noaug or :stroke)")
    augment = flag != "noaug"
    stroke = flag == "stroke"
    run_dir = EXPERIMENT_DIR / spec.replace(":", "_")
    run_dir.mkdir(parents=True, exist_ok=True)
    (x_train, y_train), (x_val, y_val) = data

    tf.keras.utils.set_random_seed(args.seed)
    model = CANDIDATES[arch_name](tf, len(names), augment=augment, l2=0.0, stroke=stroke)
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=args.learning_rate, weight_decay=args.weight_decay),
        loss="sparse_categorical_crossentropy",
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    model_path = run_dir / "model.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(model_path, monitor="val_accuracy", mode="max", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=args.patience, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5),
        tf.keras.callbacks.CSVLogger(str(run_dir / "history.csv")),
    ]
    print(f"\n=== {spec}: {model.count_params():,} parameters, augmentation={'on' if augment else 'off'}{' + stroke width' if stroke else ''} ===", flush=True)
    start = time.time()
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=args.epochs,
                        batch_size=args.batch_size, shuffle=True, callbacks=callbacks, verbose=2)
    train_seconds = time.time() - start
    model.save(model_path)

    probabilities = model.predict(x_val, batch_size=1024, verbose=0)
    report = validation_report(probabilities, y_val, names)
    export(model_path, run_dir / "model")
    summary = {
        "run": spec,
        "architecture": arch_name,
        "augmentation": augment,
        "stroke_width_augmentation": stroke,
        "parameters": int(model.count_params()),
        "model_bytes": model_path.stat().st_size,
        "epochs_completed": len(history.epoch),
        "best_epoch": int(np.argmax(history.history["val_accuracy"]) + 1),
        "train_seconds": round(train_seconds, 1),
        "keras_latency_ms": round(keras_latency_ms(model, x_val), 3),
        "numpy_latency_ms": numpy_latency_ms(run_dir / "model", x_val),
        "numpy_parity_max_abs_diff": numpy_parity(run_dir / "model", model, x_val[:512]),
        "protocol": {"optimizer": "AdamW", "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
                     "batch_size": args.batch_size, "max_epochs": args.epochs, "patience": args.patience,
                     "seed": args.seed, "train_examples": int(len(y_train))},
        "validation": report,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"  val top-1 {report['top_1_accuracy']:.4f}  macro-F1 {report['macro_f1']:.4f}  NLL {report['nll']:.4f}"
          f"  ECE {report['ece_raw']:.4f}  {summary['keras_latency_ms']} ms", flush=True)
    return summary


def existing_checkpoint_summary(tf, x_val, y_val, names) -> dict | None:
    if not EXISTING_MODEL.exists():
        return None
    model = tf.keras.models.load_model(EXISTING_MODEL, compile=False)
    probabilities = model.predict(x_val, batch_size=1024, verbose=0)
    return {
        "run": "existing_checkpoint",
        "architecture": "baseline_cnn (original training run: Adam, 30 epochs)",
        "augmentation": True,
        "parameters": int(model.count_params()),
        "model_bytes": EXISTING_MODEL.stat().st_size,
        "keras_latency_ms": round(keras_latency_ms(model, x_val), 3),
        "validation": validation_report(probabilities, y_val, names),
    }


def write_comparison(rows: list[dict]) -> None:
    EXPERIMENT_DIR.mkdir(exist_ok=True)
    (EXPERIMENT_DIR / "comparison.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Architecture comparison (validation split, 32,000 drawings)",
        "",
        "Model selection uses these validation numbers only. The selected model is scored on the test split once by `ml/promote.py`.",
        "",
        "| Run | Params | Val top-1 | Val top-3 | Macro F1 | NLL | ECE raw | T | ECE (T) | Keras ms | NumPy ms | Epochs |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        v = r["validation"]
        numpy_ms = r.get("numpy_latency_ms")
        lines.append(
            f"| {r['run']} | {r['parameters']:,} | {v['top_1_accuracy']:.4f} | {v['top_3_accuracy']:.4f} | "
            f"{v['macro_f1']:.4f} | {v['nll']:.4f} | {v['ece_raw']:.4f} | {v['temperature_fit_on_validation']} | "
            f"{v['ece_temperature_scaled']:.4f} | {r.get('keras_latency_ms', '-')} | "
            f"{'-' if numpy_ms is None else round(numpy_ms, 2)} | {r.get('epochs_completed', '-')} |"
        )
    (EXPERIMENT_DIR / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    from architectures import CANDIDATES

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", default=list(CANDIDATES))
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true", help="2 epochs on 5%% of the training data.")
    args = parser.parse_args()

    metadata = json.loads((args.data_dir / "metadata.json").read_text(encoding="utf-8"))
    names = metadata["class_names"]
    x_train, y_train = load_split(args.data_dir / "train.npz", len(names))
    x_val, y_val = load_split(args.data_dir / "validation.npz", len(names))
    if args.smoke:
        keep = np.random.default_rng(args.seed).permutation(len(y_train))[: len(y_train) // 20]
        x_train, y_train = x_train[keep], y_train[keep]
        args.epochs = 2
    print(f"train {len(y_train):,}  validation {len(y_val):,}  classes {len(names)}")
    print("per-class train counts:", np.bincount(y_train, minlength=len(names)).tolist())

    random.seed(args.seed)
    np.random.seed(args.seed)
    import tensorflow as tf

    rows = []
    comparison_path = EXPERIMENT_DIR / "comparison.json"
    if comparison_path.exists():  # keep earlier runs so experiments can be resumed one at a time
        rows = [r for r in json.loads(comparison_path.read_text()) if r["run"] not in args.runs]
    if not any(r["run"] == "existing_checkpoint" for r in rows):
        existing = existing_checkpoint_summary(tf, x_val, y_val, names)
        if existing:
            rows.insert(0, existing)
            write_comparison(rows)
    for spec in args.runs:
        rows.append(run_one(tf, spec, args, ((x_train, y_train), (x_val, y_val)), names))
        write_comparison(rows)  # after every run, so partial results survive interruption
    print(f"\nWrote {EXPERIMENT_DIR / 'comparison.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
