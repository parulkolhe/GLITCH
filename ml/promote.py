"""Versioned model registry: evaluate a candidate and promote it to the game.

    python ml/promote.py --candidate experiments/compact_bn_cnn --version compact-v2
    python ml/promote.py --register-current --version baseline-v1   # register what's already deployed

What it does (NumPy engine only; TensorFlow is not required):
1. Loads the candidate's exported weights (model.weights.npz + model.arch.json).
2. Fits a calibration temperature on VALIDATION and the uncertainty threshold on
   VALIDATION (ml/confidence.py).
3. Scores the TEST split once and records top-1/top-3/macro-F1/ECE/per-class F1.
4. Copies the files into backend/model/versions/<version>/ and appends to
   backend/model/registry.json. With --activate (and only if the candidate's
   VALIDATION top-1 is not worse than the active model's), it becomes the model
   the API serves. The previous model is kept, so a rollback is one command:
       python ml/promote.py --activate-version baseline-v1

The test split is used for reporting, never for choosing between candidates.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend.ml.engine import NumpyCNN  # noqa: E402
from confidence import choose_uncertain_threshold  # noqa: E402
from metrics import (  # noqa: E402
    apply_temperature, calibration_bins, confusion_matrix, difficulty_tiers, fit_temperature,
    negative_log_likelihood, per_class_scores, top_confusions, top_k_accuracy,
)

MODEL_DIR = ROOT / "backend" / "model"
REGISTRY = MODEL_DIR / "registry.json"
DATA_DIR = ROOT / "data" / "quickdraw" / "processed"


def load_npz(path: Path):
    with np.load(path, allow_pickle=False) as data:
        return data["images"][..., None], data["labels"].astype(np.int64)


def split_metrics(probabilities, labels, names, temperature) -> dict:
    predicted = probabilities.argmax(axis=1)
    confusion = confusion_matrix(labels, predicted, len(names))
    per_class = per_class_scores(confusion, names)
    ece_raw, _ = calibration_bins(probabilities, labels)
    scaled = apply_temperature(probabilities, temperature)
    ece_scaled, bins = calibration_bins(scaled, labels)
    return {
        "examples": int(len(labels)),
        "top_1_accuracy": float((predicted == labels).mean()),
        "top_3_accuracy": top_k_accuracy(probabilities, labels, 3),
        "macro_f1": float(np.mean([r["f1"] for r in per_class])),
        "macro_precision": float(np.mean([r["precision"] for r in per_class])),
        "macro_recall": float(np.mean([r["recall"] for r in per_class])),
        "nll_raw": negative_log_likelihood(probabilities, labels),
        "nll_temperature_scaled": negative_log_likelihood(scaled, labels),
        "ece_raw": ece_raw,
        "ece_temperature_scaled": ece_scaled,
        "reliability_temperature_scaled": bins,
        "per_class": per_class,
        "top_confusions": top_confusions(confusion, names, limit=12),
    }, confusion


def read_registry() -> dict:
    if REGISTRY.exists():
        return json.loads(REGISTRY.read_text(encoding="utf-8"))
    return {"active": None, "versions": []}


def write_registry(registry: dict) -> None:
    REGISTRY.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")


def evaluate_and_register(weights: Path, arch: Path, version: str, note: str, names: list[str]) -> dict:
    engine = NumpyCNN.load(weights, arch)
    if engine.output_classes != len(names):
        raise ValueError("Model output width does not match class_names.json")
    x_val, y_val = load_npz(DATA_DIR / "validation.npz")
    x_test, y_test = load_npz(DATA_DIR / "test.npz")
    start = time.time()
    p_val = engine.predict(x_val, 1024)
    temperature = fit_temperature(p_val, y_val)
    policy = choose_uncertain_threshold(apply_temperature(p_val, temperature), y_val)
    p_test = engine.predict(x_test, 1024)  # scored once, for the report
    test, confusion = split_metrics(p_test, y_test, names, temperature)
    validation, _ = split_metrics(p_val, y_val, names, temperature)
    latency = []
    for i in range(100):
        t0 = time.perf_counter()
        engine.run(x_val[i:i + 1])
        latency.append((time.perf_counter() - t0) * 1000)

    target = MODEL_DIR / "versions" / version
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(weights, target / "model.weights.npz")
    shutil.copy2(arch, target / "model.arch.json")
    (target / "calibration.json").write_text(json.dumps({"temperature": temperature, "fitted_on": "validation"}, indent=2) + "\n")
    (target / "confidence_policy.json").write_text(json.dumps(policy, indent=2) + "\n")
    tiers = difficulty_tiers(test["per_class"], key="f1")
    (target / "class_difficulty.json").write_text(json.dumps({
        "source": "held-out test per-class F1, split into thirds",
        "tiers": tiers, "f1": {r["class"]: round(r["f1"], 4) for r in test["per_class"]},
    }, indent=2) + "\n")
    (target / "test_metrics.json").write_text(json.dumps({"evaluation_split": "held-out test", "temperature": temperature, **test}, indent=2) + "\n")
    (target / "validation_metrics.json").write_text(json.dumps({"evaluation_split": "validation", **validation}, indent=2) + "\n")
    np.savetxt(target / "confusion_matrix.csv", confusion, fmt="%d", delimiter=",", header=",".join(names), comments="")

    entry = {
        "version": version,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": note,
        "architecture": engine.name,
        "parameters": engine.parameter_count,
        "temperature": temperature,
        "uncertain_below": policy["uncertain_below"],
        "validation_top_1": validation["top_1_accuracy"],
        "validation_macro_f1": validation["macro_f1"],
        "test_top_1": test["top_1_accuracy"],
        "test_top_3": test["top_3_accuracy"],
        "test_macro_f1": test["macro_f1"],
        "test_ece_temperature_scaled": test["ece_temperature_scaled"],
        "numpy_latency_ms_median": float(np.median(latency)),
        "evaluation_seconds": round(time.time() - start, 1),
    }
    registry = read_registry()
    registry["versions"] = [v for v in registry["versions"] if v["version"] != version] + [entry]
    write_registry(registry)
    return entry


def activate(version: str, force: bool = False) -> None:
    registry = read_registry()
    entries = {v["version"]: v for v in registry["versions"]}
    if version not in entries:
        raise ValueError(f"Unknown version {version}")
    current = entries.get(registry.get("active"))
    if current and not force and entries[version]["validation_top_1"] < current["validation_top_1"]:
        raise ValueError(
            f"{version} has lower validation top-1 ({entries[version]['validation_top_1']:.4f}) than the active "
            f"{current['version']} ({current['validation_top_1']:.4f}); pass --force to activate anyway.")
    registry["active"] = version
    write_registry(registry)
    print(f"Active model is now {version}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidate", type=Path, help="experiments/<run> directory containing model.weights.npz")
    parser.add_argument("--register-current", action="store_true", help="register backend/model/doodle_classifier.*")
    parser.add_argument("--version", help="version label, e.g. compact-v2")
    parser.add_argument("--note", default="")
    parser.add_argument("--activate", action="store_true")
    parser.add_argument("--activate-version")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        names = json.loads((MODEL_DIR / "class_names.json").read_text(encoding="utf-8"))
        if args.activate_version:
            activate(args.activate_version, args.force)
            return 0
        if not args.version:
            parser.error("--version is required")
        if args.register_current:
            weights, arch = MODEL_DIR / "doodle_classifier.weights.npz", MODEL_DIR / "doodle_classifier.arch.json"
        elif args.candidate:
            weights, arch = args.candidate / "model.weights.npz", args.candidate / "model.arch.json"
        else:
            parser.error("pass --candidate or --register-current")
        entry = evaluate_and_register(weights, arch, args.version, args.note, names)
        print(json.dumps({k: v for k, v in entry.items()}, indent=2))
        if args.activate:
            activate(args.version, args.force)
        return 0
    except (OSError, ValueError, KeyError) as exc:
        print(f"Promotion failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
