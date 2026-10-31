"""Verify that the NumPy inference engine reproduces the trained Keras model.

    python ml/verify_engine.py                       # baseline: exported weights vs recorded results
    python ml/verify_engine.py --keras               # also compare against TensorFlow directly (needs TF)

Checks, on the full held-out test split (32,000 drawings):
* top-1 accuracy equals the accuracy recorded in backend/model/test_metrics.json;
* probabilities match backend/model/predictions.npz (saved by ml/evaluate.py as float16);
* with --keras, max |p_keras - p_numpy| on the test split.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.ml.engine import NumpyCNN  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "backend" / "model")
    parser.add_argument("--keras", action="store_true")
    args = parser.parse_args()
    md = args.model_dir
    engine = NumpyCNN.load(md / "doodle_classifier.weights.npz", md / "doodle_classifier.arch.json")
    with np.load(ROOT / "data" / "quickdraw" / "processed" / "test.npz") as d:
        x, y = d["images"][..., None], d["labels"].astype(int)
    start = time.time()
    p = engine.predict(x, 1024)
    acc = float((p.argmax(1) == y).mean())
    result = {"parameters": engine.parameter_count, "test_examples": int(len(y)), "numpy_top1": acc,
              "seconds": round(time.time() - start, 1)}
    recorded = json.loads((md / "test_metrics.json").read_text()).get("top_1_accuracy")
    if recorded is not None:
        result["recorded_top1"] = recorded
        result["top1_matches_record"] = abs(acc - recorded) < 1e-9
    saved = md / "predictions.npz"
    if saved.exists():
        with np.load(saved) as s:
            ps = s["probabilities"].astype(np.float32)
        result["max_abs_diff_vs_saved_float16"] = float(np.abs(ps - p).max())
        result["argmax_agreement_vs_saved"] = float((ps.argmax(1) == p.argmax(1)).mean())
    if args.keras:
        import tensorflow as tf

        model = tf.keras.models.load_model(md / "doodle_classifier.keras", compile=False)
        pk = model.predict(x, batch_size=1024, verbose=0)
        result["max_abs_diff_vs_keras"] = float(np.abs(pk - p).max())
    print(json.dumps(result, indent=2))
    ok = result.get("top1_matches_record", True) and result.get("max_abs_diff_vs_saved_float16", 0) < 1e-3
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
