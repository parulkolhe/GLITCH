"""Does the game's drawing pipeline give the CNN what it was trained on?

    python ml/pipeline_parity.py                       # 100 test drawings per class
    python ml/pipeline_parity.py --canvas-dir reports/canvas_exports   # also profile real game exports

Two measurements (NumPy engine only, no TensorFlow):

1. Round trip. Each held-out test bitmap is turned into what the game would
   send: upsampled to 600x600, thresholded into hard strokes, inverted to dark
   ink on white, encoded as PNG. That goes through backend/ml/preprocessing.py
   (the exact code the API runs) and the model. If preprocessing matched the
   training format perfectly, round-trip accuracy would equal raw accuracy.
   The gap measures how much the canvas path distorts the representation.

2. Input statistics. Ink coverage, bounding-box fill and stroke intensity of
   (a) training-format bitmaps, (b) round-tripped bitmaps and, optionally,
   (c) PNGs exported from the real game canvas (see tests/e2e), so drift
   between what players draw and what the model saw is visible.

Writes reports/pipeline_parity.json.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.ml.model import load_bundle  # noqa: E402
from backend.ml.preprocessing import preprocess_png  # noqa: E402


def to_canvas_png(bitmap: np.ndarray, size: int = 600) -> str:
    big = Image.fromarray(bitmap.reshape(28, 28).astype(np.uint8), "L").resize((size, size), Image.Resampling.BILINEAR)
    ink = np.asarray(big) > 96  # hard strokes, like a canvas pen
    paper = np.where(ink, 0, 255).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(paper, "L").convert("RGB").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def stats(bitmaps: np.ndarray) -> dict:
    x = bitmaps.reshape(len(bitmaps), 28, 28).astype(np.float32)
    ink = x > 32
    coverage = ink.mean(axis=(1, 2))
    fills, intensity = [], []
    for img, mask in zip(x, ink):
        ys, xs = np.where(mask)
        if len(ys) == 0:
            continue
        h, w = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
        fills.append(max(h, w) / 28.0)
        intensity.append(float(img[mask].mean()))
    return {
        "ink_coverage_mean": round(float(coverage.mean()), 4),
        "ink_coverage_p10_p90": [round(float(np.percentile(coverage, 10)), 4), round(float(np.percentile(coverage, 90)), 4)],
        "bbox_extent_mean": round(float(np.mean(fills)), 4),
        "stroke_intensity_mean": round(float(np.mean(intensity)), 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-class", type=int, default=100)
    parser.add_argument("--canvas-dir", type=Path, help="folder of PNG exports from the game canvas")
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "pipeline_parity.json")
    args = parser.parse_args()

    bundle = load_bundle()
    with np.load(ROOT / "data" / "quickdraw" / "processed" / "test.npz") as data:
        images, labels = data["images"], data["labels"].astype(int)
    rng = np.random.default_rng(0)
    pick = np.concatenate([rng.choice(np.where(labels == c)[0], args.per_class, replace=False) for c in range(len(bundle.class_names))])
    raw = images[pick][..., None]
    y = labels[pick]
    tripped = np.concatenate([preprocess_png(to_canvas_png(b)) for b in raw])
    p_raw = bundle.engine.predict(raw)
    p_trip = bundle.engine.predict(tripped)
    acc_raw = float((p_raw.argmax(1) == y).mean())
    acc_trip = float((p_trip.argmax(1) == y).mean())
    agree = float((p_raw.argmax(1) == p_trip.argmax(1)).mean())
    report = {
        "model_version": bundle.version,
        "drawings": int(len(y)),
        "raw_test_accuracy": round(acc_raw, 4),
        "round_trip_accuracy": round(acc_trip, 4),
        "accuracy_gap": round(acc_raw - acc_trip, 4),
        "top1_agreement_raw_vs_round_trip": round(agree, 4),
        "mean_abs_pixel_difference": round(float(np.abs(raw.astype(float) - tripped.astype(float)).mean()), 2),
        "per_class_round_trip_accuracy": {name: round(float((p_trip.argmax(1)[y == i] == i).mean()), 3) for i, name in enumerate(bundle.class_names)},
        "input_statistics": {"training_format": stats(raw), "round_trip": stats(tripped)},
    }
    if args.canvas_dir and args.canvas_dir.is_dir():
        exports = []
        for path in sorted(args.canvas_dir.glob("*.png")):
            exports.append(preprocess_png(base64.b64encode(path.read_bytes()).decode()))
        if exports:
            report["input_statistics"]["game_canvas_exports"] = {**stats(np.concatenate(exports)), "count": len(exports)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "per_class_round_trip_accuracy"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
