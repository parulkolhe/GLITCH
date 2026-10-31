"""Score the model on real drawings made in the game's canvas.

The held-out test set contains Quick, Draw! bitmaps. This script measures the
number that actually matters for the game: accuracy on drawings made in our own
canvas and sent through our own preprocessing. It also scores the original
(pre-fix) preprocessing on the same drawings, so the effect of the fix is measured
rather than assumed.

    1. Open collect.html in the browser and draw (2 rounds x 32 classes ~ 15 min).
    2. python ml/eval_canvas.py path/to/drawings.json
    (Merge several people's files by passing them all.)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend.ml.model import load_bundle, temperature_scale  # noqa: E402
from backend.ml.preprocessing import _decode_png, preprocess_png  # noqa: E402
from metrics import confusion_matrix, per_class_scores, top_confusions  # noqa: E402


def preprocess_png_legacy(image_base64: str) -> np.ndarray:
    """The project's original preprocessing (grayscale intensity, LANCZOS), kept for comparison."""
    gray = np.asarray(_decode_png(image_base64), dtype=np.uint8)
    foreground = gray < 245
    ys, xs = np.where(foreground)
    crop = Image.fromarray(gray[ys.min():ys.max() + 1, xs.min():xs.max() + 1], mode="L")
    width, height = crop.size
    margin = max(2, int(round(max(width, height) * 0.12)))
    side = max(width, height) + 2 * margin
    paper = Image.new("L", (side, side), 255)
    paper.paste(crop, ((side - width) // 2, (side - height) // 2))
    resized = paper.resize((28, 28), Image.Resampling.LANCZOS)
    return (255 - np.asarray(resized, dtype=np.uint8)).reshape(1, 28, 28, 1)


def load_drawings(paths: list[Path], class_names: list[str]) -> list[dict]:
    drawings = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        for item in data.get("drawings", []):
            if item.get("label") in class_names and isinstance(item.get("image"), str):
                drawings.append(item)
    return drawings


def score(model, bitmaps: np.ndarray, labels: np.ndarray, names: list[str], temperature: float) -> dict:
    probabilities = temperature_scale(model.predict(bitmaps), temperature)
    predicted = probabilities.argmax(axis=1)
    top3 = np.argsort(probabilities, axis=1)[:, -3:]
    confusion = confusion_matrix(labels, predicted, len(names))
    return {
        "top_1_accuracy": float((predicted == labels).mean()),
        "top_3_accuracy": float(np.any(top3 == labels[:, None], axis=1).mean()),
        "mean_confidence": float(probabilities.max(axis=1).mean()),
        "per_class": [row for row in per_class_scores(confusion, names) if row["support"]],
        "top_confusions": top_confusions(confusion, names, limit=8),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+", type=Path, help="drawings.json files exported from collect.html")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "backend" / "model")
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "canvas_eval.json")
    args = parser.parse_args()

    try:
        bundle = load_bundle(args.model_dir)
    except Exception as error:  # noqa: BLE001
        print(f"Model unavailable: {error}", file=sys.stderr)
        return 1
    model, names = bundle.engine, bundle.class_names
    drawings = load_drawings(args.files, names)
    if not drawings:
        print("No usable drawings found.", file=sys.stderr)
        return 1
    labels = np.array([names.index(item["label"]) for item in drawings])
    current = np.concatenate([preprocess_png(item["image"]) for item in drawings])
    legacy = np.concatenate([preprocess_png_legacy(item["image"]) for item in drawings])
    temperature = bundle.temperature
    report = {
        "drawings": len(drawings),
        "drawers": sorted({item.get("drawer", "anonymous") for item in drawings}),
        "temperature": temperature,
        "current_preprocessing": score(model, current, labels, names, temperature),
        "original_preprocessing": score(model, legacy, labels, names, temperature),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    new, old = report["current_preprocessing"], report["original_preprocessing"]
    print(f"{len(drawings)} canvas drawings from {len(report['drawers'])} people")
    print(f"Current preprocessing : top-1 {new['top_1_accuracy']:.3f}  top-3 {new['top_3_accuracy']:.3f}")
    print(f"Original preprocessing: top-1 {old['top_1_accuracy']:.3f}  top-3 {old['top_3_accuracy']:.3f}")
    print(f"Saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
