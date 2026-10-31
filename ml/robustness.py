"""How much do realistic drawing variations hurt the deployed CNN?

    python ml/robustness.py --per-class 60

Applies one transformation family at a time, at increasing strength, to
VALIDATION bitmaps and reports top-1 accuracy. This is how the training
augmentation ranges (ml/train.py: translation 8%, rotation 0.045 turns = 16 deg,
zoom 8%) are judged: inside the trained range accuracy should hold; outside it,
the drop shows where more augmentation could help. Stroke thickness is tested
too because it is the variation the game's preprocessing controls.

Writes reports/robustness.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.ml.model import load_bundle  # noqa: E402


def translate(x, frac):
    s = frac * 28
    return np.stack([ndimage.shift(img, (s, s), order=1, mode="constant") for img in x])


def rotate(x, deg):
    return np.stack([ndimage.rotate(img, deg, reshape=False, order=1, mode="constant") for img in x])


def scale(x, factor):
    out = []
    for img in x:
        z = ndimage.zoom(img, factor, order=1)
        canvas = np.zeros((28, 28), np.float32)
        if z.shape[0] >= 28:
            o = (z.shape[0] - 28) // 2
            canvas = z[o:o + 28, o:o + 28]
        else:
            o = (28 - z.shape[0]) // 2
            canvas[o:o + z.shape[0], o:o + z.shape[1]] = z
        out.append(canvas)
    return np.stack(out)


def thickness(x, delta):
    if delta == 0:
        return x
    op = ndimage.grey_dilation if delta > 0 else ndimage.grey_erosion
    return np.stack([op(img, size=(abs(delta) + 1, abs(delta) + 1)) for img in x])


FAMILIES = {
    "translation (fraction of width)": (translate, [0, 0.04, 0.08, 0.12, 0.16, 0.24]),
    "rotation (degrees)": (rotate, [0, 8, 16, 24, 32, 45]),
    "scale (zoom factor)": (scale, [0.6, 0.75, 0.9, 1.0, 1.1, 1.25]),
    "stroke thickness (pixels)": (thickness, [-1, 0, 1, 2]),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-class", type=int, default=60)
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "robustness.json")
    args = parser.parse_args()
    bundle = load_bundle()
    with np.load(ROOT / "data" / "quickdraw" / "processed" / "validation.npz") as d:
        images, labels = d["images"].astype(np.float32), d["labels"].astype(int)
    rng = np.random.default_rng(3)
    idx = np.concatenate([rng.choice(np.where(labels == c)[0], args.per_class, replace=False) for c in range(len(bundle.class_names))])
    x, y = images[idx], labels[idx]
    report = {"model_version": bundle.version, "split": "validation", "drawings": int(len(y)), "families": {}}
    for name, (fn, levels) in FAMILIES.items():
        rows = []
        for level in levels:
            t = np.clip(fn(x, level), 0, 255).astype(np.uint8)[..., None]
            acc = float((bundle.engine.predict(t).argmax(1) == y).mean())
            rows.append({"level": level, "accuracy": round(acc, 4)})
            print(f"{name:34s} {level:>6}: {acc:.4f}", flush=True)
        report["families"][name] = rows
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
