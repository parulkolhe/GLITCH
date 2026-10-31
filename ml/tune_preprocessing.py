"""Tune the canvas preprocessing so game drawings look like the training data.

    python ml/tune_preprocessing.py --per-class 40

Protocol
1. Simulate canvas drawings (ml/canvas_sim.py) from the VALIDATION split.
2. Grid-search margin x dilation, scoring top-1 accuracy of the deployed CNN
   on the preprocessed simulations. Select the best config on validation only.
3. Score the current and the selected config ONCE on simulated drawings from
   the TEST split, plus the raw test bitmaps for reference.
4. Write reports/preprocessing_tuning.json. Applying the result means editing
   MARGIN / STROKE_DILATION in backend/ml/preprocessing.py (a reviewed change,
   not an automatic one).
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend.ml.model import load_bundle  # noqa: E402
from backend.ml import preprocessing as prep  # noqa: E402
from canvas_sim import canvas_mask  # noqa: E402

MARGINS = [0.02, 0.05, 0.08, 0.12]
DILATIONS = [1, 5, 9, 13, 17]


def sample(split: str, per_class: int, classes: int, seed: int):
    with np.load(ROOT / "data" / "quickdraw" / "processed" / f"{split}.npz") as d:
        images, labels = d["images"], d["labels"].astype(int)
    rng = np.random.default_rng(seed)
    idx = np.concatenate([rng.choice(np.where(labels == c)[0], per_class, replace=False) for c in range(classes)])
    masks = []
    for i in idx:
        m = canvas_mask(images[i], rng)
        ys, xs = np.where(m)
        masks.append(m[ys.min():ys.max() + 1, xs.min():xs.max() + 1] if len(ys) else m)
    return images[idx], labels[idx], masks


def accuracy(engine, masks, labels, margin, dilation) -> float:
    batch = np.concatenate([prep.ink_to_bitmap(m, margin, dilation) for m in masks])
    return float((engine.predict(batch).argmax(1) == labels).mean())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-class", type=int, default=40)
    parser.add_argument("--margins", type=float, nargs="+", default=MARGINS)
    parser.add_argument("--dilations", type=int, nargs="+", default=DILATIONS)
    parser.add_argument("--merge", action="store_true", help="add these grid points to an existing report and reselect over all of them")
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "preprocessing_tuning.json")
    args = parser.parse_args()
    bundle = load_bundle()
    n = len(bundle.class_names)
    t0 = time.time()
    _, y_val, m_val = sample("validation", args.per_class, n, seed=1)
    print(f"simulated {len(m_val)} validation drawings in {time.time() - t0:.0f}s", flush=True)
    grid = json.loads(args.out.read_text())["grid"] if args.merge and args.out.exists() else []
    done = {(r["margin"], r["dilation"]) for r in grid}
    for margin, dilation in itertools.product(args.margins, args.dilations):
        if (margin, dilation) in done:
            continue
        acc = accuracy(bundle.engine, m_val, y_val, margin, dilation)
        grid.append({"margin": margin, "dilation": dilation, "validation_accuracy": round(acc, 4)})
        print(f"  margin {margin:.2f} dilation {dilation:2d}: {acc:.4f}", flush=True)
    best = max(grid, key=lambda r: r["validation_accuracy"])
    current = next((r for r in grid if r["margin"] == prep.MARGIN and r["dilation"] == prep.STROKE_DILATION), None)
    if current is None:
        current = {"margin": prep.MARGIN, "dilation": prep.STROKE_DILATION,
                   "validation_accuracy": round(accuracy(bundle.engine, m_val, y_val, prep.MARGIN, prep.STROKE_DILATION), 4)}
    raw_test, y_test, m_test = sample("test", args.per_class, n, seed=2)
    report = {
        "model_version": bundle.version,
        "simulation": "QuickDraw bitmaps -> Zhang-Suen centrelines -> redrawn on a 600px canvas with the game's 10px pen at 260-520px size",
        "per_class": args.per_class,
        "selection_split": "validation",
        "grid": grid,
        "current": {**current, "test_accuracy": round(accuracy(bundle.engine, m_test, y_test, current["margin"], current["dilation"]), 4)},
        "selected": {**best, "test_accuracy": round(accuracy(bundle.engine, m_test, y_test, best["margin"], best["dilation"]), 4)},
        "raw_test_bitmap_accuracy_same_sample": round(float((bundle.engine.predict(raw_test[..., None]).argmax(1) == y_test).mean()), 4),
        "seconds": round(time.time() - t0, 1),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "grid"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
