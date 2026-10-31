"""Draw the result figures used in the presentation.

    python ml/plots.py                      # reads backend/model, writes reports/figures

Figures that need saved probabilities (reliability diagram, confidence
histogram) are drawn only after `python ml/evaluate.py` has written
predictions.npz.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import apply_temperature, calibration_bins, per_class_scores  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# Validated reference palette (light mode): categorical slots 1-2, text, grid.
BLUE, ORANGE = "#2a78d6", "#eb6834"
TEXT, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e2dc", "#fcfcfb"
SEQUENTIAL_BLUE = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def style(plt) -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 11,
        "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlecolor": TEXT, "axes.titlelocation": "left",
    })


def read_history(path: Path) -> dict[str, list[float]]:
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {key: [float(row[key]) for row in rows if row.get(key)] for key in rows[0] if key}


def read_confusion(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    return rows[0][1:], np.array([[int(value) for value in row[1:]] for row in rows[1:]])


def training_curves(plt, history: dict, out: Path) -> None:
    epochs = history["epoch"]
    figure, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2))
    for axis, key, title in ((left, "accuracy", "Accuracy by epoch"), (right, "loss", "Loss by epoch")):
        train, validation = history[key], history[f"val_{key}"]
        axis.plot(epochs, train, color=BLUE, linewidth=2, marker="o", markersize=5, label="Training")
        axis.plot(epochs, validation, color=ORANGE, linewidth=2, marker="o", markersize=5, label="Validation")
        axis.annotate(f"{validation[-1]:.3f}", (epochs[-1], validation[-1]), textcoords="offset points",
                      xytext=(6, 0), va="center", color=TEXT, fontsize=10)
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.set_xticks(epochs)
        axis.legend(frameon=False, labelcolor=TEXT)
    if "learning_rate" in history:
        changes = [e for e, prev, cur in zip(epochs[1:], history["learning_rate"], history["learning_rate"][1:]) if cur != prev]
        for epoch in changes:
            for axis in (left, right):
                axis.axvline(epoch - 0.5, color=MUTED, linewidth=1, linestyle="--")
            left.text(epoch - 0.45, left.get_ylim()[0], " learning rate lowered", color=MUTED, fontsize=9, va="bottom")
    figure.tight_layout()
    figure.savefig(out, dpi=200)
    plt.close(figure)


def per_class_f1(plt, rows: list[dict], out: Path) -> None:
    rows = sorted(rows, key=lambda row: row["f1"])
    figure, axis = plt.subplots(figsize=(8, 9))
    values = [row["f1"] for row in rows]
    axis.barh([row["class"] for row in rows], values, color=BLUE, height=0.72)
    for index, value in enumerate(values):
        axis.text(value + 0.005, index, f"{value:.2f}", va="center", fontsize=9, color=TEXT)
    mean = float(np.mean(values))
    axis.axvline(mean, color=MUTED, linewidth=1, linestyle="--")
    axis.text(mean, len(rows) - 0.3, f" macro F1 {mean:.3f}", color=MUTED, fontsize=9, va="bottom")
    axis.set_xlim(0.4, 1.04)
    axis.set_title("Per-class F1 on the held-out test set")
    axis.set_xlabel("F1 score")
    axis.grid(axis="y", visible=False)
    figure.tight_layout()
    figure.savefig(out, dpi=200)
    plt.close(figure)


def confusion_heatmap(plt, names: list[str], confusion: np.ndarray, out: Path) -> None:
    from matplotlib.colors import LinearSegmentedColormap

    share = confusion / confusion.sum(axis=1, keepdims=True)
    figure, axis = plt.subplots(figsize=(10, 9))
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQUENTIAL_BLUE)
    image = axis.imshow(share, cmap=cmap, vmin=0, vmax=1)
    axis.set_xticks(range(len(names)), names, rotation=90, fontsize=8)
    axis.set_yticks(range(len(names)), names, fontsize=8)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.grid(False)
    axis.set_title("Confusion matrix (share of each true class)")
    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.02)
    colorbar.outline.set_visible(False)
    figure.tight_layout()
    figure.savefig(out, dpi=200)
    plt.close(figure)


def top_confusion_bars(plt, names: list[str], confusion: np.ndarray, out: Path, limit: int = 10) -> None:
    off = confusion.copy()
    np.fill_diagonal(off, 0)
    if not off.any():
        print("No misclassifications: skipping top_confusions.png")
        return
    order = np.argsort(off, axis=None)[::-1][:limit]
    pairs = [np.unravel_index(flat, off.shape) for flat in order]
    labels = [f"{names[t]} → {names[p]}" for t, p in pairs][::-1]
    counts = [int(off[t, p]) for t, p in pairs][::-1]
    figure, axis = plt.subplots(figsize=(8, 4.8))
    axis.barh(labels, counts, color=BLUE, height=0.7)
    for index, value in enumerate(counts):
        axis.text(value + 2, index, f"{value} ({value / 10:.1f}%)", va="center", fontsize=9, color=TEXT)
    axis.set_xlim(0, max(counts) * 1.25)
    axis.set_title("Most common mistakes (true → predicted)")
    axis.set_xlabel("Test drawings out of 1,000 per class")
    axis.grid(axis="y", visible=False)
    figure.tight_layout()
    figure.savefig(out, dpi=200)
    plt.close(figure)


def reliability(plt, probabilities: np.ndarray, labels: np.ndarray, temperature: float, out: Path) -> None:
    figure, axis = plt.subplots(figsize=(6.2, 5.6))
    axis.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle="--", label="Perfect calibration")
    for probs, color, name in ((probabilities, BLUE, "Raw softmax"),
                               (apply_temperature(probabilities, temperature), ORANGE, f"Temperature-scaled (T={temperature})")):
        ece, bins = calibration_bins(probs, labels)
        points = [(b["mean_confidence"], b["accuracy"]) for b in bins if b["count"] >= 30]
        axis.plot(*zip(*points), color=color, linewidth=2, marker="o", markersize=6, label=f"{name}, ECE {ece:.3f}")
    axis.set_xlim(0, 1.01)
    axis.set_ylim(0, 1.01)
    axis.set_xlabel("Average confidence in bin")
    axis.set_ylabel("Actual accuracy in bin")
    axis.set_title("Reliability diagram (test set)")
    axis.legend(frameon=False, loc="upper left", fontsize=9, labelcolor=TEXT)
    figure.tight_layout()
    figure.savefig(out, dpi=200)
    plt.close(figure)


def confidence_histogram(plt, probabilities: np.ndarray, labels: np.ndarray, out: Path) -> None:
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == labels
    bins = np.linspace(0, 1, 21)
    figure, axis = plt.subplots(figsize=(7.5, 4.2))
    axis.hist(confidence[correct], bins=bins, color=BLUE, alpha=0.85, label=f"Correct ({correct.sum():,})",
              weights=np.full(correct.sum(), 100 / max(1, correct.sum())), edgecolor=SURFACE, linewidth=2)
    axis.hist(confidence[~correct], bins=bins, color=ORANGE, alpha=0.85, label=f"Wrong ({(~correct).sum():,})",
              weights=np.full((~correct).sum(), 100 / max(1, (~correct).sum())), edgecolor=SURFACE, linewidth=2)
    axis.set_xlabel("Top-1 confidence")
    axis.set_ylabel("% of predictions in group")
    axis.set_title("Confidence separates right from wrong predictions")
    axis.legend(frameon=False, labelcolor=TEXT)
    figure.tight_layout()
    figure.savefig(out, dpi=200)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "backend" / "model")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "reports" / "figures")
    args = parser.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style(plt)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    history_path = args.model_dir / "training_history.csv"
    if history_path.exists():
        training_curves(plt, read_history(history_path), args.out_dir / "training_curves.png")
        written.append("training_curves.png")
    confusion_path = args.model_dir / "confusion_matrix.csv"
    if confusion_path.exists():
        names, confusion = read_confusion(confusion_path)
        per_class_f1(plt, per_class_scores(confusion, names), args.out_dir / "per_class_f1.png")
        confusion_heatmap(plt, names, confusion, args.out_dir / "confusion_matrix.png")
        top_confusion_bars(plt, names, confusion, args.out_dir / "top_confusions.png")
        written += ["per_class_f1.png", "confusion_matrix.png", "top_confusions.png"]
    predictions_path = args.model_dir / "predictions.npz"
    if predictions_path.exists():
        with np.load(predictions_path) as data:
            probabilities, labels = data["probabilities"].astype(np.float64), data["labels"].astype(np.int64)
        calibration_path = args.model_dir / "calibration.json"
        temperature = json.loads(calibration_path.read_text())["temperature"] if calibration_path.exists() else 1.0
        reliability(plt, probabilities, labels, temperature, args.out_dir / "reliability.png")
        confidence_histogram(plt, probabilities, labels, args.out_dir / "confidence_histogram.png")
        written += ["reliability.png", "confidence_histogram.png"]
    else:
        print("predictions.npz not found: run `python ml/evaluate.py --fit-temperature` for calibration figures.")
    print("Wrote:", ", ".join(written) or "nothing", "->", args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
