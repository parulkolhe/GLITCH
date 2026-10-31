"""Pure-NumPy evaluation helpers shared by evaluate.py, plots.py and the tests.

Nothing here imports TensorFlow, so every function can be unit-tested quickly.
"""

from __future__ import annotations

import numpy as np


def confusion_matrix(labels: np.ndarray, predicted: np.ndarray, num_classes: int) -> np.ndarray:
    """Rows are true classes, columns are predicted classes."""
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(matrix, (np.asarray(labels), np.asarray(predicted)), 1)
    return matrix


def per_class_scores(confusion: np.ndarray, class_names: list[str]) -> list[dict]:
    """Precision, recall, F1 and support for every class."""
    rows = []
    for index, name in enumerate(class_names):
        true_positive = int(confusion[index, index])
        support = int(confusion[index].sum())
        predicted_count = int(confusion[:, index].sum())
        precision = true_positive / predicted_count if predicted_count else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({"class": name, "precision": precision, "recall": recall, "f1": f1, "support": support})
    return rows


def top_k_accuracy(probabilities: np.ndarray, labels: np.ndarray, k: int) -> float:
    k = min(k, probabilities.shape[1])
    top = np.argpartition(probabilities, -k, axis=1)[:, -k:]
    return float(np.any(top == np.asarray(labels)[:, None], axis=1).mean())


def top_confusions(confusion: np.ndarray, class_names: list[str], limit: int = 10) -> list[dict]:
    """Most frequent (true -> predicted) mistakes, largest first."""
    off_diagonal = confusion.copy()
    np.fill_diagonal(off_diagonal, 0)
    order = np.argsort(off_diagonal, axis=None)[::-1][:limit]
    rows = []
    for flat in order:
        true_index, predicted_index = np.unravel_index(flat, off_diagonal.shape)
        count = int(off_diagonal[true_index, predicted_index])
        if count == 0:
            break
        rows.append({
            "true": class_names[true_index],
            "predicted": class_names[predicted_index],
            "count": count,
            "share_of_true_class": count / max(1, int(confusion[true_index].sum())),
        })
    return rows


def apply_temperature(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    """Temperature-scale softmax outputs.

    The saved model ends in softmax, so log(p) equals the logits up to a
    per-row constant, and softmax(log(p) / T) is exactly temperature scaling.
    T > 1 softens over-confident outputs; T = 1 leaves them unchanged.
    """
    if temperature <= 0:
        raise ValueError("Temperature must be positive.")
    logits = np.log(np.clip(np.asarray(probabilities, dtype=np.float64), 1e-12, 1.0)) / temperature
    logits -= logits.max(axis=-1, keepdims=True)
    scaled = np.exp(logits)
    return scaled / scaled.sum(axis=-1, keepdims=True)


def negative_log_likelihood(probabilities: np.ndarray, labels: np.ndarray) -> float:
    picked = np.asarray(probabilities)[np.arange(len(labels)), np.asarray(labels)]
    return float(-np.log(np.clip(picked, 1e-12, 1.0)).mean())


def fit_temperature(probabilities: np.ndarray, labels: np.ndarray) -> float:
    """Find the temperature that minimises NLL (coarse grid, then a fine grid)."""
    def best(grid: np.ndarray) -> float:
        losses = [negative_log_likelihood(apply_temperature(probabilities, t), labels) for t in grid]
        return float(grid[int(np.argmin(losses))])

    coarse = best(np.linspace(0.5, 5.0, 46))
    return round(best(np.linspace(max(0.05, coarse - 0.1), coarse + 0.1, 41)), 3)


def calibration_bins(probabilities: np.ndarray, labels: np.ndarray, bins: int = 15) -> tuple[float, list[dict]]:
    """Expected calibration error and the reliability-diagram table.

    Each prediction goes into a bin by its top-1 confidence. ECE is the
    support-weighted gap between average confidence and accuracy per bin.
    """
    probabilities = np.asarray(probabilities)
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == np.asarray(labels)
    edges = np.linspace(0.0, 1.0, bins + 1)
    table, ece, total = [], 0.0, len(labels)
    for low, high in zip(edges[:-1], edges[1:]):
        in_bin = (confidence > low) & (confidence <= high)
        count = int(in_bin.sum())
        if count:
            mean_confidence = float(confidence[in_bin].mean())
            accuracy = float(correct[in_bin].mean())
            ece += count / total * abs(mean_confidence - accuracy)
        else:
            mean_confidence = accuracy = None
        table.append({"low": float(low), "high": float(high), "count": count,
                      "mean_confidence": mean_confidence, "accuracy": accuracy})
    return float(ece), table


def difficulty_tiers(per_class: list[dict], key: str = "f1") -> dict[str, str]:
    """Split classes into easy / medium / hard thirds by a per-class score.

    The game's difficulty controller uses these tiers to choose prompts: a class
    the model recognises reliably is an easy prompt, a frequently confused one
    is hard.
    """
    ranked = sorted(per_class, key=lambda row: row[key], reverse=True)
    third = max(1, round(len(ranked) / 3))
    tiers = {}
    for position, row in enumerate(ranked):
        tiers[row["class"]] = "easy" if position < third else "medium" if position < 2 * third else "hard"
    return tiers
