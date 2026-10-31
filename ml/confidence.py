"""Confidence policy: when should GLITCH say "I'm not sure"?

Softmax scores are not calibrated probabilities, so the game never uses raw
scores directly. The policy is fitted on the VALIDATION split, after
temperature scaling, and stored in backend/model/confidence_policy.json.

Definitions (top-1 calibrated confidence = c):
* coverage(t)            fraction of drawings with c >= t
* selective_accuracy(t)  top-1 accuracy among drawings with c >= t
* below_accuracy(t)      top-1 accuracy among drawings with c < t

The "uncertain" threshold is the largest t on a 0.01 grid for which the model
is right less than half the time on drawings below t (below_accuracy < 0.5),
i.e. under t the CNN's top guess is more likely wrong than right. GLITCH then
says it is unsure instead of asserting a label.

Uncertainty never decides success: a drawing counts as recognised only when
the top-1 class equals the target. The policy only shapes feedback and hints.
"""

from __future__ import annotations

import numpy as np


def risk_coverage(probabilities: np.ndarray, labels: np.ndarray, grid: np.ndarray | None = None) -> list[dict]:
    grid = np.round(np.arange(0.0, 1.0, 0.01), 2) if grid is None else grid
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == labels
    rows = []
    for t in grid:
        above = confidence >= t
        below = ~above
        rows.append({
            "threshold": float(t),
            "coverage": float(above.mean()),
            "selective_accuracy": float(correct[above].mean()) if above.any() else None,
            "below_accuracy": float(correct[below].mean()) if below.any() else None,
            "below_count": int(below.sum()),
        })
    return rows


def choose_uncertain_threshold(probabilities: np.ndarray, labels: np.ndarray, max_below_accuracy: float = 0.5) -> dict:
    curve = risk_coverage(probabilities, labels)
    chosen = 0.0
    for row in curve:
        if row["below_accuracy"] is not None and row["below_count"] >= 50 and row["below_accuracy"] < max_below_accuracy:
            chosen = row["threshold"]
    at = next(r for r in curve if r["threshold"] == chosen)
    return {
        "uncertain_below": chosen,
        "rule": f"largest threshold where validation top-1 accuracy below it is < {max_below_accuracy:.0%} (min 50 drawings)",
        "coverage_at_threshold": at["coverage"],
        "selective_accuracy_at_threshold": at["selective_accuracy"],
        "accuracy_below_threshold": at["below_accuracy"],
        "drawings_below_threshold": at["below_count"],
        "curve": curve[::5],
    }
