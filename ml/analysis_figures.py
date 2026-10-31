"""Figures for the GLITCH analyses (run after the analysis scripts).

    python ml/analysis_figures.py

Reads reports/*.json and backend/model/versions/<active>/confidence_policy.json,
writes reports/figures/{risk_coverage,robustness,preprocessing_tuning}.png.
Missing inputs are skipped with a message.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plots import BLUE, MUTED, ORANGE, SEQUENTIAL_BLUE, TEXT, style  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"


def active_dir() -> Path:
    registry = json.loads((ROOT / "backend/model/registry.json").read_text())
    return ROOT / "backend/model/versions" / registry["active"]


def risk_coverage(plt) -> str | None:
    path = active_dir() / "confidence_policy.json"
    if not path.exists():
        return None
    policy = json.loads(path.read_text())
    curve = [r for r in policy["curve"] if r["selective_accuracy"] is not None]
    t = [r["threshold"] for r in curve]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(t, [r["coverage"] for r in curve], color=BLUE, lw=2.2, label="share of drawings GLITCH answers confidently")
    ax.plot(t, [r["selective_accuracy"] for r in curve], color=ORANGE, lw=2.2, label="accuracy on those drawings")
    below = [(r["threshold"], r["below_accuracy"]) for r in curve if r["below_accuracy"] is not None]
    ax.plot([b[0] for b in below], [b[1] for b in below], color=MUTED, lw=1.6, ls="--", label="accuracy below the threshold")
    ax.axvline(policy["uncertain_below"], color=TEXT, lw=1)
    ax.text(policy["uncertain_below"] + 0.01, 0.06, f"“unsure” below {policy['uncertain_below']:.2f}", color=TEXT, fontsize=10)
    ax.set(xlabel="calibrated top-1 score threshold", ylim=(0, 1.02), xlim=(0, 1),
           title="Uncertainty threshold, fitted on validation")
    ax.legend(frameon=False, fontsize=9, loc="lower left", bbox_to_anchor=(0, 0.12))
    fig.tight_layout()
    fig.savefig(FIG / "risk_coverage.png", dpi=160)
    plt.close(fig)
    return "risk_coverage.png"


def robustness(plt) -> str | None:
    path = ROOT / "reports" / "robustness.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    fams = list(data["families"].items())
    fig, axes = plt.subplots(1, len(fams), figsize=(3.2 * len(fams), 3.4), sharey=True)
    trained = {"translation (fraction of width)": 0.08, "rotation (degrees)": 16.2, "scale (zoom factor)": (0.92, 1.08)}
    for ax, (name, rows) in zip(axes, fams):
        x = [r["level"] for r in rows]
        ax.plot(x, [r["accuracy"] for r in rows], color=BLUE, lw=2.2, marker="o", ms=4)
        if name in trained:
            v = trained[name]
            if isinstance(v, tuple):
                ax.axvspan(v[0], v[1], color=SEQUENTIAL_BLUE[1], alpha=0.8, lw=0)
            else:
                ax.axvspan(min(x), v, color=SEQUENTIAL_BLUE[1], alpha=0.8, lw=0)
        ax.set_title(name, fontsize=10)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("validation top-1")
    fig.suptitle("Robustness of the deployed CNN (shaded: range covered by training augmentation)", x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "robustness.png", dpi=160)
    plt.close(fig)
    return "robustness.png"


def tuning(plt) -> str | None:
    path = ROOT / "reports" / "preprocessing_tuning.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    margins = sorted({r["margin"] for r in data["grid"]})
    dil = sorted({r["dilation"] for r in data["grid"]})
    grid = np.full((len(margins), len(dil)), np.nan)
    for r in data["grid"]:
        grid[margins.index(r["margin"]), dil.index(r["dilation"])] = r["validation_accuracy"]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    from matplotlib.colors import LinearSegmentedColormap

    raw = grid
    grid = np.ma.masked_invalid(grid)
    im = ax.imshow(grid, cmap=LinearSegmentedColormap.from_list("seq", SEQUENTIAL_BLUE), aspect="auto")
    for i in range(len(margins)):
        for j in range(len(dil)):
            v = raw[i, j]
            if np.isnan(v):
                ax.text(j, i, "–", ha="center", va="center", fontsize=9, color=MUTED)
                continue
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=9, color="white" if v > float(grid.mean()) else TEXT)
    ax.set_xticks(range(len(dil)), [str(d) for d in dil])
    ax.set_yticks(range(len(margins)), [f"{m:.2f}" for m in margins])
    ax.set(xlabel="stroke dilation (px at 256)", ylabel="crop margin", title="Canvas preprocessing grid (simulated validation drawings; – = not tested)")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    fig.savefig(FIG / "preprocessing_tuning.png", dpi=160)
    plt.close(fig)
    return "preprocessing_tuning.png"


def main() -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style(plt)
    FIG.mkdir(parents=True, exist_ok=True)
    made = [name for name in (risk_coverage(plt), robustness(plt), tuning(plt)) if name]
    print("Wrote:", ", ".join(made) if made else "nothing (run the analysis scripts first)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
