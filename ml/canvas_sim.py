"""Simulate what a QuickDraw drawing looks like when drawn in the game canvas.

QuickDraw bitmaps are 28x28 renders with ~2px anti-aliased strokes. A player
in the game draws on a 600x600 canvas with a 10px pen (frontend
src/canvas/strokes.ts). To evaluate the game's preprocessing fairly we
recover each bitmap's centrelines (Zhang-Suen thinning at 4x resolution),
place the drawing at a random size and position on a 600x600 canvas, and
redraw it with the game's pen width. The result is a binary ink mask, the
same thing backend/ml/preprocessing.py extracts from a real canvas PNG.
"""

from __future__ import annotations

import numpy as np
from PIL import Image
from scipy import ndimage

CANVAS = 600
PEN = 10


def zhang_suen(mask: np.ndarray, max_iter: int = 60) -> np.ndarray:
    """Vectorised Zhang-Suen thinning of a boolean image."""
    img = np.pad(mask.astype(np.uint8), 1)
    for _ in range(max_iter):
        changed = False
        for step in (0, 1):
            p = img
            p2, p3, p4 = p[:-2, 1:-1], p[:-2, 2:], p[1:-1, 2:]
            p5, p6, p7 = p[2:, 2:], p[2:, 1:-1], p[2:, :-2]
            p8, p9 = p[1:-1, :-2], p[:-2, :-2]
            centre = p[1:-1, 1:-1]
            neighbours = [p2, p3, p4, p5, p6, p7, p8, p9]
            b = sum(n.astype(np.int16) for n in neighbours)
            seq = neighbours + [p2]
            a = sum(((seq[i] == 0) & (seq[i + 1] == 1)).astype(np.int16) for i in range(8))
            if step == 0:
                cond = (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                cond = (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
            remove = (centre == 1) & (b >= 2) & (b <= 6) & (a == 1) & cond
            if remove.any():
                changed = True
                img[1:-1, 1:-1][remove] = 0
        if not changed:
            break
    return img[1:-1, 1:-1].astype(bool)


def centrelines(bitmap: np.ndarray, scale: int = 4, threshold: int = 70) -> np.ndarray:
    big = Image.fromarray(bitmap.reshape(28, 28).astype(np.uint8), "L").resize((28 * scale, 28 * scale), Image.Resampling.BILINEAR)
    return zhang_suen(np.asarray(big) > threshold)


def canvas_mask(bitmap: np.ndarray, rng: np.random.Generator, pen: int = PEN) -> np.ndarray:
    """Binary 600x600 ink mask of the drawing as a player might draw it."""
    skel = centrelines(bitmap)
    ys, xs = np.where(skel)
    if len(ys) == 0:
        return np.zeros((CANVAS, CANVAS), bool)
    ys = ys - ys.min()
    xs = xs - xs.min()
    extent = max(ys.max(), xs.max(), 1)
    size = rng.uniform(260, 520)            # players fill roughly half to most of the canvas
    k = size / extent
    oy = rng.uniform(20, CANVAS - 20 - ys.max() * k)
    ox = rng.uniform(20, CANVAS - 20 - xs.max() * k)
    out = np.zeros((CANVAS, CANVAS), bool)
    # draw each skeleton pixel as a filled square of side k (keeps lines connected), then thicken to the pen width
    py = np.clip((oy + ys * k).astype(int), 0, CANVAS - 1)
    px = np.clip((ox + xs * k).astype(int), 0, CANVAS - 1)
    step = int(np.ceil(k))
    for dy in range(step):
        for dx in range(step):
            out[np.clip(py + dy, 0, CANVAS - 1), np.clip(px + dx, 0, CANVAS - 1)] = True
    # thin back to ~1px lines along the square chains, then apply the pen
    out = ndimage.binary_erosion(out, iterations=max(0, step // 2 - 1)) if step > 3 else out
    return ndimage.maximum_filter(out, size=pen)
