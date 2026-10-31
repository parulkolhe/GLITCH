"""Novelty: Grad-CAM explanations and uncertainty of a prediction.

* Grad-CAM (Selvaraju et al., 2017): the last convolution's feature maps,
  weighted by the average gradient of the predicted class, show which strokes
  drove the guess ("What GLITCH looked at").
* Uncertainty: normalised entropy of the calibrated softmax and the margin
  between the top two classes. A small margin means the CNN was torn.
"""

from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image


GRADCAM_LAYER = "block3_conv"
OVERLAY_SIZE = 196  # 28 x 7 pixels, so every model pixel stays a crisp square


# ---------------------------------------------------------------- uncertainty
def uncertainty(probabilities: np.ndarray, class_names: list[str]) -> dict:
    """Normalised entropy (0 = certain, 1 = uniform) and top-2 margin."""
    p = np.clip(np.asarray(probabilities, dtype=np.float64).reshape(-1), 1e-12, 1.0)
    p = p / p.sum()
    entropy = float(-(p * np.log(p)).sum() / np.log(len(p))) if len(p) > 1 else 0.0
    order = np.argsort(p)[::-1]
    margin = float(p[order[0]] - p[order[1]]) if len(p) > 1 else 1.0
    if margin < 0.15 or entropy > 0.6:
        level = "high"
    elif margin < 0.35 or entropy > 0.35:
        level = "medium"
    else:
        level = "low"
    return {
        "entropy": round(entropy, 4),
        "margin": round(margin, 4),
        "top_two": [class_names[int(order[0])], class_names[int(order[1])]] if len(p) > 1 else [class_names[0]],
        "level": level,
    }


# -------------------------------------------------------------------- Grad-CAM
def last_conv_layer(model):
    """block3_conv in this project's models, otherwise the last Conv2D layer."""
    try:
        return model.get_layer(GRADCAM_LAYER)
    except ValueError:
        pass
    import tensorflow as tf

    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer
    raise ValueError("The model has no convolution layer for Grad-CAM.")


def build_gradcam_model(model):
    """One model that returns (last conv feature maps, class probabilities)."""
    import tensorflow as tf

    return tf.keras.Model(inputs=model.inputs, outputs=[last_conv_layer(model).output, model.output])


def gradcam(cam_model, bitmap: np.ndarray, class_index: int) -> np.ndarray:
    """Heatmap in [0, 1] at the conv layer's resolution (7 x 7 here)."""
    import tensorflow as tf

    inputs = tf.convert_to_tensor(bitmap)
    with tf.GradientTape() as tape:
        feature_maps, probabilities = cam_model(inputs, training=False)
        # Log-probability avoids vanishing gradients when the softmax is saturated.
        class_score = tf.math.log(probabilities[:, class_index] + 1e-9)
    gradients = tape.gradient(class_score, feature_maps)
    weights = tf.reduce_mean(gradients, axis=(0, 1, 2))            # one weight per channel
    cam = tf.nn.relu(tf.reduce_sum(feature_maps[0] * weights, axis=-1)).numpy()
    peak = float(cam.max())
    return cam / peak if peak > 0 else np.zeros_like(cam)


def focus_region(heatmap: np.ndarray) -> str:
    """Plain-language region of the drawing the heatmap concentrates on."""
    heat = np.asarray(heatmap, dtype=np.float64)
    if heat.size == 0 or heat.sum() <= 0:
        return "the whole drawing"
    rows, cols = np.indices(heat.shape)
    y = float((rows * heat).sum() / heat.sum()) / max(1, heat.shape[0] - 1)
    x = float((cols * heat).sum() / heat.sum()) / max(1, heat.shape[1] - 1)
    vertical = "top" if y < 0.4 else "bottom" if y > 0.6 else "middle"
    horizontal = "left" if x < 0.4 else "right" if x > 0.6 else "centre"
    if vertical == "middle" and horizontal == "centre":
        return "the centre of the drawing"
    if vertical == "middle":
        return f"the {horizontal} side of the drawing"
    if horizontal == "centre":
        return f"the {vertical} of the drawing"
    return f"the {vertical}-{horizontal} of the drawing"


def overlay_png(bitmap: np.ndarray, heatmap: np.ndarray, size: int = OVERLAY_SIZE) -> str:
    """The 28 x 28 image the CNN saw (dark strokes on white) with the heatmap in red/yellow."""
    strokes = np.asarray(bitmap, dtype=np.float64).reshape(bitmap.shape[-3], bitmap.shape[-2])
    paper = Image.fromarray((255 - strokes).clip(0, 255).astype(np.uint8), "L").resize((size, size), Image.NEAREST)
    heat_image = Image.fromarray((np.clip(heatmap, 0, 1) * 255).astype(np.uint8), "L").resize((size, size), Image.BILINEAR)
    base = np.asarray(paper.convert("RGB"), dtype=np.float64)
    heat = np.asarray(heat_image, dtype=np.float64)[..., None] / 255.0
    # Colour ramp: transparent -> red -> yellow as attention rises.
    colour = np.concatenate([np.full_like(heat, 255.0), 255.0 * np.clip(heat * 1.6 - 0.6, 0, 1), np.zeros_like(heat)], axis=-1)
    alpha = 0.65 * heat
    blended = (base * (1 - alpha) + colour * alpha).clip(0, 255).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(blended, "RGB").save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
