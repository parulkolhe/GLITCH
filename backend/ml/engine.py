"""Pure-NumPy inference for the project's Keras CNNs.

Why this exists
---------------
TensorFlow is a heavy, platform-sensitive dependency for a single-player game
whose model has ~91k parameters. `ml/export_weights.py` converts a trained
`.keras` checkpoint into two portable files:

* ``<name>.weights.npz`` - every trained tensor, keyed ``<layer>/<index>``
* ``<name>.arch.json``  - the layer graph taken from the Keras config

`NumpyCNN` replays that graph exactly in inference mode (dropout and random
augmentation are identity, BatchNorm uses its moving statistics). Parity with
Keras is verified by `ml/verify_engine.py`, which reproduces the checkpoint's
recorded held-out test accuracy and its saved test-set probabilities.

Supported layers cover every architecture produced by `ml/train.py` and
`ml/architectures.py`: InputLayer, Rescaling, Random* (no-op), Conv2D,
DepthwiseConv2D, SeparableConv2D, BatchNormalization, Activation, ReLU,
MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, Flatten, Dropout,
SpatialDropout2D, Dense and Add.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


IDENTITY_AT_INFERENCE = {
    "InputLayer", "Dropout", "SpatialDropout2D", "GaussianNoise",
    "RandomTranslation", "RandomRotation", "RandomZoom", "RandomFlip", "RandomContrast", "RandomStrokeWidth",
}


class EngineError(ValueError):
    """Raised when an exported model cannot be replayed."""


def _activation(name: str | None, x: np.ndarray) -> np.ndarray:
    if name in (None, "linear"):
        return x
    if name == "relu":
        return np.maximum(x, 0.0)
    if name == "softmax":
        shifted = x - x.max(axis=-1, keepdims=True)
        e = np.exp(shifted)
        return e / e.sum(axis=-1, keepdims=True)
    if name == "sigmoid":
        return 1.0 / (1.0 + np.exp(-x))
    if name == "swish" or name == "silu":
        return x / (1.0 + np.exp(-x))
    if name == "gelu":
        return 0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))
    raise EngineError(f"Unsupported activation: {name}")


def _pad_same(x: np.ndarray, kh: int, kw: int, sh: int, sw: int) -> np.ndarray:
    """TensorFlow 'same' padding (extra pixel goes bottom/right)."""
    _, h, w, _ = x.shape
    out_h, out_w = -(-h // sh), -(-w // sw)
    pad_h = max((out_h - 1) * sh + kh - h, 0)
    pad_w = max((out_w - 1) * sw + kw - w, 0)
    top, left = pad_h // 2, pad_w // 2
    return np.pad(x, ((0, 0), (top, pad_h - top), (left, pad_w - left), (0, 0)))


def _windows(x: np.ndarray, kh: int, kw: int, sh: int, sw: int, padding: str) -> np.ndarray:
    if padding == "same":
        x = _pad_same(x, kh, kw, sh, sw)
    win = sliding_window_view(x, (kh, kw), axis=(1, 2))  # N, H', W', C, kh, kw
    return win[:, ::sh, ::sw]


def conv2d(x, kernel, bias, strides=(1, 1), padding="same"):
    kh, kw, cin, cout = kernel.shape
    win = _windows(x, kh, kw, strides[0], strides[1], padding)
    n, oh, ow = win.shape[:3]
    cols = np.ascontiguousarray(win).reshape(n * oh * ow, cin * kh * kw)
    # window layout is (C, kh, kw); reorder the kernel to match
    k = kernel.transpose(2, 0, 1, 3).reshape(cin * kh * kw, cout)
    out = cols @ k
    if bias is not None:
        out += bias
    return out.reshape(n, oh, ow, cout)


def depthwise_conv2d(x, kernel, bias, strides=(1, 1), padding="same"):
    kh, kw, cin, mult = kernel.shape
    win = _windows(x, kh, kw, strides[0], strides[1], padding)  # N,H,W,C,kh,kw
    out = np.einsum("nhwcij,ijcm->nhwcm", win, kernel, optimize=True)
    out = out.reshape(*out.shape[:3], cin * mult)
    if bias is not None:
        out += bias
    return out


def pool2d(x, size, strides, padding, mode):
    win = _windows(x, size[0], size[1], strides[0], strides[1], padding)
    return win.max(axis=(-2, -1)) if mode == "max" else win.mean(axis=(-2, -1))


@dataclass
class Layer:
    name: str
    kind: str
    config: dict
    inputs: list[str]
    weights: list[np.ndarray]


def _inbound_names(layer_cfg: dict) -> list[str]:
    nodes = layer_cfg.get("inbound_nodes") or []
    if not nodes:
        return []
    node = nodes[0]
    names: list[str] = []

    def collect(obj):
        if isinstance(obj, dict):
            if obj.get("class_name") == "__keras_tensor__":
                names.append(obj["config"]["keras_history"][0])
            else:
                for value in obj.values():
                    collect(value)
        elif isinstance(obj, (list, tuple)):
            # Keras 2 format: [[layer_name, node, tensor, kwargs], ...]
            if len(obj) >= 3 and isinstance(obj[0], str) and isinstance(obj[1], int):
                names.append(obj[0])
            else:
                for value in obj:
                    collect(value)

    collect(node.get("args", node) if isinstance(node, dict) else node)
    return names


class NumpyCNN:
    """Replays an exported Keras functional graph with NumPy."""

    def __init__(self, architecture: dict, weights: dict[str, list[np.ndarray]]):
        layers_cfg = architecture.get("layers")
        if not isinstance(layers_cfg, list) or not layers_cfg:
            raise EngineError("Architecture has no layers.")
        self.name = architecture.get("name", "model")
        self.layers: list[Layer] = []
        previous = None
        for cfg in layers_cfg:
            name = cfg["config"]["name"]
            inputs = _inbound_names(cfg)
            if not inputs and cfg["class_name"] != "InputLayer" and previous is not None:
                inputs = [previous]  # Sequential config
            self.layers.append(Layer(name, cfg["class_name"], cfg["config"], inputs, [
                np.asarray(w, dtype=np.float32) for w in weights.get(name, [])
            ]))
            previous = name
        outputs = architecture.get("output_layers")
        if outputs and isinstance(outputs[0], list):
            outputs = outputs[0]  # [[name, node, tensor]] (multi-output style)
        self.output_name = outputs[0] if outputs else self.layers[-1].name
        self.input_name = self.layers[0].name
        self.by_name = {layer.name: layer for layer in self.layers}
        for layer in self.layers:
            if layer.kind not in IDENTITY_AT_INFERENCE | {
                "Rescaling", "Conv2D", "DepthwiseConv2D", "SeparableConv2D", "BatchNormalization",
                "Activation", "ReLU", "MaxPooling2D", "AveragePooling2D", "GlobalAveragePooling2D",
                "Flatten", "Dense", "Add",
            }:
                raise EngineError(f"Layer {layer.name} has unsupported type {layer.kind}.")
        self.output_classes = self._infer_output_width()

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, weights_path: Path, arch_path: Path) -> "NumpyCNN":
        architecture = json.loads(Path(arch_path).read_text(encoding="utf-8"))
        with np.load(weights_path, allow_pickle=False) as data:
            grouped: dict[str, list[tuple[int, np.ndarray]]] = {}
            for key in data.files:
                layer, _, index = key.rpartition("/")
                grouped.setdefault(layer, []).append((int(index), data[key]))
        weights = {k: [arr for _, arr in sorted(v, key=lambda item: item[0])] for k, v in grouped.items()}
        return cls(architecture, weights)

    def _infer_output_width(self) -> int:
        out = self.by_name[self.output_name]
        if out.kind == "Dense":
            return int(out.config["units"])
        for layer in reversed(self.layers):
            if layer.kind == "Dense":
                return int(layer.config["units"])
        raise EngineError("Cannot determine output width.")

    @property
    def parameter_count(self) -> int:
        return int(sum(w.size for layer in self.layers for w in layer.weights))

    # ---------------------------------------------------------------- execution
    def _apply(self, layer: Layer, tensors: list[np.ndarray]) -> np.ndarray:
        cfg, w, kind = layer.config, layer.weights, layer.kind
        x = tensors[0]
        if kind in IDENTITY_AT_INFERENCE:
            return x.astype(np.float32) if kind == "InputLayer" else x
        if kind == "Rescaling":
            return x.astype(np.float32) * float(cfg.get("scale", 1.0)) + float(cfg.get("offset", 0.0))
        if kind == "Conv2D":
            if tuple(cfg.get("dilation_rate", (1, 1))) != (1, 1) or cfg.get("groups", 1) != 1:
                raise EngineError("Dilated or grouped convolutions are not supported.")
            bias = w[1] if cfg.get("use_bias", True) else None
            y = conv2d(x, w[0], bias, tuple(cfg["strides"]), cfg["padding"])
            return _activation(cfg.get("activation"), y)
        if kind == "DepthwiseConv2D":
            bias = w[1] if cfg.get("use_bias", True) else None
            y = depthwise_conv2d(x, w[0], bias, tuple(cfg["strides"]), cfg["padding"])
            return _activation(cfg.get("activation"), y)
        if kind == "SeparableConv2D":
            y = depthwise_conv2d(x, w[0], None, tuple(cfg["strides"]), cfg["padding"])
            bias = w[2] if cfg.get("use_bias", True) else None
            y = conv2d(y, w[1], bias, (1, 1), "valid")
            return _activation(cfg.get("activation"), y)
        if kind == "BatchNormalization":
            index = 0
            gamma = w[index] if cfg.get("scale", True) else None
            index += 1 if cfg.get("scale", True) else 0
            beta = w[index] if cfg.get("center", True) else None
            index += 1 if cfg.get("center", True) else 0
            mean, var = w[index], w[index + 1]
            y = (x - mean) / np.sqrt(var + float(cfg.get("epsilon", 1e-3)))
            if gamma is not None:
                y = y * gamma
            if beta is not None:
                y = y + beta
            return y
        if kind == "Activation":
            return _activation(cfg["activation"], x)
        if kind == "ReLU":
            return np.maximum(x, 0.0)
        if kind in ("MaxPooling2D", "AveragePooling2D"):
            size = tuple(cfg.get("pool_size", (2, 2)))
            strides = tuple(cfg.get("strides") or size)
            return pool2d(x, size, strides, cfg.get("padding", "valid"), "max" if kind == "MaxPooling2D" else "avg")
        if kind == "GlobalAveragePooling2D":
            return x.mean(axis=(1, 2))
        if kind == "Flatten":
            return x.reshape(x.shape[0], -1)
        if kind == "Dense":
            y = x @ w[0]
            if cfg.get("use_bias", True):
                y = y + w[1]
            return _activation(cfg.get("activation"), y)
        if kind == "Add":
            total = tensors[0]
            for other in tensors[1:]:
                total = total + other
            return total
        raise EngineError(f"Unsupported layer {kind}")

    def run(self, batch: np.ndarray, keep: tuple[str, ...] = ()) -> dict[str, np.ndarray]:
        """Forward pass. Returns the output plus any named intermediate tensors."""
        values: dict[str, np.ndarray] = {}
        for layer in self.layers:
            if layer.kind == "InputLayer":
                values[layer.name] = np.asarray(batch, dtype=np.float32)
                continue
            values[layer.name] = self._apply(layer, [values[name] for name in layer.inputs])
        result = {"output": values[self.output_name]}
        for name in keep:
            if name in values:
                result[name] = values[name]
        return result

    def predict(self, batch: np.ndarray, batch_size: int = 512) -> np.ndarray:
        parts = [self.run(batch[i:i + batch_size])["output"] for i in range(0, len(batch), batch_size)]
        return np.concatenate(parts, axis=0)

    # ----------------------------------------------------------------- Grad-CAM
    def gradcam_head(self) -> tuple[str, list[Layer]] | None:
        """Find `last spatial layer -> GAP -> Dense... -> softmax` for analytic Grad-CAM."""
        chain: list[Layer] = []
        layer = self.by_name[self.output_name]
        while layer.kind in ("Dense", "Dropout", "Activation", "ReLU"):
            chain.append(layer)
            if len(layer.inputs) != 1:
                return None
            layer = self.by_name[layer.inputs[0]]
        if layer.kind != "GlobalAveragePooling2D" or len(layer.inputs) != 1:
            return None
        return layer.inputs[0], list(reversed(chain))

    def gradcam(self, batch: np.ndarray, class_index: int | None = None) -> tuple[np.ndarray, np.ndarray] | None:
        """Exact Grad-CAM for GAP-headed CNNs, computed analytically (no autodiff).

        Because the head is GAP followed by dense layers, d(logit_c)/dA[i,j,k]
        is identical at every position: g_k / (H*W), where g is the gradient of
        the pre-softmax logit with respect to the pooled feature vector.
        Grad-CAM's channel weights are therefore g_k/(H*W) and the map is
        ReLU(sum_k w_k A_k). Returns (cam in [0,1] at feature resolution, probabilities).
        class_index=None explains the top-1 class (one forward pass in total).
        """
        head = self.gradcam_head()
        if head is None:
            return None
        feature_name, dense_chain = head
        out = self.run(batch[:1], keep=(feature_name,))
        if class_index is None:
            class_index = int(np.argmax(out["output"][0]))
        features = out[feature_name][0]  # H, W, K
        pooled = features.mean(axis=(0, 1))
        # forward through the dense chain recording pre-activations
        activations = [pooled]
        pre = []
        x = pooled
        dense_layers = [layer for layer in dense_chain if layer.kind == "Dense"]
        for layer in dense_layers:
            z = x @ layer.weights[0] + (layer.weights[1] if layer.config.get("use_bias", True) else 0.0)
            pre.append(z)
            x = _activation(layer.config.get("activation"), z) if layer is not dense_layers[-1] else z
            activations.append(x)
        # backward: gradient of logit c wrt pooled vector
        grad = np.zeros_like(pre[-1])
        grad[class_index] = 1.0
        for i in range(len(dense_layers) - 1, -1, -1):
            layer = dense_layers[i]
            if i != len(dense_layers) - 1:
                act = layer.config.get("activation")
                if act == "relu":
                    grad = grad * (pre[i] > 0)
                elif act not in (None, "linear"):
                    return None
            grad = layer.weights[0] @ grad
        h, w, _ = features.shape
        weights = grad / (h * w)
        cam = np.maximum((features * weights).sum(axis=-1), 0.0)
        peak = float(cam.max())
        cam = cam / peak if peak > 0 else cam
        return cam.astype(np.float32), out["output"][0]
