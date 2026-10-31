"""Engine, export, confidence-policy and canvas-simulation tests (NumPy only)."""

import json
import unittest
from pathlib import Path

import numpy as np

from backend.ml.engine import NumpyCNN, conv2d, depthwise_conv2d, pool2d
from backend.ml.preprocessing import ink_to_bitmap
from ml.confidence import choose_uncertain_threshold, risk_coverage
from ml.export_weights import map_h5_to_layers, snake_case

ROOT = Path(__file__).resolve().parents[2]


def layer(cls, name, inbound, **config):
    entry = {"class_name": cls, "config": {"name": name, **config}}
    if inbound is None:
        entry["inbound_nodes"] = []
    else:
        inbound = inbound if isinstance(inbound, list) else [inbound]
        args = [{"class_name": "__keras_tensor__", "config": {"keras_history": [n, 0, 0]}} for n in inbound]
        entry["inbound_nodes"] = [{"args": [args] if len(args) > 1 else args, "kwargs": {}}]
    return entry


class EngineLayerTests(unittest.TestCase):
    def test_same_padding_with_stride_two_matches_tensorflow_shape_rule(self):
        x = np.random.default_rng(0).normal(size=(1, 7, 7, 2)).astype(np.float32)
        out = conv2d(x, np.ones((3, 3, 2, 4), np.float32), None, (2, 2), "same")
        self.assertEqual(out.shape, (1, 4, 4, 4))  # ceil(7 / 2)
        self.assertEqual(pool2d(np.zeros((1, 7, 7, 3)), (2, 2), (2, 2), "valid", "max").shape, (1, 3, 3, 3))

    def test_depthwise_conv_is_a_per_channel_convolution(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=(1, 6, 6, 3)).astype(np.float32)
        k = rng.normal(size=(3, 3, 3, 1)).astype(np.float32)
        out = depthwise_conv2d(x, k, None)
        for c in range(3):
            single = conv2d(x[..., c:c + 1], k[:, :, c:c + 1, :], None)
            np.testing.assert_allclose(out[..., c:c + 1], single, rtol=1e-5, atol=1e-5)

    def test_residual_separable_graph_runs_and_matches_manual_computation(self):
        rng = np.random.default_rng(2)
        arch = {"name": "tiny_res", "output_layers": ["out", 0, 0], "layers": [
            layer("InputLayer", "bitmap", None, batch_shape=[None, 8, 8, 1]),
            layer("Rescaling", "norm", "bitmap", scale=1 / 255, offset=0.0),
            layer("Conv2D", "stem", "norm", filters=4, kernel_size=[3, 3], strides=[1, 1], padding="same", activation="relu", use_bias=False),
            layer("SeparableConv2D", "sep", "stem", filters=4, kernel_size=[3, 3], strides=[1, 1], padding="same", activation="linear", use_bias=True),
            layer("Add", "add", ["sep", "stem"]),
            layer("ReLU", "relu", "add"),
            layer("GlobalAveragePooling2D", "gap", "relu"),
            layer("Dense", "out", "gap", units=3, activation="softmax", use_bias=True),
        ]}
        w = {
            "stem": [rng.normal(size=(3, 3, 1, 4))],
            "sep": [rng.normal(size=(3, 3, 4, 1)), rng.normal(size=(1, 1, 4, 4)), rng.normal(size=4)],
            "out": [rng.normal(size=(4, 3)), np.zeros(3)],
        }
        engine = NumpyCNN(arch, w)
        x = rng.integers(0, 255, (2, 8, 8, 1)).astype(np.uint8)
        p = engine.run(x)["output"]
        np.testing.assert_allclose(p.sum(axis=1), 1.0, rtol=1e-6)
        s = np.maximum(conv2d(x / 255, w["stem"][0], None), 0)
        sep = conv2d(depthwise_conv2d(s, w["sep"][0], None), w["sep"][1], w["sep"][2], (1, 1), "valid")
        g = np.maximum(sep + s, 0).mean(axis=(1, 2))
        z = g @ w["out"][0]
        e = np.exp(z - z.max(axis=1, keepdims=True))
        np.testing.assert_allclose(p, e / e.sum(axis=1, keepdims=True), rtol=1e-5, atol=1e-6)


class ExportTests(unittest.TestCase):
    def test_keras_auto_names_are_reproduced(self):
        self.assertEqual(snake_case("Conv2D"), "conv2d")
        self.assertEqual(snake_case("BatchNormalization"), "batch_normalization")
        self.assertEqual(snake_case("GlobalAveragePooling2D"), "global_average_pooling2d")
        self.assertEqual(snake_case("SeparableConv2D"), "separable_conv2d")

    def test_h5_names_map_to_config_layers_in_order(self):
        arch = {"layers": [
            {"class_name": "Conv2D", "config": {"name": "block1_conv"}},
            {"class_name": "Conv2D", "config": {"name": "block2_conv"}},
            {"class_name": "Dense", "config": {"name": "embedding"}},
        ]}
        flat = {"layers/conv2d/vars/0": np.ones(1), "layers/conv2d_1/vars/0": np.full(1, 2.0), "layers/dense/vars/1": np.zeros(2),
                "layers/dense/vars/0": np.zeros(3)}
        mapped = map_h5_to_layers(arch, flat)
        self.assertEqual(float(mapped["block2_conv"][0][0]), 2.0)
        self.assertEqual([a.shape for a in mapped["embedding"]], [(3,), (2,)], "vars are ordered by index")

    @unittest.skipUnless((ROOT / "backend/model/doodle_classifier.weights.npz").exists(), "exported weights not present")
    def test_exported_baseline_has_the_documented_parameter_count(self):
        engine = NumpyCNN.load(ROOT / "backend/model/doodle_classifier.weights.npz", ROOT / "backend/model/doodle_classifier.arch.json")
        self.assertEqual(engine.parameter_count, 91136)
        self.assertEqual(engine.output_classes, 32)


class ConfidencePolicyTests(unittest.TestCase):
    def test_threshold_sits_where_low_confidence_predictions_are_mostly_wrong(self):
        rng = np.random.default_rng(0)
        n = 4000
        conf = rng.uniform(0.2, 1.0, n)
        correct = rng.random(n) < conf          # accuracy rises with confidence (calibrated)
        labels = np.zeros(n, int)
        probs = np.zeros((n, 2))
        probs[np.arange(n), np.where(correct, 0, 1)] = conf
        probs[np.arange(n), np.where(correct, 1, 0)] = 1 - conf
        policy = choose_uncertain_threshold(probs, labels)
        self.assertLess(policy["accuracy_below_threshold"], 0.5)
        self.assertGreater(policy["uncertain_below"], 0.3)
        curve = risk_coverage(probs, labels)
        self.assertEqual(curve[0]["coverage"], 1.0)
        self.assertGreaterEqual(curve[60]["selective_accuracy"], curve[0]["selective_accuracy"])


class PreprocessingCoreTests(unittest.TestCase):
    def test_margin_controls_how_much_of_the_frame_the_drawing_fills(self):
        ink = np.zeros((200, 200), bool)
        ink[50:150, 50:52] = True
        ink[50:52, 50:150] = True
        tight = ink_to_bitmap(ink, margin=0.02, dilation=5)[0, ..., 0] > 0
        loose = ink_to_bitmap(ink, margin=0.2, dilation=5)[0, ..., 0] > 0
        span = lambda m: np.ptp(np.where(m.any(axis=0))[0])
        self.assertGreater(span(tight), span(loose))


class CanvasSimulationTests(unittest.TestCase):
    def test_simulated_canvas_drawing_has_pen_width_strokes(self):
        from ml.canvas_sim import canvas_mask

        bitmap = np.zeros((28, 28), np.uint8)
        bitmap[6:22, 13:15] = 255   # a vertical bar
        mask = canvas_mask(bitmap, np.random.default_rng(0))
        self.assertEqual(mask.shape, (600, 600))
        rows = np.where(mask.any(axis=1))[0]
        self.assertGreater(np.ptp(rows), 150, "the bar is drawn at canvas scale")
        widths = mask[rows[len(rows) // 2]].sum()
        self.assertLess(widths, 30, "strokes are pen-width, not upscaled-bitmap width")


if __name__ == "__main__":
    unittest.main()
