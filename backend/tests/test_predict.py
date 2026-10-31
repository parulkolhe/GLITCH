"""Preprocessing, scoring, engine and API tests. No TensorFlow required.

The API tests build a tiny random model in the exported NumPy format, so they
run anywhere. Tests marked "trained model" run only when the real weights are
present in backend/model.
"""

import base64
import io
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

import backend.main as main_module
from backend.ml.engine import NumpyCNN, conv2d
from backend.ml.model import DEFAULT_MODEL_DIR, load_bundle, temperature_scale, top_predictions
from backend.ml.preprocessing import DrawingInputError, preprocess_png


def encode_png(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def circle_drawing(colour="#211b2b", width=8, offset=(0, 0), size=(600, 600), alpha=255) -> str:
    base = Image.new("RGBA", size, (255, 255, 255, 255))
    layer = Image.new("RGBA", size, (0, 0, 0, 0))
    rgb = tuple(int(colour[i:i + 2], 16) for i in (1, 3, 5))
    dx, dy = offset
    ImageDraw.Draw(layer).ellipse((150 + dx, 150 + dy, 350 + dx, 350 + dy), outline=rgb + (alpha,), width=width)
    return encode_png(Image.alpha_composite(base, layer).convert("RGB"))


# ------------------------------------------------------------ tiny test model
def _layer(cls, name, inbound, **config):
    entry = {"class_name": cls, "config": {"name": name, **config}, "name": name}
    entry["inbound_nodes"] = [] if inbound is None else [{"args": [
        {"class_name": "__keras_tensor__", "config": {"keras_history": [inbound, 0, 0]}}], "kwargs": {}}]
    return entry


def tiny_architecture(num_classes: int) -> dict:
    return {"name": "tiny_cnn", "layers": [
        _layer("InputLayer", "bitmap", None, batch_shape=[None, 28, 28, 1]),
        _layer("Rescaling", "normalize_0_1", "bitmap", scale=1 / 255, offset=0.0),
        _layer("RandomTranslation", "augment_translation", "normalize_0_1"),
        _layer("Conv2D", "block1_conv", "augment_translation", filters=4, kernel_size=[3, 3], strides=[1, 1],
               padding="same", activation="relu", use_bias=True),
        _layer("BatchNormalization", "block1_bn", "block1_conv", epsilon=1e-3, scale=True, center=True),
        _layer("MaxPooling2D", "block1_pool", "block1_bn", pool_size=[2, 2], strides=[2, 2], padding="valid"),
        _layer("Conv2D", "block3_conv", "block1_pool", filters=6, kernel_size=[3, 3], strides=[1, 1],
               padding="same", activation="relu", use_bias=True),
        _layer("GlobalAveragePooling2D", "global_avg_pool", "block3_conv"),
        _layer("Dropout", "dropout_features", "global_avg_pool", rate=0.3),
        _layer("Dense", "embedding", "dropout_features", units=8, activation="relu", use_bias=True),
        _layer("Dense", "class_probabilities", "embedding", units=num_classes, activation="softmax", use_bias=True),
    ], "output_layers": ["class_probabilities", 0, 0]}


def tiny_weights(num_classes: int, seed: int = 3) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "block1_conv/0": rng.normal(0, 0.5, (3, 3, 1, 4)), "block1_conv/1": np.zeros(4),
        "block1_bn/0": np.ones(4), "block1_bn/1": np.zeros(4), "block1_bn/2": np.zeros(4), "block1_bn/3": np.ones(4),
        "block3_conv/0": rng.normal(0, 0.5, (3, 3, 4, 6)), "block3_conv/1": np.zeros(6),
        "embedding/0": rng.normal(0, 0.5, (6, 8)), "embedding/1": np.zeros(8),
        "class_probabilities/0": rng.normal(0, 0.5, (8, num_classes)), "class_probabilities/1": np.zeros(num_classes),
    }


def build_model_dir(directory: Path, class_names: list[str], temperature: float = 1.5, uncertain: float = 0.6) -> None:
    np.savez(directory / "doodle_classifier.weights.npz", **{k: v.astype(np.float32) for k, v in tiny_weights(len(class_names)).items()})
    (directory / "doodle_classifier.arch.json").write_text(json.dumps(tiny_architecture(len(class_names))))
    (directory / "class_names.json").write_text(json.dumps(class_names))
    (directory / "calibration.json").write_text(json.dumps({"temperature": temperature}))
    (directory / "confidence_policy.json").write_text(json.dumps({"uncertain_below": uncertain}))
    tiers = {name: ("easy", "medium", "hard")[i % 3] for i, name in enumerate(class_names)}
    (directory / "class_difficulty.json").write_text(json.dumps({"tiers": tiers}))


# ---------------------------------------------------------------- the tests
class PreprocessingTests(unittest.TestCase):
    def test_output_matches_model_input_format(self):
        tensor = preprocess_png(circle_drawing())
        self.assertEqual(tensor.shape, (1, 28, 28, 1))
        self.assertEqual(tensor.dtype, np.uint8)
        self.assertEqual(int(tensor.max()), 255, "strongest stroke pixel should be full intensity, like the dataset")
        self.assertEqual(int(tensor[0, 0, 0, 0]), 0, "background should be black after inversion")

    def test_ink_colour_and_pencil_opacity_do_not_change_the_bitmap(self):
        black = preprocess_png(circle_drawing("#211b2b"))
        np.testing.assert_array_equal(black, preprocess_png(circle_drawing("#c5ff65")))
        np.testing.assert_array_equal(black, preprocess_png(circle_drawing("#c5ff65", alpha=166)))

    def test_position_on_canvas_does_not_change_the_bitmap(self):
        np.testing.assert_array_equal(preprocess_png(circle_drawing()), preprocess_png(circle_drawing(offset=(180, 120))))

    def test_thin_brush_survives_downsampling(self):
        self.assertGreater(int((preprocess_png(circle_drawing(width=2)) > 64).sum()), 60)

    def test_transparent_canvas_is_composited_on_white(self):
        image = Image.new("RGBA", (200, 200), (0, 0, 0, 0))
        ImageDraw.Draw(image).line((40, 40, 160, 160), fill=(0, 0, 0, 255), width=8)
        self.assertLess(int((preprocess_png(encode_png(image)) > 0).sum()), 28 * 28 // 2)

    def test_data_url_prefix_is_accepted(self):
        self.assertEqual(preprocess_png("data:image/png;base64," + circle_drawing()).shape, (1, 28, 28, 1))

    def test_blank_and_invalid_inputs_are_rejected(self):
        with self.assertRaises(DrawingInputError):
            preprocess_png(encode_png(Image.new("RGB", (128, 128), "white")))
        with self.assertRaises(DrawingInputError):
            preprocess_png("not base64!!")
        jpeg = io.BytesIO()
        Image.new("RGB", (32, 32), "black").save(jpeg, format="JPEG")
        with self.assertRaises(DrawingInputError):
            preprocess_png(base64.b64encode(jpeg.getvalue()).decode())


class ScoreTests(unittest.TestCase):
    def test_predictions_are_sorted_and_labeled(self):
        result = top_predictions(np.array([0.1, 0.7, 0.2]), ["cat", "dog", "fish"])
        self.assertEqual([row["class_name"] for row in result], ["dog", "fish", "cat"])
        self.assertAlmostEqual(sum(row["score"] for row in result), 1.0, places=5)

    def test_temperature_scaling_keeps_ranking(self):
        probabilities = np.array([0.05, 0.9, 0.05])
        np.testing.assert_allclose(temperature_scale(probabilities, 1.0), probabilities)
        softened = temperature_scale(probabilities, 2.0)
        self.assertAlmostEqual(float(softened.sum()), 1.0, places=6)
        self.assertLess(softened[1], 0.9)
        self.assertEqual(int(softened.argmax()), 1)

    def test_invalid_score_vectors_are_rejected(self):
        with self.assertRaises(ValueError):
            top_predictions(np.array([0.5, 0.2]), ["a", "b", "c"])
        with self.assertRaises(ValueError):
            top_predictions(np.array([0.9, 0.9]), ["a", "b"])


class EngineTests(unittest.TestCase):
    def test_conv2d_matches_a_direct_loop(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(1, 5, 6, 2)).astype(np.float32)
        k = rng.normal(size=(3, 3, 2, 3)).astype(np.float32)
        out = conv2d(x, k, None, (1, 1), "same")
        padded = np.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)))
        expected = np.zeros((1, 5, 6, 3), np.float32)
        for i in range(5):
            for j in range(6):
                expected[0, i, j] = np.tensordot(padded[0, i:i + 3, j:j + 3, :], k, axes=([0, 1, 2], [0, 1, 2]))
        np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-5)

    def test_engine_outputs_a_softmax_and_gradcam_map(self):
        weights: dict = {}
        for key, value in sorted(tiny_weights(4).items()):
            weights.setdefault(key.split("/")[0], []).append(value)
        engine = NumpyCNN(tiny_architecture(4), weights)
        batch = preprocess_png(circle_drawing())
        p = engine.run(batch)["output"]
        self.assertEqual(p.shape, (1, 4))
        self.assertAlmostEqual(float(p.sum()), 1.0, places=5)
        cam, probs = engine.gradcam(batch)
        self.assertEqual(cam.shape, (14, 14))
        self.assertTrue(0.0 <= float(cam.min()) and float(cam.max()) <= 1.0)
        np.testing.assert_allclose(probs, p[0], rtol=1e-6)

    def test_gradcam_matches_numerical_gradient(self):
        """Analytic channel weights equal finite-difference gradients of the logit."""
        weights: dict = {}
        for key, value in sorted(tiny_weights(4, seed=11).items()):
            weights.setdefault(key.split("/")[0], []).append(value.astype(np.float64))
        engine = NumpyCNN(tiny_architecture(4), weights)
        emb_w, emb_b = engine.by_name["embedding"].weights
        out_w, out_b = engine.by_name["class_probabilities"].weights
        rng = np.random.default_rng(5)
        pooled = np.abs(rng.normal(size=6))
        logit = lambda g: (np.maximum(g @ emb_w + emb_b, 0) @ out_w + out_b)[2]
        numeric = np.array([(logit(pooled + e * 1e-5) - logit(pooled - e * 1e-5)) / 2e-5 for e in np.eye(6)])
        analytic = emb_w @ (out_w[:, 2] * ((pooled @ emb_w + emb_b) > 0))
        np.testing.assert_allclose(analytic, numeric, rtol=1e-4, atol=1e-6)


class ApiTests(unittest.TestCase):
    CLASSES = ["cat", "dog", "fish", "house"]

    @classmethod
    def setUpClass(cls):
        cls.tempdir = tempfile.TemporaryDirectory()
        build_model_dir(Path(cls.tempdir.name), cls.CLASSES)
        cls.original_dir = main_module.MODEL_DIR
        main_module.MODEL_DIR = Path(cls.tempdir.name)
        cls.client = TestClient(main_module.app)
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)
        main_module.MODEL_DIR = cls.original_dir
        cls.tempdir.cleanup()

    def test_health_ready_and_model_info(self):
        self.assertEqual(self.client.get("/health/ready").status_code, 200)
        info = self.client.get("/api/model").json()
        self.assertEqual(info["classes"], self.CLASSES)
        self.assertEqual(info["temperature"], 1.5)
        self.assertEqual(info["uncertain_below"], 0.6)
        self.assertEqual(set(info["difficulty"].values()) <= {"easy", "medium", "hard"}, True)
        self.assertIn("never updates", info["note"])

    def test_predict_returns_ranked_calibrated_scores(self):
        response = self.client.post("/api/predict", json={"image_base64": circle_drawing(), "target": "fish", "request_id": "abc"})
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        scores = [row["score"] for row in body["predictions"]]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertEqual(len(body["predictions"]), 4)  # top-5 capped at class count
        self.assertAlmostEqual(sum(scores), 1.0, places=4)
        self.assertEqual(body["request_id"], "abc")
        self.assertEqual(body["target"], "fish")
        self.assertIn(body["target_rank"], range(1, 5))
        self.assertEqual(body["uncertain"], scores[0] < 0.6)
        self.assertTrue(body["model_input"].startswith("data:image/png;base64,"))
        self.assertEqual(len(body["attention"]), 14)
        self.assertGreater(body["inference_ms"], 0)

    def test_same_drawing_gives_identical_scores(self):
        a = self.client.post("/api/predict", json={"image_base64": circle_drawing()}).json()
        b = self.client.post("/api/predict", json={"image_base64": circle_drawing(offset=(60, 40))}).json()
        self.assertEqual(a["predictions"], b["predictions"])

    def test_bad_inputs_are_rejected(self):
        blank = encode_png(Image.new("RGB", (128, 128), "white"))
        self.assertEqual(self.client.post("/api/predict", json={"image_base64": blank}).status_code, 422)
        self.assertEqual(self.client.post("/api/predict", json={"image_base64": "%%%"}).status_code, 422)
        self.assertEqual(self.client.post("/api/predict", json={}).status_code, 422)
        self.assertEqual(self.client.post("/api/predict", json={"image_base64": circle_drawing(), "target": "unicorn"}).status_code, 422)
        too_big = self.client.post("/api/predict", content=b"{" + b" " * 3_600_000 + b"}", headers={"content-type": "application/json"})
        self.assertEqual(too_big.status_code, 413)

    def test_legacy_routes_still_answer(self):
        self.assertEqual(self.client.post("/predict", json={"image_base64": circle_drawing()}).status_code, 200)
        self.assertEqual(set(self.client.get("/game/meta").json()["difficulty"]), set(self.CLASSES))


class MissingModelTests(unittest.TestCase):
    def test_api_reports_unavailable_instead_of_guessing(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "class_names.json").write_text(json.dumps(["a", "b"]))
            original = main_module.MODEL_DIR
            main_module.MODEL_DIR = Path(directory)
            try:
                with TestClient(main_module.app) as client:
                    self.assertFalse(client.get("/health").json()["model_ready"])
                    self.assertEqual(client.get("/health/ready").status_code, 503)
                    self.assertEqual(client.post("/api/predict", json={"image_base64": circle_drawing()}).status_code, 503)
            finally:
                main_module.MODEL_DIR = original


def _has_trained_model() -> bool:
    try:
        load_bundle(DEFAULT_MODEL_DIR)
        return True
    except Exception:
        return False


@unittest.skipUnless(_has_trained_model(), "trained model not present")
class TrainedModelTests(unittest.TestCase):
    """End-to-end recognition of simple synthetic drawings through the real pipeline."""

    @classmethod
    def setUpClass(cls):
        main_module.MODEL_DIR = DEFAULT_MODEL_DIR
        cls.client = TestClient(main_module.app)
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)

    def _top(self, draw_fn, target):
        image = Image.new("RGB", (600, 600), "white")
        draw_fn(ImageDraw.Draw(image))
        body = self.client.post("/api/predict", json={"image_base64": encode_png(image), "target": target}).json()
        return body

    def test_house(self):
        body = self._top(lambda d: (d.rectangle((180, 280, 420, 480), outline="black", width=10),
                                    d.line([(160, 290), (300, 150), (440, 290)], fill="black", width=10)), "house")
        self.assertEqual(body["predictions"][0]["class_name"], "house")

    def test_star(self):
        points = [(300 + 200 * math.cos(math.radians(-90 + i * 144)), 300 + 200 * math.sin(math.radians(-90 + i * 144))) for i in range(6)]
        self.assertEqual(self._top(lambda d: d.line(points, fill="black", width=10), "star")["predictions"][0]["class_name"], "star")


if __name__ == "__main__":
    unittest.main()
