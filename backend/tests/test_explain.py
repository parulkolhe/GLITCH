"""Uncertainty and Grad-CAM helpers. Needs only NumPy and Pillow (no TensorFlow)."""

import base64
import io
import unittest

import numpy as np
from PIL import Image

from backend.ml.explain import OVERLAY_SIZE, focus_region, overlay_png, uncertainty


NAMES = ["cat", "dog", "fish", "house"]


class UncertaintyTests(unittest.TestCase):
    def test_confident_prediction_has_low_uncertainty(self):
        result = uncertainty(np.array([0.94, 0.03, 0.02, 0.01]), NAMES)
        self.assertEqual(result["level"], "low")
        self.assertEqual(result["top_two"], ["cat", "dog"])
        self.assertAlmostEqual(result["margin"], 0.91, places=4)
        self.assertLess(result["entropy"], 0.25)

    def test_torn_prediction_has_high_uncertainty(self):
        result = uncertainty(np.array([0.05, 0.46, 0.04, 0.45]), NAMES)
        self.assertEqual(result["level"], "high")
        self.assertEqual(result["top_two"], ["dog", "house"])
        self.assertLess(result["margin"], 0.15)

    def test_uniform_scores_have_entropy_one(self):
        result = uncertainty(np.full(4, 0.25), NAMES)
        self.assertAlmostEqual(result["entropy"], 1.0, places=4)
        self.assertEqual(result["level"], "high")


class GradCamHelperTests(unittest.TestCase):
    def test_focus_region_names_where_the_heat_is(self):
        heat = np.zeros((7, 7))
        heat[0, 0] = 1
        self.assertEqual(focus_region(heat), "the top-left of the drawing")
        heat = np.zeros((7, 7))
        heat[3, 3] = 1
        self.assertEqual(focus_region(heat), "the centre of the drawing")
        heat = np.zeros((7, 7))
        heat[6, 3] = 1
        self.assertEqual(focus_region(heat), "the bottom of the drawing")
        self.assertEqual(focus_region(np.zeros((7, 7))), "the whole drawing")

    def test_overlay_is_a_png_data_url_of_the_expected_size(self):
        bitmap = np.zeros((28, 28, 1), dtype=np.uint8)
        bitmap[10:18, 10:18, 0] = 255
        heat = np.zeros((7, 7))
        heat[3, 3] = 1.0
        url = overlay_png(bitmap, heat)
        self.assertTrue(url.startswith("data:image/png;base64,"))
        image = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
        self.assertEqual(image.size, (OVERLAY_SIZE, OVERLAY_SIZE))
        pixels = np.asarray(image.convert("RGB"))
        corner, centre = pixels[2, 2], pixels[OVERLAY_SIZE // 2, OVERLAY_SIZE // 2]
        self.assertEqual(tuple(corner), (255, 255, 255), "no stroke and no heat stays white paper")
        self.assertGreater(int(centre[0]), int(centre[2]), "hot stroke pixels are tinted red/yellow")


if __name__ == "__main__":
    unittest.main()
