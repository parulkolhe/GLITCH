import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ml.metrics import (
    apply_temperature,
    calibration_bins,
    confusion_matrix,
    difficulty_tiers,
    fit_temperature,
    per_class_scores,
    top_confusions,
    top_k_accuracy,
)
from ml.prepare_data import CONFIG_PATH, load_config, split_indices
from ml.train import AUGMENTATION_PREFIXES, augmentation_layer_names, build_model, load_split


class DatasetConfigTests(unittest.TestCase):
    def test_default_config_uses_balanced_32_class_plan(self):
        config = load_config(CONFIG_PATH)
        self.assertEqual(len(config["classes"]), 32)
        self.assertEqual(len(set(config["classes"])), 32)
        self.assertEqual(config["max_examples_per_class"], 10_000)
        self.assertEqual(config["split"], {"train": 0.8, "validation": 0.1, "test": 0.1})

    def test_split_indices_are_reproducible_and_disjoint(self):
        fractions = {"train": 0.8, "validation": 0.1, "test": 0.1}
        first = split_indices(10_000, fractions, np.random.default_rng(42))
        second = split_indices(10_000, fractions, np.random.default_rng(42))
        self.assertEqual({key: len(value) for key, value in first.items()}, {
            "train": 8_000, "validation": 1_000, "test": 1_000,
        })
        self.assertTrue(all(np.array_equal(first[key], second[key]) for key in first))
        sets = [set(values.tolist()) for values in first.values()]
        self.assertFalse(sets[0] & sets[1])
        self.assertFalse(sets[0] & sets[2])
        self.assertFalse(sets[1] & sets[2])
        self.assertEqual(len(set.union(*sets)), 10_000)


class SplitValidationTests(unittest.TestCase):
    def test_load_split_accepts_expected_bitmap_format(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.npz"
            np.savez_compressed(path, images=np.zeros((3, 28, 28), dtype=np.uint8), labels=np.array([0, 1, 2], dtype=np.int16))
            images, labels = load_split(path, expected_classes=3)
            self.assertEqual(images.shape, (3, 28, 28, 1))
            self.assertEqual(images.dtype, np.uint8)
            self.assertEqual(labels.tolist(), [0, 1, 2])

    def test_load_split_rejects_invalid_image_size(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.npz"
            np.savez_compressed(path, images=np.zeros((1, 20, 20), dtype=np.uint8), labels=np.array([0], dtype=np.int16))
            with self.assertRaises(ValueError):
                load_split(path, expected_classes=2)


HAS_TF = importlib.util.find_spec("tensorflow") is not None


@unittest.skipUnless(HAS_TF, "TensorFlow is not installed (only needed for training)")
class ModelTests(unittest.TestCase):
    def test_saved_model_pipeline_normalizes_raw_bitmap_bytes(self):
        import tensorflow as tf

        model = build_model(num_classes=3, seed=42)
        normalizer = tf.keras.Model(model.input, model.get_layer("normalize_0_1").output)
        sample = np.zeros((1, 28, 28, 1), dtype=np.uint8)
        sample[0, 0, 0, 0] = 255
        normalized = normalizer(sample, training=False).numpy()
        self.assertEqual(float(normalized[0, 1, 1, 0]), 0.0)
        self.assertEqual(float(normalized[0, 0, 0, 0]), 1.0)
        probabilities = model(sample, training=False).numpy()
        self.assertEqual(probabilities.shape, (1, 3))
        self.assertAlmostEqual(float(probabilities.sum()), 1.0, places=5)
        self.assertEqual(model.get_layer("embedding").units, 128)

    def test_augmentation_is_inactive_at_inference(self):
        model = build_model(num_classes=3, seed=1)
        sample = np.random.default_rng(0).integers(0, 255, (2, 28, 28, 1), dtype=np.uint8)
        np.testing.assert_allclose(model(sample, training=False).numpy(), model(sample, training=False).numpy())

    def test_resuming_controls_augmentation_and_keeps_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "base.keras"
            base = build_model(num_classes=3, seed=3)
            base.save(path)
            plain = build_model(num_classes=3, seed=3, initial_model=path, augment_finetuning=False)
            augmented = build_model(num_classes=3, seed=3, initial_model=path, augment_finetuning=True)
            self.assertEqual(augmentation_layer_names(plain), [], "no-augmentation fine-tuning must have no augmentation layers")
            self.assertEqual(len(augmentation_layer_names(augmented)), 3, "augmentation must be applied once, not twice")
            self.assertTrue(all(name.startswith(AUGMENTATION_PREFIXES) for name in augmentation_layer_names(augmented)))
            np.testing.assert_array_equal(plain.get_layer("block1_conv").get_weights()[0], base.get_layer("block1_conv").get_weights()[0])


class MetricTests(unittest.TestCase):
    def test_per_class_scores_and_confusions_on_a_hand_checked_example(self):
        labels = np.array([0, 0, 0, 1, 1, 2])
        predicted = np.array([0, 0, 1, 1, 0, 2])
        matrix = confusion_matrix(labels, predicted, 3)
        np.testing.assert_array_equal(matrix, [[2, 1, 0], [1, 1, 0], [0, 0, 1]])
        rows = {row["class"]: row for row in per_class_scores(matrix, ["a", "b", "c"])}
        self.assertAlmostEqual(rows["a"]["precision"], 2 / 3)
        self.assertAlmostEqual(rows["a"]["recall"], 2 / 3)
        self.assertAlmostEqual(rows["b"]["f1"], 0.5)  # P = 1/2, R = 1/2
        self.assertAlmostEqual(rows["c"]["f1"], 1.0)
        self.assertEqual(top_confusions(matrix, ["a", "b", "c"])[0]["count"], 1)

    def test_top_k_accuracy(self):
        probabilities = np.array([[0.6, 0.3, 0.1], [0.5, 0.3, 0.2]])
        self.assertEqual(top_k_accuracy(probabilities, np.array([1, 2]), 1), 0.0)
        self.assertEqual(top_k_accuracy(probabilities, np.array([1, 2]), 2), 0.5)
        self.assertEqual(top_k_accuracy(probabilities, np.array([1, 2]), 3), 1.0)

    def test_calibration_error_is_zero_when_confidence_matches_accuracy(self):
        # 80% confident everywhere and right 80% of the time.
        probabilities = np.tile([0.8, 0.2], (100, 1))
        labels = np.array([0] * 80 + [1] * 20)
        ece, _ = calibration_bins(probabilities, labels)
        self.assertAlmostEqual(ece, 0.0, places=6)

    def test_temperature_fit_softens_an_overconfident_model(self):
        rng = np.random.default_rng(0)
        labels = rng.integers(0, 4, 2000)
        correct = rng.random(2000) < 0.7                       # right 70% of the time...
        predicted = np.where(correct, labels, (labels + 1) % 4)
        probabilities = np.full((2000, 4), 0.01)
        probabilities[np.arange(2000), predicted] = 0.97        # ...but 97% confident
        temperature = fit_temperature(probabilities, labels)
        self.assertGreater(temperature, 1.0)
        before, _ = calibration_bins(probabilities, labels)
        after, _ = calibration_bins(apply_temperature(probabilities, temperature), labels)
        self.assertLess(after, before)
        np.testing.assert_array_equal(apply_temperature(probabilities, temperature).argmax(1), probabilities.argmax(1))

    def test_difficulty_tiers_split_classes_into_thirds(self):
        rows = [{"class": name, "f1": score} for name, score in zip("abcdef", [0.9, 0.8, 0.7, 0.6, 0.5, 0.4])]
        tiers = difficulty_tiers(rows)
        self.assertEqual([tiers[name] for name in "abcdef"], ["easy", "easy", "medium", "medium", "hard", "hard"])


if __name__ == "__main__":
    unittest.main()
