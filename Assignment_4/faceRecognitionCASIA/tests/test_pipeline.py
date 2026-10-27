import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import config as conf
from dataset import FaceListDataset, make_verification_pairs
from evaluation import cosin_metric, read_threshold, save_threshold, threshold_search
from face_verification import FaceVerifier, NoFaceError, align_face
import test as lfw_test


class DataAndScoringTests(unittest.TestCase):
    def test_casia_splits_and_pair_labels(self):
        train = FaceListDataset(conf.train_list, conf.train_root, conf.test_transform)
        val = FaceListDataset(conf.val_list, conf.train_root, conf.test_transform)
        self.assertEqual(len(train), 433041)
        self.assertEqual(len(val), 21150)
        self.assertFalse({path for path, _ in train.samples} & {path for path, _ in val.samples})
        pairs = make_verification_pairs(val)
        self.assertEqual(len(pairs), 21150)
        self.assertEqual([sum(y == label for _, _, y in pairs) for label in (0, 1)], [10575, 10575])

    def test_threshold_search_matches_brute_force(self):
        rng = np.random.default_rng(7)
        for _ in range(10):
            scores = rng.integers(-2, 4, size=100).astype(float)
            labels = rng.integers(0, 2, size=100)
            accuracy, threshold = threshold_search(scores, labels)
            candidates = np.r_[np.unique(scores), np.nextafter(scores.max(), np.inf)]
            expected = max(np.mean((scores >= candidate) == labels) for candidate in candidates)
            self.assertAlmostEqual(accuracy, expected)
            self.assertAlmostEqual(float(np.mean((scores >= threshold) == labels)), accuracy)

    def test_cosine_and_checkpoint_bound_threshold(self):
        self.assertAlmostEqual(cosin_metric([1, 0], [1, 1]), 2 ** -0.5)
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "weights.pth"
            checkpoint.write_bytes(b"test checkpoint")
            original_path = conf.validation_threshold
            try:
                conf.validation_threshold = str(Path(temporary) / "threshold.json")
                save_threshold(conf, checkpoint, 0.42, validation_accuracy=0.9)
                self.assertEqual(read_threshold(conf, checkpoint, "aligned"), 0.42)
                checkpoint.write_bytes(b"different checkpoint")
                with self.assertRaisesRegex(ValueError, "checkpoint_sha256"):
                    read_threshold(conf, checkpoint, "aligned")
            finally:
                conf.validation_threshold = original_path

    def test_alignment_maps_reference_landmarks_to_output(self):
        reference = np.array([
            [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
            [41.5493, 92.3655], [70.7299, 92.2041],
        ]) * (128 / 112)
        pixels = np.zeros((160, 160, 3), dtype=np.uint8)
        pixels[:, :, 0] = np.arange(160)[None, :]
        pixels[:, :, 1] = np.arange(160)[:, None]
        image = Image.fromarray(pixels)
        aligned = align_face(image, reference + [10, 8], (128, 128))
        self.assertEqual(aligned.size, (128, 128))
        # Output (50, 50) must sample source (60, 58), not its inverse.
        red, green, _ = aligned.getpixel((50, 50))
        self.assertAlmostEqual(red, 60, delta=1)
        self.assertAlmostEqual(green, 58, delta=1)
        with self.assertRaises(NoFaceError):
            align_face(image, np.full((5, 2), np.nan), (128, 128))

    def test_lfw_keeps_supplied_threshold_even_when_it_scores_poorly(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pairs = root / "pairs.txt"
            pairs.write_text("a.jpg b.jpg 1\na.jpg c.jpg 0\n", encoding="utf-8")
            features = {
                str(root / "a.jpg"): torch.tensor([1.0, 0.0]),
                str(root / "b.jpg"): torch.tensor([0.6, 0.8]),
                str(root / "c.jpg"): torch.tensor([1.0, 0.0]),
            }
            with patch.object(lfw_test, "threshold_search", side_effect=AssertionError("test-set fitting")):
                accuracy, threshold = lfw_test.compute_accuracy(features, pairs, root, 0.7)
            self.assertEqual(accuracy, 0.0)
            self.assertEqual(threshold, 0.7)


class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.aligned = FaceVerifier(aligned=True, require_threshold=False)
        cls.detected = FaceVerifier(aligned=False, require_threshold=False)

    def test_real_weights_produce_finite_embedding(self):
        path = Path(__file__).resolve().parents[1] / "samples/009.jpg"
        feature = self.aligned.feature(path)
        self.assertEqual(feature.shape, (conf.embedding_size,))
        self.assertTrue(np.all(np.isfinite(feature)))

    def test_detect_and_align_reference_face(self):
        path = Path(conf.train_root) / "0000045/001.jpg"
        feature = self.detected.feature(path)
        self.assertEqual(feature.shape, (conf.embedding_size,))
        self.assertTrue(np.all(np.isfinite(feature)))

    def test_missing_and_blank_images(self):
        with self.assertRaises(FileNotFoundError):
            self.detected.feature("/does/not/exist.jpg")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "blank.png"
            Image.new("RGB", (128, 128), "white").save(path)
            with self.assertRaises(NoFaceError):
                self.detected.feature(path)

    def test_verify_return_contract_and_invalid_inputs(self):
        face = Path(conf.train_root) / "0000045/001.jpg"
        self.detected.threshold = 0.5
        result = self.detected.verify(face, face)
        self.assertIs(type(result), int)
        self.assertEqual(result, 1)
        with tempfile.TemporaryDirectory() as temporary:
            blank = Path(temporary) / "blank.png"
            Image.new("RGB", (128, 128), "white").save(blank)
            self.assertEqual(self.detected.verify(blank, face), 0)
            with self.assertRaises(FileNotFoundError):
                self.detected.verify(blank, Path(temporary) / "missing.jpg")
            corrupt = Path(temporary) / "corrupt.jpg"
            corrupt.write_text("this is not an image")
            with self.assertRaises(UnidentifiedImageError):
                self.detected.verify(face, corrupt)


if __name__ == "__main__":
    unittest.main()
