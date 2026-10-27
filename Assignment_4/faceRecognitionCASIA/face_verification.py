"""Assignment API: Y = FaceVerification(X1, X2), with image paths as input."""

from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from config import config as conf
from evaluation import cosin_metric, read_threshold
from model import build_backbone, load_backbone


class NoFaceError(ValueError):
    """A readable image contains no usable face."""


@dataclass(frozen=True)
class VerificationResult:
    """Detailed result for the GUI; the assignment API still returns an int."""

    prediction: int
    similarity: float | None
    threshold: float
    faces_detected: bool


def read_image(image_path):
    path = Path(image_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    with Image.open(path) as source:
        return ImageOps.exif_transpose(source).convert("RGB")


def align_face(image, points, output_shape):
    """Fit a similarity transform using both eyes, nose, and mouth corners."""
    points = np.asarray(points, dtype=np.float64)
    if points.shape != (5, 2) or not np.all(np.isfinite(points)):
        raise NoFaceError("Invalid facial landmarks.")
    height, width = output_shape
    if height != width:
        raise ValueError("Five-point alignment requires square model input.")
    reference = np.array([
        [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
        [41.5493, 92.3655], [70.7299, 92.2041],
    ]) * (width / 112.0)
    design, targets = [], []
    for (x, y), (u, v) in zip(reference, points):
        design.extend([[x, -y, 1, 0], [y, x, 0, 1]])
        targets.extend([u, v])
    # Pillow expects the inverse map: coordinates in output -> source image.
    a, b, tx, ty = np.linalg.lstsq(design, targets, rcond=None)[0]
    if not np.all(np.isfinite([a, b, tx, ty])) or a * a + b * b < 1e-12:
        raise NoFaceError("Degenerate facial landmarks.")
    return image.transform(
        (width, height), Image.Transform.AFFINE, (a, -b, tx, b, a, ty),
        resample=Image.Resampling.BILINEAR,
    )


class FaceVerifier:
    """Load the network once; use the largest detected face in each image."""

    def __init__(self, aligned=False, require_threshold=True):
        self.device = conf.device
        self.aligned = aligned
        self.checkpoint = Path(conf.checkpoints) / conf.test_model
        self.threshold = read_threshold(
            conf, self.checkpoint, "aligned" if aligned else "detected"
        ) if require_threshold else None
        self.net = build_backbone(conf)
        load_backbone(self.net, self.checkpoint, "cpu")
        self.net.to(self.device).eval()
        self.detector = None
        if not aligned:
            from facenet_pytorch import MTCNN
            self.detector = MTCNN(
                min_face_size=conf.detection_min_face_size,
                thresholds=conf.detection_thresholds,
                factor=conf.detection_factor,
                device=self.device,
            ).eval()

    def detect(self, image):
        if min(image.size) < conf.detection_min_face_size:
            raise NoFaceError("Image is smaller than the detector's minimum face size.")
        with torch.inference_mode():
            boxes, _, landmarks = self.detector.detect(image, landmarks=True)
        if boxes is None or len(boxes) == 0 or landmarks is None:
            raise NoFaceError("No face detected.")
        index = int(np.argmax((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])))
        return align_face(image, landmarks[index], conf.input_shape[1:])

    def features(self, paths):
        """Extract a batch; calibration with detection uses feature() per image."""
        images = [read_image(path) for path in paths]
        if not self.aligned:
            images = [self.detect(image) for image in images]
        batch = torch.stack([conf.test_transform(image) for image in images]).to(self.device)
        with torch.inference_mode():
            return self.net(batch).cpu().numpy()

    def feature(self, path):
        return self.features([path])[0]

    def compare(self, first_path, second_path):
        if self.threshold is None:
            raise ValueError("Verification requires a calibrated threshold.")
        # Validate both paths before returning 0 for an undetectable face.
        images = [read_image(first_path), read_image(second_path)]
        try:
            if not self.aligned:
                images = [self.detect(image) for image in images]
        except NoFaceError:
            return VerificationResult(0, None, self.threshold, False)
        batch = torch.stack([conf.test_transform(image) for image in images]).to(self.device)
        with torch.inference_mode():
            first, second = self.net(batch).cpu().numpy()
        similarity = cosin_metric(first, second)
        return VerificationResult(int(similarity >= self.threshold), similarity, self.threshold, True)

    def verify(self, first_path, second_path):
        return self.compare(first_path, second_path).prediction


@lru_cache(maxsize=1)
def _load_runtime():
    return FaceVerifier()


def compare_faces(X1, X2):
    """Use the same cached runtime as FaceVerification, with result details."""
    return _load_runtime().compare(X1, X2)


def FaceVerification(X1, X2):
    """Return Python int 1 for the same person, otherwise 0.

    Both inputs are image paths. No detectable face returns 0; missing/corrupt
    files and missing/mismatched model artifacts raise informative exceptions.
    """
    return _load_runtime().verify(X1, X2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Detect, align, and verify two face images.")
    parser.add_argument("image1")
    parser.add_argument("image2")
    parser.add_argument("--checkpoint", default=conf.test_model)
    parser.add_argument("--threads", type=int, default=4, help="PyTorch CPU threads.")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    conf.test_model = args.checkpoint
    print(FaceVerification(args.image1, args.image2))
