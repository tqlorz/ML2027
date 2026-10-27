"""Cosine scoring and validation-only threshold calibration."""

import hashlib
import json
from pathlib import Path

import numpy as np


def cosin_metric(first, second):
    first, second = np.asarray(first), np.asarray(second)
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    if denominator == 0 or not np.isfinite(denominator):
        raise ValueError("Cannot compare zero-length or non-finite embeddings.")
    score = float(np.dot(first, second) / denominator)
    if not np.isfinite(score):
        raise ValueError("Non-finite cosine similarity.")
    return score


def threshold_search(scores, labels):
    """Maximize accuracy in O(N log N), with prediction = score >= threshold."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    if scores.ndim != 1 or scores.shape != labels.shape or scores.size == 0:
        raise ValueError("Scores and labels must be matching nonempty vectors.")
    if not np.all(np.isfinite(scores)) or not np.all(np.isin(labels, [0, 1])):
        raise ValueError("Scores must be finite and labels must be 0/1.")
    order = np.argsort(scores)
    sorted_scores, sorted_labels = scores[order], labels[order]
    thresholds, starts = np.unique(sorted_scores, return_index=True)
    changes = np.add.reduceat(1 - 2 * sorted_labels, starts)
    correct = np.r_[np.sum(labels) + np.r_[0, np.cumsum(changes)[:-1]],
                    np.sum(labels) + np.sum(changes)]
    thresholds = np.r_[thresholds, np.nextafter(scores.max(), np.inf)]
    best = int(np.argmax(correct))
    return float(correct[best] / scores.size), float(thresholds[best])


def checkpoint_digest(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preprocessing_metadata(conf, mode):
    if mode not in ("aligned", "detected"):
        raise ValueError(f"Unknown preprocessing mode: {mode}")
    metadata = {"preprocessing": mode, "transform_version": "gray_resize_normalize_0.5_v1"}
    if mode == "detected":
        metadata["detector"] = {
            "name": "facenet-pytorch-2.6.0-mtcnn",
            "min_face_size": conf.detection_min_face_size,
            "thresholds": list(conf.detection_thresholds),
            "factor": conf.detection_factor,
            "alignment": "five_point_similarity_v1",
            "face_selection": "largest",
        }
    return metadata


def _metadata(conf, checkpoint, mode):
    return {
        "backbone": conf.backbone,
        "embedding_size": conf.embedding_size,
        "input_shape": list(conf.input_shape),
        "checkpoint_sha256": checkpoint_digest(checkpoint),
        **preprocessing_metadata(conf, mode),
    }


def _threshold_path(conf, mode):
    return Path(conf.validation_threshold if mode == "aligned" else conf.verification_threshold)


def save_threshold(conf, checkpoint, threshold, preprocessing="aligned", **details):
    if not np.isfinite(threshold):
        raise ValueError("Threshold must be finite.")
    path = _threshold_path(conf, preprocessing)
    metadata = {**details, **_metadata(conf, checkpoint, preprocessing), "threshold": float(threshold)}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_threshold(conf, checkpoint, preprocessing="detected"):
    path = _threshold_path(conf, preprocessing)
    if not path.is_file():
        raise FileNotFoundError(
            f"Threshold file not found: {path}. Run calibrate.py --mode {preprocessing} "
            f"--checkpoint {Path(checkpoint).name} first."
        )
    metadata = json.loads(path.read_text(encoding="utf-8"))
    for key, value in _metadata(conf, checkpoint, preprocessing).items():
        if metadata.get(key) != value:
            raise ValueError(f"Threshold metadata mismatch for {key}; recalibrate for this model.")
    threshold = float(metadata["threshold"])
    if not np.isfinite(threshold):
        raise ValueError("Threshold must be finite.")
    return threshold
