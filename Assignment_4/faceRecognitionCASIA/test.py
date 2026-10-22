from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from config import config as conf
from model import build_backbone, load_backbone


def _preprocess(images: list, transform) -> torch.Tensor:
    tensors = []
    for path in images:
        with Image.open(path) as image:
            tensors.append(transform(image))
    return torch.stack(tensors)


def featurize(images: list, transform, net, device) -> dict:
    data = _preprocess(images, transform).to(device)
    with torch.inference_mode():
        features = net(data).cpu()
    return dict(zip(images, features))


def load_pairs(pair_list, test_root):
    """Read LFW pairs.txt or three-column 'image1 image2 label' lists."""
    root = Path(test_root)
    pairs = []
    with open(pair_list, encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            words = line.split()
            if not words:
                continue
            # LFW header: '10 300' (or a single pair count for dev lists).
            if not pairs and len(words) in (1, 2) and all(w.isdigit() for w in words):
                continue
            if len(words) == 3 and not words[1].isdigit():
                image1, image2 = root / words[0], root / words[1]
                label = int(words[2])
                if label not in (0, 1):
                    raise ValueError(f"Invalid pair label on line {line_number}: {line.strip()}")
            elif len(words) == 3:
                name, index1, index2 = words
                image1 = root / name / f"{name}_{int(index1):04d}.jpg"
                image2 = root / name / f"{name}_{int(index2):04d}.jpg"
                label = 1
            elif len(words) == 4:
                name1, index1, name2, index2 = words
                image1 = root / name1 / f"{name1}_{int(index1):04d}.jpg"
                image2 = root / name2 / f"{name2}_{int(index2):04d}.jpg"
                label = 0
            else:
                raise ValueError(f"Invalid pair on line {line_number}: {line.strip()}")
            pairs.append((str(image1), str(image2), label))
    if not pairs:
        raise ValueError(f"No image pairs found in {pair_list}")
    return pairs


def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))


def threshold_search(y_score, y_true):
    y_score = np.asarray(y_score, dtype=float)
    y_true = np.asarray(y_true)
    if len(y_score) == 0 or len(y_score) != len(y_true):
        raise ValueError("Scores and labels must have the same nonzero length.")
    best_acc = -1.0
    best_th = 0.0
    thresholds = np.append(np.unique(y_score), np.nextafter(y_score.max(), np.inf))
    for threshold in thresholds:
        accuracy = np.mean((y_score >= threshold) == y_true)
        if accuracy > best_acc:
            best_acc, best_th = float(accuracy), float(threshold)
    return best_acc, best_th


def compute_accuracy(feature_dict, pair_list, test_root):
    similarities = []
    labels = []
    for image1, image2, label in load_pairs(pair_list, test_root):
        feature1 = feature_dict[image1].cpu().numpy()
        feature2 = feature_dict[image2].cpu().numpy()
        similarities.append(cosin_metric(feature1, feature2))
        labels.append(label)
    return threshold_search(similarities, labels)


def test():
    weights_path = Path(conf.checkpoints) / conf.test_model
    net = build_backbone(conf).to(conf.device)
    load_backbone(net, weights_path, conf.device)
    net.eval()

    pairs = load_pairs(conf.test_list, conf.test_root)
    images = sorted({image for pair in pairs for image in pair[:2]})
    missing = [image for image in images if not Path(image).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} LFW images are missing, e.g. {missing[0]}. "
            "Check test_root in config.py and extract datasets.zip."
        )
    if conf.test_batch_size < 1:
        raise ValueError("test_batch_size must be positive.")

    print(f"Device: {conf.device}, pairs: {len(pairs)}, unique images: {len(images)}")
    feature_dict = {}
    for start in tqdm(range(0, len(images), conf.test_batch_size), desc="Extracting features"):
        group = images[start : start + conf.test_batch_size]
        feature_dict.update(featurize(group, conf.test_transform, net, conf.device))

    accuracy, threshold = compute_accuracy(feature_dict, conf.test_list, conf.test_root)
    print(f"Model: {weights_path}")
    print(f"Best accuracy on these pairs: {accuracy:.4%}, threshold: {threshold:.6f}")
    return accuracy, threshold


if __name__ == "__main__":
    test()
