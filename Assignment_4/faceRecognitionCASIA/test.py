from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from config import config as conf
from model import build_backbone, load_backbone
from evaluation import cosin_metric, threshold_search, read_threshold
from face_verification import FaceVerification, FaceVerifier, NoFaceError


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


def compute_accuracy(feature_dict, pair_list, test_root, threshold):
    similarities = []
    labels = []
    for image1, image2, label in load_pairs(pair_list, test_root):
        feature1 = feature_dict[image1].cpu().numpy()
        feature2 = feature_dict[image2].cpu().numpy()
        similarities.append(cosin_metric(feature1, feature2))
        labels.append(label)
    accuracy = float(np.mean((np.asarray(similarities) >= threshold) == labels))
    return accuracy, threshold


def test(detect=False):
    weights_path = Path(conf.checkpoints) / conf.test_model
    threshold = read_threshold(conf, weights_path, "detected" if detect else "aligned")

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
    if detect:
        verifier = FaceVerifier()
        features = {}
        failures = 0
        for image in tqdm(images, desc="Detecting test faces"):
            try:
                features[image] = verifier.feature(image)
            except NoFaceError:
                features[image] = None
                failures += 1
        correct = 0
        for first, second, label in pairs:
            prediction = 0
            if features[first] is not None and features[second] is not None:
                prediction = int(cosin_metric(features[first], features[second]) >= threshold)
            correct += prediction == label
        accuracy = correct / len(pairs)
        print(f"End-to-end accuracy: {accuracy:.4%}; no-face images: {failures}")
        return accuracy, threshold

    net = build_backbone(conf).to(conf.device)
    load_backbone(net, weights_path, conf.device)
    net.eval()
    feature_dict = {}
    for start in tqdm(range(0, len(images), conf.test_batch_size), desc="Extracting features"):
        group = images[start : start + conf.test_batch_size]
        feature_dict.update(featurize(group, conf.test_transform, net, conf.device))

    accuracy, threshold = compute_accuracy(feature_dict, conf.test_list, conf.test_root, threshold)
    print(f"Model: {weights_path}")
    print(f"Accuracy at fixed CASIA-validation threshold: {accuracy:.4%}, threshold: {threshold:.6f}")
    print("This is fixed-threshold evaluation, not LFW ten-fold cross-validation.")
    return accuracy, threshold


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test LFW without fitting the threshold on LFW.")
    parser.add_argument("--checkpoint", default=conf.test_model)
    parser.add_argument("--detect", action="store_true")
    parser.add_argument("--pairs", default=conf.test_list)
    parser.add_argument("--test-root", default=conf.test_root)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    conf.test_model, conf.test_list, conf.test_root = args.checkpoint, args.pairs, args.test_root
    test(detect=args.detect)
