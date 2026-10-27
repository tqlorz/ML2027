"""Calibrate aligned-image and detection-pipeline thresholds using CASIA val.txt."""

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from config import config as conf
from dataset import FaceListDataset, make_verification_pairs
from evaluation import cosin_metric, save_threshold, threshold_search
from face_verification import FaceVerifier, NoFaceError


def calibrate(mode):
    dataset = FaceListDataset(conf.val_list, conf.train_root, conf.test_transform)
    pairs = make_verification_pairs(dataset)
    verifier = FaceVerifier(aligned=(mode == "aligned"), require_threshold=False)
    images = sorted({path for first, second, _ in pairs for path in (first, second)})
    features, failures = {}, []
    if mode == "aligned":
        for start in tqdm(range(0, len(images), conf.test_batch_size), desc="CASIA aligned features"):
            paths = images[start:start + conf.test_batch_size]
            features.update(zip(paths, verifier.features(paths)))
    else:
        for path in tqdm(images, desc="CASIA detect/align features"):
            try:
                features[path] = verifier.feature(path)
            except NoFaceError:
                failures.append(path)
    original_pairs = len(pairs)
    pairs = [(a, b, label) for a, b, label in pairs if a in features and b in features]
    if {label for _, _, label in pairs} != {0, 1}:
        raise ValueError("Not enough usable validation pairs of both classes.")
    scores = [cosin_metric(features[a], features[b]) for a, b, _ in pairs]
    accuracy, threshold = threshold_search(scores, [label for _, _, label in pairs])
    save_threshold(
        conf, verifier.checkpoint, threshold, preprocessing=mode,
        validation_accuracy=accuracy, validation_list=str(conf.val_list),
        pairs=len(pairs), validation_images=len(images), detection_failures=len(failures),
        excluded_pairs=original_pairs - len(pairs),
        model_source=f"existing checkpoint: {Path(verifier.checkpoint).name}; calibration does not train weights",
    )
    print(f"{mode}: validation accuracy={accuracy:.4%}; threshold={threshold:.6f}; pairs={len(pairs)}")
    print(f"No-face images={len(failures)}; excluded pairs={original_pairs - len(pairs)}")
    return accuracy, threshold


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["aligned", "detected", "both"], default="both")
    parser.add_argument("--checkpoint", default=conf.test_model)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    conf.test_model = args.checkpoint
    for mode in (["aligned", "detected"] if args.mode == "both" else [args.mode]):
        calibrate(mode)
