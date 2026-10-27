import os
import os.path as osp
import random

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

from model import build_backbone, load_backbone
from model.metric import ArcFace, CosFace
from model.loss import FocalLoss
from dataset import load_data, make_verification_pairs
from evaluation import cosin_metric, threshold_search, save_threshold
from config import config as conf


# Data Setup
def verification_metrics(net, loader, dataset, pairs, device):
    net.eval()
    features = []
    with torch.inference_mode():
        for data, _ in tqdm(loader, desc="Validation", leave=False):
            features.extend(net(data.to(device)).cpu().numpy())
    feature_dict = {
        str(dataset.root / relative_path): feature
        for (relative_path, _), feature in zip(dataset.samples, features)
    }
    scores = [cosin_metric(feature_dict[a], feature_dict[b]) for a, b, _ in pairs]
    accuracy, threshold = threshold_search(scores, [label for _, _, label in pairs])
    net.train()
    return accuracy, threshold


def train():
    random.seed(conf.seed)
    torch.manual_seed(conf.seed)
    if conf.train_batch_size < 2:
        raise ValueError("train_batch_size must be at least 2 for BatchNorm.")
    dataloader, class_num = load_data(conf, training=True)
    val_loader, _ = load_data(conf, training=False)
    train_data, val_data = dataloader.dataset, val_loader.dataset
    overlap = {path for path, _ in train_data.samples} & {path for path, _ in val_data.samples}
    if overlap:
        raise ValueError(f"Train/validation image overlap: {next(iter(overlap))}")
    val_pairs = make_verification_pairs(val_data)
    if len(dataloader) == 0:
        raise ValueError("Training data has fewer images than train_batch_size.")
    embedding_size = conf.embedding_size
    device = conf.device

    # Network Setup
    net = build_backbone(conf).to(device)

    if conf.metric == "arcface":
        metric = ArcFace(embedding_size, class_num).to(device)
    else:
        metric = CosFace(embedding_size, class_num).to(device)

    if str(device).startswith("cuda") and torch.cuda.device_count() > 1:
        chunks = torch.empty(conf.train_batch_size).chunk(torch.cuda.device_count())
        if min(chunk.numel() for chunk in chunks) < 2:
            raise ValueError("Multi-GPU training requires at least 2 samples per GPU.")
        net = nn.DataParallel(net)
        metric = nn.DataParallel(metric)

    # Training Setup
    if conf.loss == "focal_loss":
        criterion = FocalLoss(gamma=2)
    else:
        criterion = nn.CrossEntropyLoss()

    if conf.optimizer == "sgd":
        optimizer = optim.SGD(
            [{"params": net.parameters()}, {"params": metric.parameters()}],
            lr=conf.lr,
            weight_decay=conf.weight_decay,
        )
    else:
        optimizer = optim.Adam(
            [{"params": net.parameters()}, {"params": metric.parameters()}],
            lr=conf.lr,
            weight_decay=conf.weight_decay,
        )

    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=conf.lr_step, gamma=conf.lr_decay
    )

    # Checkpoints Setup
    checkpoints = conf.checkpoints
    os.makedirs(checkpoints, exist_ok=True)

    if conf.restore:
        weights_path = osp.join(checkpoints, conf.restore_model)
        load_backbone(net, weights_path, device)
        print(f"Loaded backbone for fine-tuning: {weights_path}")

    print(f"Device: {device}, identities: {class_num}, images: {len(dataloader.dataset)}")
    net.train()
    metric.train()
    best_val_accuracy = -1.0

    for e in range(conf.epoch):
        total_loss = 0.0
        total_samples = 0
        for data, labels in tqdm(dataloader, desc=f"Epoch {e + 1}/{conf.epoch}"):
            data = data.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            embeddings = net(data)
            thetas = metric(embeddings, labels)
            loss = criterion(thetas, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
            total_samples += len(labels)

        print(f"Epoch {e + 1}/{conf.epoch}, loss: {total_loss / total_samples:.4f}")
        backbone_path = osp.join(checkpoints, f"{e}.pth")
        backbone = net.module if isinstance(net, nn.DataParallel) else net
        torch.save(backbone.state_dict(), backbone_path)
        val_accuracy, threshold = verification_metrics(
            net, val_loader, val_data, val_pairs, device
        )
        print(f"Validation pair accuracy: {val_accuracy:.4%}; threshold: {threshold:.6f}")
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_path = osp.join(checkpoints, "best.pth")
            torch.save(backbone.state_dict(), best_path)
            save_threshold(
                conf, best_path, threshold, preprocessing="aligned",
                validation_accuracy=val_accuracy, epoch=e + 1,
                validation_list=str(conf.val_list), pairs=len(val_pairs),
                model_source="trained by train.py",
            )
            print(f"Saved best model: {best_path}")
        scheduler.step()


if __name__ == "__main__":
    train()
