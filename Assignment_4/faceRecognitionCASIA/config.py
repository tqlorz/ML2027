from pathlib import Path

import torch
import torchvision.transforms as T

BASE_DIR = Path(__file__).resolve().parent


class Config:
    # network settings
    backbone = "fmobile"  # [resnet, fmobile]
    metric = "cosface"  # [cosface, arcface]
    embedding_size = 512
    drop_ratio = 0.5

    # data preprocess
    input_shape = [1, 128, 128]
    train_transform = T.Compose(
        [
            T.Grayscale(),
            T.RandomHorizontalFlip(),
            T.Resize((144, 144)),
            T.RandomCrop(input_shape[1:]),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5]),
        ]
    )
    test_transform = T.Compose(
        [
            T.Grayscale(),
            T.Resize(input_shape[1:]),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5]),
        ]
    )

    # dataset
    train_root = str(BASE_DIR / "datasets/CASIA-WebFace-Align-96")
    test_root = str(
        BASE_DIR / "datasets/scikit_learn_data/lfw_home/lfw_funeled_cropped"
    )
    test_list = str(BASE_DIR / "datasets/scikit_learn_data/lfw_home/pairs.txt")

    # training settings
    checkpoints = str(BASE_DIR / "checkpoints")
    restore = False  # True loads backbone weights for fine-tuning, not full resume
    restore_model = "18.pth"
    test_model = "18.pth"

    train_batch_size = 64
    test_batch_size = 64

    epoch = 10
    optimizer = "sgd"  # ['sgd', 'adam']
    lr = 1e-1
    lr_step = 10
    lr_decay = 0.1
    weight_decay = 5e-4
    loss = "focal_loss"  # ['focal_loss', 'cross_entropy']
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pin_memory = device == "cuda"
    num_workers = 4  # dataloader


config = Config()
