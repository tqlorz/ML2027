from pathlib import Path

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


def load_data(conf, training=True):
    if training:
        dataroot = conf.train_root
        transform = conf.train_transform
        batch_size = conf.train_batch_size
    else:
        dataroot = conf.test_root
        transform = conf.test_transform
        batch_size = conf.test_batch_size

    if not Path(dataroot).is_dir():
        raise FileNotFoundError(
            f"Dataset directory not found: {dataroot}. "
            "Extract datasets.zip in the project directory first."
        )
    data = ImageFolder(dataroot, transform=transform)
    class_num = len(data.classes)
    loader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=training,
        drop_last=training,  # BatchNorm1d requires more than one training sample
        pin_memory=conf.pin_memory and str(conf.device).startswith("cuda"),
        num_workers=conf.num_workers,
    )
    return loader, class_num
