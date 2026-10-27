from collections import defaultdict
from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader, Dataset


class FaceListDataset(Dataset):
    """Load original identity labels without renumbering either split."""

    def __init__(self, list_path, root, transform):
        self.root = Path(root)
        self.transform = transform
        self.samples = []
        with open(list_path, encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                fields = line.split()
                if len(fields) != 2:
                    raise ValueError(f"Invalid image list line {line_number}: {line.strip()}")
                relative_path, label = fields
                if int(label) < 0:
                    raise ValueError(f"Negative identity on line {line_number}.")
                self.samples.append((relative_path.replace("\\", "/"), int(label)))
        if not self.samples:
            raise ValueError(f"No images in {list_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        relative_path, label = self.samples[index]
        with Image.open(self.root / relative_path) as source:
            image = self.transform(source.convert("RGB"))
        return image, label


def make_verification_pairs(dataset):
    """One same-person and one different-person pair per eligible identity."""
    by_identity = defaultdict(list)
    for relative_path, label in dataset.samples:
        by_identity[label].append(str(dataset.root / relative_path))
    identities = sorted(by_identity)
    if len(identities) < 2:
        raise ValueError("Verification validation requires at least two identities.")
    pairs = []
    for index, identity in enumerate(identities):
        images = by_identity[identity]
        if len(images) < 2:
            continue
        other = identities[(index + 1) % len(identities)]
        pairs.append((images[0], images[1], 1))
        pairs.append((images[0], by_identity[other][0], 0))
    if not pairs:
        raise ValueError("Validation needs an identity with at least two images.")
    return pairs


def load_data(conf, training=True):
    root = Path(conf.train_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {root}; extract datasets.zip first.")
    data = FaceListDataset(
        conf.train_list if training else conf.val_list,
        root,
        conf.train_transform if training else conf.test_transform,
    )
    # Labels have gaps; counting distinct labels would create too few outputs.
    class_num = max(label for _, label in data.samples) + 1
    loader = DataLoader(
        data,
        batch_size=conf.train_batch_size if training else conf.test_batch_size,
        shuffle=training,
        drop_last=training,  # BatchNorm1d requires more than one training sample
        pin_memory=conf.pin_memory and str(conf.device).startswith("cuda"),
        num_workers=conf.num_workers,
    )
    return loader, class_num
