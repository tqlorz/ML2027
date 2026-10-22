from pathlib import Path

import torch

from .fmobilenet import FaceMobileNet
from .resnet import ResIRSE
from .metric import ArcFace, CosFace
from .loss import FocalLoss


def build_backbone(conf):
    if conf.backbone == "fmobile":
        return FaceMobileNet(conf.embedding_size)
    if conf.backbone == "resnet":
        return ResIRSE(conf.embedding_size, conf.drop_ratio)
    raise ValueError(f"Unknown backbone: {conf.backbone}")


def load_backbone(net, weights_path, device):
    """Load both legacy DataParallel weights and plain backbone weights."""
    if not Path(weights_path).is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {weights_path}. "
            "Train first and set test_model/restore_model in config.py."
        )
    state = torch.load(weights_path, map_location=device, weights_only=True)
    state = {key.removeprefix("module."): value for key, value in state.items()}
    backbone = net.module if isinstance(net, torch.nn.DataParallel) else net
    backbone.load_state_dict(state)
