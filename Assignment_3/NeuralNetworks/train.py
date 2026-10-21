import argparse
import json
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model.resnet import ResNet18

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def ensure_data(data_root: Path):
    if (data_root / "train").is_dir() and (data_root / "test").is_dir():
        return
    raise FileNotFoundError(
        f"Expected train/ and test/ directories under {data_root}. "
        "Extract the CIFAR-10 archive before running training."
    )


def make_loaders(data_root: Path, batch_size: int, workers: int):
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    train_set = datasets.ImageFolder(data_root / "train", transform=train_transform)
    test_set = datasets.ImageFolder(data_root / "test", transform=test_transform)
    if train_set.classes != test_set.classes:
        raise ValueError("Train and test class directories do not match")
    if len(train_set.classes) != 10:
        raise ValueError("Expected 10 CIFAR-10 classes")
    print(
        f"Dataset: {len(train_set)} train, {len(test_set)} test | {train_set.classes}"
    )
    kwargs = dict(
        batch_size=batch_size, num_workers=workers, pin_memory=torch.cuda.is_available()
    )
    if workers > 0:
        kwargs["persistent_workers"] = True
    return (
        DataLoader(train_set, shuffle=True, **kwargs),
        DataLoader(test_set, shuffle=False, **kwargs),
        train_set.classes,
    )


@torch.inference_mode()
def evaluate(model, loader, device, criterion):
    model.eval()
    loss_sum = correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss_sum += criterion(logits, labels).item() * labels.size(0)
        correct += logits.argmax(1).eq(labels).sum().item()
        total += labels.size(0)
    return loss_sum / total, 100.0 * correct / total


def smoke_test(model, device):
    """Check one synthetic training step without loading data or reporting accuracy."""
    model.train()
    images = torch.randn(2, 3, 32, 32, device=device)
    labels = torch.tensor([0, 1], device=device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    optimizer.zero_grad(set_to_none=True)
    logits = model(images)
    if logits.shape != (2, 10):
        raise RuntimeError(f"Unexpected output shape: {tuple(logits.shape)}")
    loss = nn.functional.cross_entropy(logits, labels)
    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite loss")
    loss.backward()
    optimizer.step()
    model.eval()
    with torch.inference_mode():
        model(images)
    print(
        f"Smoke test passed | output {tuple(logits.shape)} | "
        f"parameters {sum(p.numel() for p in model.parameters()):,}"
    )


def main():
    project = Path(__file__).resolve().parent
    default_data = project / "CIFAR10"
    if not default_data.is_dir():
        default_data = project.parent / "CIFAR10"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=default_data)
    parser.add_argument(
        "--output", type=Path, default=project / "checkpoints" / "resnet18_cifar10.pt"
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--threads", type=int, default=4, help="CPU threads for tensor operations"
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Evaluate a saved checkpoint without training",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a synthetic forward/backward step without dataset",
    )
    args = parser.parse_args()
    if (
        args.epochs < 1
        or args.batch_size < 1
        or args.workers < 0
        or args.threads < 1
        or args.lr <= 0
    ):
        parser.error(
            "epochs, batch-size, threads and lr must be positive; workers must be nonnegative"
        )
    if args.eval_only and args.smoke_test:
        parser.error("--eval-only and --smoke-test cannot be combined")
    torch.set_num_threads(args.threads)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    selected_device = (
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else args.device
    )
    if selected_device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    device = torch.device(selected_device)
    print(f"Device: {device}")
    model = ResNet18().to(device)
    if args.smoke_test:
        smoke_test(model, device)
        return
    ensure_data(args.data_root)
    train_loader, test_loader, classes = make_loaders(
        args.data_root, args.batch_size, args.workers
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    if args.eval_only:
        checkpoint = torch.load(args.output, map_location=device, weights_only=True)
        if checkpoint["classes"] != classes:
            raise ValueError("Checkpoint class ordering does not match dataset")
        model.load_state_dict(checkpoint["model"])
        loss, accuracy = evaluate(model, test_loader, device, criterion)
        print(f"Test loss: {loss:.4f} | Test accuracy: {accuracy:.2f}%")
        return

    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4, nesterov=True
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = correct = total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.item() * labels.size(0)
            correct += logits.detach().argmax(1).eq(labels).sum().item()
            total += labels.size(0)
        scheduler.step()
        print(
            f"Epoch {epoch:03d}/{args.epochs} | train loss {loss_sum / total:.4f} "
            f"| train accuracy {100 * correct / total:.2f}% | lr {scheduler.get_last_lr()[0]:.5g}"
        )
        torch.save(
            {
                "model": model.state_dict(),
                "classes": classes,
                "epoch": epoch,
                "args": {
                    k: str(v) if isinstance(v, Path) else v
                    for k, v in vars(args).items()
                },
            },
            args.output,
        )

    loss, accuracy = evaluate(model, test_loader, device, criterion)
    print(f"Test loss: {loss:.4f} | Test accuracy: {accuracy:.2f}%")
    result_path = args.output.with_suffix(".results.json")
    result_path.write_text(
        json.dumps(
            {
                "test_loss": loss,
                "test_accuracy_percent": accuracy,
                "test_samples": len(test_loader.dataset),
                "train_samples": len(train_loader.dataset),
                "epochs": args.epochs,
                "classes": classes,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Test results saved to {result_path}")


if __name__ == "__main__":
    main()
