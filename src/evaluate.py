import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from config import CLASSES, CKPT_DIR
from dataset import get_dataloaders
from model import CIFAR10CNN
from utils import get_device, load_checkpoint


@torch.no_grad()
def evaluate_full(model, loader, device):
    model.eval()
    n = len(CLASSES)
    correct_per_class = [0] * n
    total_per_class = [0] * n
    confusion = np.zeros((n, n), dtype=int)

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images).argmax(dim=1)
        for true, pred in zip(labels.tolist(), preds.tolist()):
            total_per_class[true] += 1
            confusion[true][pred] += 1
            if true == pred:
                correct_per_class[true] += 1

    overall = sum(correct_per_class) / sum(total_per_class)
    per_class = [c / t for c, t in zip(correct_per_class, total_per_class)]
    return overall, per_class, confusion


def plot_confusion(confusion):
    plt.figure(figsize=(8, 7))
    sns.heatmap(
        confusion,
        annot=True,
        fmt="d",
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        cmap="mako",
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("checkpoints/confusion_matrix.png", dpi=120)
    plt.show()


if __name__ == "__main__":
    device = get_device()
    _, test_loader = get_dataloaders()

    model = CIFAR10CNN().to(device)
    load_checkpoint(model, CKPT_DIR / "best.pth", device)

    overall, per_class, confusion = evaluate_full(model, test_loader, device)
    print(f"Overall test accuracy: {overall:.2%}\n")
    print("Per-class accuracy:")
    for name, acc in zip(CLASSES, per_class):
        bar = "#" * int(acc * 40)
        print(f"  {name:10s} {acc:6.2%} {bar}")

    plot_confusion(confusion)
