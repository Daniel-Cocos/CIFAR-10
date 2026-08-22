import torch
import torch.nn as nn

from config import EPOCHS, LEARNING_RATE, WEIGHT_DECAY, CKPT_DIR
from dataset import get_dataloaders
from model import CIFAR10CNN
from utils import get_device, save_checkpoint


def train():
    device = get_device()
    train_loader, test_loader = get_dataloaders()

    model = CIFAR10CNN().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_acc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()  # clear old gradients
            loss.backward()  # compute new gradients
            optimizer.step()  # update weights

            running_loss += loss.item() * images.size(0)

        scheduler.step()  # decay learning rate

        # quick validation after each epoch
        val_acc = evaluate_accuracy(model, test_loader, device)
        avg_loss = running_loss / len(train_loader.dataset)
        print(
            f"Epoch {epoch+1:02d}/{EPOCHS} | "
            f"loss {avg_loss:.4f} | val acc {val_acc:.2%}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(
                model, CKPT_DIR / "best.pth", extra={"acc": best_acc, "epoch": epoch}
            )

    print(f"Best validation accuracy: {best_acc:.2%}")


@torch.no_grad()  # no gradients for faster execution and less memory
def evaluate_accuracy(model, loader, device):
    model.eval()  # disable Dropout
    correct = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images).argmax(dim=1)  # pick highest-scoring class
        correct += (preds == labels).sum().item()
    return correct / len(loader.dataset)


if __name__ == "__main__":
    train()
