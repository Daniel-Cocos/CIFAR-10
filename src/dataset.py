from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from config import MEAN, STD, BATCH_SIZE, NUM_WORKERS, DATA_DIR


def get_transforms(train: bool) -> transforms.Compose:
    base = [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    if train:
        aug = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4)]
        return transforms.Compose(aug + base)
    return transforms.Compose(base)


def get_dataloaders():
    train_set = datasets.CIFAR10(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=get_transforms(train=True),
    )
    test_set = datasets.CIFAR10(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=get_transforms(train=False),
    )

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    return train_loader, test_loader


if __name__ == "__main__":
    train_loader, _ = get_dataloaders()
    images, labels = next(iter(train_loader))
    print("Batch shape:", images.shape)  # [128, 3, 32, 32]
    print("Label range:", labels.min().item(), "to", labels.max().item())
    print("Pixel range:", images.min().item(), "to", images.max().item())
