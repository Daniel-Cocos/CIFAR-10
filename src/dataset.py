import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from config import MEAN, STD, BATCH_SIZE, NUM_WORKERS, DATA_DIR

def get_transforms(train: bool) -> transforms.Compose:
    base = [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    if train:
        aug = [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4)
                ]
        return transforms.Compose(aug + base)
    return transforms.Compose(base)
