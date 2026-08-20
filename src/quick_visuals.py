import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets, transforms, utils

look = datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=transforms.ToTensor(),
)
loader = torch.utils.data.DataLoader(look, batch_size=8, shuffle=True)
images, labels = next(iter(loader))

# torchvision images are (C, H, W) but matplotlib wants (H, W, C)
grid = utils.make_grid(images)
plt.figure(figsize=(8, 4))
plt.imshow(np.transpose(grid.numpy(), (1, 2, 0)))
plt.title([look.classes[l] for l in labels])
plt.axis("off")
plt.show()
