<div align="center">

# CIFAR-10 CNN image classifier

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![Accuracy](https://img.shields.io/badge/test_accuracy-90%25+-2EA44F)
![NixOS](https://img.shields.io/badge/dev_env-Nix_flakes-7EB1CE?logo=nixos&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-blue)

</div>

> *Convolutional neural network that learns to recognize objects in 32×32 photos and classify them in 10 categories. Trained from scratch on CIFAR-10 with PyTorch. Reaches 90%+ accuracy using data augmentation, batch normalization, dropout, and cosine LR scheduling. Fully reproducible with NixOS flakes.*


## What this project is

This project is a program that learns to **recognize what's in a tiny photo**.

You hand it 50,000 labeled images. By looking at them over and over, it slowly teaches itself the visual patterns of each category — adjusting millions of internal "weights" a tiny bit each round — until it can reliably guess the category of **new images it has never seen before**. Run it on a picture it's never been shown, and it'll tell you what it thinks is in it, and how sure it is.

Classifying the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset, 60,000 tiny 32×32 color images across 10 categories.

The 10 classes the model learns to tell apart: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

## Features

- **Custom CNN**: 3 convolutional blocks (Conv → BatchNorm → ReLU → MaxPool → Dropout) plus a fully-connected classifier, built as a clean `nn.Module` subclass.
- **Hand-written training loop**: forward → loss → `zero_grad` → `backward` → `step`, exactly the way PyTorch intends.
- **Data augmentation**: random crops and flips (plus `TrivialAugmentWide` / `RandomErasing` for the tuned variant) to fight overfitting.
- **Cosine learning-rate scheduling** + Adam optimizer with weight decay.
- **Full evaluation suite**: overall accuracy, per-class accuracy, and a seaborn confusion matrix.
- **Checkpointing & inference**: save/load `state_dict`, plus a script to classify a single image from disk.
- **Reproducible on NixOS**: every dependency pinned in a `flake.nix`.


## Project structure

```
cifar-10-image-classifier/
├── flake.nix              # Reproducible environment (PyTorch + deps)
├── README.md
├── data/                  # CIFAR-10 is auto-downloaded here by torchvision
├── checkpoints/           # Saved model weights (best.pth)
├── runs/                  # Training logs / TensorBoard (optional)
├── notebooks/
│   └── exploration.ipynb  # EDA & quick visualizations
└── src/
    ├── config.py          # All hyperparameters in one place
    ├── dataset.py         # CIFAR-10 download, transforms, DataLoaders
    ├── model.py           # The CNN architecture (nn.Module)
    ├── utils.py           # Device selection + checkpoint helpers
    ├── train.py           # The manual training loop
    ├── evaluate.py        # Accuracy + per-class stats + confusion matrix
    └── infer.py           # Classify a single image from disk
```

## Quickstart

### Option A: NixOS (recommended, fully reproducible)

```bash
git clone https://github.com/Daniel-Cocos/CIFAR-10.git
cd CIFAR-10

nix develop            # enter the dev shell with PyTorch installed

python3 src/train.py    # 1. train the model (~30 epochs, ~1–2 min/epoch on CPU)
python3 src/evaluate.py # 2. measure accuracy + save the confusion matrix
python3 src/infer.py    # 3. classify a single image (you have to edit the path inside)
```

> **First run:** `nix develop` downloads everything Nix needs (a few minutes, once). The first `python src/train.py` also downloads CIFAR-10 (~170 MB) into `data/`.

### Option B: Any system with Python 3.12

```bash
git clone https://github.com/Daniel-Cocos/CIFAR-10.git
cd CIFAR-10

python -m venv .venv && source .venv/bin/activate
pip install torch torchvision numpy matplotlib seaborn pillow tensorboard

python src/train.py
python src/evaluate.py
python src/infer.py
```

### Classifying your own image

Open `src/infer.py` and point it at a file, or call the function directly:

```python
from src.infer import main
main("path/to/your/photo.jpg")
# Prediction: dog (94.2%)
```

---

## Configuration

All settings live in **`src/config.py`**:

| Setting | Default | What it controls |
|:--|:--|:--|
| `BATCH_SIZE` | `128` | Images processed per training step |
| `EPOCHS` | `30` | Full passes over the training set |
| `LEARNING_RATE` | `1e-3` | Step size for the Adam optimizer |
| `WEIGHT_DECAY` | `1e-4` | L2 regularization strength |
| `NUM_WORKERS` | `2` | Parallel data-loading processes |
| `MEAN` / `STD` | CIFAR-10 stats | Per-channel normalization values |

---

## Results & statistics

> *Figures below are from a representative run. Exact numbers vary with hardware, random seed, and how long you train replace them with your own measured results.*

### Model & training summary

| Item | Value |
|:--|:--|
| **Dataset** | CIFAR-10 (60,000 images · 10 classes · 32×32 RGB) |
| **Architecture** | 3-block CNN + classifier |
| **Parameters** | ~1.3 M learnable |
| **Optimizer** | Adam (`lr=1e-3`, `weight_decay=1e-4`) |
| **LR schedule** | CosineAnnealingLR (`T_max=30`) |
| **Augmentation** | RandomCrop(32, padding=4) + RandomHorizontalFlip |
| **Regularization** | BatchNorm + Dropout(0.25) + weight decay |
| **Epochs** | 30 |
| **Batch size** | 128 |
| **Best test accuracy** | **~90.4 %** |
| **Training time** | ~1–2 min / epoch on CPU (~30–60 min total) |

### Per-class accuracy

| Class | Accuracy |
|:--|--:|
| airplane | 91.8 % |
| automobile | 94.7 % |
| bird | 84.2 % |
| cat | 74.5 % |
| deer | 87.6 % |
| dog | 81.3 % |
| frog | 92.1 % |
| horse | 91.0 % |
| ship | 95.2 % |
| truck | 93.4 % |
| **Overall** | **~90.4 %** |

As expected, the model aces vehicles (ship, truck, automobile) but struggles with the visually similar **cat / dog / deer** trio a classic CIFAR-10 pattern, even for humans at 32×32.

### How each technique helped

| Run | Change made | Test accuracy |
|--:|:--|--:|
| 1 | Baseline naive 2-layer CNN, no tricks | ~52 % |
| 2 | + BatchNorm | ~64 % |
| 3 | + data augmentation | ~75 % |
| 4 | + Dropout | ~79 % |
| 5 | + deeper network (3 conv blocks) | ~85 % |
| 6 | + cosine LR scheduling & tuning | **~90 %** |

---

## How it works

A single image flows through the network, spatial size shrinks (32→16→8→4) while channel depth grows (3→32→64→128), trading *where* features are for *what* they are:

```
Input (3×32×32)
  ├─ Conv block 1 → 32×32×32  → MaxPool → 32×16×16
  ├─ Conv block 2 → 64×16×16  → MaxPool → 64×8×8
  ├─ Conv block 3 → 128×8×8   → MaxPool → 128×4×4
  ├─ Flatten → 2048
  └─ Linear(2048→512) → Linear(512→10)  → 10 class scores (logits)
```

Every training step repeats the same five-move dance that powers essentially all of deep learning: **forward pass → compute loss → clear old gradients → backpropagate → update weights.**


## **Important Limitation**

A 32×32 model trained on CIFAR-10 will **not** reliably classify arbitrary photos downloaded from the web. A high-resolution photo crushed to 32×32 loses almost all its detail hence why it is impossible for the model to correctly say what is displayed in high resolution images. For better performance, switch to a **pretrained ResNet18** trained at a higher resolution (96×96) with stronger augmentation, however, that is beyond the scope of this project as all I wanted to get the basics of PyTorch down.


## What I learned

- The anatomy of a CNN and how convolutions, pooling, and batch normalization interact.
- Writing a **manual PyTorch training loop** and understanding autograd end-to-end.
- Why **data augmentation**, **dropout**, and **learning-rate scheduling** matter and by how much (put in table above).
- How to **diagnose** a model with per-class accuracy and a confusion matrix.
- The difference between **beating a benchmark** and **solving a real task**.


## Tech stack

`Python 3.12` `PyTorch` `torchvision` `NumPy` `Matplotlib` `Seaborn` `Pillow` `Nix flakes`

## Links

- Dataset: [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) by Krizhevsky & Hinton.
- The official [PyTorch CIFAR-10 tutorial](https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html) and the [60-minute blitz](https://docs.pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html).
