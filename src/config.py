from pathlib import Path

# Paths
DATA_DIR = Path("data")
CKPT_DIR = Path("checkpoints")

# Data
BATCH_SIZE = 128
NUM_WORKERS = 2

# Training
EPOCHS = 30
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

# Pre-compute means and stds
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)

CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)
NUM_CLASSES = len(CLASSES)
