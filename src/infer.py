from PIL import Image
import torch
import torchvision.transforms as T

from config import CLASSES, MEAN, STD, CKPT_DIR
from model import CIFAR10CNN
from utils import get_device, load_checkpoint


def main(image_path: str = "img.png"):
    device = get_device()
    model = CIFAR10CNN().to(device)
    load_checkpoint(model, CKPT_DIR / "best.pth", device)
    model.eval()

    preprocess = T.Compose([
        T.Resize((32, 32)),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])

    img = preprocess(Image.open(image_path).convert("RGB"))
    img = img.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(img)
        pred = logits.argmax(dim=1).item()
        probs = torch.softmax(logits, dim=1)[0]

    print(f"Prediction: {CLASSES[pred]} ({probs[pred]:.1%})")
    for name, p in sorted(zip(CLASSES, probs.tolist()), key=lambda x: -x[1])[:3]:
        print(f"  {name:10s} {p:.1%}")


if __name__ == "__main__":
    main()
