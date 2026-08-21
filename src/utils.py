import torch


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(model, path, extra=None):
    state = {"model": model.state_dict()}
    if extra:
        state.update(extra)
    torch.save(state, path)
    print("Saved checkpoint ->", path)


def load_checkpoint(model, path, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    return ckpt
