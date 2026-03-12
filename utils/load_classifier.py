from models.classifier import Classifier
import torch

def load_classifier(checkpoint_path, device):
    clf = Classifier(in_channels=5*14, n_classes=4).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    clf.load_state_dict(state, strict=False)
    clf.eval()
    for p in clf.parameters():
        p.requires_grad = False
    return clf