import argparse
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, roc_auc_score
from models.classifier import Classifier
from models.cGAN import Gen1D
from utils.device_selection import device_selection

@torch.no_grad()
def _from_4class(y4: torch.Tensor):
    y4 = y4.long().view(-1)
    yv = ((y4 >> 1) & 1).float()
    ya = (y4 & 1).float()
    return yv, ya

def bin_metrics(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")
    return acc, prec, rec, f1, auc

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--save-path", type=str, default="./experiments/c_gan_best.pth")
    p.add_argument("--clf-path", type=str, default="./experiments/classfier_best.pth")
    p.add_argument("--num-samples", type=int, default=100000)
    p.add_argument("--batch", type=int, default=256)
    return p.parse_args()

def main():
    args = parse_args()
    device = device_selection()

    # ----- Classifier 로드 -----
    clf = Classifier(in_channels=5*14, n_classes=4).to(device)
    clf_ckpt = torch.load(args.clf_path, map_location=device)
    clf_state = clf_ckpt["model"] if "model" in clf_ckpt else clf_ckpt
    clf.load_state_dict(clf_state, strict=False)
    clf.eval()

    # ----- Generator 로드 -----
    ckpt = torch.load(args.save_path, map_location="cpu")
    z_dim = ckpt.get("z_dim", 128)
    n_classes = ckpt.get("n_classes", 4)
    G = Gen1D(z_dim=z_dim, n_class=n_classes, C_out=5*14, T_len=1280).to(device)
    G.load_state_dict(ckpt["G_state_dict"])
    G.eval()

    # ----- 생성 & 평가 -----
    pv_list, pa_list, yv_list, ya_list = [], [], [], []
    with torch.no_grad():
        for i in range(0, args.num_samples, args.batch):
            b = min(args.batch, args.num_samples - i)
            z = torch.randn(b, z_dim, device=device)
            labels = torch.randint(0, n_classes, (b,), device=device)

            xg = G(z, labels)
            out = clf(xg)

            pv = torch.sigmoid(out["val"]).cpu()
            pa = torch.sigmoid(out["aro"]).cpu()
            yv, ya = _from_4class(labels)
            yv = yv.cpu(); ya = ya.cpu()

            pv_list.append(pv); pa_list.append(pa)
            yv_list.append(yv); ya_list.append(ya)

    # concat
    pv = torch.cat(pv_list).numpy()
    pa = torch.cat(pa_list).numpy()
    yv = torch.cat(yv_list).numpy().astype(int)
    ya = torch.cat(ya_list).numpy().astype(int)

    # joint 4-class
    p_val = (pv >= 0.5).astype(int)
    p_aro = (pa >= 0.5).astype(int)
    y4 = (2*yv + ya)
    pred4 = (2*p_val + p_aro)

    acc4 = accuracy_score(y4, pred4)
    f1m4 = f1_score(y4, pred4, average="macro")
    print(f"[Joint 4-class] Acc={acc4:.4f}  F1(macro)={f1m4:.4f}")

    # per-head
    acc, prec, rec, f1, auc = bin_metrics(yv, pv)
    print(f"[Val] Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f}  AUC={auc:.4f}")
    acc, prec, rec, f1, auc = bin_metrics(ya, pa)
    print(f"[Aro] Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f}  AUC={auc:.4f}")

if __name__ == "__main__":
    main()