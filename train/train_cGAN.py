import pandas as pd
import numpy as np
import random
import time
import os
import argparse
from utils.device_selection import device_selection
from data_processing.data_load import initial_dreamer_load
from data_processing.data_spilt import gen_data_split
from models.classifier import Classifier
from models.cGAN import Gen1D, Disc1D
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

import torch
import torch.nn as nn

# ─────────────────────────────
# Loss functions
# ─────────────────────────────
bce_logits = nn.BCEWithLogitsLoss()

def d_loss_fn(d_real, d_fake):
    return bce_logits(d_real, torch.ones_like(d_real)) + \
           bce_logits(d_fake, torch.zeros_like(d_fake))

def g_loss_fn(d_fake):
    return bce_logits(d_fake, torch.ones_like(d_fake))


# Metric utils
# ─────────────────────────────
@torch.no_grad()
def _from_4class(y4: torch.Tensor):
    y4 = y4.long().view(-1)
    yv = ((y4 >> 1) & 1).float()
    ya = (y4 & 1).float()
    return yv, ya

@torch.no_grad()
def _bin_metrics(y_true_t: torch.Tensor, y_prob_t: torch.Tensor, thr: float = 0.5):
    y_true = y_true_t.cpu().numpy().astype("int64")
    y_prob = y_prob_t.cpu().numpy()
    y_pred = (y_prob >= thr).astype("int64")
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")
    acc = (y_true == y_pred).mean()
    return {"acc": float(acc), "prec": float(prec), "rec": float(rec), "f1": float(f1), "auc": float(auc)}


def run_train_cgan(epochs, batch_size, z_dim, n_class, c_in, t_len, lr, betas, min_auc, save_path, seed):
    device = device_selection()


    clf = Classifier(in_channels=5*14, n_classes=4).to(device)
    clf_ckpt = torch.load("./experiments/classfier_best.pth", map_location=device)
    state = clf_ckpt["model"] if isinstance(clf_ckpt, dict) and "model" in clf_ckpt else clf_ckpt
    clf.load_state_dict(state, strict=False)
    clf.eval()

    # 데이터
    train_loader, _ = gen_data_split(df=initial_dreamer_load(), batch_size=batch_size, seed=seed)

    # 모델
    G = Gen1D(z_dim=z_dim, n_class=n_class, C_out=c_in, T_len=t_len).to(device)
    D = Disc1D(C_in=c_in, n_class=n_class).to(device)

    optG = torch.optim.Adam(G.parameters(), lr=lr, betas=betas)
    optD = torch.optim.Adam(D.parameters(), lr=lr, betas=betas)

    # 고정 평가 샘플
    K = 128
    with torch.no_grad():
        labels_eval = torch.arange(n_class, device=device).repeat_interleave(K)
        z_eval = torch.randn(len(labels_eval), z_dim, device=device)

    best = -1.0
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 학습 루프
    for ep in range(1, epochs + 1):
        t0 = time.time()
        G.train(); D.train()
        d_loss_acc, g_loss_acc, n_batches = 0.0, 0.0, 0

        for real, y in train_loader:
            real, y = real.to(device), y.to(device)
            b = real.size(0)

            # D step
            z = torch.randn(b, z_dim, device=device)
            fake = G(z, y).detach()
            d_real = D(real, y)
            d_fake = D(fake, y)
            loss_d = d_loss_fn(d_real, d_fake)
            optD.zero_grad(set_to_none=True); loss_d.backward(); optD.step()

            # G step
            z = torch.randn(b, z_dim, device=device)
            fake = G(z, y)
            d_fake = D(fake, y)
            loss_g = g_loss_fn(d_fake)
            optG.zero_grad(set_to_none=True); loss_g.backward(); optG.step()

            d_loss_acc += loss_d.item()
            g_loss_acc += loss_g.item()
            n_batches += 1

        dt = time.time() - t0
        d_epoch = d_loss_acc / max(n_batches, 1)
        g_epoch = g_loss_acc / max(n_batches, 1)
        print(f"[{ep:02d}/{epochs}]  dt={dt:.2f}s  D={d_epoch:.4f}  G={g_epoch:.4f}")

        # 검증
        G.eval()
        with torch.no_grad():
            xg = G(z_eval, labels_eval)
            out = clf(xg)
            pv = torch.sigmoid(out["val"])
            pa = torch.sigmoid(out["aro"])
            yv, ya = _from_4class(labels_eval)
            mv = _bin_metrics(yv, pv)
            ma = _bin_metrics(ya, pa)
            f1_mean = 0.5 * (mv["f1"] + ma["f1"])

        print(
            f"eval(K={K}): \n"
            f"V[F1={mv['f1']:.4f}, AUC={mv['auc']:.4f}]  \n"
            f"A[F1={ma['f1']:.4f}, AUC={ma['auc']:.4f}]  \n"
            f"avgF1={f1_mean:.4f}\n"
        )

        # 모델 저장
        if f1_mean > best and mv["auc"] > min_auc and ma["auc"] > min_auc:
            best = f1_mean
            torch.save({
                "epoch": ep,
                "K": K,
                "score_avg_f1": float(best),
                "d_loss_epoch": d_epoch,
                "g_loss_epoch": g_epoch,
                "val_metrics": mv,
                "aro_metrics": ma,
                "G_state_dict": G.state_dict(),
                "D_state_dict": D.state_dict(),
                "optG_state_dict": optG.state_dict(),
                "optD_state_dict": optD.state_dict(),
                "z_dim": z_dim,
                "device": str(device),
                "note": "selection by avg F1 over WHOLE eval set (labels mixed)",
            }, save_path)
            print(f"   ↳ improved, saved {{G,D,optG,optD}}+metrics -> {save_path}")

        G.train()


# ─────────────────────────────
# Argparse
# ─────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--z-dim", type=int, default=128)
    p.add_argument("--n-class", type=int, default=4)
    p.add_argument("--c-in", type=int, default=70)
    p.add_argument("--t-len", type=int, default=1280)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--betas", type=float, nargs=2, default=(0.5, 0.999))
    p.add_argument("--min-auc", type=float, default=0.6)
    p.add_argument("--save-path", type=str, default="./experiments/c_gan_best.pth")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_train_cgan(
        epochs=args.epochs,
        batch_size=args.batch_size,
        z_dim=args.z_dim,
        n_class=args.n_class,
        c_in=args.c_in,
        t_len=args.t_len,
        lr=args.lr,
        betas=tuple(args.betas),
        min_auc=args.min_auc,
        save_path=args.save_path,
        seed=args.seed,
    )