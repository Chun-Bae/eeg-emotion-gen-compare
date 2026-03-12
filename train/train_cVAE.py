#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, time, argparse
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

from utils.device_selection import device_selection
from data_processing.data_load import initial_dreamer_load
from data_processing.data_spilt import gen_data_split
from models.cVAE import CondVAE1D
from utils.load_classifier import load_classifier


# ---------------------------
# Helpers
# ---------------------------
@torch.no_grad()
def _from_4class(y4: torch.Tensor):
    y4 = y4.long().view(-1)
    yv = (y4 // 2).float()
    ya = (y4 %  2).float()
    return yv, ya

def _bin_metrics(y_true, y_prob, thr=0.5):
    y_true = np.asarray(y_true).astype(np.uint8)
    y_prob = np.asarray(y_prob).astype(np.float32)
    y_pred = (y_prob >= thr).astype(np.uint8)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = float("nan")
    return {"prec": float(prec), "rec": float(rec), "f1": float(f1), "auc": float(auc)}


# ---------------------------
# Losses (선택 가능)
# ---------------------------
def recon_loss(xhat, x, recon_type="smoothl1", huber_beta=0.02):
    if recon_type == "l1":
        return F.l1_loss(xhat, x, reduction="mean")
    elif recon_type == "mse":
        return F.mse_loss(xhat, x, reduction="mean")
    elif recon_type == "smoothl1":
        return F.smooth_l1_loss(xhat, x, beta=huber_beta, reduction="mean")
    elif recon_type == "mix":
        return 0.5 * F.l1_loss(xhat, x, reduction="mean") + 0.5 * F.mse_loss(xhat, x, reduction="mean")
    else:
        raise ValueError(f"Unknown recon_type: {recon_type}")

def kl_loss(mu, logv):
    return -0.5 * torch.mean(1 + logv - mu.pow(2) - logv.exp())

def kl_loss_freebits(mu, logv, fb_nats=0.2):
    # free-bits: 각 차원 fb_nats까지는 '공짜', 초과분만 벌점
    kl_pd = -0.5 * (1 + logv - mu.pow(2) - logv.exp())   # (B, z_dim)
    kl_eff = torch.clamp(kl_pd - fb_nats, min=0.0)       # ReLU(kl_pd - fb)
    return kl_eff.sum(dim=1).mean()

@torch.no_grad()
def kl_loss_raw(mu, logv):
    return (-0.5 * (1 + logv - mu.pow(2) - logv.exp())).sum(dim=1).mean()


# ---------------------------
# Train VAE (per-epoch 평가 포함)
# ---------------------------
def train_vae(
    vae, loader, device,
    epochs=80, lr=2e-4, save_path="./experiments/vae_best.pth",
    clf=None, eval_K=128, min_auc=0.60, eval_save_path="./experiments/vae_best_by_clf_eval.pth",
    # beta-anneal
    beta0=0.0, beta1=1.0, warm=10,
    # free-bits
    use_free_bits=False, free_bits_nats=0.2,
    # recon
    recon_type="smoothl1", huber_beta=0.02,
    # scale penalty
    alpha_scale=0.05,
    # misc
    z_dim=128, n_class=4
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if clf is not None:
        clf.eval()
        for p in clf.parameters():
            p.requires_grad = False

    opt = torch.optim.Adam(vae.parameters(), lr=lr)
    best = float("inf"); best_state = None

    # 평가 기준 최고 점수 로드(있으면 이어받기)
    best_eval = -1.0

    for ep in range(1, epochs+1):
        t0 = time.time()
        vae.train()
        Lr, Lk, Lt, n_batches = 0.0, 0.0, 0.0, 0

        # β warmup
        beta = beta0 + (beta1 - beta0) * min(1.0, ep / warm)

        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            xhat, mu, logv = vae(xb, yb)

            rec = recon_loss(xhat, xb, recon_type=recon_type, huber_beta=huber_beta)
            kl  = kl_loss_freebits(mu, logv, free_bits_nats) if use_free_bits else kl_loss(mu, logv)

            # 출력 진폭 수축 방지(채널별 표준편차 매칭)
            std_x   = xb.std(dim=(0,2), unbiased=False)                       # target
            std_hat = xhat.std(dim=(0,2), unbiased=False).clamp_min(1e-6)     # pred
            L_scale = F.l1_loss(std_hat, std_x)

            # 최종 손실
            loss = rec + beta * kl + alpha_scale * L_scale

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            Lr += rec.item(); Lk += kl.item(); Lt += loss.item(); n_batches += 1

        dt = time.time() - t0
        print(f"[{ep:03d}/{epochs}] dt={dt:.2f}s  Recon={Lr/n_batches:.4f}  KL={Lk/n_batches:.4f}  Total={Lt/n_batches:.4f}  (beta={beta:.3f})")

        # ---- 분류기 기반 평가 (prior, 진단용)
        if clf is not None:
            vae.eval()
            with torch.no_grad():
                z_eval = torch.randn(eval_K, z_dim, device=device)
                labels_eval = torch.randint(low=0, high=n_class, size=(eval_K,), device=device)
                xg = vae.decode(z_eval, labels_eval)

                out = clf(xg)   # {"val":(K,), "aro":(K,)}
                pv = torch.sigmoid(out["val"]).view(-1)
                pa = torch.sigmoid(out["aro"]).view(-1)
                yv, ya = _from_4class(labels_eval)

                mv = _bin_metrics(yv.cpu().numpy(), pv.cpu().numpy())
                ma = _bin_metrics(ya.cpu().numpy(), pa.cpu().numpy())
                f1_mean = 0.5 * (mv["f1"] + ma["f1"])

            print(
                f"eval(K={eval_K}): "
                f"V[F1={mv['f1']:.4f}, AUC={mv['auc']:.4f}]  "
                f"A[F1={ma['f1']:.4f}, AUC={ma['auc']:.4f}]  "
                f"avgF1={f1_mean:.4f}"
            )

        # ---- 디버그 (posterior 한 배치)
        with torch.no_grad():
            xb_dbg, yb_dbg = next(iter(loader))
            xb_dbg, yb_dbg = xb_dbg.to(device), yb_dbg.to(device)
            xhat_dbg, mu_dbg, logv_dbg = vae(xb_dbg, yb_dbg)
            print(f"[dbg] mu|abs_mean={mu_dbg.abs().mean():.3f}, logv|mean={logv_dbg.mean():.3f}, "
                  f"std real={xb_dbg.std():.3f} gen={xhat_dbg.std():.3f}, KL_raw={kl_loss_raw(mu_dbg, logv_dbg).item():.3f}")

        # ---- (선택) 평가 기준 저장
        if clf is not None:
            if (f1_mean > best_eval) and (mv["auc"] >= min_auc) and (ma["auc"] >= min_auc):
                best_eval = f1_mean
                torch.save({
                    "epoch": ep,
                    "score_avg_f1": float(best_eval),
                    "val_metrics": mv,
                    "aro_metrics": ma,
                    "K": int(eval_K),
                    "z_dim": int(z_dim),
                    "device": str(device),
                    "VAE_state_dict": vae.state_dict(),
                    "note": "selection by avg F1 over synthetic eval set (labels mixed)",
                }, eval_save_path)
                print(f"   ↳ improved, saved {{VAE}}+metrics -> {eval_save_path}")

        # ---- (A) 손실 기준 저장 (옵션, 필요하면 유지)
        cur = Lt / n_batches
        if cur < best:
            best = cur
            best_state = {
                "epoch": ep,
                "model": vae.state_dict(),
                "optim": opt.state_dict(),
                "beta": beta,
                "note": "CondVAE EEG",
            }
        vae.train()

    # 손실 기준 최종 저장
    if best_state is not None:
        torch.save(best_state, save_path)
        print(f"✅ Saved best(by loss) VAE to {save_path} (total={best:.4f})")
    else:
        print("No improvement; nothing saved.")


# ---------------------------
# Argparse
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser("Train Conditional VAE for EEG (with per-epoch classifier eval)")
    # 모델/데이터
    p.add_argument("--C_in", type=int, default=70)
    p.add_argument("--T_len", type=int, default=1280)
    p.add_argument("--n_class", type=int, default=4)
    p.add_argument("--z_dim", type=int, default=128)
    p.add_argument("--emb_dim", type=int, default=32)
    p.add_argument("--base", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=64)
    # 학습
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=2e-4)
    # beta-annealing
    p.add_argument("--beta0", type=float, default=0.0)
    p.add_argument("--beta1", type=float, default=1.0)
    p.add_argument("--warm", type=int, default=10)
    # free-bits
    p.add_argument("--use_free_bits", action="store_true", default=False)
    p.add_argument("--free_bits_nats", type=float, default=0.2)
    # 재구성
    p.add_argument("--recon_type", type=str, default="smoothl1", choices=["smoothl1","l1","mse","mix"])
    p.add_argument("--huber_beta", type=float, default=0.02)
    # scale penalty
    p.add_argument("--alpha_scale", type=float, default=0.05)
    # 평가
    p.add_argument("--eval_K", type=int, default=128)
    p.add_argument("--min_auc", type=float, default=0.55)
    # 경로
    p.add_argument("--save_path", type=str, default="./experiments/vae_best.pth")
    p.add_argument("--eval_save_path", type=str, default="./experiments/vae_best_by_clf_eval.pth")
    p.add_argument("--clf_path", type=str, default="./experiments/classifier_best.pth")
    return p.parse_args()


# ---------------------------
# Main
# ---------------------------
def main():
    args = parse_args()

    # --- Config / Env
    C_in, T_len = args.C_in, args.T_len
    n_class, z_dim = args.n_class, args.z_dim
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr

    beta0, beta1, warm = args.beta0, args.beta1, args.warm
    use_free_bits = args.use_free_bits
    free_bits_nats = args.free_bits_nats

    recon_type = args.recon_type
    huber_beta = args.huber_beta
    alpha_scale = args.alpha_scale

    eval_K = args.eval_K
    min_auc = args.min_auc
    save_path = args.save_path
    eval_save_path = args.eval_save_path
    clf_path = args.clf_path

    # --- Device, Data, Model, Classifier
    device = device_selection()
    print(f"============================\n선택된 디바이스              {device}\n============================")

    train_loader, _ = gen_data_split(df=initial_dreamer_load(), batch_size=batch_size, seed=42)

    vae = CondVAE1D(C_in=C_in, T_len=T_len, n_class=n_class,
                    z_dim=z_dim, emb_dim=args.emb_dim, base=args.base).to(device)

    clf = load_classifier(checkpoint_path=clf_path, device=device)
    clf.eval()
    for p in clf.parameters():
        p.requires_grad = False

    # --- Train
    train_vae(
        vae, train_loader, device,
        epochs=epochs, lr=lr, save_path=save_path,
        clf=clf, eval_K=eval_K, min_auc=min_auc, eval_save_path=eval_save_path,
        beta0=beta0, beta1=beta1, warm=warm,
        use_free_bits=use_free_bits, free_bits_nats=free_bits_nats,
        recon_type=recon_type, huber_beta=huber_beta,
        alpha_scale=alpha_scale,
        z_dim=z_dim, n_class=n_class
    )


if __name__ == "__main__":
    main()