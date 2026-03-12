#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, argparse
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, roc_auc_score

# --- project imports (네 구조에 맞춤)
from utils.device_selection import device_selection
from data_processing.data_load import initial_dreamer_load
from data_processing.data_spilt import gen_data_split
from models.cVAE import CondVAE1D
from utils.load_classifier import load_classifier
2
# ---------------------------
# utils
# ---------------------------
@torch.no_grad()
def _from_4class(y4: torch.Tensor):
    y4 = y4.long().view(-1)
    yv = (y4 // 2).float()
    ya = (y4 %  2).float()
    return yv, ya

def _bin_metrics(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")
    return acc, prec, rec, f1, auc

# ---------------------------
# VAE → Classifier (batched eval)
# ---------------------------
@torch.no_grad()
def vae_eval_batched(
    vae, clf, device,
    num_samples=100000, batch=256,
    z_dim=128, n_class=4,
    mode="prior",              # "prior" | "posterior"
    loader_for_posterior=None, # posterior일 때 real 배치 1개 필요
    labels_provider=None,      # 라벨 함수: lambda b: torch.randint(0,n_class,(b,),device)
    rescale=False              # 평가용 재스케일링
):
    clf.eval(); vae.eval()

    # 라벨 공급자 기본값
    if labels_provider is None:
        labels_provider = lambda b: torch.randint(0, n_class, (b,), device=device)

    # posterior: real 배치 1개에서 (mu, y) 추출
    if mode == "posterior":
        assert loader_for_posterior is not None, "posterior 모드에는 loader_for_posterior가 필요합니다."
        xb_eval, yb_eval = next(iter(loader_for_posterior))
        xb_eval, yb_eval = xb_eval.to(device), yb_eval.to(device)
        _, mu_eval, _ = vae(xb_eval, yb_eval)
        num_samples = mu_eval.size(0)  # posterior는 이 배치 크기로 고정

    pv_list, pa_list, yv_list, ya_list = [], [], [], []

    # (선택) rescale용 통계: posterior면 real 배치 통계 사용
    if rescale and mode == "posterior":
        mean_r = xb_eval.mean(dim=(0,2), keepdim=True)
        std_r  = xb_eval.std(dim=(0,2), keepdim=True).clamp_min(1e-6)

    for i in range(0, num_samples, batch):
        b = min(batch, num_samples - i)

        if mode == "prior":
            z = torch.randn(b, z_dim, device=device)
            labels = labels_provider(b)
            xg = vae.decode(z, labels)
        else:
            z = mu_eval[i:i+b]
            labels = yb_eval[i:i+b]
            xg = vae.decode(z, labels)

        # (선택) 평가-전용 rescale
        if rescale:
            mean_g = xg.mean(dim=(0,2), keepdim=True)
            std_g  = xg.std(dim=(0,2), keepdim=True).clamp_min(1e-6)
            xg = (xg - mean_g) / std_g
            if mode == "posterior":
                xg = xg * std_r + mean_r

        out = clf(xg)
        pv = torch.sigmoid(out["val"]).detach().cpu().numpy()
        pa = torch.sigmoid(out["aro"]).detach().cpu().numpy()
        yv = ((labels // 2).float().cpu().numpy()).astype(int)
        ya = ((labels %  2).float().cpu().numpy()).astype(int)

        pv_list.append(pv); pa_list.append(pa)
        yv_list.append(yv); ya_list.append(ya)

    # concat
    pv = np.concatenate(pv_list, axis=0)
    pa = np.concatenate(pa_list, axis=0)
    yv = np.concatenate(yv_list, axis=0)
    ya = np.concatenate(ya_list, axis=0)

    # joint 4-class
    p_val = (pv >= 0.5).astype(int)
    p_aro = (pa >= 0.5).astype(int)
    y4 = 2*yv + ya
    pred4 = 2*p_val + p_aro

    acc4 = accuracy_score(y4, pred4)
    f1m4 = f1_score(y4, pred4, average="macro")

    v_acc, v_prec, v_rec, v_f1, v_auc = _bin_metrics(yv, pv)
    a_acc, a_prec, a_rec, a_f1, a_auc = _bin_metrics(ya, pa)

    tag = f"{mode}{' +rescale' if rescale else ''}"

    print(f"[VAE→CLF {tag}] Joint4 Acc={acc4:.4f} F1m={f1m4:.4f} (N={len(y4)})")
    print(f"  [Val] Acc={v_acc:.4f} Prec={v_prec:.4f} Rec={v_rec:.4f} "
        f"F1={v_f1:.4f} AUC={v_auc:.4f}")
    print(f"  [Aro] Acc={a_acc:.4f} Prec={a_prec:.4f} Rec={a_rec:.4f} "
        f"F1={a_f1:.4f} AUC={a_auc:.4f}")

    return {
        "acc4": acc4, "f1m4": f1m4,
        "val": {"Acc": v_acc, "Prec": v_prec, "Rec": v_rec, "F1": v_f1, "AUC": v_auc},
        "aro": {"Acc": a_acc, "Prec": a_prec, "Rec": a_rec, "F1": a_f1, "AUC": a_auc},
        "N": int(len(y4))
    }

# ---------------------------
# argparse
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser("Evaluate cVAE by batched generation → classifier")
    # Model spec
    p.add_argument("--C_in", type=int, default=70)
    p.add_argument("--T_len", type=int, default=1280)
    p.add_argument("--n_class", type=int, default=4)
    p.add_argument("--z_dim", type=int, default=128)
    p.add_argument("--emb_dim", type=int, default=32)
    p.add_argument("--base", type=int, default=128)
    # Paths
    p.add_argument("--vae_ckpt", type=str, default="./experiments/vae_best_by_clf_eval.pth")
    p.add_argument("--clf_path", type=str, default="./experiments/classifier_best.pth")
    # Eval config
    p.add_argument("--mode", type=str, choices=["prior","posterior"], default="prior")
    p.add_argument("--num_samples", type=int, default=100000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--rescale", action="store_true", default=False)
    p.add_argument("--balanced_labels", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size_loader", type=int, default=64, help="posterior 모드에서 로더 배치 크기")
    return p.parse_args()

# ---------------------------
# main
# ---------------------------
def main():
    args = parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    device = device_selection()
    print(f"============================\n선택된 디바이스              {device}\n============================")

    # --- 모델/클래스파이어 로드
    vae = CondVAE1D(C_in=args.C_in, T_len=args.T_len, n_class=args.n_class,
                    z_dim=args.z_dim, emb_dim=args.emb_dim, base=args.base).to(device)

    ckpt = torch.load(args.vae_ckpt, map_location=device)
    state_key = "VAE_state_dict" if "VAE_state_dict" in ckpt else "model"
    vae.load_state_dict(ckpt[state_key], strict=False)
    vae.eval()
    print(f"✅ Loaded VAE checkpoint from {args.vae_ckpt} (epoch={ckpt.get('epoch','?')})")

    clf = load_classifier(checkpoint_path=args.clf_path, device=device)
    clf.eval()
    for p in clf.parameters():
        p.requires_grad = False

    # --- posterior 모드면 로더 준비
    loader_for_posterior = None
    if args.mode == "posterior":
        df = initial_dreamer_load()
        loader_for_posterior, _ = gen_data_split(df=df, batch_size=args.batch_size_loader, seed=args.seed)

    # --- 라벨 공급자 (균형 라벨)
    def balanced_labels_provider(b):
        if not args.balanced_labels:
            return torch.randint(0, args.n_class, (b,), device=device)
        reps = max(1, (b + args.n_class - 1) // args.n_class)
        arr = np.tile(np.arange(args.n_class, dtype=np.int64), reps)[:b]
        return torch.from_numpy(arr).to(device)

    # --- 평가 실행
    _ = vae_eval_batched(
        vae, clf, device,
        num_samples=args.num_samples, batch=args.batch,
        z_dim=args.z_dim, n_class=args.n_class,
        mode=args.mode,
        loader_for_posterior=loader_for_posterior,
        labels_provider=balanced_labels_provider,
        rescale=args.rescale
    )

if __name__ == "__main__":
    main()