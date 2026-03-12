from models.classifier import Classifier
from utils.device_selection import device_selection
from data_processing.data_load import initial_dreamer_load
from data_processing.data_spilt import clf_data_split
import argparse, os, time
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

import random
import torch
import torch.nn as nn

# ======= metrics =======
def step_metrics(logits, targets):
    """
    logits: (B,) raw logit
    targets: (B,) float(0/1) or long(0/1)
    """
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    preds = (probs >= 0.5).astype("int64")
    y     = targets.detach().cpu().numpy().astype("int64")

    prec, rec, f1, _ = precision_recall_fscore_support(
        y, preds, average="binary", zero_division=0
    )
    try:
        auc = roc_auc_score(y, probs)   # 한쪽 클래스만 나오면 예외
    except ValueError:
        auc = float("nan")
    acc = (preds == y).mean()
    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1, "auc": auc}


def _avg_metric_list(m_list, key):
    return float(np.mean([m[key] for m in m_list])) if m_list else 0.0


def _pack_epoch_metrics(agg):
    # agg: {"val":[dict...], "aro":[dict...]}
    return {
        "val": {k: _avg_metric_list(agg["val"], k) for k in ["acc","prec","rec","f1","auc"]},
        "aro": {k: _avg_metric_list(agg["aro"], k) for k in ["acc","prec","rec","f1","auc"]},
    }



# ======= train / eval =======
def train_one_epoch(model, loader, opt, device, criterion):
    model.train()
    total_loss, n = 0.0, 0
    agg = {"val": [], "aro": []}

    for Xb, yv, ya in loader:
        Xb, yv, ya = Xb.to(device), yv.to(device).float(), ya.to(device).float()

        opt.zero_grad()
        out = model(Xb)  # {"val": (B,), "aro": (B,)}
        loss = criterion(out["val"], yv) + criterion(out["aro"], ya)
        loss.backward()
        opt.step()

        bs = Xb.size(0)
        total_loss += loss.item() * bs
        n += bs

        agg["val"].append(step_metrics(out["val"], yv))
        agg["aro"].append(step_metrics(out["aro"], ya))

    return total_loss / max(n, 1), _pack_epoch_metrics(agg)


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss, n = 0.0, 0
    agg = {"val": [], "aro": []}

    for Xb, yv, ya in loader:
        Xb, yv, ya = Xb.to(device), yv.to(device).float(), ya.to(device).float()

        out = model(Xb)
        loss = criterion(out["val"], yv) + criterion(out["aro"], ya)

        bs = Xb.size(0)
        total_loss += loss.item() * bs
        n += bs

        agg["val"].append(step_metrics(out["val"], yv))
        agg["aro"].append(step_metrics(out["aro"], ya))

    return total_loss / max(n, 1), _pack_epoch_metrics(agg)


def train_valid_log(epoch, epochs, tr_loss, va_loss, tr_m, va_m, dt):
    print(
        "\n" + "=" * 60 +
        f"\n[Epoch {epoch:02d}/{epochs}]  Time: {dt:.2f}s" +
        "\n" + "-" * 60 +
        f"\n | Loss       | Train: {tr_loss:.4f}   Val: {va_loss:.4f}" +
        "\n" + "-" * 60 +
        "\n |      V (Valence)      |      A (Arousal)     " +
        "\n" + "-" * 60 +
        f"\n | Acc : {tr_m['val']['acc']:<6.4f} / {va_m['val']['acc']:<6.4f} "
        f"| {tr_m['aro']['acc']:<6.4f} / {va_m['aro']['acc']:<6.4f}" +
        f"\n | Prec: {tr_m['val']['prec']:<6.4f} / {va_m['val']['prec']:<6.4f} "
        f"| {tr_m['aro']['prec']:<6.4f} / {va_m['aro']['prec']:<6.4f}" +
        f"\n | Rec : {tr_m['val']['rec']:<6.4f} / {va_m['val']['rec']:<6.4f} "
        f"| {tr_m['aro']['rec']:<6.4f} / {va_m['aro']['rec']:<6.4f}" +
        f"\n | F1  : {tr_m['val']['f1']:<6.4f} / {va_m['val']['f1']:<6.4f} "
        f"| {tr_m['aro']['f1']:<6.4f} / {va_m['aro']['f1']:<6.4f}" +
        f"\n | AUC : {tr_m['val']['auc']:<6.4f} / {va_m['val']['auc']:<6.4f} "
        f"| {tr_m['aro']['auc']:<6.4f} / {va_m['aro']['auc']:<6.4f}" +
        "\n" + "=" * 60
    )



def run_train(model, train_loader, val_loader, device,
                 epochs=100, lr=1e-3, weight_decay=1e-5,
                 save_path="./experiments/classfier_best.pth", seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
        # ======= training loop =======
    # 모델/옵티마이저/로스
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {
        "epoch": [],
        "train_loss": [], "val_loss": [],

        # V(Valence) - train/val
        "V_acc": [],   "V_prec": [],   "V_rec": [],   "V_f1": [],   "V_auc": [],
        "V_val_acc": [], "V_val_prec": [], "V_val_rec": [], "V_val_f1": [], "V_val_auc": [],

        # A(Arousal) - train/val
        "A_acc": [],   "A_prec": [],   "A_rec": [],   "A_f1": [],   "A_auc": [],
        "A_val_acc": [], "A_val_prec": [], "A_val_rec": [], "A_val_f1": [], "A_val_auc": [],
    }

    best_score = -1.0
    best_state = None

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_m = train_one_epoch(model, train_loader, optimizer, device, criterion)
        va_loss, va_m = evaluate(model, val_loader, device, criterion)
        dt = time.time() - t0

        # 로깅 출력
        train_valid_log(epoch, epochs, tr_loss, va_loss, tr_m, va_m, dt)

        # 통합 지표(Valence/Arousal 평균)
        def mean_task(m, key):
            return 0.5 * (m["val"][key] + m["aro"][key])

        history["epoch"].append(epoch)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)

        # --- V(Valence) 개별 ---
        for k in ["acc","prec","rec","f1","auc"]:
            history[f"V_{k}"].append(tr_m["val"][k])
            history[f"V_val_{k}"].append(va_m["val"][k])

        # --- A(Arousal) 개별 ---
        for k in ["acc","prec","rec","f1","auc"]:
            history[f"A_{k}"].append(tr_m["aro"][k])
            history[f"A_val_{k}"].append(va_m["aro"][k])

        val_f1_mean = mean_task(va_m, "f1")
        if val_f1_mean > best_score:
            best_score = val_f1_mean
            best_state = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_f1_mean": float(best_score),
                "note": "selection = mean(F1_val, F1_aro)",
                "history": {k: list(v) for k, v in history.items()},
            }
    # ======= save =======
    final_state = {
        "best_state": best_state,
        "full_history": {k: list(v) for k, v in history.items()}
    }
    os.makedirs("./experiments", exist_ok=True)
    torch.save(final_state, save_path)

    if best_state is not None:
        print(f"\nBest mean F1={best_state['best_f1_mean']:.4f} (epoch {best_state['epoch']}) saved to {save_path}")
    else:
        print("\nNo improvement recorded.")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--save-path", type=str, default="./experiments/classfier_best.pth")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    device = device_selection()
    model = Classifier(in_channels=5*14, n_classes=4).to(device)
    train_loader, val_loader = clf_data_split(df=initial_dreamer_load(), batch_size=args.batch_size, seed=args.seed)

    run_train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        save_path=args.save_path,
        seed=args.seed,
    )