#!/usr/bin/env python
# coding: utf-8

# In[4]:


# ---------------------------
# 패키지 
# ---------------------------

# ---- basic ----
import pandas as pd
import numpy as np
import random
import time
import gc

# ---- custom ----
from utils.bandpass_filter import bandpass_filter
from models.classifier import Classifier_EEG, RhythmSpecificDeepCNN, RhythmSpecificDeepCNN22
from models.cDCGAN import G2D, D2D

# ---- sklearn ----
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

# ---- scipy ----
import scipy.io

# ---- torch ----
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, Dataset, DataLoader


# In[5]:


# ---------------------------
# 원시 데이터 가져오기
# ---------------------------
mat = scipy.io.loadmat('./data/DREAMER.mat', struct_as_record=False, squeeze_me=True)
dreamer = mat['DREAMER']


# In[6]:


# ---------------------------
# 데이터 전처리
# ---------------------------
segment_len = 1280              # 10s @ 128 Hz
overlap = 256                   # 겹침 길이(프레임) = 2s
hop = segment_len - overlap     # = 1152 (슬라이딩 간격)

bands = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 14),
    'beta':  (14, 30),
    'gamma': (30, 50)
}

df = []
for subject in dreamer.Data:
    eeg_list = subject.EEG.stimuli
    V, A, D  = subject.ScoreValence, subject.ScoreArousal, subject.ScoreDominance

    for vid_idx, eeg in enumerate(eeg_list):
        T, C = eeg.shape  # (time, channels=14)

        for start in range(0, T - segment_len + 1, hop):
            seg = eeg[start:start+segment_len, :]  # (1280, 14)

            band_list = []
            for (low, high) in bands.values():
                # 각 채널에 밴드패스 적용  (1280, 14)
                filtered = np.stack(
                    [bandpass_filter(seg[:, ch], low, high, fs=128) for ch in range(C)],
                    axis=-1
                )
                band_list.append(filtered)

            # (1280, 5, 14)
            X = np.stack(band_list, axis=1).astype(np.float32)

            df.append({
                'eeg_band_data': X,
                'valence': float(V[vid_idx]),
                'arousal': float(A[vid_idx]),
                'dominance': float(D[vid_idx]),
            })

df = pd.DataFrame(df)
df['y'] = (
    2 * (df['valence'] >= 2.5).astype(int) +
        (df['arousal'] >= 2.5).astype(int)
)
df['y_val'] = (df['valence'].to_numpy() >= 2.5).astype(np.int64)
df['y_aro'] = (df['arousal'].to_numpy() >= 2.5).astype(np.int64)


# In[7]:


# ---------------------------
# 디바이스 선택
# ---------------------------
if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Using device:", device)


# In[8]:


# ---------------------------
# Hyperparams
# ---------------------------
seed_num = 42
batch_size = 512
n_classes = 4

# ---- 공용 ----
epochs = 10
sample_n = 8
betas = (0.5, 0.999)
momentum = 0.9
weight_decay = 1e-4


# In[9]:


# ---------------------------
# seed 고정
# ---------------------------
torch.manual_seed(seed_num)
np.random.seed(seed_num)
random.seed(seed_num)


# In[10]:


# ---------------------------
# Dataset object
# ---------------------------
class EEGDataset(Dataset):
    def __init__(self, X, y_val, y_aro):
        self.X = torch.tensor(X, dtype=torch.float32)  
        self.y_val = torch.tensor(y_val, dtype=torch.float32)
        self.y_aro = torch.tensor(y_aro, dtype=torch.float32)
    
    def __len__(self): 
        return len(self.X)
    
    def __getitem__(self, i):
        return self.X[i], self.y_val[i], self.y_aro[i]


# In[11]:


# ---------------------------
# Data Split
# ---------------------------
X = np.stack(df['eeg_band_data'].values).astype(np.float32)  # (B, 1280, 5, 14)
X = np.transpose(X, (0, 2, 3, 1))                             # (B, 5, 14, 1280)
B, R, C, T = X.shape
X = X.reshape(B, R*C, T).astype(np.float32)                   # (B, 70, 1280)

y = df['y'].values.astype(np.int64)                           # 0~3  (2*V + A)
y_val_bin = (y // 2).astype(np.int64)                         # 0/1
y_aro_bin = (y % 2).astype(np.int64)                          # 0/1

idx = np.arange(len(X))
idx_tr, idx_va = train_test_split(idx, test_size=0.2, random_state=seed_num, stratify=y)
# ---------------------------
# 스케일링 : 표준화
# ---------------------------

X_train, X_val = X[idx_tr], X[idx_va]
y_train_val, y_val_val = y_val_bin[idx_tr], y_val_bin[idx_va]
y_train_aro, y_val_aro = y_aro_bin[idx_tr], y_aro_bin[idx_va]

# (B, 70, 1280) -> (B*1280, 70) 로 바꿔서 채널 표준화
X_train_2d = np.swapaxes(X_train, 1, 2).reshape(-1, R*C) 
X_val_2d   = np.swapaxes(X_val,   1, 2).reshape(-1, R*C)

scaler = StandardScaler()
X_train_2d = scaler.fit_transform(X_train_2d)
X_val_2d   = scaler.transform(X_val_2d)

# 다시 (B, 70, 1280)로 복구
X_train = X_train_2d.reshape(X_train.shape[0], T, R*C).swapaxes(1, 2).astype(np.float32)
X_val   = X_val_2d.reshape(  X_val.shape[0],   T, R*C).swapaxes(1, 2).astype(np.float32)

# ---------------------------
# DataLoader
# ---------------------------
train_dataset = EEGDataset(X_train, y_train_val, y_train_aro)
val_dataset   = EEGDataset(X_val,   y_val_val,   y_val_aro)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  drop_last=False)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, drop_last=False)


# In[12]:


# ---------------------------
# Free
# ---------------------------
for var in [
    'dreamer', 'mat',
    'segment_len', 'overlap', 'hop',
    'bands',
    'subject', 'eeg_list', 'V', 'A', 'D',
    'vid_idx', 'eeg', 'T', 'C', 'start', 'seg',
    'band_list', 'low', 'high', 'filtered',
    'df',
    'scaler',
    'X_scaled',
]:
    if var in globals():
        del globals()[var]

gc.collect()


# In[13]:


# EPS = 1e-12
# bce = nn.BCEWithLogitsLoss()   # 필요시 pos_weight=... 로 클래스 불균형 보정

# # ==== metrics ====
# def step_metrics(logits, targets):
#     """
#     logits: (B,)  raw logit
#     targets: (B,)  float(0/1) or long(0/1)
#     """
#     probs = torch.sigmoid(logits).detach().cpu().numpy()
#     preds = (probs >= 0.5).astype("int64")
#     y     = targets.detach().cpu().numpy().astype("int64")

#     prec, rec, f1, _ = precision_recall_fscore_support(y, preds, average="binary", zero_division=0)
#     try:
#         auc = roc_auc_score(y, probs)   # 한쪽 클래스만 나오면 예외
#     except ValueError:
#         auc = float("nan")
#     acc = (preds == y).mean()
#     return {"acc": acc, "prec": prec, "rec": rec, "f1": f1, "auc": auc}

# def _avg_metric_list(m_list, key):
#     return float(np.mean([m[key] for m in m_list])) if m_list else 0.0

# def _pack_epoch_metrics(agg):
#     # agg: {"val":[dict...], "aro":[dict...]}
#     return {
#         "val": {k: _avg_metric_list(agg["val"], k) for k in ["acc","prec","rec","f1","auc"]},
#         "aro": {k: _avg_metric_list(agg["aro"], k) for k in ["acc","prec","rec","f1","auc"]},
#     }

# # ==== train / eval ====
# def train_one_epoch(model, loader, opt, device):
#     model.train()
#     total_loss, n = 0.0, 0
#     agg = {"val": [], "aro": []}

#     for Xb, yv, ya in loader:
#         Xb, yv, ya = Xb.to(device), yv.to(device).float(), ya.to(device).float()

#         opt.zero_grad()
#         out = model(Xb)  # {"val": (B,), "aro": (B,)}
#         loss = bce(out["val"], yv) + bce(out["aro"], ya)
#         loss.backward()
#         opt.step()

#         bs = Xb.size(0)
#         total_loss += loss.item() * bs
#         n += bs

#         agg["val"].append(step_metrics(out["val"], yv))
#         agg["aro"].append(step_metrics(out["aro"], ya))

#     return total_loss / max(n, 1), _pack_epoch_metrics(agg)

# @torch.no_grad()
# def evaluate(model, loader, device):
#     model.eval()
#     total_loss, n = 0.0, 0
#     agg = {"val": [], "aro": []}

#     for Xb, yv, ya in loader:
#         Xb, yv, ya = Xb.to(device), yv.to(device).float(), ya.to(device).float()

#         out = model(Xb)
#         loss = bce(out["val"], yv) + bce(out["aro"], ya)

#         bs = Xb.size(0)
#         total_loss += loss.item() * bs
#         n += bs

#         agg["val"].append(step_metrics(out["val"], yv))
#         agg["aro"].append(step_metrics(out["aro"], ya))

#     return total_loss / max(n, 1), _pack_epoch_metrics(agg)

# # ==== pretty print ====
# def pretty_log(epoch, epochs, tr_loss, va_loss, tr_m, va_m, dt):
#     print(
#         "\n" + "=" * 60 +
#         f"\n[Epoch {epoch:02d}/{epochs}]  Time: {dt:.2f}s" +
#         "\n" + "-" * 60 +
#         f"\n | Loss       | Train: {tr_loss:.4f}   Val: {va_loss:.4f}" +
#         "\n" + "-" * 60 +
#         "\n |      V (Valence)      |      A (Arousal)     " +
#         "\n" + "-" * 60 +
#         f"\n | Acc : {tr_m['val']['acc']:<6.4f} / {va_m['val']['acc']:<6.4f} "
#         f"| {tr_m['aro']['acc']:<6.4f} / {va_m['aro']['acc']:<6.4f}" +
#         f"\n | Prec: {tr_m['val']['prec']:<6.4f} / {va_m['val']['prec']:<6.4f} "
#         f"| {tr_m['aro']['prec']:<6.4f} / {va_m['aro']['prec']:<6.4f}" +
#         f"\n | Rec : {tr_m['val']['rec']:<6.4f} / {va_m['val']['rec']:<6.4f} "
#         f"| {tr_m['aro']['rec']:<6.4f} / {va_m['aro']['rec']:<6.4f}" +
#         f"\n | F1  : {tr_m['val']['f1']:<6.4f} / {va_m['val']['f1']:<6.4f} "
#         f"| {tr_m['aro']['f1']:<6.4f} / {va_m['aro']['f1']:<6.4f}" +
#         f"\n | AUC : {tr_m['val']['auc']:<6.4f} / {va_m['val']['auc']:<6.4f} "
#         f"| {tr_m['aro']['auc']:<6.4f} / {va_m['aro']['auc']:<6.4f}" +
#         "\n" + "=" * 60
#     )

# # ==== training loop ====
# model = RhythmSpecificDeepCNN22(in_channels=5*14, n_classes=4).to(device)  # 네 정의 그대로, 출력만 멀티헤드로 바꿔둔 버전 사용
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
# epochs = 100

# best_score = -1.0
# best_state = None

# for epoch in range(1, epochs + 1):
#     t0 = time.time()
#     tr_loss, tr_m = train_one_epoch(model, train_loader, optimizer, device)
#     va_loss, va_m = evaluate(model, val_loader, device)
#     dt = time.time() - t0

#     pretty_log(epoch, epochs, tr_loss, va_loss, tr_m, va_m, dt)

#     # ---- 베스트 선정 기준: Val/Aro F1의 평균 (원하면 AUC나 Acc로 바꿔도 됨)
#     val_f1_mean = 0.5 * (va_m["val"]["f1"] + va_m["aro"]["f1"])
#     if val_f1_mean > best_score:
#         best_score = val_f1_mean
#         best_state = {
#             "epoch": epoch,
#             "model": model.state_dict(),
#             "optimizer": optimizer.state_dict(),
#             "best_f1_mean": best_score,
#             "note": "selection = mean(F1_val, F1_aro)",
#         }

# # ==== save ====
# import os
# os.makedirs("./experiments", exist_ok=True)
# if best_state is not None:
#     path = "./experiments/rhythm_cnn_best.pth"
#     torch.save(best_state, path)
#     print(f"\nBest mean F1={best_state['best_f1_mean']:.4f} (epoch {best_state['epoch']}) saved to {path}")
# else:
#     print("\nNo improvement recorded.")


# In[ ]:





# In[25]:


# ============================================================
# Conditional VAE for EEG (70 x 1280) with classifier guidance
# - StandardScaler 스케일 가정 (출력 활성 없음)
# - β-annealing, L1+MSE recon, (옵션) free-bits
# - Classifier-consistency auxiliary loss (멀티헤드 분류기 freeze)
# ============================================================

import os, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_recall_fscore_support

# ---------------------------
# Config / Env
# ---------------------------
C_in, T_len = 70, 1280
n_class, z_dim = 4, 128
batch_size = 64
epochs = 80
lr = 2e-4

# β-annealing (KL warmup)
beta0, beta1, warm = 0.0, 1.0, 30
use_free_bits = True      # posterior collapse 방지 옵션
free_bits_nats = 0.5      # per-dim 최소 KL (nats)
lambda_cls = 0.2          # classifier consistency loss 가중치

# ---------------------------
# Dataset (y4 = 2*val + aro)
# ---------------------------
class EEGCondDataset(Dataset):
    def __init__(self, X, y4):
        self.X  = torch.tensor(X,  dtype=torch.float32)
        self.y4 = torch.tensor(y4, dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y4[i]

# ---------------------------
# Conditional VAE
# ---------------------------
class CondVAE1D(nn.Module):
    def __init__(self, C_in=70, T_len=1280, n_class=4, z_dim=128, emb_dim=32, base=128):
        super().__init__()
        self.T_len = T_len
        self.emb = nn.Embedding(n_class, emb_dim)

        # ----- Encoder -----
        Cin_enc = C_in + emb_dim  # 채널 concat
        self.enc = nn.Sequential(
            nn.Conv1d(Cin_enc, base,   5, padding=2, stride=2), nn.ReLU(),   # 1280→640
            nn.Conv1d(base,   base*2,  5, padding=2, stride=2), nn.ReLU(),   # 640→320
            nn.Conv1d(base*2, base*4,  5, padding=2, stride=2), nn.ReLU(),   # 320→160
            nn.Conv1d(base*4, base*4,  3, padding=1, stride=2), nn.ReLU(),   # 160→80
            nn.AdaptiveAvgPool1d(1)                                           
        )
        self.fc_mu   = nn.Linear(base*4, z_dim)
        self.fc_logv = nn.Linear(base*4, z_dim)

        # ----- Decoder -----
        self.fc0 = nn.Linear(z_dim + emb_dim, base*4*80)  # seed length 80
        self.dec = nn.Sequential(
            nn.ConvTranspose1d(base*4, base*4, 4, stride=2, padding=1), nn.ReLU(),  # 80→160
            nn.ConvTranspose1d(base*4, base*2, 4, stride=2, padding=1), nn.ReLU(),  # 160→320
            nn.ConvTranspose1d(base*2, base,   4, stride=2, padding=1), nn.ReLU(),  # 320→640
            nn.ConvTranspose1d(base,   base,   4, stride=2, padding=1), nn.ReLU(),  # 640→1280
            nn.Conv1d(base, C_in, 1)  # 활성 없음 (표준화 스케일)
        )

    def encode(self, x, y):
        B, C, T = x.shape
        e = self.emb(y)                                # (B, emb)
        e_rep = e.unsqueeze(-1).expand(B, e.size(1), T)# (B, emb, T)
        h = torch.cat([x, e_rep], dim=1)
        h = self.enc(h).squeeze(-1)                    # (B, base*4)
        mu, logv = self.fc_mu(h), self.fc_logv(h)
        return mu, logv

    def reparameterize(self, mu, logv):
        std = torch.exp(0.5 * logv)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, y):
        e = self.emb(y)
        zc = torch.cat([z, e], dim=1)
        h = self.fc0(zc).view(z.size(0), -1, 80)
        xhat = self.dec(h)                             # (B, C, 1280)
        return xhat

    def forward(self, x, y):
        mu, logv = self.encode(x, y)
        z = self.reparameterize(mu, logv)
        xhat = self.decode(z, y)
        return xhat, mu, logv

# ---------------------------
# 분류기 로드 (freeze): RhythmSpecificDeepCNN22 (멀티헤드)
# ---------------------------
# from your_models import RhythmSpecificDeepCNN22
class RhythmSpecificDeepCNN22(nn.Module):
    # 너의 프로젝트 정의를 사용해. (여기선 placeholder)
    def __init__(self, in_channels=70, n_classes=4):
        super().__init__()
        raise NotImplementedError("프로젝트의 RhythmSpecificDeepCNN22를 import 하세요.")

def load_classifier(checkpoint_path):
    clf = RhythmSpecificDeepCNN22(in_channels=5*14, n_classes=4).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    clf.load_state_dict(state, strict=False)
    clf.eval()
    for p in clf.parameters():
        p.requires_grad = False
    return clf

# ---------------------------
# Losses
# ---------------------------
def recon_loss(xhat, x):
    # L1 + MSE 혼합
    return 0.5 * F.l1_loss(xhat, x, reduction="mean") + 0.5 * F.mse_loss(xhat, x, reduction="mean")

def kl_loss(mu, logv):
    # 평균 KL (nats)
    return -0.5 * torch.mean(1 + logv - mu.pow(2) - logv.exp())

def kl_loss_freebits(mu, logv, fb_nats=0.5):
    # per-dim KL에 최소치(free-bits) 적용 후 배치 평균
    kl_per_dim = -0.5 * (1 + logv - mu.pow(2) - logv.exp())   # (B, z_dim)
    kl_per_sample = torch.clamp(kl_per_dim, min=fb_nats).sum(dim=1)  # (B,)
    return kl_per_sample.mean()

@torch.no_grad()
def bin_targets_from_y4(y4):
    # y4 ∈ {0,1,2,3} → val, aro
    y_val = (y4 // 2).float()
    y_aro = (y4 %  2).float()
    return y_val, y_aro

def classifier_consistency_loss(clf, xhat, y4):
    # clf는 멀티헤드 dict 출력 {"val":(B,), "aro":(B,)}
    with torch.no_grad():
        y_val, y_aro = bin_targets_from_y4(y4)
    out = clf(xhat)
    L_val = F.binary_cross_entropy_with_logits(out["val"], y_val.to(xhat.device))
    L_aro = F.binary_cross_entropy_with_logits(out["aro"], y_aro.to(xhat.device))
    return L_val + L_aro

# ---------------------------
# Train VAE
# ---------------------------
def train_vae(vae, loader, clf=None, epochs=80, lr=2e-4, save_path="./experiments/vae_best.pth"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    opt = torch.optim.Adam(vae.parameters(), lr=lr)
    best = float("inf"); best_state = None

    for ep in range(1, epochs+1):
        t0 = time.time()
        vae.train()
        Lr, Lk, Lc, Lt, n_batches = 0.0, 0.0, 0.0, 0.0, 0

        # β warmup
        beta = beta0 + (beta1 - beta0) * min(1.0, ep / warm)

        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            xhat, mu, logv = vae(xb, yb)

            rec = recon_loss(xhat, xb)
            kl  = kl_loss_freebits(mu, logv, free_bits_nats) if use_free_bits else kl_loss(mu, logv)

            loss = rec + beta * kl
            if clf is not None and lambda_cls > 0:
                # 분류기 일치 보조 손실
                L_cls = classifier_consistency_loss(clf, xhat, yb)
                loss = loss + lambda_cls * L_cls
            else:
                L_cls = torch.tensor(0.0, device=device)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            Lr += rec.item(); Lk += kl.item(); Lc += L_cls.item(); Lt += loss.item(); n_batches += 1

        dt = time.time() - t0
        print(f"[{ep:03d}/{epochs}] dt={dt:.2f}s  Recon={Lr/n_batches:.4f}  KL={Lk/n_batches:.4f}  CLS={Lc/n_batches:.4f}  Total={Lt/n_batches:.4f}  (beta={beta:.3f})")

        # best by total loss (평균)
        cur = Lt / n_batches
        if cur < best:
            best = cur
            best_state = {
                "epoch": ep,
                "model": vae.state_dict(),
                "optim": opt.state_dict(),
                "beta": beta,
                "note": "CondVAE EEG with warmup+freebits+clf-guidance",
            }

    if best_state is not None:
        torch.save(best_state, save_path)
        print(f"✅ Saved best VAE to {save_path} (total={best:.4f})")
    else:
        print("No improvement; nothing saved.")


# In[26]:


# ---------------------------
# Sampling
# ---------------------------
@torch.no_grad()
def vae_generate(vae, labels, z_dim=128):
    vae.eval()
    labels = torch.as_tensor(labels, dtype=torch.long, device=device)
    z = torch.randn(labels.shape[0], z_dim, device=device)
    x = vae.decode(z, labels)   # (N, 70, 1280)
    return x.cpu().numpy()

# ---------------------------
# Evaluate with classifier (멀티헤드)
# ---------------------------
@torch.no_grad()
def evaluate_with_classifier(clf, X_synth, y4_synth):
    X_t = torch.tensor(X_synth, dtype=torch.float32, device=device)
    clf.eval()
    out = clf(X_t)
    pred_val = (torch.sigmoid(out["val"]) >= 0.5).long()
    pred_aro = (torch.sigmoid(out["aro"]) >= 0.5).long()
    preds_4  = (2 * pred_val + pred_aro).cpu().numpy()

    # joint 4-class
    acc4 = accuracy_score(y4_synth, preds_4)
    f1m4 = f1_score(y4_synth, preds_4, average="macro")

    # per-head
    y_val = (y4_synth // 2)
    y_aro = (y4_synth % 2)
    p_val = pred_val.cpu().numpy()
    p_aro = pred_aro.cpu().numpy()

    metrics = {}
    for name, y_true, p_bin, logit in [
        ("Val", y_val, p_val, out["val"]),
        ("Aro", y_aro, p_aro, out["aro"]),
    ]:
        acc = accuracy_score(y_true, p_bin)
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, p_bin, average="binary", zero_division=0)
        try:
            auc = roc_auc_score(y_true, torch.sigmoid(logit).cpu().numpy())
        except ValueError:
            auc = float("nan")
        metrics[name] = {"Acc": acc, "Prec": prec, "Rec": rec, "F1": f1, "AUC": auc}

    return acc4, f1m4, metrics

# ============================================================
# ==== Plug your data & classifier and run ===================
# ============================================================

# 1) 데이터 준비 (X_train: (N,70,1280) 표준화, y4: (N,) in {0..3})
# 이미 너한테 있음: X_train, y
train_ds = EEGCondDataset(X_train, y)
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)

# 2) 분류기 로드 (freeze)
clf_path = "./experiments/rhythm_cnn_best.pth"
clf = load_classifier(clf_path)


# In[30]:


# 3) VAE 생성 & 학습
vae = CondVAE1D(C_in=C_in, T_len=T_len, n_class=n_class, z_dim=z_dim, emb_dim=32, base=128).to(device)
# 분류기 일치 손실 사용하려면 clf 인스턴스 전달, 아니면 None
# train_vae(vae, train_dl, clf=clf, epochs=epochs, lr=lr, save_path="./experiments/vae_best.pth")
# 분류기 일치 없이:
train_vae(vae, train_dl, clf=None, epochs=epochs, lr=lr, save_path="./experiments/vae_best.pth")


# In[32]:


# 4) 합성 & 평가 (균형 샘플 예시)
labels_bal = np.concatenate([np.full(250, c, dtype=np.int64) for c in range(n_class)], axis=0)
X_synth = vae_generate(vae, labels_bal, z_dim=z_dim)
y_synth = labels_bal
acc4, f1m4, head = evaluate_with_classifier(clf, X_synth, y_synth)
print(f"[VAE Synth → Classifier] Joint 4-class Acc={acc4:.4f}  F1(macro)={f1m4:.4f}")
for k,v in head.items():
    print(f"[{k}] Acc={v['Acc']:.4f} Prec={v['Prec']:.4f} Rec={v['Rec']:.4f} F1={v['F1']:.4f} AUC={v['AUC']:.4f}")


# In[ ]:




