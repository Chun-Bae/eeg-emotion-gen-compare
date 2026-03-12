#!/usr/bin/env python
# coding: utf-8

# In[5]:


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
from models.classifier import Classifier

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


# In[6]:


# ---------------------------
# 원시 데이터 가져오기
# ---------------------------
mat = scipy.io.loadmat('./data/DREAMER.mat', struct_as_record=False, squeeze_me=True)
dreamer = mat['DREAMER']


# In[7]:


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


# In[8]:


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


# In[9]:


# ---------------------------
# Hyperparams
# ---------------------------
seed_num = 42
batch_size = 128
n_classes = 4

epochs = 10
sample_n = 8
betas = (0.5, 0.999)
momentum = 0.9 
weight_decay = 1e-4
EPS = 1e-12


# In[10]:


# ---------------------------
# seed 고정
# ---------------------------
torch.manual_seed(seed_num)
np.random.seed(seed_num)
random.seed(seed_num)


# In[11]:


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


# In[12]:


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

# 1차: train / temp (temp = val+test)
idx_tr, idx_temp = train_test_split(
    idx, test_size=0.3, random_state=seed_num, stratify=y
)

# 2차: temp -> val / test
idx_va, idx_te = train_test_split(
    idx_temp, test_size=0.5, random_state=seed_num, stratify=y[idx_temp]
)

print(f"train: {len(idx_tr)}, val: {len(idx_va)}, test: {len(idx_te)}")

# ---------------------------
# Split Data
# ---------------------------
X_train, X_val, X_test = X[idx_tr], X[idx_va], X[idx_te]
y_train_val, y_val_val, y_test_val = y_val_bin[idx_tr], y_val_bin[idx_va], y_val_bin[idx_te]
y_train_aro, y_val_aro, y_test_aro = y_aro_bin[idx_tr], y_aro_bin[idx_va], y_aro_bin[idx_te]

# ---------------------------
# Scaling (표준화)
# ---------------------------
# (B, 70, 1280) -> (B*1280, 70) 로 바꿔서 채널 표준화
X_train_2d = np.swapaxes(X_train, 1, 2).reshape(-1, R*C)
X_val_2d   = np.swapaxes(X_val,   1, 2).reshape(-1, R*C)
X_test_2d  = np.swapaxes(X_test,  1, 2).reshape(-1, R*C)

scaler = StandardScaler()
X_train_2d = scaler.fit_transform(X_train_2d)
X_val_2d   = scaler.transform(X_val_2d)
X_test_2d  = scaler.transform(X_test_2d)

# 다시 (B, 70, 1280)로 복구
X_train = X_train_2d.reshape(X_train.shape[0], T, R*C).swapaxes(1, 2).astype(np.float32)
X_val   = X_val_2d.reshape(  X_val.shape[0],   T, R*C).swapaxes(1, 2).astype(np.float32)
X_test  = X_test_2d.reshape( X_test.shape[0],  T, R*C).swapaxes(1, 2).astype(np.float32)

# ---------------------------
# DataLoader
# ---------------------------
train_dataset = EEGDataset(X_train, y_train_val, y_train_aro)
val_dataset   = EEGDataset(X_val,   y_val_val,   y_val_aro)
test_dataset  = EEGDataset(X_test,  y_test_val,  y_test_aro)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  drop_last=False)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, drop_last=False)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, drop_last=False)


# In[13]:


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
    'X', 'y',
]:
    if var in globals():
        del globals()[var]

gc.collect()


# In[15]:


bce = nn.BCEWithLogitsLoss()

# ==== metrics ====
def step_metrics(logits, targets):
    """
    logits: (B,)  raw logit
    targets: (B,)  float(0/1) or long(0/1)
    """
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    preds = (probs >= 0.5).astype("int64")
    y     = targets.detach().cpu().numpy().astype("int64")

    prec, rec, f1, _ = precision_recall_fscore_support(y, preds, average="binary", zero_division=0)
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

# ==== train / eval ====
def train_one_epoch(model, loader, opt, device):
    model.train()
    total_loss, n = 0.0, 0
    agg = {"val": [], "aro": []}

    for Xb, yv, ya in loader:
        Xb, yv, ya = Xb.to(device), yv.to(device).float(), ya.to(device).float()

        opt.zero_grad()
        out = model(Xb)  # {"val": (B,), "aro": (B,)}
        loss = bce(out["val"], yv) + bce(out["aro"], ya)
        loss.backward()
        opt.step()

        bs = Xb.size(0)
        total_loss += loss.item() * bs
        n += bs

        agg["val"].append(step_metrics(out["val"], yv))
        agg["aro"].append(step_metrics(out["aro"], ya))

    return total_loss / max(n, 1), _pack_epoch_metrics(agg)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, n = 0.0, 0
    agg = {"val": [], "aro": []}

    for Xb, yv, ya in loader:
        Xb, yv, ya = Xb.to(device), yv.to(device).float(), ya.to(device).float()

        out = model(Xb)
        loss = bce(out["val"], yv) + bce(out["aro"], ya)

        bs = Xb.size(0)
        total_loss += loss.item() * bs
        n += bs

        agg["val"].append(step_metrics(out["val"], yv))
        agg["aro"].append(step_metrics(out["aro"], ya))

    return total_loss / max(n, 1), _pack_epoch_metrics(agg)

# ==== pretty print ====
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

# ==== training loop ====
model = Classifier(in_channels=5*14, n_classes=4).to(device)  # 네 정의 그대로, 출력만 멀티헤드로 바꿔둔 버전 사용
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
epochs = 100

best_score = -1.0
best_state = None

for epoch in range(1, epochs + 1):
    t0 = time.time()
    tr_loss, tr_m = train_one_epoch(model, train_loader, optimizer, device)
    va_loss, va_m = evaluate(model, val_loader, device)
    dt = time.time() - t0

    train_valid_log(epoch, epochs, tr_loss, va_loss, tr_m, va_m, dt)

    # ---- 베스트 선정 기준: Val/Aro F1의 평균 (원하면 AUC나 Acc로 바꿔도 됨)
    val_f1_mean = 0.5 * (va_m["val"]["f1"] + va_m["aro"]["f1"])
    if val_f1_mean > best_score:
        best_score = val_f1_mean
        best_state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_f1_mean": best_score,
            "note": "selection = mean(F1_val, F1_aro)",
        }

# ==== save ====
import os
os.makedirs("./experiments", exist_ok=True)
if best_state is not None:
    path = "./experiments/rhythm_cnn_best_test.pth"
    torch.save(best_state, path)
    print(f"\nBest mean F1={best_state['best_f1_mean']:.4f} (epoch {best_state['epoch']}) saved to {path}")
else:
    print("\nNo improvement recorded.")


# In[16]:


# ==== test evaluation ====
@torch.no_grad()
def evaluate_test(model, loader, device, max_samples=1000):
    model.eval()
    agg = {"val": [], "aro": []}
    n = 0

    for Xb, yv, ya in loader:
        Xb, yv, ya = Xb.to(device), yv.to(device).float(), ya.to(device).float()
        out = model(Xb)

        agg["val"].append(step_metrics(out["val"], yv))
        agg["aro"].append(step_metrics(out["aro"], ya))

        n += Xb.size(0)
        if n >= max_samples:
            break

    return _pack_epoch_metrics(agg)


# ==== load best model and evaluate ====
best_ckpt = torch.load("./experiments/rhythm_cnn_best_test.pth", map_location=device)
model.load_state_dict(best_ckpt["model"])

test_metrics = evaluate_test(model, test_loader, device, max_samples=1000)

print("\n" + "=" * 60)
print(" Test Evaluation (first 1000 samples)")
print("-" * 60)
print(" Valence (V)".ljust(30), "|", end=" ")
for k in ["acc","prec","rec","f1","auc"]:
    print(f"{k.upper()}: {test_metrics['val'][k]:.4f}", end="  ")
print("\n Arousal (A)".ljust(30), "|", end=" ")
for k in ["acc","prec","rec","f1","auc"]:
    print(f"{k.upper()}: {test_metrics['aro'][k]:.4f}", end="  ")
print("\n" + "=" * 60)


# In[ ]:




