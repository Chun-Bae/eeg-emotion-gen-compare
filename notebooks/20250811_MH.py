#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from models.classifier import Classifier_EEG, EEGMultiHeadClassifier
from models.cDCGAN import G2D, D2D

# ---- sklearn ----
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

# ---- scipy ----
import scipy.io

# ---- torch ----
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, Dataset, DataLoader


# In[2]:


# ---------------------------
# 원시 데이터 가져오기
# ---------------------------
mat = scipy.io.loadmat('./data/DREAMER.mat', struct_as_record=False, squeeze_me=True)
dreamer = mat['DREAMER']


# In[3]:


# ---------------------------
# 데이터 전처리
# ---------------------------
segment_len = 1280              # 10s @ 128 Hz
overlap = 128                   # 겹침 길이(프레임) = 1s
hop = segment_len - overlap     # = 1152 (슬라이딩 간격)

bands = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta':  (13, 30),
    'gamma': (30, 45)
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

df["y_val"] = (df["valence"]  >= 3).astype(int).values
df["y_aro"] = (df["arousal"]  >= 3).astype(int).values
df["y_dom"] = (df["dominance"]>= 3).astype(int).values



# In[5]:


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


# In[6]:


# ---------------------------
# Hyperparams
# ---------------------------
seed_num = 42
batch_size = 128
n_classes = 4

# ---- 공용 ----
epochs = 10
sample_n = 8
betas = (0.5, 0.999)
momentum = 0.9
weight_decay = 1e-4

# ---- cDCGAN ----
z_dim = 128
lr_G =2e-4
lr_D = 2e-4
n_critic = 2
gen_per_class = 100


# In[7]:


# ---------------------------
# seed 고정
# ---------------------------
torch.manual_seed(seed_num)
np.random.seed(seed_num)
random.seed(seed_num)


# In[8]:


# ---------------------------
# Dataset object
# ---------------------------
class EEGDataset(Dataset):
    def __init__(self, X, y_val, y_aro, y_dom):
        self.X = torch.tensor(X, dtype=torch.float32).permute(0, 3, 2, 1)  # -> (N,14,5,1280)
        self.y_val = torch.tensor(y_val).long()
        self.y_aro = torch.tensor(y_aro).long()
        self.y_dom = torch.tensor(y_dom).long()

    def __len__(self):
        return len(self.y_val)

    def __getitem__(self, idx):
        return self.X[idx], (self.y_val[idx], self.y_aro[idx], self.y_dom[idx])


# In[9]:


# ---------------------------
# Data Split
# ---------------------------
X = np.stack(df['eeg_band_data'].values).astype(np.float32)
y_combo = df["y_val"] * 4 + df["y_aro"] * 2 + df["y_dom"]

X_train, X_val, idx_train, idx_val = train_test_split(
    X, np.arange(len(X)), test_size=0.2, random_state=seed_num, stratify=y_combo
)


# In[10]:


# ---------------------------
# 스케일링 : -1 ~ 1
# ---------------------------
X_train = X_train.reshape(-1, 5*14)
X_val   = X_val.reshape(-1, 5*14)

scaler = MinMaxScaler(feature_range=(-1, 1))
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)

X_train = X_train.reshape(-1, 1280, 5, 14)
X_val   = X_val.reshape(-1, 1280, 5, 14)


# In[11]:


# ---------------------------
# DataLoader
# ---------------------------
y_val_train, y_val_val = df["y_val"].values[idx_train], df["y_val"].values[idx_val]
y_aro_train, y_aro_val = df["y_aro"].values[idx_train], df["y_aro"].values[idx_val]
y_dom_train, y_dom_val = df["y_dom"].values[idx_train], df["y_dom"].values[idx_val]

train_dataset = EEGDataset(X_train, y_val_train, y_aro_train, y_dom_train)
val_dataset = EEGDataset(X_val, y_val_val, y_aro_val, y_dom_val)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  pin_memory=False)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, pin_memory=False)


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
    'X', 'y',
]:
    if var in globals():
        del globals()[var]

gc.collect()


# In[ ]:


# -----------------------------
# Model / Optim / Loss
# -----------------------------
model = EEGMultiHeadClassifier(
    use_heads=["val","aro","dom"],
    num_classes_per_head={"val":2, "aro":2, "dom":2},
    attn_heads=4,
).to(device)

# (옵션) 클래스 가중치가 필요하면 각 헤드별 weight를 넣어주세요.
# ce_v = nn.CrossEntropyLoss(weight=torch.tensor([w0,w1,w2], device=device))
# ce_a = nn.CrossEntropyLoss(weight=torch.tensor([w0,w1,w2], device=device))
# ce_d = nn.CrossEntropyLoss(weight=torch.tensor([w0,w1,w2], device=device))
ce_v = nn.CrossEntropyLoss()
ce_a = nn.CrossEntropyLoss()
ce_d = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
epochs = 30

# -----------------------------
# Helpers
# -----------------------------
def metrics_from_preds(y_true_list, y_pred_list):
    """
    y_true_list, y_pred_list: torch.Tensor list (batch별 축적)
    return: dict(acc, prec, rec, f1) with macro average
    """
    y_true = torch.cat(y_true_list).cpu().numpy()
    y_pred = torch.cat(y_pred_list).cpu().numpy()
    acc = (y_true == y_pred).mean() if len(y_true) else 0.0
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1}

def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    total_loss = 0.0
    total_samples = 0

    preds = {"val":[], "aro":[], "dom":[]}
    trues = {"val":[], "aro":[], "dom":[]}

    for xb, (yv, ya, yd) in loader:
        xb = xb.to(device, non_blocking=True)
        yv = yv.to(device, non_blocking=True)
        ya = ya.to(device, non_blocking=True)
        yd = yd.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            out = model(xb)  # dict: "val"/"aro"/"dom"
            lv = ce_v(out["val"], yv)
            la = ce_a(out["aro"], ya)
            ld = ce_d(out["dom"], yd)
            loss = lv + la + ld

            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        # 누적
        bs = xb.size(0)
        total_loss += loss.item() * bs
        total_samples += bs

        # 예측 축적
        preds["val"].append(out["val"].argmax(1).detach())
        preds["aro"].append(out["aro"].argmax(1).detach())
        preds["dom"].append(out["dom"].argmax(1).detach())

        trues["val"].append(yv.detach())
        trues["aro"].append(ya.detach())
        trues["dom"].append(yd.detach())

    avg_loss = total_loss / max(1, total_samples)

    # 헤드별 지표
    head_metrics = {h: metrics_from_preds(trues[h], preds[h]) for h in ["val","aro","dom"]}
    return avg_loss, head_metrics

# -----------------------------
# Train
# -----------------------------
for epoch in range(1, epochs+1):
    t0 = time.time()

    train_loss, train_m = run_epoch(train_loader, train=True)
    val_loss,   val_m   = run_epoch(val_loader,   train=False)

    dt = time.time() - t0
    print(
        "\n" + "=" * 60 +
        f"\n[Epoch {epoch:02d}/{epochs}]  Time: {dt:.2f}s" +
        "\n" + "-" * 60 +
        f"\n| Loss       | Train: {train_loss:.4f}   Val: {val_loss:.4f}" +
        "\n" + "-" * 60 +
        f"\n| V (Valence)   | Acc: {train_m['val']['acc']:.4f} / {val_m['val']['acc']:.4f}"
        f" | Prec: {train_m['val']['prec']:.4f} / {val_m['val']['prec']:.4f}"
        f" | Rec: {train_m['val']['rec']:.4f} / {val_m['val']['rec']:.4f}"
        f" | F1: {train_m['val']['f1']:.4f} / {val_m['val']['f1']:.4f}"
        +
        f"\n| A (Arousal)   | Acc: {train_m['aro']['acc']:.4f} / {val_m['aro']['acc']:.4f}"
        f" | Prec: {train_m['aro']['prec']:.4f} / {val_m['aro']['prec']:.4f}"
        f" | Rec: {train_m['aro']['rec']:.4f} / {val_m['aro']['rec']:.4f}"
        f" | F1: {train_m['aro']['f1']:.4f} / {val_m['aro']['f1']:.4f}"
        +
        f"\n| D (Dominance) | Acc: {train_m['dom']['acc']:.4f} / {val_m['dom']['acc']:.4f}"
        f" | Prec: {train_m['dom']['prec']:.4f} / {val_m['dom']['prec']:.4f}"
        f" | Rec: {train_m['dom']['rec']:.4f} / {val_m['dom']['rec']:.4f}"
        f" | F1: {train_m['dom']['f1']:.4f} / {val_m['dom']['f1']:.4f}"
        +
        "\n" + "=" * 60
    )


# In[ ]:




