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
from scipy.signal import welch, butter, filtfilt, iirnotch
from scipy.stats import kurtosis, skew

# ---- torch ----
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, Dataset, DataLoader


# In[3]:


mat = scipy.io.loadmat('./data/DREAMER.mat', struct_as_record=False, squeeze_me=True)
dreamer = mat['DREAMER']


# In[4]:


# ================================================
# EEG-only feature extraction (DREAMER)
# - Features per channel: α PSD, β PSD, γ PSD, DFA  → 4
# - Channels: 14 → 14 × 4 = 56-dim per segment
# - Fast path: vectorized filtering & Welch; DFA on downsampled signal
# ================================================

import os, glob, numpy as np, pandas as pd
from tqdm.auto import tqdm
from scipy.signal import butter, iirnotch, sosfiltfilt, welch

# ---------------------------
# Parameters
# ---------------------------
FS        = 128                   # sampling rate (EEG)
SEG_SEC   = 30
SEG_LEN   = SEG_SEC * FS          # 30s = 3840 (논문)
HOP       = SEG_LEN               # no overlap
BANDS     = {'delta':(0.5,4), 'theta':(4,8), 'alpha':(8,13), 'beta':(13,30), 'gamma':(30,45)}
USE_DOWNSAMPLE_FOR_DFA = True     # DFA 연산 가속용 다운샘플 사용 권장
DFA_DS_FACTOR          = 2        # 128→64 Hz (필요시 4로 더 줄여도 됨)
CKPT_DIR  = "./feat_ckpt_eeg56"
SAVE_EVERY = 500                  # 저장 주기(세그먼트)

os.makedirs(CKPT_DIR, exist_ok=True)

# ---------------------------
# Filters (design once)
# ---------------------------
def design_filters(fs=FS):
    sos_bp = butter(4, [0.5/(fs/2), 45/(fs/2)], btype='band', output='sos')
    b0, a0 = iirnotch(50/(fs/2), Q=30.0)
    # iirnotch는 sos가 아니라 ba라서, sosfiltfilt에 직접 넣기 어렵습니다.
    # 간단히 notch→bandpass 순서로 filtfilt를 두 번 적용해도 되지만,
    # 여기서는 notch를 'ba' 필터로 filtfilt, bandpass는 SOS로 sosfiltfilt로 처리합니다.
    return (b0, a0), sos_bp

NOTCH_BA, SOS_BP = design_filters()

# ---------------------------
# Preprocessing
# ---------------------------
from scipy.signal import filtfilt

def preprocess_seg_matrix(seg):     # seg: (L,14)
    # 50Hz notch (ba) → bandpass 0.5–45 Hz (sos)
    x = filtfilt(NOTCH_BA[0], NOTCH_BA[1], seg, axis=0)
    x = sosfiltfilt(SOS_BP, x, axis=0)
    return x

# ---------------------------
# Welch masks (compute once)
# ---------------------------
def make_band_mask(seg_len, fs=FS, nperseg=256, noverlap=128):
    f, _ = welch(np.zeros(seg_len), fs=fs, nperseg=nperseg, noverlap=noverlap)
    masks, df = {}, f[1]-f[0]
    for k,(lo,hi) in BANDS.items():
        m = (f >= lo) & (f < hi)
        masks[k] = (m, df)
    return masks

MASKS = make_band_mask(SEG_LEN, fs=FS)

# α/β/γ의 인덱스 (BANDS dict 순서 기준)
_BAND_KEYS = list(BANDS.keys())
IDX_ALPHA  = _BAND_KEYS.index('alpha')
IDX_BETA   = _BAND_KEYS.index('beta')
IDX_GAMMA  = _BAND_KEYS.index('gamma')

# ---------------------------
# DFA (downsampled for speed)
# ---------------------------
def dfa_1d(x, min_scale=4, max_scale=None, n_scales=18):
    """Simple DFA for one 1D array (fast-ish)."""
    x = np.asarray(x, float)
    x = x - np.mean(x)
    y = np.cumsum(x)
    N = len(y)
    if max_scale is None:
        max_scale = N // 4
    # log-spaced scales
    scales = np.unique(np.logspace(np.log10(min_scale), np.log10(max_scale), n_scales).astype(int))
    F = []
    for s in scales:
        if s < 2:
            continue
        nseg = N // s
        if nseg < 2:
            continue
        rms = []
        for k in range(nseg):
            seg = y[k*s:(k+1)*s]
            t = np.arange(s)
            p = np.polyfit(t, seg, 1)
            trend = np.polyval(p, t)
            rms.append(np.sqrt(np.mean((seg - trend)**2)))
        F.append(np.mean(rms) + 1e-12)
    if len(F) == 0:
        return 0.0
    F = np.array(F)
    scales = scales[:len(F)]
    slope = np.polyfit(np.log(scales + 1e-12), np.log(F + 1e-12), 1)[0]
    return float(slope)

def dfa_per_channel(seg, ds_factor=DFA_DS_FACTOR):
    """seg: (L,14) → dfa: (14,), with optional downsampling for speed"""
    x = seg
    if USE_DOWNSAMPLE_FOR_DFA and ds_factor > 1:
        x = x[::ds_factor, :]
    out = np.empty(x.shape[1], dtype=float)
    for ch in range(x.shape[1]):
        out[ch] = dfa_1d(x[:, ch])
    return out  # (14,)

# ---------------------------
# Label mapping (1–2→0, 3→1, 4–5→2)
# ---------------------------
def map_vad(x):
    x = float(x)
    return 0 if x <= 2 else (1 if x == 3 else 2)

# ---------------------------
# Count total segments
# ---------------------------
def count_segments(dreamer, seg_len=SEG_LEN, hop=HOP):
    total = 0
    for subject in dreamer.Data:
        for eeg in subject.EEG.stimuli:
            T, _ = eeg.shape
            if T >= seg_len:
                total += 1 + (T - seg_len) // hop
    return total

# ---------------------------
# Resume support (optional)
# ---------------------------
def already_done_keys(ckpt_dir=CKPT_DIR):
    keys = set()
    parts = sorted(glob.glob(os.path.join(ckpt_dir, "features_part_*.pkl")))
    for p in parts:
        try:
            dfp = pd.read_pickle(p)
        except Exception:
            continue
        for r in dfp[["subject_id","video_idx","start"]].itertuples(index=False):
            keys.add((r.subject_id, r.video_idx, r.start))
    return keys

# =========================================================
# Main loop: EEG-only features → 56-dim per segment
# =========================================================
TOTAL_SEGS = count_segments(dreamer, SEG_LEN, HOP)
pbar = tqdm(total=TOTAL_SEGS, desc="EEG feature extraction (EEG 56-dim)", ncols=100)

done = already_done_keys()
rows = []
buffer_rows, part_idx, processed = [], 0, 0

for subject in dreamer.Data:
    eeg_list = subject.EEG.stimuli
    V, A, D  = subject.ScoreValence, subject.ScoreArousal, subject.ScoreDominance
    sid = getattr(subject, "ID", None)

    for vid_idx, eeg in enumerate(eeg_list):
        T, C = eeg.shape  # (time, 14)
        for start in range(0, T - SEG_LEN + 1, HOP):
            key = (sid, vid_idx, start)
            if key in done:
                pbar.update(1)
                continue

            seg = eeg[start:start+SEG_LEN, :]      # (3840, 14)

            # 1) Preprocess (vectorized)
            seg_p = preprocess_seg_matrix(seg)     # (L, 14)

            # 2) Welch once for all channels
            f, psd = welch(seg_p, fs=FS, nperseg=256, noverlap=128, axis=0)  # (F,14)

            # 3) Band powers (vectorized), get alpha/beta/gamma only
            bp_all = []
            for k in BANDS.keys():
                m, df = MASKS[k]
                bp = (psd[m, :].sum(axis=0) * df) + 1e-12  # (14,)
                bp_all.append(np.log(bp))
            bp_all = np.stack(bp_all, axis=1)  # (14, 5)
            alpha = bp_all[:, IDX_ALPHA:IDX_ALPHA+1]   # (14,1)
            beta  = bp_all[:, IDX_BETA:IDX_BETA+1]     # (14,1)
            gamma = bp_all[:, IDX_GAMMA:IDX_GAMMA+1]   # (14,1)

            # 4) DFA (per-channel; on downsampled seg for speed)
            dfa_vals = dfa_per_channel(seg_p)[:, None]  # (14,1)

            # 5) Concat → (14,4) → 56
            ch_feat = np.concatenate([alpha, beta, gamma, dfa_vals], axis=1)  # (14,4)
            feat_vec = ch_feat.reshape(-1).astype(np.float32)                 # (56,)

            rec = {
                "subject_id": sid,
                "video_idx": vid_idx,
                "start": start,
                "features_eeg": feat_vec,            # (56,)
                "label_val": map_vad(V[vid_idx]),
                "label_aro": map_vad(A[vid_idx]),
                "label_dom": map_vad(D[vid_idx]),
            }
            rows.append(rec)
            buffer_rows.append(rec)

            processed += 1
            pbar.update(1)

            # checkpoint
            if (processed % SAVE_EVERY) == 0:
                out_path = os.path.join(CKPT_DIR, f"features_part_{part_idx:04d}.pkl")
                pd.DataFrame(buffer_rows).to_pickle(out_path)
                buffer_rows.clear()
                part_idx += 1
                pbar.set_postfix(saved=os.path.basename(out_path))

pbar.close()

# flush remaining
if buffer_rows:
    out_path = os.path.join(CKPT_DIR, f"features_part_{part_idx:04d}.pkl")
    pd.DataFrame(buffer_rows).to_pickle(out_path)
    buffer_rows.clear()

df_feat = pd.DataFrame(rows)
print("샘플 수:", len(df_feat))
print("예시 벡터 shape:", df_feat.iloc[0]["features_eeg"].shape)  # (56,)


# In[7]:


import pandas as pd, json
df = df_feat.copy()
df["features_eeg"] = df["features_eeg"].apply(lambda x: json.dumps(list(map(float, x))))
df.to_csv("./data/eeg_feat.csv", index=False)  # 필요시 compression="gzip"


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




