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
import math
import os
from tqdm.auto import tqdm


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


# In[4]:


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


# In[5]:


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


# In[6]:


# ---------------------------
# seed 고정
# ---------------------------
torch.manual_seed(seed_num)
np.random.seed(seed_num)
random.seed(seed_num)


# In[7]:


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


# In[8]:


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


# In[9]:


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


# In[10]:


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


# In[11]:


# ============================================================
# Conditional DDPM for EEG (70 x 1280)
# - Class-conditional U-Net1D
# - Cosine noise schedule
# - (옵션) Classifier-Free Guidance (CFG)
# - 표준화 스케일 가정 (출력 활성 없음)
# ============================================================

# ---------------------------
# Env / Config
# ---------------------------
C_in, T_len = 70, 1280     # EEG shape
n_class      = 4
batch_size   = 64
epochs       = 50
lr           = 2e-4
timesteps    = 1000        # diffusion steps
p_uncond     = 0.1         # CFG: cond drop prob (훈련용)
cfg_scale    = 2.0         # CFG scale (샘플링용, 1.0=끄기)

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
# Utilities: time embedding (sinusoidal)
# ---------------------------
def sinusoidal_time_embedding(t, dim):
    # t: (B,), returns (B, dim)
    half = dim // 2
    freqs = torch.exp(
        torch.linspace(math.log(1.0), math.log(10000.0), steps=half, device=t.device)
    )
    angles = t[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0,1))
    return emb

# ---------------------------
# Building block: ResBlock1D + FiLM conditioning (time + class)
# ---------------------------
class ResBlock1D(nn.Module):
    def __init__(self, cin, cout, t_dim, c_dim, groups=8):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, cin)
        self.conv1 = nn.Conv1d(cin, cout, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, cout)
        self.conv2 = nn.Conv1d(cout, cout, 3, padding=1)

        # FiLM-style (time + class) → scale/shift
        self.t_proj = nn.Linear(t_dim, cout*2)
        self.c_proj = nn.Linear(c_dim, cout*2)

        self.act = nn.SiLU()
        self.skip = (cin != cout)
        if self.skip:
            self.skip_conv = nn.Conv1d(cin, cout, 1)

    def forward(self, x, t_emb, c_emb):
        h = self.act(self.norm1(x))
        h = self.conv1(h)

        # FiLM
        t_scale, t_shift = self.t_proj(t_emb).chunk(2, dim=1)
        c_scale, c_shift = self.c_proj(c_emb).chunk(2, dim=1)
        scale = (t_scale + c_scale).unsqueeze(-1)
        shift = (t_shift + c_shift).unsqueeze(-1)

        h = self.act(self.norm2(h))
        h = h * (1 + scale) + shift
        h = self.conv2(h)

        if self.skip:
            x = self.skip_conv(x)
        return x + h

# ---------------------------
# U-Net1D (down/up with skips)
# ---------------------------
class UNet1D(nn.Module):
    def __init__(self, in_ch=70, base=64, t_dim=128, c_dim=64):
        super().__init__()
        self.in_conv = nn.Conv1d(in_ch, base, 3, padding=1)

        # Down
        self.rb1 = ResBlock1D(base, base, t_dim, c_dim)
        self.down1 = nn.Conv1d(base, base*2, 4, stride=2, padding=1)   # 1280->640

        self.rb2 = ResBlock1D(base*2, base*2, t_dim, c_dim)
        self.down2 = nn.Conv1d(base*2, base*4, 4, stride=2, padding=1) # 640->320

        self.rb3 = ResBlock1D(base*4, base*4, t_dim, c_dim)
        self.down3 = nn.Conv1d(base*4, base*4, 4, stride=2, padding=1) # 320->160

        self.rb4 = ResBlock1D(base*4, base*4, t_dim, c_dim)            # bottleneck

        # Up (nearest upsample + conv to stabilize on MPS)
        self.up3 = nn.Conv1d(base*4, base*4, 3, padding=1)
        self.ub3 = ResBlock1D(base*4+base*4, base*4, t_dim, c_dim)

        self.up2 = nn.Conv1d(base*4, base*2, 3, padding=1)
        self.ub2 = ResBlock1D(base*2+base*2, base*2, t_dim, c_dim)

        self.up1 = nn.Conv1d(base*2, base, 3, padding=1)
        self.ub1 = ResBlock1D(base+base, base, t_dim, c_dim)

        self.out = nn.Conv1d(base, in_ch, 1)

        # embeddings
        self.t_mlp = nn.Sequential(
            nn.Linear(t_dim, t_dim), nn.SiLU(),
            nn.Linear(t_dim, t_dim)
        )
        self.c_emb = nn.Embedding(n_class, c_dim)

        # null token for CFG
        self.null_emb = nn.Parameter(torch.zeros(1, c_dim))

    def forward(self, x, t, y=None):
        # x: (B,C,L), t: (B,) ∈ [0..T-1], y: (B,) class or None (uncond)
        t = t.float()
        t_emb = sinusoidal_time_embedding(t, self.t_mlp[0].in_features)
        t_emb = self.t_mlp(t_emb)

        if y is None:
            c_emb = self.null_emb.expand(x.size(0), -1)   # (B, c_dim)
        else:
            c_emb = self.c_emb(y)

        # Down
        x0 = self.in_conv(x)                    # (B,base,1280)
        d1 = self.rb1(x0, t_emb, c_emb)
        x1 = self.down1(d1)                     # 640
        d2 = self.rb2(x1, t_emb, c_emb)
        x2 = self.down2(d2)                     # 320
        d3 = self.rb3(x2, t_emb, c_emb)
        x3 = self.down3(d3)                     # 160

        mid = self.rb4(x3, t_emb, c_emb)

        # Up
        u = F.interpolate(mid, scale_factor=2, mode="nearest"); u = self.up3(u)
        u = torch.cat([u, d3], dim=1); u = self.ub3(u, t_emb, c_emb)

        u = F.interpolate(u, scale_factor=2, mode="nearest"); u = self.up2(u)
        u = torch.cat([u, d2], dim=1); u = self.ub2(u, t_emb, c_emb)

        u = F.interpolate(u, scale_factor=2, mode="nearest"); u = self.up1(u)
        u = torch.cat([u, d1], dim=1); u = self.ub1(u, t_emb, c_emb)

        return self.out(u)  # predict noise ε

# ---------------------------
# Noise schedule (cosine)
# ---------------------------
def cosine_beta_schedule(timesteps):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
    alphas_cumprod = torch.cos((x / timesteps + 0.008) / 1.008 * math.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 1e-6, 0.999).float() 

betas = cosine_beta_schedule(timesteps).to(device)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
one_over_sqrt_alpha = torch.sqrt(1.0 / alphas)

# ---------------------------
# DDPM helpers
# ---------------------------
def q_sample(x0, t, noise):
    # x_t = sqrt(a_bar_t) * x0 + sqrt(1-a_bar_t) * noise
    fac1 = extract(sqrt_alphas_cumprod, t, x0.shape)
    fac2 = extract(sqrt_one_minus_alphas_cumprod, t, x0.shape)
    return fac1 * x0 + fac2 * noise

def extract(a, t, x_shape):
    # a[t] → reshaped to x_shape
    b = t.shape[0]
    out = a.gather(-1, t).float()
    return out.reshape(b, *((1,)*(len(x_shape)-1)))

@torch.no_grad()
def p_sample(model, x_t, t, y, cfg_scale=1.0):
    # εθ(x_t, t, y) with optional CFG: ε = (1+s)*ε(y) - s*ε(∅)
    eps_cond = model(x_t, t, y)
    if cfg_scale != 1.0:
        eps_uncond = model(x_t, t, None)
        eps = (1+cfg_scale)*eps_cond - cfg_scale*eps_uncond
    else:
        eps = eps_cond

    a_t = extract(alphas, t, x_t.shape)
    a_bar_t = extract(alphas_cumprod, t, x_t.shape)
    sqrt_one_minus_a_bar_t = torch.sqrt(torch.clamp(1 - a_bar_t, 1e-8))
    sqrt_inv_a_t = torch.sqrt(1.0 / a_t)

    # pred x0
    x0_pred = (x_t - sqrt_one_minus_a_bar_t * eps) / torch.sqrt(torch.clamp(a_bar_t, 1e-8))

    # posterior mean
    if (t == 0).all():
        noise = 0
    else:
        noise = torch.randn_like(x_t)
    beta_t = extract(betas, t, x_t.shape)
    mean = sqrt_inv_a_t * (x_t - (beta_t / sqrt_one_minus_a_bar_t) * eps)
    var = beta_t
    return mean + torch.sqrt(var) * noise

@torch.no_grad()
def sample_ddpm(model, labels, cfg_scale=1.0, show_progress=True, every=1, callback=None):
    """
    DDPM 샘플링 with progress bar.
    - show_progress: tqdm 진행바 표시
    - every: tqdm postfix/콜백 업데이트 주기(스텝)
    - callback(step, x): 중간 상태를 보고 싶을 때 호출 (예: 시각화/저장)
    """
    model.eval()
    B = len(labels)
    x = torch.randn(B, C_in, T_len, device=device)
    y = torch.as_tensor(labels, dtype=torch.long, device=device)

    it = range(timesteps - 1, -1, -1)
    if show_progress:
        it = tqdm(it, total=timesteps, desc="DDPM sampling", ncols=100, smoothing=0.1)

    for i in it:
        t = torch.full((B,), i, device=device, dtype=torch.long)
        x = p_sample(model, x, t, y, cfg_scale=cfg_scale)

        if show_progress and (i % every == 0):
            it.set_postfix_str(f"t={i}")

        if callback is not None and (i % every == 0):
            # 필요하면 CPU로 내려서 넘겨도 됨: callback(i, x.detach().cpu().numpy())
            callback(i, x)

    return x.detach().cpu().numpy()

# ---------------------------
# Model / Optim
# ---------------------------
model = UNet1D(in_ch=C_in, base=64, t_dim=128, c_dim=64).to(device)
opt = torch.optim.Adam(model.parameters(), lr=lr)

# ---------------------------
# Train loop (GAN 느낌 로그)
# ---------------------------
def train_diffusion(model, loader, epochs=100):
    model.train()
    for ep in range(1, epochs+1):
        t0 = time.time()
        n_batches = 0
        loss_acc = 0.0
        for x0, y in loader:
            x0, y = x0.to(device), y.to(device)
            b = x0.size(0)

            # t ~ Uniform{0..T-1}
            t = torch.randint(0, timesteps, (b,), device=device)
            noise = torch.randn_like(x0)
            x_t = q_sample(x0, t, noise)

            # classifier-free guidance: 조건 드롭
            use_cond = torch.rand(b, device=device) > p_uncond
            y_in = y.clone()
            y_in[~use_cond] = 0  # 실제로는 forward에서 None을 쓰지만, batch-level로 나눠 처리 어려우니 trick ↓
            # 트릭: 두 번 forward해서 합치기 (cond/uncond)
            eps_pred = torch.zeros_like(x0)

            if use_cond.any():
                eps_pred[use_cond] = model(x_t[use_cond], t[use_cond], y[use_cond])
            if (~use_cond).any():
                eps_pred[~use_cond] = model(x_t[~use_cond], t[~use_cond], None)

            loss = F.mse_loss(eps_pred, noise)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            loss_acc += loss.item(); n_batches += 1

        dt = time.time() - t0
        print(f"[{ep:03d}/{epochs}] dt={dt:.2f}s  Loss={loss_acc/n_batches:.4f}")


# In[12]:


# ---------------------------
# Usage
# ---------------------------
# 예시) 학습
train_ds = EEGCondDataset(X_train, y)  # y: (N,) in {0..3}
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
train_diffusion(model, train_dl, epochs=epochs)


# In[21]:


# 예시) 저장 / 로드
os.makedirs("./experiments", exist_ok=True)
torch.save({"model": model.state_dict()}, "./experiments/diffusion_unet1d.pth")
ckpt = torch.load("./experiments/diffusion_unet1d.pth", map_location=device)
model.load_state_dict(ckpt["model"]); model.eval()


# In[24]:


# 2) 분류기 로드 (freeze)

def load_classifier(checkpoint_path):
    clf = RhythmSpecificDeepCNN22(in_channels=5*14, n_classes=4).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    clf.load_state_dict(state, strict=False)
    clf.eval()
    for p in clf.parameters():
        p.requires_grad = False
    return clf


clf_path = "./experiments/rhythm_cnn_best.pth"
clf = load_classifier(clf_path)


# In[25]:


# ───────────────────────────────────────────────────────────
# 공용: 멀티헤드 분류기 평가 함수
# ───────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_with_classifier(clf, X_synth, y4_synth):
    """
    X_synth: (N, 70, 1280) float32 (표준화 스케일)
    y4_synth: (N,) int64 in {0,1,2,3}
    """
    X_t = torch.tensor(X_synth, dtype=torch.float32, device=device)
    clf.eval()
    out = clf(X_t)  # dict: {"val":(N,), "aro":(N,)}

    pred_val = (torch.sigmoid(out["val"]) >= 0.5).long()
    pred_aro = (torch.sigmoid(out["aro"]) >= 0.5).long()
    preds_4  = (2 * pred_val + pred_aro).cpu().numpy()

    # joint 4-class
    acc4 = accuracy_score(y4_synth, preds_4)
    f1m4 = f1_score(y4_synth, preds_4, average="macro")

    # per-head
    y_val = (y4_synth // 2)
    y_aro = (y4_synth %  2)
    p_val = pred_val.cpu().numpy()
    p_aro = pred_aro.cpu().numpy()

    head = {}
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
        head[name] = {"Acc": acc, "Prec": prec, "Rec": rec, "F1": f1, "AUC": auc}

    return acc4, f1m4, head

# ───────────────────────────────────────────────────────────
# 공용: 균형 라벨 생성
# ───────────────────────────────────────────────────────────
def make_balanced_labels(n_per_class=250, n_class=4, dtype=np.int64):
    return np.concatenate([np.full(n_per_class, c, dtype=dtype) for c in range(n_class)], axis=0)

# ───────────────────────────────────────────────────────────
# 공용: “생성→평가→예쁜 프린트” 래퍼
# gen_fn(labels: np.ndarray[int]) -> np.ndarray float32 (N,70,1280)
# ───────────────────────────────────────────────────────────
def evaluate_synth_balanced(clf, gen_fn, n_per_class=250, n_class=4, title="Synth → Classifier"):
    labels_bal = make_balanced_labels(n_per_class=n_per_class, n_class=n_class)
    X_synth = gen_fn(labels_bal)  # (N,70,1280)
    y_synth = labels_bal

    acc4, f1m4, head = evaluate_with_classifier(clf, X_synth, y_synth)

    print(f"[{title}] Joint 4-class  Acc={acc4:.4f}  F1(macro)={f1m4:.4f}")
    for k, v in head.items():
        print(f"[{k}] Acc={v['Acc']:.4f}  Prec={v['Prec']:.4f}  Rec={v['Rec']:.4f}  F1={v['F1']:.4f}  AUC={v['AUC']:.4f}")
    return acc4, f1m4, head


# In[29]:


@torch.no_grad()
def ddpm_gen(labels):
    return sample_ddpm(model, labels, cfg_scale=2.0)  # (N,70,1280)

_ = evaluate_synth_balanced(clf, ddpm_gen, n_per_class=250, n_class=4, title="DDPM Synth → Classifier")


# In[ ]:




