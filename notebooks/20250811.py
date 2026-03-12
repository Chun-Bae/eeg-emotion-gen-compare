#!/usr/bin/env python
# coding: utf-8

# In[29]:


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
from models.classifier import Classifier_EEG
from models.cDCGAN import G2D, D2D

# ---- sklearn ----
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

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
df['y'] = (
    2 * (df['valence'] >= 2.5).astype(int) +
        (df['arousal'] >= 2.5).astype(int)
)
df=df.drop({'arousal','dominance','valence'},axis=1)


# In[4]:


# ---------------------------
# 스케일링 : -1 ~ 1
# ---------------------------
X = np.stack(df['eeg_band_data'].values).astype(np.float32)
X = X.reshape(-1, 5 * 14)

scaler = MinMaxScaler(feature_range=(-1, 1))
X_scaled = scaler.fit_transform(X)

X = X_scaled.reshape(-1, 1280, 5, 14)
y = df['y'].values


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


# In[32]:


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
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).permute(0, 3, 2, 1)  # (B, 14, 5, 1280)
        self.y = torch.tensor(y).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# In[9]:


# ---------------------------
# DataLoader
# ---------------------------
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=seed_num, stratify=y)

train_dataset = EEGDataset(X_train, y_train)
val_dataset = EEGDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


# In[10]:


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


# In[11]:


# ---------------------------
# 분류 모델 불러오기
# ---------------------------
ckpt = torch.load("experiments/250810_EEG_classifier_46.pth", map_location="cpu")
model_clf = Classifier_EEG().to(device)
model_clf.load_state_dict(ckpt["model_state_dict"])
model_clf.eval()


# In[33]:


# ---------------------------
# 학습 정의
# ---------------------------

# ---- model ----
G = G2D(z_dim=z_dim, n_classes=n_classes).to(device)
D = D2D(n_classes=n_classes).to(device)

# ---- 최적화 설정 ----
opt_G = torch.optim.AdamW(G.parameters(), lr=lr_G, betas=betas, weight_decay=weight_decay)
opt_D = torch.optim.AdamW(D.parameters(), lr=lr_D, betas=betas, weight_decay=weight_decay)
# mac에서는 적용 안됨 (mps 없음)
scaler_G = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))
scaler_D = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))


# In[40]:


# ---------------------------
# 학습 프로세스
# ---------------------------

for epoch in range(1, epochs+1):
    G.train(); D.train()
    t0 = time.time()
    loss_G_avg = 0.0
    loss_D_avg = 0.0
    nb = 0

    for x_real, y in train_loader:
        nb += 1
        x_real = x_real.to(device, non_blocking=True)  # (B,14,5,1280)
        y      = y.to(device, non_blocking=True)
        bsz    = x_real.size(0)

        # -------------------------
        # 1) Update Discriminator (hinge)
        # -------------------------
        for _ in range(n_critic):
            z = torch.randn(bsz, z_dim, device=device)
            with torch.cuda.amp.autocast(enabled=(device=="cuda")):
                # fake
                x_fake = G(z, y).detach()                    # stop grad to G
                d_real = D(x_real, y)                        # (B,)
                d_fake = D(x_fake, y)                        # (B,)
                # hinge: max(0, 1 - D(real)) + max(0, 1 + D(fake))
                loss_D = torch.relu(1.0 - d_real).mean() + torch.relu(1.0 + d_fake).mean()

            opt_D.zero_grad(set_to_none=True)
            scaler_D.scale(loss_D).backward()
            scaler_D.step(opt_D)
            scaler_D.update()

        # -------------------------
        # 2) Update Generator (hinge)
        # -------------------------
        z = torch.randn(bsz, z_dim, device=device)
        with torch.cuda.amp.autocast(enabled=(device=="cuda")):
            x_fake = G(z, y)
            d_fake = D(x_fake, y)
            # hinge-G: maximize D(fake)  <=> minimize -D(fake)
            loss_G = - d_fake.mean()

        opt_G.zero_grad(set_to_none=True)
        scaler_G.scale(loss_G).backward()
        scaler_G.step(opt_G)
        scaler_G.update()

        loss_D_avg += loss_D.item()
        loss_G_avg += loss_G.item()

    loss_D_avg /= max(1, nb)
    loss_G_avg /= max(1, nb)
    took = time.time() - t0
    print(f"[Epoch {epoch:03d}] D: {loss_D_avg:.4f} | G: {loss_G_avg:.4f} | {took:.1f}s")

print("Done.")
# 20번째


# In[14]:


# # ---------------------------
# # 모델 저장
# # ---------------------------
# torch.save(G.state_dict(), "./experiments/250811_EEG_cDCGAN_G.pth")
# torch.save(D.state_dict(), "./experiments/250811_EEG_cDCGAN_D.pth")


# In[41]:


G.to(device).eval()
model_clf.to(device).eval()


@torch.no_grad()
def gen_only(G, n_classes, z_dim, per_class, device):
    xs, ys = [], []
    for c in range(n_classes):
        z = torch.randn(per_class, z_dim, device=device)
        y = torch.full((per_class,), c, dtype=torch.long, device=device)
        x = G(z, y).clamp(-1, 1)         # (N,14,5,1280)
        xs.append(x.cpu())
        ys.append(y.cpu())
    return torch.cat(xs, 0), torch.cat(ys, 0)

# 1) 생성
Xg, yg = gen_only(G, n_classes, z_dim, gen_per_class, device)

# 2) 로더
gen_loader = DataLoader(TensorDataset(Xg, yg), batch_size=batch_size, shuffle=False)

@torch.no_grad()
def evaluate_on_generated():
    y_true, y_pred = [], []
    for xb, yb in gen_loader:
        xb = xb.to(device)
        logits = model_clf(xb)          # (B, n_classes)
        y_true.append(yb)
        y_pred.append(logits.argmax(1).cpu())
    y_true = torch.cat(y_true).numpy()
    y_pred = torch.cat(y_pred).numpy()
    print(y_true)
    print()
    print(y_pred)
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro")
    return acc, f1m

acc, f1m = evaluate_on_generated()
print(f"[Generated only] acc={acc:.3f}  f1(macro)={f1m:.3f}")


# In[36]:


Xg


# In[37]:


Xg_min = Xg.min().item()
Xg_max = Xg.max().item()
Xg_mean = Xg.mean().item()
Xg_std = Xg.std().item()
print(f"Xg stats  min={Xg_min:.3f}  max={Xg_max:.3f}  mean={Xg_mean:.3f}  std={Xg_std:.3f}")


# In[ ]:




