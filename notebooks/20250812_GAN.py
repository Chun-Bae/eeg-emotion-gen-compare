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
from utils.device_selection import device_selection
from data_processing.data_load import initial_dreamer_load
from data_processing.data_spilt import clf_data_split
from data_processing.data_object import 
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

np.random.seed(42)
random.seed(42)


# In[9]:


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


# In[10]:


device = device_selection()
clf = Classifier(in_channels=5*14, n_classes=4).to(device)
train_loader, val_loader = clf_data_split(df=initial_dreamer_load(), batch_size=64, seed=42)


# In[26]:


z_dim   = 128
n_class = 4       # (val, aro) 조합 → 0..3
C_in    = 70      # 5 bands * 14 ch
T_len   = 1280

# ---------------------------
# Conditional BatchNorm1d
# ---------------------------
class CBN1d(nn.Module):
    def __init__(self, num_features, n_class, emb_dim=64):
        super().__init__()
        self.bn   = nn.BatchNorm1d(num_features, affine=False)
        self.emb  = nn.Embedding(n_class, emb_dim)
        self.gama = nn.Linear(emb_dim, num_features)
        self.beta = nn.Linear(emb_dim, num_features)
        nn.init.zeros_(self.beta.weight); nn.init.ones_(self.gama.weight)
        nn.init.zeros_(self.beta.bias);   nn.init.zeros_(self.gama.bias)
    def forward(self, x, y):
        # x: (B,C,L), y: (B,)
        h = self.bn(x)
        e = self.emb(y)                    # (B,emb)
        g = self.gama(e).unsqueeze(-1)     # (B,C,1)
        b = self.beta(e).unsqueeze(-1)     # (B,C,1)
        return g * h + b

# ---------------------------
# Generator (upsample x2 * 4 → 80→1280)
# ---------------------------
class Gen1D(nn.Module):
    def __init__(self, z_dim=128, n_class=4, C_out=70, T_len=1280, base_ch=256):
        super().__init__()
        self.T0 = T_len // 16              # 80
        self.linear = nn.Linear(z_dim, base_ch * self.T0)

        # 업블록: (C_in → C_out), upsample×2
        self.cbn1 = CBN1d(base_ch, n_class)
        self.conv1 = nn.Conv1d(base_ch, base_ch, 3, padding=1)

        self.cbn2 = CBN1d(base_ch, n_class)
        self.conv2 = nn.Conv1d(base_ch, base_ch//2, 3, padding=1)

        self.cbn3 = CBN1d(base_ch//2, n_class)
        self.conv3 = nn.Conv1d(base_ch//2, base_ch//4, 3, padding=1)

        self.cbn4 = CBN1d(base_ch//4, n_class)
        self.conv4 = nn.Conv1d(base_ch//4, base_ch//4, 3, padding=1)

        self.to_out = nn.Conv1d(base_ch//4, C_out, 1)

    def _up(self, x):  # nearest upsample ×2
        return F.interpolate(x, scale_factor=2, mode="nearest")

    def forward(self, z, y):
        # z:(B,z_dim), y:(B,)
        h = self.linear(z).view(z.size(0), -1, self.T0)          # (B,256,80)

        h = self._up(F.relu(self.cbn1(h, y)))
        h = F.relu(self.conv1(h))                                # (B,256,160)

        h = self._up(F.relu(self.cbn2(h, y)))
        h = F.relu(self.conv2(h))                                # (B,128,320)

        h = self._up(F.relu(self.cbn3(h, y)))
        h = F.relu(self.conv3(h))                                # (B,64,640)

        h = self._up(F.relu(self.cbn4(h, y)))
        h = F.relu(self.conv4(h))                                # (B,64,1280)

        x = self.to_out(h)                                       # (B,70,1280)
        # 표준화된 실수 스케일 가정 → tanh 없음
        return x

# ---------------------------
# Discriminator (Projection D)
# ---------------------------
def SN(m):  # spectral norm helper
    return nn.utils.spectral_norm(m)

class Disc1D(nn.Module):
    def __init__(self, C_in=70, n_class=4, base_ch=64):
        super().__init__()
        self.conv1 = SN(nn.Conv1d(C_in,   base_ch,   5, padding=2))
        self.conv2 = SN(nn.Conv1d(base_ch, base_ch*2,5, padding=2))
        self.conv3 = SN(nn.Conv1d(base_ch*2, base_ch*4,5, padding=2))
        self.conv4 = SN(nn.Conv1d(base_ch*4, base_ch*4,3, padding=1))

        self.pool  = nn.AvgPool1d(2)
        self.lin   = SN(nn.Linear(base_ch*4, 1))
        self.emb   = nn.Embedding(n_class, base_ch*4)  # projection

    def forward(self, x, y):
        # x:(B,70,1280), y:(B,)
        h = F.leaky_relu(self.conv1(x), 0.2); h = self.pool(h)   # L/2
        h = F.leaky_relu(self.conv2(h), 0.2); h = self.pool(h)   # L/4
        h = F.leaky_relu(self.conv3(h), 0.2); h = self.pool(h)   # L/8
        h = F.leaky_relu(self.conv4(h), 0.2); h = self.pool(h)   # L/16

        h = h.mean(dim=-1)                                       # GAP → (B,C)
        out = self.lin(h).squeeze(1)                             # (B,)
        # projection term
        proj = (self.emb(y) * h).sum(dim=1)                      # (B,)
        return out + proj

# ---------------------------
# Loss (non-saturating logistic)
# ---------------------------
bce_logits = nn.BCEWithLogitsLoss()

def d_loss_fn(d_real, d_fake):
    return bce_logits(d_real, torch.ones_like(d_real)) + \
           bce_logits(d_fake, torch.zeros_like(d_fake))

def g_loss_fn(d_fake):
    return bce_logits(d_fake, torch.ones_like(d_fake))

# ---------------------------
# Small dataset wrapper example
# ---------------------------
class EEGCondDataset(Dataset):
    # X: (N,70,1280)  (표준화된 실수), y4: (N,) in {0,1,2,3}
    def __init__(self, X, y4):
        self.X  = torch.tensor(X, dtype=torch.float32)
        self.y4 = torch.tensor(y4, dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y4[i]

# ---------------------------
# Build
# ---------------------------
G = Gen1D(z_dim=z_dim, n_class=n_class, C_out=C_in, T_len=T_len).to(device)
D = Disc1D(C_in=C_in, n_class=n_class).to(device)

optG = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
optD = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))


# In[27]:


train_ds = EEGCondDataset(X_train, y)
train_dl = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=True)

epochs = 50
for ep in range(1, epochs+1):
    t0 = time.time()
    G.train(); D.train()
    d_loss_acc, g_loss_acc, n_batches = 0.0, 0.0, 0

    for real, y in train_dl:
        real, y = real.to(device), y.to(device)
        b = real.size(0)
        # ---------------- D step ----------------
        z = torch.randn(b, z_dim, device=device)
        fake = G(z, y).detach()
        d_real = D(real, y)
        d_fake = D(fake, y)
        loss_d = d_loss_fn(d_real, d_fake)

        optD.zero_grad(set_to_none=True)
        loss_d.backward()
        optD.step()

        # ---------------- G step ----------------
        z = torch.randn(b, z_dim, device=device)
        fake = G(z, y)
        d_fake = D(fake, y)
        loss_g = g_loss_fn(d_fake)

        optG.zero_grad(set_to_none=True)
        loss_g.backward()
        optG.step()

        d_loss_acc += loss_d.item()
        g_loss_acc += loss_g.item()
        n_batches  += 1

    dt = time.time() - t0
    print(f"[{ep:03d}/{epochs}]  dt={dt:.2f}s  D={d_loss_acc/n_batches:.4f}  G={g_loss_acc/n_batches:.4f}")


# In[32]:


G.eval()

# ---------------------------
# 2. Generator로 데이터 생성
# ---------------------------
num_samples = 1000
z = torch.randn(num_samples, z_dim, device=device)
labels = torch.randint(0, n_classes, (num_samples,), device=device)

with torch.no_grad():
    X_gen = G(z, labels).cpu().numpy()  # (num_samples, 70, 1280)
    y_gen = labels.cpu().numpy()

# ---------------------------
# 3. CNN 모델 불러오기
# ---------------------------
path = "./experiments/rhythm_cnn_best.pth"

ckpt = torch.load(path, map_location=device)
state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

model = RhythmSpecificDeepCNN22(in_channels=5*14, n_classes=4).to(device)

# strict=True 권장 (구조가 정확히 같아야 함)
missing, unexpected = model.load_state_dict(state, strict=False)
if missing or unexpected:
    print("[load_state_dict] missing:", missing)
    print("[load_state_dict] unexpected:", unexpected)
else:
    print("Loaded weights OK.")
model.eval()

# ---------------------------
# 4. CNN 모델로 생성 데이터 평가
# ---------------------------
X_gen_tensor = torch.tensor(X_gen, dtype=torch.float32, device=device)
y_gen_tensor = torch.tensor(y_gen, dtype=torch.long, device=device)

with torch.no_grad():
    out = model(X_gen_tensor)  # dict: {"val": (B,), "aro": (B,)}

    # 각 헤드 이진 예측 (threshold=0.5)
    pred_val = (torch.sigmoid(out["val"]) >= 0.5).long()  # (B,)
    pred_aro = (torch.sigmoid(out["aro"]) >= 0.5).long()  # (B,)

    # 2비트 결합 → 0..3
    preds = (2 * pred_val + pred_aro).cpu().numpy()

y4 = y_gen  # 생성 시 조건 라벨 0..3

# joint 4-class
acc4 = accuracy_score(y4, preds)
f1m4 = f1_score(y4, preds, average="macro")
print(f"[Joint 4-class] Acc={acc4:.4f}  F1(macro)={f1m4:.4f}")

# per-head (이진)
y_val = (y4 // 2)
y_aro = (y4 % 2)
p_val = pred_val.cpu().numpy()
p_aro = pred_aro.cpu().numpy()

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
    print(f"[{name}] Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f}  AUC={auc:.4f}")


# In[14]:


path = "./experiments/rhythm_cnn_best.pth"

ckpt = torch.load(path, map_location=device)
state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

model = RhythmSpecificDeepCNN22(in_channels=5*14, n_classes=4).to(device)


# In[18]:


import torch
import torch.onnx

# 모델을 eval 모드로 전환
model.eval()

# 더미 입력 (batch size 1)
x = torch.zeros(1, 70, 1280).to(device)

# ONNX export
torch.onnx.export(
    model, 
    x, 
    "cnn.onnx",
    input_names=['CNN'], 
    output_names=['Output'],
    opset_version=11,          # 최신 호환 버전
    do_constant_folding=True,  # 불필요한 상수 연산 제거
    dynamic_axes={
        'CNN': {0: 'batch_size'},     # batch dimension 가변
        'Output': {0: 'batch_size'}
    }
)
print("Exported to cnn.onnx")


# In[ ]:




