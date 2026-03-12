#!/usr/bin/env python
# coding: utf-8

# In[12]:


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
from models.classifier import Classifier_EEG, RhythmSpecificDeepCNN
from models.cDCGAN import G2D, D2D

# ---- sklearn ----
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# ---- scipy ----
import scipy.io

# ---- torch ----
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, Dataset, DataLoader


# In[13]:


# ---------------------------
# 원시 데이터 가져오기
# ---------------------------
mat = scipy.io.loadmat('./data/DREAMER.mat', struct_as_record=False, squeeze_me=True)
dreamer = mat['DREAMER']


# In[14]:


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
df=df.drop({'arousal','dominance','valence'},axis=1)


# In[15]:


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


# In[16]:


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


# In[17]:


# ---------------------------
# seed 고정
# ---------------------------
torch.manual_seed(seed_num)
np.random.seed(seed_num)
random.seed(seed_num)


# In[18]:


# ---------------------------
# Dataset object
# ---------------------------
class EEGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# In[19]:


# ---------------------------
# Data Split
# ---------------------------
X = np.stack(df['eeg_band_data'].values).astype(np.float32)
X = np.transpose(X, (0, 2, 3, 1))          
B, R, C, T = X.shape
X = X.reshape(B, R*C, T) 
y = df['y'].values

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=seed_num, stratify=y)
# ---------------------------
# 스케일링 : 표준화
# ---------------------------
X_train = np.swapaxes(X_train, 1, 2).reshape(-1, R*C)
X_val   = np.swapaxes(X_val,   1, 2).reshape(-1, R*C)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)

X_train = X_train.reshape(X_train.shape[0] // T, T, R*C).swapaxes(1, 2).astype(np.float32)
X_val   = X_val.reshape(X_val.shape[0]     // T, T, R*C).swapaxes(1, 2).astype(np.float32)

# ---------------------------
# DataLoader
# ---------------------------
train_dataset = EEGDataset(X_train, y_train)
val_dataset = EEGDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


# In[20]:


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


# In[ ]:


EPS = 1e-12

def update_cm(cm, preds, y, n_classes):
    with torch.no_grad():
        k = (preds.long() * n_classes + y.long()).to(torch.int64)
        binc = torch.bincount(k, minlength=n_classes * n_classes)
        binc_np = binc.cpu().numpy().reshape(n_classes, n_classes)  # numpy 변환
        cm += binc_np
    return cm

def metrics_from_cm(cm):
    cm = cm.astype(np.float64)
    tp = np.diag(cm)
    fp = cm.sum(axis=1) - tp
    fn = cm.sum(axis=0) - tp
    tn = cm.sum() - (tp + fp + fn)

    prec_c = tp / (tp + fp + EPS)
    rec_c  = tp / (tp + fn + EPS)
    f1_c   = 2 * prec_c * rec_c / (prec_c + rec_c + EPS)

    prec_macro = prec_c.mean()
    rec_macro  = rec_c.mean()
    f1_macro   = f1_c.mean()
    acc        = tp.sum() / (cm.sum() + EPS)
    return float(acc), float(prec_macro), float(rec_macro), float(f1_macro)

def accuracy(logits, y):
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()

def train_one_epoch(model, loader, opt, criterion, device, n_classes):
    model.train()
    total_loss, n = 0.0, 0
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)

    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        opt.zero_grad()
        logits = model(Xb)
        loss = criterion(logits, yb)
        loss.backward()
        opt.step()

        bs = yb.size(0)
        total_loss += loss.item() * bs
        n += bs

        preds = logits.argmax(dim=1)
        cm = update_cm(cm, preds.cpu(), yb.cpu(), n_classes)

    acc, prec, rec, f1 = metrics_from_cm(cm)
    return total_loss / n, {"acc": acc, "prec": prec, "rec": rec, "f1": f1}

@torch.no_grad()
def evaluate(model, loader, criterion, device, n_classes):
    model.eval()
    total_loss, n = 0.0, 0
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)

    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        logits = model(Xb)
        loss = criterion(logits, yb)

        bs = yb.size(0)
        total_loss += loss.item() * bs
        n += bs

        preds = logits.argmax(dim=1)
        cm = update_cm(cm, preds.cpu(), yb.cpu(), n_classes)

    acc, prec, rec, f1 = metrics_from_cm(cm)
    return total_loss / n, {"acc": acc, "prec": prec, "rec": rec, "f1": f1}
# ---------------------------
# 준비 & 학습 (논문 하이퍼파라미터)
# ---------------------------
in_channels = 5 * 14
n_classes   = int(np.unique(y_train).size)
model = RhythmSpecificDeepCNN(in_channels, n_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
epochs = 100

best_val_acc = 0.0
best_state = None

for epoch in range(1, epochs + 1):
    t0 = time.time()
    tr_loss, tr_m = train_one_epoch(model, train_loader, optimizer, criterion, device, n_classes)
    va_loss, va_m = evaluate(model, val_loader, criterion, device, n_classes)
    dt = time.time() - t0
    print(
        "\n" + "=" * 60 +
        f"\n[Epoch {epoch:02d}/{epochs}]  Time: {dt:.2f}s" +
        "\n" + "-" * 60 +
        "\n" +
        f" | Loss      | Train: {tr_loss:.4f}   Val: {va_loss:.4f}" +
        "\n" + "-" * 60 +
        "\n" +
        f" | Acc       | Train: {tr_m['acc']:.4f}   Val: {va_m['acc']:.4f}\n"
        f" | Prec      | Train: {tr_m['prec']:.4f}   Val: {va_m['prec']:.4f}\n"
        f" | Rec       | Train: {tr_m['rec']:.4f}   Val: {va_m['rec']:.4f}\n"
        f" | F1        | Train: {tr_m['f1']:.4f}   Val: {va_m['f1']:.4f}" +
        "\n" + "=" * 60
    )

    if va_m['acc'] > best_val_acc:
        best_val_acc = va_m['acc']
        best_state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_acc": best_val_acc,
            "in_channels": in_channels,
            "n_classes": n_classes,
        }

if best_state is not None:
    torch.save(best_state, "./experiments/rhythm_cnn_best.pth")
    print(f"Best val acc={best_val_acc:.4f} (epoch {best_state['epoch']}) saved to rhythm_cnn_best.pth")
else:
    print("No improvement recorded.")


# In[ ]:





# In[ ]:




