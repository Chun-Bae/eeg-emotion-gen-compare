#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import random
import time
import gc
import math
import os
from tqdm.auto import tqdm
import torch.nn.functional as F


from models.classifier import Classifier
from models.cDDPM import UNet1D
from utils.device_selection import device_selection
from data_processing.data_load import initial_dreamer_load
from data_processing.data_spilt import gen_data_split


from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score



import torch
import torch.nn as nn
np.random.seed(42)
random.seed(42)


# In[2]:





# In[2]:


device=device_selection()


# In[3]:


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
batch_size   = 128
epochs       = 100
lr           = 2e-4
timesteps    = 1000        # diffusion steps
p_uncond     = 0.1         # CFG: cond drop prob (훈련용)
cfg_scale    = 2.0         # CFG scale (샘플링용, 1.0=끄기)


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
def train_diffusion(model, loader, device, epochs=100):
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
            use_cond = torch.ones(b, dtype=torch.bool, device=device)

            y_in = y.clone()

            y_in[~use_cond] = 0  # 실제로는 forward에서 None을 쓰지만, batch-level로 나눠 처리 어려우니 trick ↓
            # 트릭: 두 번 forward해서 합치기 (cond/uncond)

            eps_pred = torch.zeros_like(x0)

            eps_pred = model(x_t, t, y)         

            loss = F.mse_loss(eps_pred, noise)

            opt.zero_grad(set_to_none=True)

            loss.backward()

            opt.step()


            loss_acc += loss.item(); n_batches += 1
            torch.save({
                "epoch": ep,
                "model_state": model.state_dict(),
                "optimizer_state": opt.state_dict(),
            }, "ddpm_checkpoint.pth")

        dt = time.time() - t0
        print(f"[{ep:03d}/{epochs}] dt={dt:.2f}s  Loss={loss_acc/n_batches:.4f}")


# In[9]:


# ---------------------------
# Usage
# ---------------------------
# 예시) 학습
train_loader, _ = gen_data_split(df=initial_dreamer_load(),  batch_size=batch_size,seed=42)
# train_diffusion(model, train_loader, device, epochs=epochs)


# In[10]:


gc.collect()


# In[4]:


import torch

torch.__version__


# In[11]:


train_diffusion(model, train_loader, device, epochs=epochs)


# In[7]:


import torch
import numpy as np
from contextlib import nullcontext

@torch.no_grad()
def ddpm_gen_in_chunks(labels, *, chunk=16, cfg_scale=None, steps=None, amp=True):
    """
    labels: np.ndarray[int] or torch.LongTensor, shape (N,)
    return: np.ndarray float32, shape (N, 70, 1280)
    """
    # ── 준비
    dev = next(model.parameters()).device
    if not torch.is_tensor(labels):
        labels = torch.as_tensor(labels, dtype=torch.long, device=dev)
    else:
        labels = labels.to(dev)

    outs = []
    # AMP 컨텍스트 (MPS/GPU면 켜고 CPU면 자동으로 꺼짐)
    if torch.cuda.is_available():
        amp_cm = torch.cuda.amp.autocast(dtype=torch.float16, enabled=amp)
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        amp_cm = torch.autocast("mps", dtype=torch.float16, enabled=amp)
    else:
        amp_cm = nullcontext()

    # ── 마이크로배치 루프
    for s in range(0, labels.shape[0], chunk):
        lab = labels[s:s+chunk]

        with amp_cm, torch.inference_mode():
            # **중요**: 학습을 cond-only로 했으면 CFG는 끄기
            x_part = sample_ddpm(model, lab, cfg_scale=1.0 if cfg_scale is None else cfg_scale, steps=steps)
            # sample_ddpm이 torch.Tensor 반환한다고 가정
            x_part = x_part.detach().cpu().to(torch.float32)

        outs.append(x_part)

        # (선택) 피크 완화
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            try: torch.mps.empty_cache()
            except: pass

    X = torch.cat(outs, dim=0).numpy()
    return X  # (N, 70, 1280) float32

# 불러오기
ckpt = torch.load("ddpm_checkpoint.pth", map_location=device)
model.load_state_dict(ckpt["model_state"])
opt.load_state_dict(ckpt["optimizer_state"])

# ───────────────────────────────────────────────────────────
# 공용: 멀티헤드 분류기 평가 함수
# ───────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_with_classifier(clf, X_synth, y4_synth, batch=64):
    import numpy as np, torch
    device = next(clf.parameters()).device

    # torch/np 모두 허용
    if isinstance(X_synth, np.ndarray):
        X_t = torch.from_numpy(X_synth).float()
    else:
        X_t = X_synth.float().cpu()  # 안전하게 CPU 기준으로 모음

    preds_val, preds_aro = [], []
    for s in range(0, X_t.shape[0], batch):
        xb = X_t[s:s+batch].to(device, non_blocking=True)
        out = clf(xb)  # {"val":(N,), "aro":(N,)}
        preds_val.append(torch.sigmoid(out["val"]).detach().cpu())
        preds_aro.append(torch.sigmoid(out["aro"]).detach().cpu())

    p_val = torch.cat(preds_val, 0).numpy() >= 0.5
    p_aro = torch.cat(preds_aro, 0).numpy() >= 0.5
    preds_4 = (2 * p_val.astype(int) + p_aro.astype(int))

    from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, roc_auc_score
    acc4 = accuracy_score(y4_synth, preds_4)
    f1m4 = f1_score(y4_synth, preds_4, average="macro")

    y_val = (y4_synth // 2)
    y_aro = (y4_synth %  2)

    head = {}
    for name, y_true, p_bin, scores in [
        ("Val", y_val, p_val, torch.cat(preds_val, 0).numpy()),
        ("Aro", y_aro, p_aro, torch.cat(preds_aro, 0).numpy()),
    ]:
        acc = accuracy_score(y_true, p_bin)
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, p_bin, average="binary", zero_division=0)
        try:
            auc = roc_auc_score(y_true, scores)
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


# In[8]:


from utils.load_classifier import load_classifier

clf = load_classifier(checkpoint_path="./experiments/classifier_best.pth", device=device)
clf.eval()

@torch.no_grad()
def ddpm_gen(labels):
    return sample_ddpm(model, labels, cfg_scale=1.0)  # (N,70,1280)

_ = evaluate_synth_balanced(
    clf,
    lambda labels: ddpm_gen_in_chunks(labels, chunk=32, cfg_scale=None, steps=50, amp=True),
    n_per_class=250,
    n_class=4,
    title="DDPM Synth → Classifier (chunked, cond-only)"
)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




