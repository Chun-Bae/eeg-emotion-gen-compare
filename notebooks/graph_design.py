#!/usr/bin/env python
# coding: utf-8

# In[19]:


import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import pandas as pd
import numpy as np

# --- 한글 폰트 경로(맥) ---
FONT_PATH = "/System/Library/Fonts/AppleSDGothicNeo.ttc"
font_prop = fm.FontProperties(fname=FONT_PATH)

# --- 더미 메트릭(우상향, 검증 포함) 생성 ---
np.random.seed(42)
n = 60
epochs = np.arange(1, n + 1)

def upcurve(start, end, noise_scale=0.01):
    base = np.linspace(start, end, n)
    noise = np.random.normal(0, noise_scale, n).cumsum() / 5
    raw = base + noise
    raw = np.clip(raw, 0, 0.999)
    return np.maximum.accumulate(raw)

def downcurve(start, end, noise_scale=0.01):
    base = np.linspace(start, end, n)  # start> end (감소)
    noise = np.random.normal(0, noise_scale, n).cumsum() / 5
    raw = base + noise
    raw = np.clip(raw, 1e-6, None)
    # 단조 감소 보정: 누적 최소를 앞으로 전파
    return np.minimum.accumulate(raw[::-1])[::-1]

# Train curves
acc_tr  = upcurve(0.72, 0.95, noise_scale=0.015)
auc_tr  = upcurve(0.75, 0.97, noise_scale=0.012)
prec_tr = upcurve(0.68, 0.93, noise_scale=0.018)
rec_tr  = upcurve(0.66, 0.92, noise_scale=0.018)
f1_tr   = upcurve(0.67, 0.925, noise_scale=0.017)
loss_tr = downcurve(0.9, 0.12, noise_scale=0.02)

# Validation curves (train보다 약간 낮게/높게)
def make_val(train, gap=0.01):
    jitter = np.abs(np.random.normal(0, gap/3, len(train)))
    val = np.clip(train - gap - jitter, 0, 0.999)
    # 단조성 유지(상승/감소를 train에 맞추고 싶다면 누적 보정)
    return np.maximum.accumulate(val) if train[0] < train[-1] else np.minimum.accumulate(val[::-1])[::-1]

acc_val  = make_val(acc_tr,  gap=0.012)
auc_val  = make_val(auc_tr,  gap=0.010)
prec_val = make_val(prec_tr, gap=0.012)
rec_val  = make_val(rec_tr,  gap=0.012)
f1_val   = make_val(f1_tr,   gap=0.011)

# 손실은 검증 쪽이 약간 더 높게 (감소형)
def make_val_loss(train_loss, gap=0.02):
    jitter = np.abs(np.random.normal(0, gap/3, len(train_loss)))
    val = np.clip(train_loss + gap + jitter, 1e-6, None)
    return np.minimum.accumulate(val[::-1])[::-1]

loss_val = make_val_loss(loss_tr, gap=0.025)

df = pd.DataFrame({
    "Epoch": epochs,
    "Accuracy": acc_tr,  "Val_Accuracy": acc_val,
    "AUC": auc_tr,       "Val_AUC": auc_val,
    "Precision": prec_tr,"Val_Precision": prec_val,
    "Recall": rec_tr,    "Val_Recall": rec_val,
    "F1": f1_tr,         "Val_F1": f1_val,
    "Loss": loss_tr,     "Val_Loss": loss_val
})

# --- Seaborn 스타일(whitegrid) + 폰트 크기 ---
sns.set_theme(font_scale=1.1)

# --- 2×3 서브플롯 생성 ---
fig, axes = plt.subplots(2, 3, figsize=(16, 10), sharex=True)
(ax_acc, ax_auc, ax_prec), (ax_rec, ax_f1, ax_loss) = axes

# 공통 유틸: 스파인 제거 & 폰트 적용
def beautify(ax, title):
    ax.set_title(title, fontproperties=font_prop, fontsize=16)
    ax.set_xlabel("Epoch", fontproperties=font_prop)
    ax.set_ylabel("값", fontproperties=font_prop)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(font_prop)
    for spine in ax.spines.values():
        spine.set_visible(False)

# ----- (1) Accuracy + Loss(보조축) -----
sns.lineplot(data=df, x="Epoch", y="Accuracy", ax=ax_acc, label="정확도")
sns.lineplot(data=df, x="Epoch", y="Val_Accuracy", ax=ax_acc, label="검증 정확도", linestyle="--")

ax_acc2 = ax_acc.twinx()
sns.lineplot(data=df, x="Epoch", y="Loss", ax=ax_acc2, label="손실", alpha=0.9)
sns.lineplot(data=df, x="Epoch", y="Val_Loss", ax=ax_acc2, label="검증 손실", linestyle="--", alpha=0.9)
ax_acc2.set_ylabel("손실", fontproperties=font_prop)

# 두 축의 범례 합치기
handles1, labels1 = ax_acc.get_legend_handles_labels()
handles2, labels2 = ax_acc2.get_legend_handles_labels()
legend = ax_acc.legend(handles1 + handles2, labels1 + labels2, prop=font_prop, loc="lower right")
ax_acc2.get_legend().remove() if ax_acc2.get_legend() else None
beautify(ax_acc, "정확도 & 손실")

# ----- (2) AUC -----
sns.lineplot(data=df, x="Epoch", y="AUC", ax=ax_auc, label="AUC")
sns.lineplot(data=df, x="Epoch", y="Val_AUC", ax=ax_auc, label="검증 AUC", linestyle="--")
ax_auc.legend(prop=font_prop, loc="lower right")
beautify(ax_auc, "AUC")

# ----- (3) Precision -----
sns.lineplot(data=df, x="Epoch", y="Precision", ax=ax_prec, label="정밀도")
sns.lineplot(data=df, x="Epoch", y="Val_Precision", ax=ax_prec, label="검증 정밀도", linestyle="--")
ax_prec.legend(prop=font_prop, loc="lower right")
beautify(ax_prec, "정밀도 (Precision)")

# ----- (4) Recall -----
sns.lineplot(data=df, x="Epoch", y="Recall", ax=ax_rec, label="재현율")
sns.lineplot(data=df, x="Epoch", y="Val_Recall", ax=ax_rec, label="검증 재현율", linestyle="--")
ax_rec.legend(prop=font_prop, loc="lower right")
beautify(ax_rec, "재현율 (Recall)")

# ----- (5) F1 -----
sns.lineplot(data=df, x="Epoch", y="F1", ax=ax_f1, label="F1-score")
sns.lineplot(data=df, x="Epoch", y="Val_F1", ax=ax_f1, label="검증 F1-score", linestyle="--")
ax_f1.legend(prop=font_prop, loc="lower right")
beautify(ax_f1, "F1-score")

# ----- (6) Loss (단독) -----
sns.lineplot(data=df, x="Epoch", y="Loss", ax=ax_loss, label="손실")
sns.lineplot(data=df, x="Epoch", y="Val_Loss", ax=ax_loss, label="검증 손실", linestyle="--")
ax_loss.legend(prop=font_prop, loc="upper right")
beautify(ax_loss, "손실 (Loss)")
ax_loss.set_ylabel("손실", fontproperties=font_prop)

plt.tight_layout()
plt.show()


# In[15]:





# In[ ]:




