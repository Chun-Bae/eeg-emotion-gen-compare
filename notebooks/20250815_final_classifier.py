#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import random
import torch

np.random.seed(42)
random.seed(42)


# In[2]:


get_ipython().system('python -m train.train_classifier --epoch 200 --save-path "experiments/classifier_best_epoch200.pth"')


# In[23]:


import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import pandas as pd

# 폰트 설정 (mac)
FONT_PATH = "/System/Library/Fonts/AppleSDGothicNeo.ttc"
font_prop = fm.FontProperties(fname=FONT_PATH)

state = torch.load("./experiments/classifier_best_epoch200.pth", map_location="cpu")
df_hist = pd.DataFrame(state["full_history"])

sns.set_theme(font_scale=1.1)

def format_plot(ax, title, y_label=""):
    ax.set_title(title, fontproperties=font_prop, fontsize=16)
    ax.set_xlabel("Epoch", fontproperties=font_prop)
    ax.set_ylabel(y_label, fontproperties=font_prop)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontproperties(font_prop)
    # 외곽선 제거
    for spine in ax.spines.values():
        spine.set_visible(False)

def plot_metric_grid(df, prefix, title_prefix):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True)
    ax_acc, ax_loss = axes

    # Accuracy
    sns.lineplot(data=df, x="epoch", y=f"{prefix}_acc", ax=ax_acc, label="accuracy")
    sns.lineplot(data=df, x="epoch", y=f"{prefix}_val_acc", ax=ax_acc, label="valid accuracy")
    ax_acc.legend(prop=font_prop, loc="lower right")
    format_plot(ax_acc, f"{title_prefix} Accuracy")

    # Loss
    sns.lineplot(data=df, x="epoch", y="train_loss", ax=ax_loss, label="loss")
    sns.lineplot(data=df, x="epoch", y="val_loss", ax=ax_loss, label="vaild loss")
    ax_loss.legend(prop=font_prop, loc="upper right")
    format_plot(ax_loss, f"{title_prefix} Loss")

    plt.tight_layout()
    plt.show()

# ==== V(Valence) ====
plot_metric_grid(df_hist, prefix="V", title_prefix="Valence")

# ==== A(Arousal) ====
plot_metric_grid(df_hist, prefix="A", title_prefix="Arousal")


# In[ ]:




