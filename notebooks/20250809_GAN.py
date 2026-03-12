#!/usr/bin/env python
# coding: utf-8

# In[2]:


import scipy.io

mat = scipy.io.loadmat('./data/DREAMER.mat', struct_as_record=False, squeeze_me=True)
print(mat.keys())


# EEG 데이터를 담고 있는 mat 파일의 객체를 확인하기 위한 과정입니다.  
# 'DREAMER' 외에는 메타데이터라 무시해도 좋습니다.

# In[3]:


dreamer = mat['DREAMER']
print("DREAMER/")
for key in dreamer.__dict__.keys():
    print("-", key)


# DREAMER 객체에 또 다시 어떤 객체들이 있는지 확인하는 과정입니다.  
# 실제 EEG 데이터는 'Data' 객체에 들어있구요. 나머지는 크게 중요한 것이 아닙니다.

# In[4]:


eeg_sampling_rate = dreamer.EEG_SamplingRate 
ecg_sampling_rate = dreamer.ECG_SamplingRate
eeg_electrodes = dreamer.EEG_Electrodes

print(f"EEG sampling rate: {eeg_sampling_rate} Hz")
print(f"ECG sampling rate: {ecg_sampling_rate} Hz")

print("EEG electrode names:", end=' ')
print(" | ".join(eeg_electrodes))


# EEG는 128Hz(1초에 128프레임 캡처)로 데이터가 담아져있음을 보여주는 정보입니다.  
# 그 외 EEG의 어떤 전극을 사용한지 보여줍니다. (논문에 이미 나와있긴함)

# In[5]:


data = dreamer.Data
print(data.shape)
subject0 = data[0]
print(subject0)


# 데이터에 크기를 확인하기 위한 코드입니다. 피험자가 23명이기 때문에 23라는 숫자가 출력되고 있습니다.  
# 그 아래 출력은 객체 고유 id 같은 거니 무시해도 됩니다.

# In[6]:


subject0 = dreamer.Data[0]
print("subject0/")
for key in subject0.__dict__.keys():
    print("-", key)


# 또 다시 피험자0 객체에는 어떤 데이터들이 들어있는 지 확인하는 과정입니다.  
# 나이, 성별, EEG data, ECG data, Valence(불쾌), Arousal(흥분), Dominance(통제력)의 정도를 나타냅니다.  
# VA 또는 VAD 모델이 존재하는데 이 수치를 기준으로 인간의 감정을 벡터로 표현할 수 있습니다.  
# 여기서 VAD가 정답 라벨(결과값)이라고 생각할 수 있습니다.

# In[7]:


eeg_stimuli = subject0.EEG.stimuli

for i, stimulus in enumerate(eeg_stimuli):
    time_frames = stimulus.shape[0]
    num_channels = stimulus.shape[1]
    duration_sec = int(time_frames / 128)

    print(f"[영상 {i:02d}] Time frames: {time_frames:5d}개 | Channels: {num_channels}개 | Duration: {duration_sec:3d}초")


# 피험자 별로 각각 영상 18개의 시청에 대한 EEG, ECG 데이터가 있고, 각각의 clip마다 영상의 길이가 다릅니다.  
# 좀 더 쉽게 설명하자면 각 영상에 대해 x(128 x 영상길이)에 대한 y(진폭 == EEG 신호) 가 존재합니다.

# In[8]:


for subj_idx, subject in enumerate(dreamer.Data):
    print(f"[Subject {subj_idx:02d}]")

    eeg_stimuli = subject.EEG.stimuli
    ecg_stimuli = subject.ECG.stimuli

    valence_scores = subject.ScoreValence
    arousal_scores = subject.ScoreArousal
    dominance_scores = subject.ScoreDominance

    for i in range(18):
        eeg = eeg_stimuli[i]
        ecg = ecg_stimuli[i]

        time_frames = eeg.shape[0]
        eeg_channels = eeg.shape[1]
        ecg_channels = ecg.shape[1] if ecg.ndim > 1 else 1
        duration = int(time_frames / 128)

        val = valence_scores[i]
        aro = arousal_scores[i]
        dom = dominance_scores[i]

        print(f"  [영상 {i:02d}] EEG: {time_frames:5d}×{eeg_channels:2d} | ECG: {time_frames:5d}×{ecg_channels:1d} | Duration: {duration:3d}초 | VAD: [{val:.1f}, {aro:.1f}, {dom:.1f}]")
    print()


# 이건 모든 피험자에 대해 정상적으로 데이터가 있는지 확인하기 위한 코드입니다.  
# 한 번 보고 넘어가도 됩니다.

# In[9]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Subject 0, Video 0
subject0 = dreamer.Data[0]
eeg = subject0.EEG.stimuli[0]
ecg = subject0.ECG.stimuli[0]
eeg_electrodes = dreamer.EEG_Electrodes

# Trim to equal length
min_len = min(len(eeg), len(ecg))
eeg = eeg[:min_len]
ecg = ecg[:min_len]

# Settings
frames_per_sec = 256
num_segments = 3

# Color palettes
eeg_colors = sns.color_palette("Blues", n_colors=eeg.shape[1])
ecg_colors = sns.color_palette("Reds", n_colors=ecg.shape[1] if ecg.ndim > 1 else 1)

# Plot setup
sns.set(style="whitegrid", font_scale=0.9)
fig, axes = plt.subplots(2, num_segments, figsize=(18, 8), sharey='row', sharex=False)

# Legend handles
eeg_handles, eeg_labels = [], []
ecg_handles, ecg_labels = [], []

for seg in range(num_segments):
    start = seg * frames_per_sec
    end = start + frames_per_sec
    x = np.arange(frames_per_sec)

    # EEG subplot
    ax_eeg = axes[0, seg]
    for ch in range(eeg.shape[1]):
        sns.lineplot(x=x, y=eeg[start:end, ch], ax=ax_eeg, color=eeg_colors[ch], label=None)
        if seg == 0:
            line_obj = ax_eeg.lines[-1]
            eeg_handles.append(line_obj)
            eeg_labels.append(f"EEG {eeg_electrodes[ch]}")
    ax_eeg.set_title(f"EEG: {seg+1}s (frame {start}–{end})")
    ax_eeg.set_xlabel("Frame")
    ax_eeg.set_ylabel("Amplitude" if seg == 0 else "")

    # ECG subplot
    ax_ecg = axes[1, seg]
    if ecg.ndim == 1:
        sns.lineplot(x=x, y=ecg[start:end], ax=ax_ecg, color=ecg_colors[0], label=None)
        if seg == 0:
            line_obj = ax_ecg.lines[-1]
            ecg_handles.append(line_obj)
            ecg_labels.append("ECG ch1")
    else:
        for ch in range(ecg.shape[1]):
            sns.lineplot(x=x, y=ecg[start:end, ch], ax=ax_ecg, color=ecg_colors[ch], label=None)
            if seg == 0:
                line_obj = ax_ecg.lines[-1]
                ecg_handles.append(line_obj)
                ecg_labels.append(f"ECG ch{ch+1}")
    ax_ecg.set_title(f"ECG: {seg+1}s (frame {start}–{end})")
    ax_ecg.set_xlabel("Frame")
    ax_ecg.set_ylabel("Amplitude" if seg == 0 else "")

# Global legend
fig.legend(handles=eeg_handles + ecg_handles,
           labels=eeg_labels + ecg_labels,
           loc='upper center',
           ncol=len(eeg_labels) + len(ecg_labels),
           bbox_to_anchor=(0.5, 1.02))

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


# EEG와 ECG를 시각화 한 코드입니다. 적당한 길이 256단위(2초)로 끊어서 나타냈습니다.

# In[10]:


# Sample EEG 정보 (피험자 0 - 영상 0)
subject0 = dreamer.Data[0]
eeg = subject0.EEG.stimuli[0]
eeg_names = dreamer.EEG_Electrodes
fs = dreamer.EEG_SamplingRate  # 128Hz

# 1초 구간 (128 프레임)
duration_sec = 1
n_samples = fs * duration_sec

# 채널 선택 (예: ch 0 = 첫 번째 채널 = Fp1 등)
ch = 0
eeg_segment = eeg[:n_samples, ch]
eeg_segment = eeg_segment - np.mean(eeg_segment)

# 주파수 변환
fft = np.fft.fft(eeg_segment)
freq = np.fft.fftfreq(n_samples, d=1/fs)
fft_magnitude = np.abs(fft)[:n_samples // 2]
fft_freq = freq[:n_samples // 2]

# 시각화
sns.set(style="whitegrid", font_scale=1.1)
fig, axs = plt.subplots(2, 1, figsize=(12, 6))

# 시간 영역
axs[0].plot(np.arange(n_samples) / fs, eeg_segment, color='blue')
axs[0].set_title(f"Time Domain EEG Signal (Channel: {eeg_names[ch]})")
axs[0].set_xlabel("Time (sec)")
axs[0].set_ylabel("Amplitude")

# 주파수 영역
axs[1].plot(fft_freq, fft_magnitude, color='red')
axs[1].set_title("Frequency Domain (FFT Magnitude)")
axs[1].set_xlabel("Frequency (Hz)")
axs[1].set_ylabel("Magnitude")

plt.tight_layout()
plt.show()


# 신호에는 시간 영역과 주파수 영역이라는 개념이 있는데, 평상시 볼 수 있는, 시간에 대한 신호 값을 나타내는 그래프를 시간 영역이라고 합니다.  
# 시간 영역에 있는 신호가 주파수를 얼마만큼 포함을 하고 있는지를 보여주는 것을 주파수 영역이라고 합니다.   
# 일반적으로 FFT(고속 푸리에 변환)으로 분해할 수 있습니다.

# In[11]:


from scipy.signal import butter, filtfilt

def apply_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
    주어진 EEG 시계열에 대해 Butterworth 밴드패스 필터를 적용합니다.
    
    Parameters:
        data (np.ndarray): 1D EEG 시계열 데이터
        lowcut (float): 하위 컷오프 주파수 (Hz)
        highcut (float): 상위 컷오프 주파수 (Hz)
        fs (int): 샘플링 주파수 (Hz)
        order (int): 필터의 차수

    Returns:
        filtered_data (np.ndarray): 필터링된 EEG 시계열
    """
    nyq = 0.5 * fs  # 나이퀴스트 주파수
    low = lowcut / nyq
    high = highcut / nyq
    
    b, a = butter(order, [low, high], btype='band')
    filtered_data = filtfilt(b, a, data)
    return filtered_data


# EEG는 일반적으로 델타, 세타, 알파, 베타, 감마파에서 특징을 두드러지게 나타내는 것으로 알려져있습니다.  
# 위 밴드패스 필터 함수는 신호 데이터에 대해서 특정 주파수 대역을 감쇄시키는 함수 입니다.  
# 아래에서 확인할 수 있는데, 일반 신호를 각각 델타,세타,알파,베타,감마파 외에는 감소시켜서 통과할 수 있도록합니다.  

# In[12]:


import pandas as pd
import numpy as np

segment_len = 1280  # 10초 기준 (128Hz)
fs = 128 # 겹치기 구간
bands = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta':  (13, 30),
    'gamma': (30, 45)
}

data_list = []

for subj_idx, subject in enumerate(dreamer.Data):
    eeg_stimuli = subject.EEG.stimuli
    val_scores = subject.ScoreValence
    aro_scores = subject.ScoreArousal
    dom_scores = subject.ScoreDominance

    for vid_idx, eeg in enumerate(eeg_stimuli):
        val, aro, dom = val_scores[vid_idx], aro_scores[vid_idx], dom_scores[vid_idx]
        total_frames, ch = eeg.shape
        total_segments = total_frames // segment_len

        for seg_idx in range(total_segments):
            seg = eeg[seg_idx * segment_len : (seg_idx + 1) * segment_len, :]  # shape: (1280, 14)

            # Bandpass filtering (example: 각 주파수 대역별 필터 적용)
            band_features = []
            for lowcut, highcut in bands.values():
                filtered_band = np.stack([
                    apply_bandpass_filter(seg[:, c], lowcut, highcut, fs)
                    for c in range(ch)
                ], axis=-1)  # shape: (1280, 14)
                band_features.append(filtered_band)

            band_features = np.stack(band_features, axis=1)  # shape: (1280, 5, 14)

            data_list.append({
                'subject_id': subj_idx,
                'video_id': vid_idx,
                'segment_idx': seg_idx,
                'eeg_band_data': band_features,
                'valence': val,
                'arousal': aro,
                'dominance': dom
            })

df = pd.DataFrame(data_list)


# 이 작업을 하는 이유는, 파이썬 분석할 때 대다수가 DataFrame이라는 파이썬 패키지를 사용합니다.  
# 처음에는 일반 numpy형태의 객체로 되어 있어서 DataFrame이라는 객체로 옮겨 담는 작업을 한 것입니다.  
# 다만, 일반 EEG 시계열은 그대로 옮긴 게 아니고 위에서 정의한 밴드패스필터를 사용애헛 특징을 5개 영역으로 나누어 추가해서 담았습니다.
#   
#   
# 또한, 하나의 eeg 데이터에 대해서 모두 사용하지 않고 세그먼트를 나눠서 진행합니다.  
# 일반적으로 EEG 논문에서도 그러한 기법을 사용합니다.  
# (ex: 5000개 x데이터가 있으면 1000개씩 500개는 겹치도록 9번 나눠서 진행)

# In[13]:


df


# df라는 변수(객체)에 데이터가 깔끔하게 담긴 것을 볼 수 있습니다

# In[14]:


df_remove = df.drop({'subject_id','video_id','segment_idx'},axis=1)
df_remove


# In[15]:


print(df_remove['arousal'].max())
print(df_remove['valence'].max())
print(df_remove['dominance'].max())
print(df_remove['arousal'].min())
print(df_remove['valence'].min())
print(df_remove['dominance'].min())


# In[64]:


threshold = 2.5

df_add_label = df_remove.copy()
df_add_label['y'] = (
    2 * (df_remove['valence'] >= threshold).astype(int) +
        (df_remove['arousal'] >= threshold).astype(int)
)

df_add_label=df_add_label.drop({'arousal','dominance','valence'},axis=1)


# In[65]:


df_add_label


# 위에 지운 필드(subject_id, video_id, segment_idx)는 특징을 담고 있다고 간주하기 어려워 지웠습니다.

# In[96]:


import numpy as np

X = np.stack(df_add_label['eeg_band_data'].values).astype(np.float32)
y = df_add_label['y'].values


# X라벨 y라벨을 따로 분리해줍니다.

# In[97]:


X


# In[98]:


from sklearn.preprocessing import StandardScaler

X_reshaped = X.reshape(-1, 5 * 14)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_reshaped)

# 다시 (8372, 1280, 5, 14)로 reshape
X = X_scaled.reshape(-1, 1280, 5, 14)


# In[99]:


y


# X 데이터를 정규화해주는 과정입니다. AI 모델 학습시킬 때 반드시 해야하는 건데,  
# 그 이유는 생략하겠습니다.

# 8372: 시계열이 나눠진 세그먼트 개수  
# 1280: 10초짜리 시계열 데이터  
# 5: 델타, 세타, 알파, 베타, 감마 특징 개수  
# 14: EEG 채널 개수  

# y는 연속형 변수이기 때문에, 총 4진 분류로 만들기 위해서 V의 중간, A의 중간을 기준으로 나눴습니다.  
# 0: V(0~3) A(0~3)  
# 1: V(0~3) A(3~6)  
# 2: V(3~6) A(0~3)  
# 3: V(3~6) A(3~6)  
# 
# D는 제외했습니다. 이유는 라벨 분리가 많아지면 학습이 더 잘 안될 수도 있고 실제로 논문에서도 VA만 나누는 간단한 실험을 하기 때문입니다.

# In[100]:


import torch
from torch.utils.data import Dataset, DataLoader

class EEGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).permute(0, 3, 2, 1)  # (B, C=14, 5, 1280)
        self.y = torch.tensor(y).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# train/val split
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

train_dataset = EEGDataset(X_train, y_train)
val_dataset = EEGDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False)


# In[101]:


import torch

# 디바이스 선택: CUDA > MPS > CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Using device:", device)


# In[138]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ---------------------------
# Dataset: (B,14,5,1280) -> (B, 70, 1280)
# ---------------------------
class EEGDataset1D(EEGDataset):
    def __init__(self, X, y):
        super().__init__(X, y)  # self.X: (B,14,5,1280), self.y: (B,)
        B, C, BANDS, T = self.X.shape
        self.X = self.X.reshape(B, C*BANDS, T)  # (B, 70, 1280)
        self.n_channels = C*BANDS
        self.T = T

# ---------------------------
# DataLoader
# ---------------------------
train_dataset_1d = EEGDataset1D(X_train, y_train)
val_dataset_1d   = EEGDataset1D(X_val,   y_val)

train_loader = DataLoader(train_dataset_1d, batch_size=1024, shuffle=True,  drop_last=True)
val_loader   = DataLoader(val_dataset_1d,   batch_size=1024, shuffle=False, drop_last=False)

# ---------------------------
# Hyperparams
# ---------------------------
device     = torch.device('mps' if torch.backends.mps.is_available() else
                          'cuda' if torch.cuda.is_available() else 'cpu')
n_classes  = 4
C_out      = train_dataset_1d.n_channels   # 70
T_len      = train_dataset_1d.T            # 1280
z_dim      = 128
emb_dim    = 32
base_ch    = 256
epochs     = 20
lr_G, lr_D = 2e-4, 2e-4       # TTUR 쓰고 싶으면 서로 다르게
n_critic   = 3                # ★ D를 더 자주 업데이트

# (선택) 학습 때 쓰던 채널별 mean/std (길이 70)
train_mean = torch.tensor(scaler.mean_,  dtype=torch.float32, device=device)  # (70,)
train_std  = torch.tensor(scaler.scale_, dtype=torch.float32, device=device)  # (70,)

# ---------------------------
# Generator (1D-DCGAN + class conditioning)
# ---------------------------
class Generator(nn.Module):
    def __init__(self, z_dim, n_classes, emb_dim, C_out, T_len, base_ch=256):
        super().__init__()
        self.class_emb = nn.Embedding(n_classes, emb_dim)
        self.T0 = T_len // 16        # 1280 -> 80
        self.base_ch = base_ch
        in_dim = z_dim + emb_dim

        self.fc = nn.Linear(in_dim, base_ch * self.T0)

        self.up1 = nn.Sequential(
            nn.ConvTranspose1d(base_ch,   base_ch//2, 4, 2, 1),
            nn.BatchNorm1d(base_ch//2), nn.ReLU(True)      # 80->160
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose1d(base_ch//2, base_ch//4, 4, 2, 1),
            nn.BatchNorm1d(base_ch//4), nn.ReLU(True)      # 160->320
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose1d(base_ch//4, base_ch//8, 4, 2, 1),
            nn.BatchNorm1d(base_ch//8), nn.ReLU(True)      # 320->640
        )
        self.up4 = nn.ConvTranspose1d(base_ch//8, C_out, 4, 2, 1)            # 640->1280

        # ★ 마지막 업샘플 레이어 가중치 스케일 업으로 초기화 (분산 확장 유도)
        nn.init.xavier_normal_(self.up4.weight, gain=5.0)
        if self.up4.bias is not None:
            nn.init.zeros_(self.up4.bias)

    def forward(self, z, y):
        yemb = self.class_emb(y)                  # (B, emb_dim)
        h = torch.cat([z, yemb], dim=1)           # (B, z+emb)
        h = self.fc(h)                            # (B, base_ch*T0)
        h = h.view(h.size(0), self.base_ch, self.T0)
        x = self.up1(h)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)                           # (B, C_out, 1280)
        return x

# ---------------------------
# Discriminator (Projection)
# D(x,y) = h(x) + <proj(f(x)), emb(y)>
# ---------------------------
class Discriminator(nn.Module):
    def __init__(self, n_classes, emb_dim, C_in, base_ch=256):
        super().__init__()
        self.class_emb = nn.Embedding(n_classes, emb_dim)

        sn = nn.utils.spectral_norm
        self.feat = nn.Sequential(
            sn(nn.Conv1d(C_in,        base_ch//4, 5, 2, 2)), nn.LeakyReLU(0.2, True),  # 1280->640
            sn(nn.Conv1d(base_ch//4,  base_ch//2, 5, 2, 2)), nn.LeakyReLU(0.2, True),  # 640->320
            sn(nn.Conv1d(base_ch//2,  base_ch,    5, 2, 2)), nn.LeakyReLU(0.2, True),  # 320->160
            sn(nn.Conv1d(base_ch,     base_ch,    5, 2, 2)), nn.LeakyReLU(0.2, True),  # 160->80
        )
        self.lin  = sn(nn.Linear(base_ch, 1))
        self.proj = sn(nn.Linear(base_ch, emb_dim))

    def forward(self, x, y):
        h = self.feat(x)              # (B, base_ch, 80)
        h = h.mean(dim=2)             # (B, base_ch) - GAP
        out = self.lin(h).squeeze(1)  # (B,)
        hy  = self.proj(h)            # (B, emb_dim)
        emb = self.class_emb(y)       # (B, emb_dim)
        return out + (hy * emb).sum(dim=1)

G = Generator(z_dim, n_classes, emb_dim, C_out, T_len, base_ch).to(device)
D = Discriminator(n_classes, emb_dim, C_out, base_ch).to(device)

# ---------------------------
# Losses & Optims
# ---------------------------
def d_hinge_loss(real_scores, fake_scores):
    return F.relu(1. - real_scores).mean() + F.relu(1. + fake_scores).mean()

def g_hinge_loss(fake_scores):
    return (-fake_scores).mean()

# ★ moment loss: 채널별 mean/std를 real에 맞추도록 유도
def moment_loss(x_fake, x_real, eps=1e-6):
    mu_f,  std_f  = x_fake.mean(dim=(0,2)), x_fake.std(dim=(0,2)) + eps
    mu_r,  std_r  = x_real.mean(dim=(0,2)), x_real.std(dim=(0,2)) + eps
    return (mu_f - mu_r).abs().mean() + (std_f - std_r).abs().mean()

lambda_mom = 1.0  # 영향 키우기

opt_G = torch.optim.Adam(G.parameters(), lr=lr_G, betas=(0.0, 0.9))
opt_D = torch.optim.Adam(D.parameters(), lr=lr_D, betas=(0.0, 0.9))

# ---------------------------
# Training loop (with n_critic)
# ---------------------------
global_step = 0
for epoch in range(1, epochs+1):
    G.train(); D.train()
    for Xb, yb in train_loader:
        Xb = Xb.to(device)   # (B,70,1280) -- 표준화 공간에서 학습 권장
        yb = yb.to(device)
        B  = Xb.size(0)

        # ----- D steps (n_critic) -----
        for _ in range(n_critic):
            z = torch.randn(B, z_dim, device=device)
            with torch.no_grad():
                X_fake = G(z, yb)
            D_real = D(Xb, yb)
            D_fake = D(X_fake, yb)
            loss_D = d_hinge_loss(D_real, D_fake)

            opt_D.zero_grad(set_to_none=True)
            loss_D.backward()
            opt_D.step()
            global_step += 1

        # ----- G step -----
        z = torch.randn(B, z_dim, device=device)
        X_fake = G(z, yb)
        D_fake = D(X_fake, yb)

        loss_G = g_hinge_loss(D_fake) + lambda_mom * moment_loss(X_fake, Xb)

        opt_G.zero_grad(set_to_none=True)
        loss_G.backward()
        opt_G.step()

    # ----- 모니터링 -----
    G.eval()
    with torch.no_grad():
        z = torch.randn(256, z_dim, device=device)
        y = torch.randint(0, n_classes, (256,), device=device)
        Xf = G(z, y)  # (256,70,1280)

        # 생성물 자체 통계(표준화 전제 공간)
        mean = Xf.mean().item()
        chstd = Xf.std(dim=(0,2)).mean().item()

        # 분류기로 hit-rate 확인(학습 시 전처리와 동일)
        Xf_in = (Xf - train_mean.view(1,-1,1)) / (train_std.view(1,-1,1) + 1e-6)
        try:
            logits = model(Xf_in)  # (B,num_classes)
        except:
            logits = model(Xf_in.view(Xf_in.size(0), 14, 5, 1280))
        hit = (logits.argmax(1) == y).float().mean().item()

    print(f"[ep {epoch:02d}] G mean={mean:.3f} ch_std={chstd:.3f} hit={hit:.2f}  "
          f"lossD={loss_D.item():.3f} lossG={loss_G.item():.3f}")


# In[162]:


# import os

# save_dir = "./experiments"
# os.makedirs(save_dir, exist_ok=True)

# # 모델 저장
# torch.save(G.state_dict(), os.path.join(save_dir, "250810_EEG_gene_G.pth"))
# torch.save(D.state_dict(), os.path.join(save_dir, "250810_EEG_gene_D.pth"))

# # 하이퍼파라미터 저장
# hyperparams = {
#     "n_classes": n_classes,
#     "C_out": C_out,
#     "T_len": T_len,
#     "z_dim": z_dim,
#     "emb_dim": emb_dim,
#     "base_ch": base_ch,
#     "epochs": epochs,
#     "lr_G": lr_G,
#     "lr_D": lr_D,
#     "n_critic": n_critic,
#     "lambda_mom": lambda_mom
# }
# torch.save(hyperparams, os.path.join(save_dir, "250810_EEG_GAN_hyperparams.pth"))


# In[58]:


# 학습이 끝난 직후(혹은 에폭마다) 평가
G.eval()
with torch.no_grad():
    z = torch.randn(256, z_dim, device=device)
    y = torch.randint(0, n_classes, (256,), device=device)
    Xf = G(z, y)

print("fake mean:", Xf.mean().item(),
      "ch_std_mean:", Xf.std(dim=(0,2)).mean().item())  # z-score면 ≈0, ≈1 기대

# 조건 반영률(각 라벨로 생성 → 분류기 예측이 같은 라벨인지)
from collections import Counter
def hit_rate_per_class(model, G, per_class=128):
    model.eval(); G.eval()
    res = {}
    with torch.no_grad():
        for c in range(n_classes):
            y = torch.full((per_class,), c, device=device, dtype=torch.long)
            z = torch.randn(per_class, z_dim, device=device)
            Xf = G(z, y)
            try: logit = model(Xf)
            except: logit = model(Xf.view(Xf.size(0),14,5,1280))
            pred = logit.argmax(1).detach().cpu().tolist()
            cnt = Counter(pred)
            res[c] = {"hit": cnt.get(c,0)/per_class, "dist": dict(cnt)}
    return res

print(hit_rate_per_class(model, G))


# In[163]:


import torch
import torch.nn as nn
import torch.nn.functional as F

# ===== cVAE =====
class Encoder(nn.Module):
    def __init__(self, C_in, z_dim=128, base_ch=256):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Conv1d(C_in,        base_ch//4, 5, 2, 2), nn.LeakyReLU(0.2, True),  # 1280->640
            nn.Conv1d(base_ch//4,  base_ch//2, 5, 2, 2), nn.LeakyReLU(0.2, True),  # 640->320
            nn.Conv1d(base_ch//2,  base_ch,    5, 2, 2), nn.LeakyReLU(0.2, True),  # 320->160
            nn.Conv1d(base_ch,     base_ch,    5, 2, 2), nn.LeakyReLU(0.2, True),  # 160->80
        )
        self.out = nn.Linear(base_ch, 2*z_dim)  # (mu, logvar)

    def forward(self, x):               # x: (B,70,1280)
        h = self.feat(x)                # (B, base_ch, 80)
        h = h.mean(dim=2)               # (B, base_ch)
        mu_logvar = self.out(h)         # (B, 2*z_dim)
        mu, logvar = mu_logvar.chunk(2, dim=1)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, z_dim, n_classes, emb_dim, C_out, T_len, base_ch=256):
        super().__init__()
        self.class_emb = nn.Embedding(n_classes, emb_dim)
        self.T0 = T_len // 16           # 80
        self.base_ch = base_ch
        in_dim = z_dim + emb_dim

        self.fc = nn.Linear(in_dim, base_ch * self.T0)
        self.net = nn.Sequential(
            nn.ConvTranspose1d(base_ch,     base_ch//2, 4, 2, 1), nn.BatchNorm1d(base_ch//2), nn.ReLU(True), # 80->160
            nn.ConvTranspose1d(base_ch//2,  base_ch//4, 4, 2, 1), nn.BatchNorm1d(base_ch//4), nn.ReLU(True), # 160->320
            nn.ConvTranspose1d(base_ch//4,  base_ch//8, 4, 2, 1), nn.BatchNorm1d(base_ch//8), nn.ReLU(True), # 320->640
            nn.ConvTranspose1d(base_ch//8,  C_out,      4, 2, 1), # 640->1280
            # z-score 데이터라 출력 제한 X → activation 없음
        )

    def forward(self, z, y):
        yemb = self.class_emb(y)                 # (B, emb_dim)
        h = torch.cat([z, yemb], dim=1)          # (B, z+emb)
        h = self.fc(h).view(-1, self.base_ch, self.T0)  # (B, base_ch, 80)
        x_hat = self.net(h)                      # (B, C_out, 1280)
        return x_hat

class CVAE(nn.Module):
    def __init__(self, C_in, n_classes, emb_dim, z_dim, T_len, base_ch=256):
        super().__init__()
        self.enc = Encoder(C_in, z_dim, base_ch)
        self.dec = Decoder(z_dim, n_classes, emb_dim, C_in, T_len, base_ch)

    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, y):
        mu, logvar = self.enc(x)
        z = self.reparam(mu, logvar)
        x_hat = self.dec(z, y)
        return x_hat, mu, logvar

def vae_loss(x, x_hat, mu, logvar, beta=1.0):
    # 재구성: MSE (z-score 데이터에 안정적)
    recon = F.mse_loss(x_hat, x, reduction='mean')
    # KL: D_KL(q(z|x)||N(0,1)) = -0.5 * sum(1 + logσ^2 - μ^2 - σ^2)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + beta*kl, recon, kl


z_dim   = 128
emb_dim = 32
beta    = 1.0  # β-VAE 조절 가능 (0.5~4 사이 튜닝 추천)

VAE = CVAE(C_out, n_classes, emb_dim, z_dim, T_len, base_ch).to(device)
opt = torch.optim.Adam(VAE.parameters(), lr=1e-3)

epochs = 10
for epoch in range(1, epochs+1):
    VAE.train()
    loss_all = recon_all = kl_all = 0.0
    for Xb, yb in train_loader:                 # Xb: (B,70,1280)
        Xb = Xb.to(device)
        yb = yb.to(device)

        x_hat, mu, logvar = VAE(Xb, yb)
        loss, recon, kl = vae_loss(Xb, x_hat, mu, logvar, beta=beta)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        loss_all  += loss.item()  * Xb.size(0)
        recon_all += recon.item() * Xb.size(0)
        kl_all    += kl.item()    * Xb.size(0)

    N = len(train_dataset_1d)
    print(f"[{epoch}/{epochs}] loss={loss_all/N:.4f}  recon={recon_all/N:.4f}  kl={kl_all/N:.4f}")


# torch라는 AI 딥러닝 학습할 때 사용하기 좋은 파이썬 패키지입니다.  
# 지금 여기선 데이터를 torch 객체로 변환하는 작업이라 생각하면 됩니다.

# In[72]:


import torch
import torch.nn as nn
import torch.nn.functional as F

class EEGCNN_GRU(nn.Module):
    def __init__(self):
        super(EEGCNN_GRU, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=14*5, out_channels=128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(128)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(256)
        self.pool2 = nn.MaxPool1d(2)

        self.gru = nn.GRU(input_size=256, hidden_size=128, batch_first=True, bidirectional=True)

        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 2, 64) 
        self.fc2 = nn.Linear(64, 4) 

    def forward(self, x):
        x = x.view(x.size(0), 14*5, 1280)  
        x = self.pool1(F.relu(self.bn1(self.conv1(x)))) 
        x = self.pool2(F.relu(self.bn2(self.conv2(x)))) 

        x = x.permute(0, 2, 1)
        out, _ = self.gru(x)  
        out = out[:, -1, :] 

        out = self.dropout(F.relu(self.fc1(out)))
        return self.fc2(out)


# AI 모델인데, 논문에 나온 모델이랑 같도록 테스트하기 위해 만들었습니다.

# In[73]:


model = EEGCNN_GRU().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(50):
    model.train()
    total_loss = 0
    correct = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        loss = criterion(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (pred.argmax(1) == yb).sum().item()

    acc = correct / len(train_dataset)
    print(f"[Epoch {epoch+1}] Loss: {total_loss:.4f}, Train Acc: {acc:.4f}")

    # validation
    model.eval()
    correct = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            correct += (pred.argmax(1) == yb).sum().item()
    val_acc = correct / len(val_dataset)
    print(f"           Validation Acc: {val_acc:.4f}")


# In[74]:


# torch.save({
#     "epoch": epoch,
#     "model_state_dict": model.state_dict(),
#     "optimizer_state_dict": optimizer.state_dict(),
#     "loss": loss
# }, "experiments/250810_EEG_classifier_48.pth")


# In[76]:


import torch

ckpt = torch.load("experiments/250810_EEG_classifier_48.pth", map_location="cpu")
print(type(ckpt))

# state_dict일 경우
if isinstance(ckpt, dict):
    for k, v in ckpt.items():
        print(k, v.shape if hasattr(v, "shape") else type(v))

for k, v in ckpt["model_state_dict"].items():
    print(k, v.shape)


# In[30]:


ckpt = torch.load("experiments/250810_EEG_classifier_55.pth", map_location="cpu")
model = EEGCNN_GRU().to(device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()


# 학습 과정인데, 정확도 상승에 진전이 없어 그만 두었습니다.  
# 지금 모델이 4진분류인데 정확도가 33퍼면 그냥 4개중 랜덤으로 하나 골라도 맞추는 수준이라서 유의미한 결과는 아니라고 보면 됩니다.  
#   
# 실제로 논문들도 다중분류를 하지 않고 이진 분류로 여러 개 대조하는 방식으로 많이 하기 때문에 실망할 수도 있습니다.  
# 그만큼 EEG로 감정을 분류하는 작업이 어렵다는 뜻이겠죠...  
# 제가 그래서 제안한 방식이 일단 2진분류를 여러 개 해보고 정확도를 높이는 방식을 제안해드리고 싶습니다.

# In[161]:


import torch
import numpy as np

G.eval(); model.eval()
num_samples=2000
with torch.no_grad():
    # 조건 라벨과 노이즈 생성
    y_fake_t = torch.randint(0, 4, (num_samples,), device=device)         # (B,)
    z_sample = torch.randn(num_samples, z_dim, device=device)             # (B, z_dim)

    # G로 생성
    X_fake_t = G(z_sample, y_fake_t).to(torch.float32)                    # (B,70,1280)


    # 분류기 추론
    logits = model(X_fake_t)                                                  # (B,num_classes)
    preds = torch.argmax(logits, dim=1)
    acc = (preds == y_fake_t).float().mean().item()


# 리포트
print(f"[SCALED] accuracy on {num_samples} samples: {acc*100:.2f}%")
print("preds:", preds.tolist())
print("gts  :", y_fake_t.tolist())

# 체크: 표준화 후 통계 (이상적이면 채널별 mean≈0, std≈1 근처)
with torch.no_grad():
    m = X_fake_t.mean(dim=(0,2))
    s = X_fake_t.std(dim=(0,2))
print("post-scale mean (first 8):", m[:8].tolist())
print("post-scale std  (first 8):", s[:8].tolist())


# In[ ]:





# In[114]:


import torch

# 학습 때 사용한 scaler 평균/표준편차 (예: 저장해둔 numpy → torch)
train_mean = torch.tensor(scaler.mean_, dtype=torch.float32, device=device)
train_std  = torch.tensor(scaler.scale_, dtype=torch.float32, device=device)
# 생성 데이터 통계
gen_mean = X_fake_t.mean(dim=(0, 2))   # 채널별 평균 (70,)
gen_std  = X_fake_t.std(dim=(0, 2))    # 채널별 표준편차 (70,)

# 평균/표준편차 비교
diff_mean = (gen_mean - train_mean).cpu().numpy()
diff_std  = (gen_std - train_std).cpu().numpy()

print("=== Mean Comparison (gen vs train) ===")
for i in range(len(train_mean)):
    print(f"ch{i:02d} : gen={gen_mean[i]:.4f} | train={train_mean[i]:.4f} | diff={diff_mean[i]:.4f}")

print("\n=== Std Comparison (gen vs train) ===")
for i in range(len(train_std)):
    print(f"ch{i:02d} : gen={gen_std[i]:.4f} | train={train_std[i]:.4f} | diff={diff_std[i]:.4f}")


# In[116]:


# scaler 사용해서 inverse_transform 후 다시 standardize
# 또는 단순 비율 조정
X_fake_scaled = X_fake_t * train_std.view(1, -1, 1) + train_mean.view(1, -1, 1)

# 그리고 학습 때 쓰던 scaler로 transform
X_fake_processed = (X_fake_scaled - train_mean.view(1, -1, 1)) / train_std.view(1, -1, 1)

logits = model(X_fake_processed)
preds = logits.argmax(dim=1)
acc = (preds == y_fake_t).float().mean().item()
print(acc)


# In[91]:


# A) 실데이터로 sanity check: 모델이 정상인지
model.eval()
Xr, yr = next(iter(val_loader))
with torch.no_grad():
    try: logits_r = model(Xr.to(device))
    except: logits_r = model(Xr.to(device).view(Xr.size(0),14,5,1280))
acc_r = (logits_r.argmax(1).cpu()==yr).float().mean().item()
print("val batch acc:", acc_r)


# In[89]:


import torch
import numpy as np

# --- 0) df_gen -> 텐서 변환 ---
X_fake_np = np.array(df_gen["generated_data"].tolist(), dtype=np.float32)  # (B,70,1280)
y_fake_np = df_gen["label"].to_numpy()

X_fake_t = torch.from_numpy(X_fake_np).to(device)          # (B,70,1280)
y_fake_t = torch.from_numpy(y_fake_np).long().to(device)   # (B,)

# --- 1) 분포 확인 (mean/std) ---
with torch.no_grad():
    mean_all = X_fake_t.mean().item()
    std_all  = X_fake_t.std().item()
    std_ch_mean = X_fake_t.std(dim=(0,2)).mean().item()  # 채널별 표준편차의 평균
print(f"[FAKE] mean={mean_all:.4f}  std_all={std_all:.4f}  ch_std_mean={std_ch_mean:.4f}")

# --- 2) (선택) 간단 보정: train 배치 통계로 채널별 mean/std 맞추기 ---
#     * 만약 1)의 std가 너무 작거나(예: ~0.4) 크면 보정 후 성능 변화를 보자.
Xr, _ = next(iter(train_loader))    # (B,70,1280) 이미 EEGDataset1D가 reshape 했으므로 바로 사용 가능
Xr = Xr.to(device)

with torch.no_grad():
    mu_r  = Xr.mean(dim=(0,2), keepdim=True)                # (1,70,1)
    std_r = Xr.std(dim=(0,2), keepdim=True) + 1e-6          # (1,70,1)

    mu_f  = X_fake_t.mean(dim=(0,2), keepdim=True)
    std_f = X_fake_t.std(dim=(0,2), keepdim=True) + 1e-6

    X_fake_cal = (X_fake_t - mu_f) / std_f * std_r + mu_r   # 채널별 캘리브레이션

# --- 3) 분류 정확도 측정 (원본/보정 둘 다) ---
model.eval()

def run_model(inp):
    with torch.no_grad():
        try:
            logits = model(inp)  # (B,70,1280) 기대
        except:
            B = inp.size(0)
            logits = model(inp.view(B,14,5,1280))  # (B,14,5,1280) 기대일 때
        return logits.argmax(1)

preds_raw = run_model(X_fake_t)
acc_raw   = (preds_raw == y_fake_t).float().mean().item()

preds_cal = run_model(X_fake_cal)
acc_cal   = (preds_cal == y_fake_t).float().mean().item()

print(f"Acc (raw fake): {acc_raw*100:.2f}%")
print(f"Acc (calibrated fake): {acc_cal*100:.2f}%")
print("preds(raw):", preds_raw.tolist())
print("preds(cal):", preds_cal.tolist())
print("gts      :", y_fake_t.tolist())


# In[87]:


# real 분포(한 배치)와 fake 분포 비교
Xr, _ = next(iter(train_loader))  # (B,70,1280)
Xr = Xr.to(device); Xf = X_fake_t  # transform 적용 전의 텐서

print("real ch_std:", Xr.std(dim=(0,2)).mean().item())
print("fake ch_std:", Xf.std(dim=(0,2)).mean().item())


# In[180]:


import torch
import numpy as np

def eval_on_generated(model, generator_or_vae, n_classes, z_dim=128, num_samples=64, device="cuda"):
    """
    - generator_or_vae: GAN의 G 또는 cVAE (dec가 있는 객체)
    - model: 학습된 분류기
    """
    # 1) 생성기 호출 통일: fn(y, z) -> X_fake (B, 70, 1280)
    def generate(y, z):
        if hasattr(generator_or_vae, "dec"):   # cVAE
            return generator_or_vae.dec(z, y)
        else:                                  # GAN(G)
            return generator_or_vae(z, y)

    # 2) 라벨/잠재 벡터 샘플
    y_sample = torch.randint(0, n_classes, (num_samples,), device=device)
    z_sample = torch.randn(num_samples, z_dim, device=device)

    # 3) 생성
    generator_or_vae.eval()
    with torch.no_grad():
        X_fake = generate(y_sample, z_sample)          # (B, 70, 1280)
        if isinstance(X_fake, tuple):                  # 혹시 부가값 반환하면 첫번째만
            X_fake = X_fake[0]
        X_fake = X_fake.detach()

    # 4) 분류기 입력 형태 맞춰 추론
    model.eval()
    with torch.no_grad():
        try:
            logits = model(X_fake)  # (B,70,1280)를 기대하는 모델
        except Exception:
            # (B,14,5,1280) 형태를 기대하는 모델
            B = X_fake.size(0)
            X_fake_b = X_fake.view(B, 14, 5, 1280)
            try:
                logits = model(X_fake_b)
            except Exception:
                # 일부 2D-CNN이 (B,1,70,1280) 같은 채널 차원을 기대할 수도 있음
                logits = model(X_fake.unsqueeze(1))

        preds = torch.argmax(logits, dim=1)

    acc = (preds == y_sample).float().mean().item()
    return acc, preds.detach().cpu().tolist(), y_sample.detach().cpu().tolist()

# ===== 사용 예시 =====
# GAN인 경우: generator_or_vae = G
# cVAE인 경우: generator_or_vae = VAE
acc, preds, gts = eval_on_generated(model, generator_or_vae=G, n_classes=4, z_dim=128, num_samples=2000, device=device)
print(f"Generated → model accuracy: {acc*100:.2f}%")
print("preds:", preds)
print("gts  :", gts)


# In[ ]:





# In[ ]:




