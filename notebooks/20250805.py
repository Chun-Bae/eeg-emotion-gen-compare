#!/usr/bin/env python
# coding: utf-8

# In[13]:


import scipy.io

mat = scipy.io.loadmat('./data/DREAMER.mat', struct_as_record=False, squeeze_me=True)
print(mat.keys())


# EEG 데이터를 담고 있는 mat 파일의 객체를 확인하기 위한 과정입니다.  
# 'DREAMER' 외에는 메타데이터라 무시해도 좋습니다.

# In[14]:


dreamer = mat['DREAMER']
print("DREAMER/")
for key in dreamer.__dict__.keys():
    print("-", key)


# DREAMER 객체에 또 다시 어떤 객체들이 있는지 확인하는 과정입니다.  
# 실제 EEG 데이터는 'Data' 객체에 들어있구요. 나머지는 크게 중요한 것이 아닙니다.

# In[15]:


eeg_sampling_rate = dreamer.EEG_SamplingRate 
ecg_sampling_rate = dreamer.ECG_SamplingRate
eeg_electrodes = dreamer.EEG_Electrodes

print(f"EEG sampling rate: {eeg_sampling_rate} Hz")
print(f"ECG sampling rate: {ecg_sampling_rate} Hz")

print("EEG electrode names:", end=' ')
print(" | ".join(eeg_electrodes))


# EEG는 128Hz(1초에 128프레임 캡처)로 데이터가 담아져있음을 보여주는 정보입니다.  
# 그 외 EEG의 어떤 전극을 사용한지 보여줍니다. (논문에 이미 나와있긴함)

# In[16]:


data = dreamer.Data
print(data.shape)
subject0 = data[0]
print(subject0)


# 데이터에 크기를 확인하기 위한 코드입니다. 피험자가 23명이기 때문에 23라는 숫자가 출력되고 있습니다.  
# 그 아래 출력은 객체 고유 id 같은 거니 무시해도 됩니다.

# In[17]:


subject0 = dreamer.Data[0]
print("subject0/")
for key in subject0.__dict__.keys():
    print("-", key)


# 또 다시 피험자0 객체에는 어떤 데이터들이 들어있는 지 확인하는 과정입니다.  
# 나이, 성별, EEG data, ECG data, Valence(불쾌), Arousal(흥분), Dominance(통제력)의 정도를 나타냅니다.  
# VA 또는 VAD 모델이 존재하는데 이 수치를 기준으로 인간의 감정을 벡터로 표현할 수 있습니다.  
# 여기서 VAD가 정답 라벨(결과값)이라고 생각할 수 있습니다.

# In[18]:


eeg_stimuli = subject0.EEG.stimuli

for i, stimulus in enumerate(eeg_stimuli):
    time_frames = stimulus.shape[0]
    num_channels = stimulus.shape[1]
    duration_sec = int(time_frames / 128)

    print(f"[영상 {i:02d}] Time frames: {time_frames:5d}개 | Channels: {num_channels}개 | Duration: {duration_sec:3d}초")


# 피험자 별로 각각 영상 18개의 시청에 대한 EEG, ECG 데이터가 있고, 각각의 clip마다 영상의 길이가 다릅니다.  
# 좀 더 쉽게 설명하자면 각 영상에 대해 x(128 x 영상길이)에 대한 y(진폭 == EEG 신호) 가 존재합니다.

# In[19]:


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

# In[20]:


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

# In[21]:


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

# In[31]:


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

# In[32]:


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

# In[35]:


df


# df라는 변수(객체)에 데이터가 깔끔하게 담긴 것을 볼 수 있습니다

# In[43]:


df_remove = df.drop({'subject_id','video_id','segment_idx'},axis=1)
df_remove


# 위에 지운 필드(subject_id, video_id, segment_idx)는 특징을 담고 있다고 간주하기 어려워 지웠습니다.

# In[44]:


import numpy as np

X = np.stack(df_remove['eeg_band_data'].values).astype(np.float32)

y_valence = df_remove['valence'].values
y_arousal = df_remove['arousal'].values
# y_dominance = df_remove['dominance'].values


# X라벨 y라벨을 따로 분리해줍니다.

# In[57]:


from sklearn.preprocessing import StandardScaler

X_reshaped = X.reshape(-1, 5 * 14)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_reshaped)

# 다시 (8372, 1280, 5, 14)로 reshape
X_scaled = X_scaled.reshape(-1, 1280, 5, 14)


# X 데이터를 정규화해주는 과정입니다. AI 모델 학습시킬 때 반드시 해야하는 건데,  
# 그 이유는 생략하겠습니다.

# In[58]:


X_scaled.shape


# 8372: 시계열이 나눠진 세그먼트 개수  
# 1280: 10초짜리 시계열 데이터  
# 5: 델타, 세타, 알파, 베타, 감마 특징 개수  
# 14: EEG 채널 개수  

# In[59]:


y_combined = (
    (y_valence > 3).astype(int) * 2 +
    (y_arousal > 3).astype(int)       
)
y_combined = np.array(y_combined).astype(np.int64)


# y는 연속형 변수이기 때문에, 총 4진 분류로 만들기 위해서 V의 중간, A의 중간을 기준으로 나눴습니다.  
# 0: V(0~3) A(0~3)  
# 1: V(0~3) A(3~6)  
# 2: V(3~6) A(0~3)  
# 3: V(3~6) A(3~6)  
# 
# D는 제외했습니다. 이유는 라벨 분리가 많아지면 학습이 더 잘 안될 수도 있고 실제로 논문에서도 VA만 나누는 간단한 실험을 하기 때문입니다.

# In[60]:


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
X_train, X_val, y_train, y_val = train_test_split(X, y_combined, test_size=0.2, random_state=42, stratify=y_combined)

train_dataset = EEGDataset(X_train, y_train)
val_dataset = EEGDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)


# torch라는 AI 딥러닝 학습할 때 사용하기 좋은 파이썬 패키지입니다.  
# 지금 여기선 데이터를 torch 객체로 변환하는 작업이라 생각하면 됩니다.

# In[64]:


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

# In[65]:


device = torch.device('mps' if torch.mps.is_available() else 'cpu')
model = EEGCNN().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(1000):
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


# 학습 과정인데, 정확도 상승에 진전이 없어 그만 두었습니다.  
# 지금 모델이 4진분류인데 정확도가 33퍼면 그냥 4개중 랜덤으로 하나 골라도 맞추는 수준이라서 유의미한 결과는 아니라고 보면 됩니다.  
#   
# 실제로 논문들도 다중분류를 하지 않고 이진 분류로 여러 개 대조하는 방식으로 많이 하기 때문에 실망할 수도 있습니다.  
# 그만큼 EEG로 감정을 분류하는 작업이 어렵다는 뜻이겠죠...  
# 제가 그래서 제안한 방식이 일단 2진분류를 여러 개 해보고 정확도를 높이는 방식을 제안해드리고 싶습니다.

# In[ ]:




