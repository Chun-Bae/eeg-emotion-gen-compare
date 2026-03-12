#!/usr/bin/env python
# coding: utf-8

# In[13]:


import scipy.io

mat = scipy.io.loadmat('./data/DREAMER.mat', struct_as_record=False, squeeze_me=True)
print(mat.keys())


# In[14]:


dreamer = mat['DREAMER']
print("DREAMER/")
for key in dreamer.__dict__.keys():
    print("-", key)


# In[15]:


eeg_sampling_rate = dreamer.EEG_SamplingRate 
ecg_sampling_rate = dreamer.ECG_SamplingRate
eeg_electrodes = dreamer.EEG_Electrodes

print(f"EEG sampling rate: {eeg_sampling_rate} Hz")
print(f"ECG sampling rate: {ecg_sampling_rate} Hz")

print("EEG electrode names:", end=' ')
print(" | ".join(eeg_electrodes))


# In[16]:


data = dreamer.Data
print(data.shape)
subject0 = data[0]
print(subject0)


# In[17]:


subject0 = dreamer.Data[0]
print("subject0/")
for key in subject0.__dict__.keys():
    print("-", key)


# In[18]:


eeg_stimuli = subject0.EEG.stimuli

for i, stimulus in enumerate(eeg_stimuli):
    time_frames = stimulus.shape[0]
    num_channels = stimulus.shape[1]
    duration_sec = int(time_frames / 128)

    print(f"[영상 {i:02d}] Time frames: {time_frames:5d}개 | Channels: {num_channels}개 | Duration: {duration_sec:3d}초")


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


# In[20]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Subject 0, Video 0
subject0 = dreamer.Data[0]
eeg = subject0.EEG.stimuli[0]
ecg = subject0.ECG.stimuli[0]
eeg_electrodes = dreamer.EEG_Electrodes  # EEG 채널 이름

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

plt.tight_layout(rect=[0, 0, 1, 0.96])  # leave space for legend
plt.show()


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


# In[22]:


from scipy.signal import iirnotch, filtfilt

def apply_notch_filter(signal, fs=128, notch_freq=50.0, Q=30.0):
    """
    50Hz 노치 필터를 적용하여 전력 잡음 제거
    입력: (samples,) 또는 (samples, channels)
    출력: 동일 shape의 필터링된 신호
    """
    b, a = iirnotch(w0=notch_freq, Q=Q, fs=fs)

    if signal.ndim == 1:
        return filtfilt(b, a, signal)
    else:
        return np.stack([filtfilt(b, a, signal[:, ch]) for ch in range(signal.shape[1])], axis=1)


# In[23]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.signal import iirnotch, filtfilt

# Notch 필터 함수 정의
def apply_notch_filter(signal, fs=128, notch_freq=50.0, Q=30.0):
    b, a = iirnotch(w0=notch_freq, Q=Q, fs=fs)
    if signal.ndim == 1:
        return filtfilt(b, a, signal)
    else:
        return np.stack([filtfilt(b, a, signal[:, ch]) for ch in range(signal.shape[1])], axis=1)

# DREAMER 데이터 불러오기
subject0 = dreamer.Data[0]
eeg = subject0.EEG.stimuli[0]
ecg = subject0.ECG.stimuli[0]
eeg_electrodes = dreamer.EEG_Electrodes

# 길이 맞추기
min_len = min(len(eeg), len(ecg))
eeg = eeg[:min_len]
ecg = ecg[:min_len]

# 🔧 Notch 필터 적용
eeg_filtered = apply_notch_filter(eeg, fs=128, notch_freq=50.0, Q=30)
ecg_filtered = apply_notch_filter(ecg, fs=256, notch_freq=50.0, Q=30)

# 시각화 설정
frames_per_sec = 128
num_segments = 3
eeg_colors = sns.color_palette("Blues", n_colors=eeg.shape[1])
ecg_colors = sns.color_palette("Reds", n_colors=ecg.shape[1] if ecg.ndim > 1 else 1)

sns.set(style="whitegrid", font_scale=0.9)
fig, axes = plt.subplots(2, num_segments, figsize=(18, 8), sharey='row', sharex=False)
eeg_handles, eeg_labels = [], []
ecg_handles, ecg_labels = [], []

for seg in range(num_segments):
    start = seg * frames_per_sec
    end = start + frames_per_sec
    x = np.arange(frames_per_sec)

    # EEG subplot
    ax_eeg = axes[0, seg]
    for ch in range(eeg.shape[1]):
        sns.lineplot(x=x, y=eeg_filtered[start:end, ch], ax=ax_eeg, color=eeg_colors[ch], label=None)
        if seg == 0:
            line_obj = ax_eeg.lines[-1]
            eeg_handles.append(line_obj)
            eeg_labels.append(f"EEG {eeg_electrodes[ch]}")
    ax_eeg.set_title(f"EEG (Filtered): {seg+1}s (frame {start}–{end})")
    ax_eeg.set_xlabel("Frame")
    ax_eeg.set_ylabel("Amplitude" if seg == 0 else "")

    # ECG subplot
    ax_ecg = axes[1, seg]
    if ecg.ndim == 1:
        sns.lineplot(x=x, y=ecg_filtered[start:end], ax=ax_ecg, color=ecg_colors[0], label=None)
        if seg == 0:
            line_obj = ax_ecg.lines[-1]
            ecg_handles.append(line_obj)
            ecg_labels.append("ECG ch1")
    else:
        for ch in range(ecg.shape[1]):
            sns.lineplot(x=x, y=ecg_filtered[start:end, ch], ax=ax_ecg, color=ecg_colors[ch], label=None)
            if seg == 0:
                line_obj = ax_ecg.lines[-1]
                ecg_handles.append(line_obj)
                ecg_labels.append(f"ECG ch{ch+1}")
    ax_ecg.set_title(f"ECG (Filtered): {seg+1}s (frame {start}–{end})")
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


# In[24]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.signal import iirnotch, filtfilt

# 🔧 노치 필터 함수 정의
def apply_notch_filter(signal, fs=128, notch_freq=50.0, Q=30.0):
    b, a = iirnotch(w0=notch_freq, Q=Q, fs=fs)
    return filtfilt(b, a, signal)

# 📥 Sample EEG 정보 (피험자 0 - 영상 0)
subject0 = dreamer.Data[0]
eeg = subject0.EEG.stimuli[0]
eeg_names = dreamer.EEG_Electrodes
fs = dreamer.EEG_SamplingRate  # 128Hz

# 🎯 1초 구간 (128 프레임)
duration_sec = 1
n_samples = fs * duration_sec
ch = 0  # 예시 채널: EEG 0 (ex. Fp1)
eeg_segment = eeg[:n_samples, ch]
eeg_segment = eeg_segment - np.mean(eeg_segment)  # DC 제거

# ✅ 노치 필터 적용
eeg_filtered = apply_notch_filter(eeg_segment, fs=fs)

# 🎧 주파수 변환 (원본)
fft_orig = np.fft.fft(eeg_segment)
freq = np.fft.fftfreq(n_samples, d=1/fs)
fft_mag_orig = np.abs(fft_orig)[:n_samples // 2]
fft_freq = freq[:n_samples // 2]

# 🎧 주파수 변환 (필터 적용 후)
fft_filt = np.fft.fft(eeg_filtered)
fft_mag_filt = np.abs(fft_filt)[:n_samples // 2]

# 🎨 시각화
sns.set(style="whitegrid", font_scale=1.1)
fig, axs = plt.subplots(2, 2, figsize=(14, 6))

# 시간 영역 - 원본
axs[0, 0].plot(np.arange(n_samples) / fs, eeg_segment, color='blue')
axs[0, 0].set_title(f"Time Domain (Original) - {eeg_names[ch]}")
axs[0, 0].set_xlabel("Time (sec)")
axs[0, 0].set_ylabel("Amplitude")

# 시간 영역 - 필터 적용
axs[0, 1].plot(np.arange(n_samples) / fs, eeg_filtered, color='green')
axs[0, 1].set_title(f"Time Domain (Filtered) - {eeg_names[ch]}")
axs[0, 1].set_xlabel("Time (sec)")
axs[0, 1].set_ylabel("Amplitude")

# 주파수 영역 - 원본
axs[1, 0].plot(fft_freq, fft_mag_orig, color='red')
axs[1, 0].set_title("Frequency Domain (Original)")
axs[1, 0].set_xlabel("Frequency (Hz)")
axs[1, 0].set_ylabel("Magnitude")
axs[1, 0].set_xlim(0, 64)
axs[1, 0].axvline(50, color='gray', linestyle='--', alpha=0.5)

# 주파수 영역 - 필터 적용
axs[1, 1].plot(fft_freq, fft_mag_filt, color='purple')
axs[1, 1].set_title("Frequency Domain (Filtered)")
axs[1, 1].set_xlabel("Frequency (Hz)")
axs[1, 1].set_ylabel("Magnitude")
axs[1, 1].set_xlim(0, 64)
axs[1, 1].axvline(50, color='gray', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()


# In[25]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# 🎛️ Butterworth Bandpass 필터 함수
def apply_bandpass_filter(signal, fs=128, lowcut=0.5, highcut=45.0, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(N=order, Wn=[low, high], btype='band')
    return filtfilt(b, a, signal)

# 📥 Sample EEG 정보 (피험자 0 - 영상 0)
subject0 = dreamer.Data[0]
eeg = subject0.EEG.stimuli[0]
eeg_names = dreamer.EEG_Electrodes
fs = dreamer.EEG_SamplingRate  # 128Hz

# 🎯 1초 구간 (128 프레임)
duration_sec = 1
n_samples = fs * duration_sec
ch = 0
eeg_segment = eeg[:n_samples, ch]
eeg_segment = eeg_segment - np.mean(eeg_segment)  # DC 제거

# ✅ 밴드패스 필터 적용 (0.5–45Hz)
eeg_bandpassed = apply_bandpass_filter(eeg_segment, fs=fs)

# 🎧 주파수 변환 (원본 vs 필터)
fft_orig = np.fft.fft(eeg_segment)
fft_filt = np.fft.fft(eeg_bandpassed)
freq = np.fft.fftfreq(n_samples, d=1/fs)
fft_freq = freq[:n_samples // 2]
fft_mag_orig = np.abs(fft_orig)[:n_samples // 2]
fft_mag_filt = np.abs(fft_filt)[:n_samples // 2]

# 🎨 시각화
sns.set(style="whitegrid", font_scale=1.1)
fig, axs = plt.subplots(2, 2, figsize=(14, 6))

# 시간 영역
axs[0, 0].plot(np.arange(n_samples) / fs, eeg_segment, color='blue')
axs[0, 0].set_title(f"Time Domain (Original) - {eeg_names[ch]}")
axs[0, 0].set_xlabel("Time (sec)")
axs[0, 0].set_ylabel("Amplitude")

axs[0, 1].plot(np.arange(n_samples) / fs, eeg_bandpassed, color='green')
axs[0, 1].set_title(f"Time Domain (Bandpass 0.5–45Hz) - {eeg_names[ch]}")
axs[0, 1].set_xlabel("Time (sec)")
axs[0, 1].set_ylabel("Amplitude")

# 주파수 영역
axs[1, 0].plot(fft_freq, fft_mag_orig, color='red')
axs[1, 0].set_title("Frequency Domain (Original)")
axs[1, 0].set_xlabel("Frequency (Hz)")
axs[1, 0].set_ylabel("Magnitude")
axs[1, 0].set_xlim(0, 64)

axs[1, 1].plot(fft_freq, fft_mag_filt, color='purple')
axs[1, 1].set_title("Frequency Domain (Bandpass Filtered)")
axs[1, 1].set_xlabel("Frequency (Hz)")
axs[1, 1].set_ylabel("Magnitude")
axs[1, 1].set_xlim(0, 64)

plt.tight_layout()
plt.show()


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


# In[ ]:


# cell 위아래로: option + 방향키


# In[35]:


df


# In[43]:


df_remove = df.drop({'subject_id','video_id','segment_idx'},axis=1)
df_remove


# In[44]:


import numpy as np

X = np.stack(df_remove['eeg_band_data'].values).astype(np.float32)

y_valence = df_remove['valence'].values
y_arousal = df_remove['arousal'].values
# y_dominance = df_remove['dominance'].values


# In[57]:


from sklearn.preprocessing import StandardScaler

X_reshaped = X.reshape(-1, 5 * 14)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_reshaped)

# 다시 (8372, 1280, 5, 14)로 reshape
X_scaled = X_scaled.reshape(-1, 1280, 5, 14)


# In[58]:


X_scaled.shape


# In[59]:


y_combined = (
    (y_valence > 3).astype(int) * 2 +  # 상위 비트
    (y_arousal > 3).astype(int)        # 하위 비트
)
y_combined = np.array(y_combined).astype(np.int64)


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

        # GRU input: (batch, seq_len, input_size)
        self.gru = nn.GRU(input_size=256, hidden_size=128, batch_first=True, bidirectional=True)

        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 2, 64)  # bidirectional → 2x hidden
        self.fc2 = nn.Linear(64, 4)  # 4-class output

    def forward(self, x):
        # x: (B, 5, 14, 1280)
        x = x.view(x.size(0), 14*5, 1280)  # → (B, 70, 1280)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))  # → (B, 128, 640)
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))  # → (B, 256, 320)

        x = x.permute(0, 2, 1)  # GRU expects: (B, seq_len, feature) → (B, 320, 256)
        out, _ = self.gru(x)    # → (B, 320, 256)
        out = out[:, -1, :]     # 마지막 time step 출력만 사용 (B, 256)

        out = self.dropout(F.relu(self.fc1(out)))
        return self.fc2(out)


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


# In[ ]:




