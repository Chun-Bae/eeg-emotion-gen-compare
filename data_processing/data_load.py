import os
import numpy as np
import pandas as pd

from utils.bandpass_filter import bandpass_filter
import scipy.io

def initial_dreamer_load():
    npz_path = "./data/filterd_EEG_VA_data.npz"
    
    if os.path.exists(npz_path):
        data = np.load(npz_path)

        X = data["eeg_band_data"]  
        valence = data["valence"]
        arousal = data["arousal"]
        dominance = data["dominance"]
        y = data["y"]
        y_val = data["y_val"]
        y_aro = data["y_aro"]

        df = pd.DataFrame({
            "eeg_band_data": list(X),
            "valence": valence,
            "arousal": arousal,
            "dominance": dominance,
            "y": y,
            "y_val": y_val,
            "y_aro": y_aro
        })

    else:
        # ---------------------------
        # 원시 데이터 가져오기
        # ---------------------------
        mat = scipy.io.loadmat('./data/DREAMER.mat', struct_as_record=False, squeeze_me=True)
        dreamer = mat['DREAMER']

        # ---------------------------
        # 데이터 전처리
        # ---------------------------
        segment_len = 1280              # 10s @ 128 Hz
        overlap = 256                   # 겹침 길이(프레임) = 2s
        hop = segment_len - overlap     # = 1024 (슬라이딩 간격)

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


        np.savez("./data/filterd_EEG_VA_data.npz",
            eeg_band_data=np.array(df['eeg_band_data'].tolist(), dtype=np.float32),
            valence=df['valence'].to_numpy(),
            arousal=df['arousal'].to_numpy(),
            dominance=df['dominance'].to_numpy(),
            y=df['y'].to_numpy(),
            y_val=df['y_val'].to_numpy(),
            y_aro=df['y_aro'].to_numpy())
        
    return df