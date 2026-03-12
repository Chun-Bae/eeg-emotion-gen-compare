import pandas as pd
import numpy as np
from data_processing.data_object import EEGDataset, EEGCondDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


from torch.utils.data import DataLoader

def clf_data_split(df, batch_size, seed):
    # ---------------------------
    # Data Split
    # ---------------------------
    X = np.stack(df['eeg_band_data'].values).astype(np.float32)  # (B, 1280, 5, 14)
    X = np.transpose(X, (0, 2, 3, 1))                             # (B, 5, 14, 1280)
    B, R, C, T = X.shape
    X = X.reshape(B, R*C, T).astype(np.float32)                   # (B, 70, 1280)

    y = df['y'].values.astype(np.int64)                           # 0~3  (2*V + A)
    y_val_bin = (y // 2).astype(np.int64)                         # 0/1
    y_aro_bin = (y % 2).astype(np.int64)                          # 0/1

    idx = np.arange(len(X))
    idx_tr, idx_va = train_test_split(idx, test_size=0.2, random_state=seed, stratify=y)
    # ---------------------------
    # 스케일링 : 표준화
    # ---------------------------

    X_train, X_val = X[idx_tr], X[idx_va]
    y_train_val, y_val_val = y_val_bin[idx_tr], y_val_bin[idx_va]
    y_train_aro, y_val_aro = y_aro_bin[idx_tr], y_aro_bin[idx_va]

    # (B, 70, 1280) -> (B*1280, 70) 로 바꿔서 채널 표준화
    X_train_2d = np.swapaxes(X_train, 1, 2).reshape(-1, R*C) 
    X_val_2d   = np.swapaxes(X_val,   1, 2).reshape(-1, R*C)

    scaler = StandardScaler()
    X_train_2d = scaler.fit_transform(X_train_2d)
    X_val_2d   = scaler.transform(X_val_2d)

    # 다시 (B, 70, 1280)로 복구
    X_train = X_train_2d.reshape(X_train.shape[0], T, R*C).swapaxes(1, 2).astype(np.float32)
    X_val   = X_val_2d.reshape(  X_val.shape[0],   T, R*C).swapaxes(1, 2).astype(np.float32)

    # ---------------------------
    # DataLoader
    # ---------------------------
    train_dataset = EEGDataset(X_train, y_train_val, y_train_aro)
    val_dataset   = EEGDataset(X_val,   y_val_val,   y_val_aro)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  drop_last=False)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, drop_last=False)

    return train_loader, val_loader


def gen_data_split(df, batch_size, seed):
    # ---------------------------
    # Data Split
    # ---------------------------
    X = np.stack(df['eeg_band_data'].values).astype(np.float32)  # (B, 1280, 5, 14)
    X = np.transpose(X, (0, 2, 3, 1))                             # (B, 5, 14, 1280)
    B, R, C, T = X.shape
    X = X.reshape(B, R*C, T).astype(np.float32)                   # (B, 70, 1280)

    y = df['y'].values.astype(np.int64)                           # 0~3  (2*V + A)
    y_val_bin = (y // 2).astype(np.int64)                         # 0/1
    y_aro_bin = (y % 2).astype(np.int64)                          # 0/1

    idx = np.arange(len(X))
    idx_tr, idx_va = train_test_split(idx, test_size=0.2, random_state=seed, stratify=y)
    # ---------------------------
    # 스케일링 : 표준화
    # ---------------------------

    X_train, X_val = X[idx_tr], X[idx_va]
    y_train_val, y_val_val = y_val_bin[idx_tr], y_val_bin[idx_va]
    y_train_aro, y_val_aro = y_aro_bin[idx_tr], y_aro_bin[idx_va]

    # (B, 70, 1280) -> (B*1280, 70) 로 바꿔서 채널 표준화
    X_train_2d = np.swapaxes(X_train, 1, 2).reshape(-1, R*C) 
    X_val_2d   = np.swapaxes(X_val,   1, 2).reshape(-1, R*C)

    scaler = StandardScaler()
    X_train_2d = scaler.fit_transform(X_train_2d)
    X_val_2d   = scaler.transform(X_val_2d)

    # 다시 (B, 70, 1280)로 복구
    X_train = X_train_2d.reshape(X_train.shape[0], T, R*C).swapaxes(1, 2).astype(np.float32)
    X_val   = X_val_2d.reshape(  X_val.shape[0],   T, R*C).swapaxes(1, 2).astype(np.float32)

    # ---------------------------
    # DataLoader
    # ---------------------------
    train_dataset = EEGCondDataset(X_train, y)
    val_dataset   = EEGCondDataset(X_val, y)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  drop_last=False)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, drop_last=False)

    return train_loader, val_loader