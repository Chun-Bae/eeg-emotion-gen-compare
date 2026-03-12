import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
    
class SpatialDropout1D(nn.Dropout2d):
    def forward(self, x):
        x = x.unsqueeze(3)   # (N, C, L, 1)
        x = super().forward(x)
        return x.squeeze(3)  # (N, C, L)

class Classifier(nn.Module):
    def __init__(self, in_channels=70, n_classes=4):
        super(Classifier, self).__init__()
        # Block 1
        self.conv1_1 = nn.Conv1d(in_channels, 32, kernel_size=5, padding=2)
        self.conv1_2 = nn.Conv1d(32, 32, kernel_size=5, padding=2)
        self.pool1   = nn.AvgPool1d(2)
        self.bn1     = nn.BatchNorm1d(32)
        self.sdrop1  = SpatialDropout1D(p=0.125)

        # Block 2
        self.conv2_1 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.conv2_2 = nn.Conv1d(64, 64, kernel_size=5, padding=2)
        self.pool2   = nn.AvgPool1d(2)
        self.bn2     = nn.BatchNorm1d(64)
        self.sdrop2  = SpatialDropout1D(p=0.25)

        # Block 3
        self.conv3_1 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.pool3   = nn.AvgPool1d(2)
        self.bn3     = nn.BatchNorm1d(128)
        self.sdrop3  = SpatialDropout1D(p=0.5)

        # Block 4
        self.conv4_1 = nn.Conv1d(128, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv1d(512, 512, kernel_size=3, padding=1)
        self.bn4     = nn.BatchNorm1d(512)
        self.drop4   = nn.Dropout(0.5)

        # Dense head
        self.fc1 = nn.Linear(512, 64)
        self.fc2 = nn.Linear(64, 8)
        self.drop_head = nn.Dropout(0.25)

        # --- 단일 분류기 → 멀티헤드(Val/Aro 각 1로짓) ---
        self.out_val = nn.Linear(8, 1)
        self.out_aro = nn.Linear(8, 1)

    def forward(self, x):  # x: (B, 70, 1280)
        # Block 1
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.pool1(x)
        x = self.bn1(x)
        x = self.sdrop1(x)

        # Block 2
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.pool2(x)
        x = self.bn2(x)
        x = self.sdrop2(x)

        # Block 3
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = self.pool3(x)
        x = self.bn3(x)
        x = self.sdrop3(x)

        # Block 4
        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = self.bn4(x)

        # Global Average Pooling
        x = x.mean(dim=-1)  # (B, 512)
        x = self.drop4(x)

        # Dense head (공유 임베딩)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.drop_head(x)

        # --- 멀티헤드 출력 ---
        logit_val = self.out_val(x).squeeze(1)  # (B,)
        logit_aro = self.out_aro(x).squeeze(1)  # (B,)
        return {"val": logit_val, "aro": logit_aro}