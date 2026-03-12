import torch
from torch.utils.data import Dataset

# ---------------------------
# Dataset object
# ---------------------------
class EEGDataset(Dataset):
    def __init__(self, X, y_val, y_aro):
        self.X = torch.tensor(X, dtype=torch.float32)  
        self.y_val = torch.tensor(y_val, dtype=torch.float32)
        self.y_aro = torch.tensor(y_aro, dtype=torch.float32)
    
    def __len__(self): 
        return len(self.X)
    
    def __getitem__(self, i):
        return self.X[i], self.y_val[i], self.y_aro[i]
    

class EEGCondDataset(Dataset):
    # 기존 데이터는 y label이 2개인데, 생성 AI 학습할 떄는 4진으로 나눠서 학습.
    # X: (N,70,1280)  (표준화된 실수), y4: (N,) in {0,1,2,3}
    def __init__(self, X, y4):
        self.X  = torch.tensor(X, dtype=torch.float32)
        self.y4 = torch.tensor(y4, dtype=torch.long)
    def __len__(self): 
        return len(self.X)

    def __getitem__(self, i): 
        return self.X[i], self.y4[i]