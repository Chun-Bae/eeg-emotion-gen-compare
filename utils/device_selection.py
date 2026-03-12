# ---------------------------
# 디바이스 선택
# ---------------------------
import torch

def device_selection():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    bar_len = 28 
    device_str = str(device)
    print("=" * bar_len)
    print(f"선택된 디바이스    {device_str.center(bar_len - 4)}")
    print("=" * bar_len)

    return device