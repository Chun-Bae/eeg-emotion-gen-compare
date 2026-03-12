import torch.nn as nn
import torch.nn.functional as F


z_dim   = 128
n_class = 4       
C_in    = 70      # 5 bands * 14 ch
T_len   = 1280

# ---------------------------
# Conditional BatchNorm1d
# ---------------------------
class CBN1d(nn.Module):
    def __init__(self, num_features, n_class, emb_dim=64):
        super().__init__()
        self.bn   = nn.BatchNorm1d(num_features, affine=False)
        self.emb  = nn.Embedding(n_class, emb_dim)
        self.gama = nn.Linear(emb_dim, num_features)
        self.beta = nn.Linear(emb_dim, num_features)
        nn.init.zeros_(self.beta.weight); nn.init.ones_(self.gama.weight)
        nn.init.zeros_(self.beta.bias);   nn.init.zeros_(self.gama.bias)
    def forward(self, x, y):
        # x: (B,C,L), y: (B,)
        h = self.bn(x)
        e = self.emb(y)                    # (B,emb)
        g = self.gama(e).unsqueeze(-1)     # (B,C,1)
        b = self.beta(e).unsqueeze(-1)     # (B,C,1)
        return g * h + b

# ---------------------------
# Generator (upsample x2 * 4 → 80→1280)
# ---------------------------
class Gen1D(nn.Module):
    def __init__(self, z_dim=128, n_class=4, C_out=70, T_len=1280, base_ch=256):
        super().__init__()
        self.T0 = T_len // 16              # 80
        self.linear = nn.Linear(z_dim, base_ch * self.T0)

        # 업블록: (C_in → C_out), upsample×2
        self.cbn1 = CBN1d(base_ch, n_class)
        self.conv1 = nn.Conv1d(base_ch, base_ch, 3, padding=1)

        self.cbn2 = CBN1d(base_ch, n_class)
        self.conv2 = nn.Conv1d(base_ch, base_ch//2, 3, padding=1)

        self.cbn3 = CBN1d(base_ch//2, n_class)
        self.conv3 = nn.Conv1d(base_ch//2, base_ch//4, 3, padding=1)

        self.cbn4 = CBN1d(base_ch//4, n_class)
        self.conv4 = nn.Conv1d(base_ch//4, base_ch//4, 3, padding=1)

        self.to_out = nn.Conv1d(base_ch//4, C_out, 1)

    def _up(self, x):  # nearest upsample ×2
        return F.interpolate(x, scale_factor=2, mode="nearest")

    def forward(self, z, y):
        # z:(B,z_dim), y:(B,)
        h = self.linear(z).view(z.size(0), -1, self.T0)          # (B,256,80)

        h = self._up(F.relu(self.cbn1(h, y)))
        h = F.relu(self.conv1(h))                                # (B,256,160)

        h = self._up(F.relu(self.cbn2(h, y)))
        h = F.relu(self.conv2(h))                                # (B,128,320)

        h = self._up(F.relu(self.cbn3(h, y)))
        h = F.relu(self.conv3(h))                                # (B,64,640)

        h = self._up(F.relu(self.cbn4(h, y)))
        h = F.relu(self.conv4(h))                                # (B,64,1280)

        x = self.to_out(h)                                       # (B,70,1280)
        return x


# ---------------------------
# Discriminator (Projection D)
# ---------------------------
def SN(m):  # spectral norm
    return nn.utils.spectral_norm(m)

class Disc1D(nn.Module):
    def __init__(self, C_in=70, n_class=4, base_ch=64):
        super().__init__()
        self.conv1 = SN(nn.Conv1d(C_in,   base_ch,   5, padding=2))
        self.conv2 = SN(nn.Conv1d(base_ch, base_ch*2,5, padding=2))
        self.conv3 = SN(nn.Conv1d(base_ch*2, base_ch*4,5, padding=2))
        self.conv4 = SN(nn.Conv1d(base_ch*4, base_ch*4,3, padding=1))

        self.pool  = nn.AvgPool1d(2)
        self.lin   = SN(nn.Linear(base_ch*4, 1))
        self.emb   = nn.Embedding(n_class, base_ch*4)  # projection

    def forward(self, x, y):
        # x:(B,70,1280), y:(B,)
        h = F.leaky_relu(self.conv1(x), 0.2); h = self.pool(h)   # L/2
        h = F.leaky_relu(self.conv2(h), 0.2); h = self.pool(h)   # L/4
        h = F.leaky_relu(self.conv3(h), 0.2); h = self.pool(h)   # L/8
        h = F.leaky_relu(self.conv4(h), 0.2); h = self.pool(h)   # L/16

        h = h.mean(dim=-1)                                       # GAP → (B,C)
        out = self.lin(h).squeeze(1)                             # (B,)
        # projection term
        proj = (self.emb(y) * h).sum(dim=1)                      # (B,)
        return out + proj