import torch
import torch.nn as nn

C_in=70
T_len = 1280
n_class = 4
z_dim = 128

# ---------------------------
# Conditional VAE
# ---------------------------
class CondVAE1D(nn.Module):
    def __init__(self, C_in=70, T_len=1280, n_class=4, z_dim=128, emb_dim=32, base=128):
        super().__init__()
        self.T_len = T_len
        self.emb = nn.Embedding(n_class, emb_dim)

        # ----- Encoder -----
        Cin_enc = C_in + emb_dim  # 채널 concat
        self.enc = nn.Sequential(
            nn.Conv1d(Cin_enc, base,   5, padding=2, stride=2), nn.ReLU(),   # 1280→640
            nn.Conv1d(base,   base*2,  5, padding=2, stride=2), nn.ReLU(),   # 640→320
            nn.Conv1d(base*2, base*4,  5, padding=2, stride=2), nn.ReLU(),   # 320→160
            nn.Conv1d(base*4, base*4,  3, padding=1, stride=2), nn.ReLU(),   # 160→80
            nn.AdaptiveAvgPool1d(1)                                           
        )
        self.fc_mu   = nn.Linear(base*4, z_dim)
        self.fc_logv = nn.Linear(base*4, z_dim)

        # ----- Decoder -----
        self.fc0 = nn.Linear(z_dim + emb_dim, base*4*80)  # seed length 80
        self.dec = nn.Sequential(
            nn.ConvTranspose1d(base*4, base*4, 4, stride=2, padding=1), nn.ReLU(),  # 80→160
            nn.ConvTranspose1d(base*4, base*2, 4, stride=2, padding=1), nn.ReLU(),  # 160→320
            nn.ConvTranspose1d(base*2, base,   4, stride=2, padding=1), nn.ReLU(),  # 320→640
            nn.ConvTranspose1d(base,   base,   4, stride=2, padding=1), nn.ReLU(),  # 640→1280
            nn.Conv1d(base, C_in, 1)  # 활성 없음 (표준화 스케일)
        )

    def encode(self, x, y):
        B, C, T = x.shape
        e = self.emb(y)                                # (B, emb)
        e_rep = e.unsqueeze(-1).expand(B, e.size(1), T)# (B, emb, T)
        h = torch.cat([x, e_rep], dim=1)
        h = self.enc(h).squeeze(-1)                    # (B, base*4)
        mu, logv = self.fc_mu(h), self.fc_logv(h)
        return mu, logv

    def reparameterize(self, mu, logv):
        std = torch.exp(0.5 * logv)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, y):
        e = self.emb(y)
        zc = torch.cat([z, e], dim=1)
        h = self.fc0(zc).view(z.size(0), -1, 80)
        xhat = self.dec(h)                             # (B, C, 1280)
        return xhat

    def forward(self, x, y):
        mu, logv = self.encode(x, y)
        z = self.reparameterize(mu, logv)
        xhat = self.decode(z, y)
        return xhat, mu, logv