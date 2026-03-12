import torch.nn as nn
import torch
import math
import torch.nn.functional as F

# ---------------------------
# Utilities: time embedding (sinusoidal)
# ---------------------------
def sinusoidal_time_embedding(t, dim):
    # t: (B,), returns (B, dim)
    half = dim // 2
    freqs = torch.exp(
        torch.linspace(math.log(1.0), math.log(10000.0), steps=half, device=t.device)
    )
    angles = t[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0,1))
    return emb

# ---------------------------
# Building block: ResBlock1D + FiLM conditioning (time + class)
# ---------------------------
class ResBlock1D(nn.Module):
    def __init__(self, cin, cout, t_dim, c_dim, groups=8, n_class=4):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, cin)
        self.conv1 = nn.Conv1d(cin, cout, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, cout)
        self.conv2 = nn.Conv1d(cout, cout, 3, padding=1)

        # FiLM-style (time + class) → scale/shift
        self.t_proj = nn.Linear(t_dim, cout*2)
        self.c_proj = nn.Linear(c_dim, cout*2)

        self.act = nn.SiLU()
        self.skip = (cin != cout)
        if self.skip:
            self.skip_conv = nn.Conv1d(cin, cout, 1)

    def forward(self, x, t_emb, c_emb):
        h = self.act(self.norm1(x))
        h = self.conv1(h)

        # FiLM
        t_scale, t_shift = self.t_proj(t_emb).chunk(2, dim=1)
        c_scale, c_shift = self.c_proj(c_emb).chunk(2, dim=1)
        scale = (t_scale + c_scale).unsqueeze(-1)
        shift = (t_shift + c_shift).unsqueeze(-1)

        h = self.act(self.norm2(h))
        h = h * (1 + scale) + shift
        h = self.conv2(h)

        if self.skip:
            x = self.skip_conv(x)
        return x + h

# ---------------------------
# U-Net1D (down/up with skips)
# ---------------------------
class UNet1D(nn.Module):
    def __init__(self, in_ch=70, base=64, t_dim=128, c_dim=64, n_class=4):
        super().__init__()
        self.in_conv = nn.Conv1d(in_ch, base, 3, padding=1)

        # Down
        self.rb1 = ResBlock1D(base, base, t_dim, c_dim)
        self.down1 = nn.Conv1d(base, base*2, 4, stride=2, padding=1)   # 1280->640

        self.rb2 = ResBlock1D(base*2, base*2, t_dim, c_dim)
        self.down2 = nn.Conv1d(base*2, base*4, 4, stride=2, padding=1) # 640->320

        self.rb3 = ResBlock1D(base*4, base*4, t_dim, c_dim)
        self.down3 = nn.Conv1d(base*4, base*4, 4, stride=2, padding=1) # 320->160

        self.rb4 = ResBlock1D(base*4, base*4, t_dim, c_dim)            # bottleneck

        # Up (nearest upsample + conv to stabilize on MPS)
        self.up3 = nn.Conv1d(base*4, base*4, 3, padding=1)
        self.ub3 = ResBlock1D(base*4+base*4, base*4, t_dim, c_dim)

        self.up2 = nn.Conv1d(base*4, base*2, 3, padding=1)
        self.ub2 = ResBlock1D(base*2+base*2, base*2, t_dim, c_dim)

        self.up1 = nn.Conv1d(base*2, base, 3, padding=1)
        self.ub1 = ResBlock1D(base+base, base, t_dim, c_dim)

        self.out = nn.Conv1d(base, in_ch, 1)

        # embeddings
        self.t_mlp = nn.Sequential(
            nn.Linear(t_dim, t_dim), nn.SiLU(),
            nn.Linear(t_dim, t_dim)
        )
        self.c_emb = nn.Embedding(n_class, c_dim)

        # null token for CFG
        self.null_emb = nn.Parameter(torch.zeros(1, c_dim))

    def forward(self, x, t, y=None):
        # x: (B,C,L), t: (B,) ∈ [0..T-1], y: (B,) class or None (uncond)
        t = t.float()

        t_emb = sinusoidal_time_embedding(t, self.t_mlp[0].in_features)
        t_emb = self.t_mlp(t_emb)

        if y is None:
            c_emb = self.null_emb.expand(x.size(0), -1)   # (B, c_dim)
        else:
            c_emb = self.c_emb(y)

        # Down
        x0 = self.in_conv(x)                    # (B,base,1280)
        d1 = self.rb1(x0, t_emb, c_emb)
        x1 = self.down1(d1)                     # 640
        d2 = self.rb2(x1, t_emb, c_emb)
        x2 = self.down2(d2)                     # 320
        d3 = self.rb3(x2, t_emb, c_emb)
        x3 = self.down3(d3)                     # 160

        mid = self.rb4(x3, t_emb, c_emb)


        # Up
        u = F.interpolate(mid, scale_factor=2, mode="nearest"); u = self.up3(u)
        u = torch.cat([u, d3], dim=1); u = self.ub3(u, t_emb, c_emb)

        u = F.interpolate(u, scale_factor=2, mode="nearest"); u = self.up2(u)
        u = torch.cat([u, d2], dim=1); u = self.ub2(u, t_emb, c_emb)

        u = F.interpolate(u, scale_factor=2, mode="nearest"); u = self.up1(u)
        u = torch.cat([u, d1], dim=1); u = self.ub1(u, t_emb, c_emb)


        return self.out(u)  # predict noise ε
