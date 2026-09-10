"""
diffusion.py
============
Conditional diffusion model for probabilistic flood mapping.

Architecture:
  - Denoising network ε_θ: lightweight conditional U-Net [16, 32, 64, 128]
  - Condition vector c: 9 channels (U-Net prediction + CYGNSS_B +
                        DEM/HAND/ACC/TWI + SAR_A + SAR_C + Delta_CYGNSS)
  - Cosine noise schedule (Nichol & Dhariwal, 2021), T=1000
  - DDIM sampling (Song et al., 2021) with SDEdit initialization

Reference:
  Malandrin & Gerlein-Safdi, IEEE TGRS, 2026
"""

import math
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ── Config ────────────────────────────────────────────────────────────────────

N_COND = 9
# ─────────────────────────────────────────────────────────────────────────────


# ── Cosine noise schedule ─────────────────────────────────────────────────────

def cosine_beta_schedule(T, s=0.008):
    """Cosine schedule (Nichol & Dhariwal 2021)."""
    steps  = torch.arange(T + 1, dtype=torch.float64)
    f      = torch.cos(((steps / T) + s) / (1 + s) * math.pi / 2) ** 2
    alphas = f / f[0]
    betas  = 1 - alphas[1:] / alphas[:-1]
    return torch.clamp(betas, 0, 0.999).float()


def precompute_schedule(betas):
    alphas      = 1.0 - betas
    alpha_bar   = torch.cumprod(alphas, dim=0)
    alpha_bar_prev = F.pad(alpha_bar[:-1], (1, 0), value=1.0)
    sqrt_ab     = torch.sqrt(alpha_bar)
    sqrt_1m_ab  = torch.sqrt(1.0 - alpha_bar)
    return {
        "betas":         betas,
        "alphas":        alphas,
        "alpha_bar":     alpha_bar,
        "alpha_bar_prev": alpha_bar_prev,
        "sqrt_ab":       sqrt_ab,
        "sqrt_1m_ab":    sqrt_1m_ab,
    }


# ── Time embedding ────────────────────────────────────────────────────────────

class SinusoidalTimeEmbedding(nn.Module):
    """Encodage sinusoïdal de t, projeté par MLP (comme Transformer)."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t):
        half    = self.dim // 2
        freqs   = torch.exp(-math.log(10000) *
                  torch.arange(half, device=t.device) / (half - 1))
        args    = t[:, None].float() * freqs[None]
        emb     = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.mlp(emb)


# ── Blocs U-Net conditionnel ──────────────────────────────────────────────────

class ResBlock(nn.Module):
    """Bloc résiduel avec injection du time embedding."""
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.conv1  = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2  = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1  = nn.GroupNorm(8, out_ch)
        self.norm2  = nn.GroupNorm(8, out_ch)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.skip   = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h  = F.silu(self.norm1(self.conv1(x)))
        h  = h + self.time_proj(t_emb)[:, :, None, None]
        h  = F.silu(self.norm2(self.conv2(h)))
        return h + self.skip(x)


# ── U-Net de débruitage léger ─────────────────────────────────────────────────

class DiffusionUNet(nn.Module):
    """
    U-Net léger [16,32,64,128] pour le débruitage conditionnel.
    Entrée : (x_t: 1ch) + (condition c: N_COND ch) + (time embedding)
    Sortie : bruit prédit ε (1ch)
    """
    def __init__(self, n_cond=N_COND, features=None, time_dim=128):
        super().__init__()
        if features is None:
            features = [16, 32, 64, 128]

        self.time_emb = SinusoidalTimeEmbedding(time_dim)
        in_ch = 1 + n_cond  # x_t + condition

        # Encodeur
        self.enc_blocks = nn.ModuleList()
        self.downs      = nn.ModuleList()
        ch = in_ch
        for f in features:
            self.enc_blocks.append(ResBlock(ch, f, time_dim))
            self.downs.append(nn.Conv2d(f, f, 3, stride=2, padding=1))
            ch = f

        # Bottleneck
        self.mid = ResBlock(ch, ch * 2, time_dim)
        ch = ch * 2

        # Décodeur
        self.dec_blocks = nn.ModuleList()
        self.ups        = nn.ModuleList()
        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(ch, f, 2, stride=2))
            self.dec_blocks.append(ResBlock(f * 2, f, time_dim))
            ch = f

        self.out = nn.Conv2d(ch, 1, 1)

    def forward(self, x_t, t, c):
        """
        x_t : (B, 1, H, W)   — carte bruitée
        t   : (B,)            — étape de bruit
        c   : (B, N_COND, H, W) — condition
        """
        x      = torch.cat([x_t, c], dim=1)
        t_emb  = self.time_emb(t)
        skips  = []

        for enc, down in zip(self.enc_blocks, self.downs):
            x = enc(x, t_emb)
            skips.append(x)
            x = down(x)

        x = self.mid(x, t_emb)

        for up, dec, skip in zip(self.ups, self.dec_blocks, reversed(skips)):
            x = dec(torch.cat([up(x), skip], dim=1), t_emb)

        return self.out(x)