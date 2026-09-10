"""
unet_v5.py
==========
U-Net V12/V13 — architecture temporelle à trois encodeurs partagés
avec cross-attention multi-niveau.

Différences vs unet_v4.py (version légère) :
  - Trois branches encodeurs partageant leurs poids (A, B, C)
  - Canaux statiques (DEM, HAND, ACC_log, TWI) projetés et injectés
    dans chaque branche via une convolution 1×1 entraînable
  - TemporalCrossAttentionV2 appliquée à TOUS les niveaux du décodeur
    (Q=feat_B, K/V=feat_A+feat_C) — gamma initialisé à 0 par niveau
  - Attention locale (window) aux deux premiers niveaux (128×128, 64×64)
    pour limiter le coût mémoire O(N²)
  - Attention globale aux deux derniers niveaux (32×32, 16×16) + bottleneck

Séparation des 10 canaux d'entrée :
  Statique (4) : DEM(0), HAND(1), ACC_log(2), TWI(3)
  Branche A (1) : S1_flood_A(7)
  Branche B (3) : CYGNSS_B(4), Mean_CYGNSS(5), Delta_CYGNSS(6)
  Branche C (2) : S1_flood_C(8), Mean_VV(9)

Chaque branche reçoit ses canaux propres + projection des canaux statiques.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════════
# Blocs de base
# ═══════════════════════════════════════════════════════════════════════════════

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.15):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


# ═══════════════════════════════════════════════════════════════════════════════
# Attention temporelle V2 — globale (pour niveaux profonds)
# ═══════════════════════════════════════════════════════════════════════════════

class TemporalCrossAttentionGlobal(nn.Module):
    """
    Cross-attention globale entre les trois branches temporelles.
    Q = feat_B, K/V = concat(feat_A, feat_C)
    Coût : O(H*W)² — réservé aux niveaux profonds (32×32 et moins).
    gamma initialisé à 0 — transparent au départ.
    """

    def __init__(self, channels):
        super().__init__()
        d = max(channels // 8, 16)
        self.q    = nn.Conv2d(channels,     d,        kernel_size=1, bias=False)
        self.k    = nn.Conv2d(channels * 2, d,        kernel_size=1, bias=False)
        self.v    = nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(channels,     channels, kernel_size=1)
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, feat_a, feat_b, feat_c):
        B, C, H, W = feat_b.shape
        feat_ac = torch.cat([feat_a, feat_c], dim=1)

        Q = self.q(feat_b).view(B, -1, H * W)
        K = self.k(feat_ac).view(B, -1, H * W)
        V = self.v(feat_ac).view(B, -1, H * W)

        scale = Q.shape[1] ** 0.5
        attn  = F.softmax(torch.bmm(Q.permute(0, 2, 1), K) / scale, dim=-1)
        out   = torch.bmm(V, attn.permute(0, 2, 1)).view(B, C, H, W)
        out   = self.proj(out)

        gamma_c = torch.tanh(self.gamma)
        return self.norm(feat_b + gamma_c * out)


# ═══════════════════════════════════════════════════════════════════════════════
# Attention temporelle V2 — locale (pour niveaux superficiels)
# ═══════════════════════════════════════════════════════════════════════════════

class TemporalCrossAttentionLocal(nn.Module):
    """
    Cross-attention locale (window attention) entre les trois branches.
    Découpe les feature maps en fenêtres de taille window_size × window_size
    et applique l'attention indépendamment dans chaque fenêtre.
    Coût : O(window_size²)² × (H*W/window_size²) — linéaire en H*W.
    Utilisé aux niveaux superficiels (64×64, 128×128).

    V16 — deux changements vs V15 :
      V17 : tanh simple sur gamma (cohérent avec module global).
         Suppression du LayerNorm Q/K et du facteur 3 — gamma_c = tanh(gamma).
         Plage ±1, gradient non nul partout, comportement uniforme sur tous niveaux.
    """

    def __init__(self, channels, window_size=8):
        super().__init__()
        self.window_size = window_size
        d = max(channels // 8, 16)
        self.q      = nn.Conv2d(channels,     d,        kernel_size=1, bias=False)
        self.k      = nn.Conv2d(channels * 2, d,        kernel_size=1, bias=False)
        self.v      = nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False)
        self.proj   = nn.Conv2d(channels,     channels, kernel_size=1)
        self.norm   = nn.GroupNorm(min(8, channels), channels)
        self.gamma  = nn.Parameter(torch.zeros(1))

    def _window_partition(self, x):
        """Découpe (B, C, H, W) en fenêtres (B*nW, C, ws, ws)."""
        B, C, H, W = x.shape
        ws = self.window_size
        x = x.view(B, C, H // ws, ws, W // ws, ws)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        return x.view(-1, C, ws, ws)

    def _window_reverse(self, x, B, H, W):
        """Reconstruit (B, C, H, W) depuis les fenêtres."""
        ws = self.window_size
        C  = x.shape[1]
        x  = x.view(B, H // ws, W // ws, C, ws, ws)
        x  = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        return x.view(B, C, H, W)

    def forward(self, feat_a, feat_b, feat_c):
        B, C, H, W = feat_b.shape
        ws = self.window_size

        # Padding si nécessaire
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_h > 0 or pad_w > 0:
            feat_a = F.pad(feat_a, (0, pad_w, 0, pad_h))
            feat_b = F.pad(feat_b, (0, pad_w, 0, pad_h))
            feat_c = F.pad(feat_c, (0, pad_w, 0, pad_h))

        _, _, Hp, Wp = feat_b.shape
        feat_ac = torch.cat([feat_a, feat_c], dim=1)

        # Projections Q, K, V
        Q_map = self.q(feat_b)
        K_map = self.k(feat_ac)
        V_map = self.v(feat_ac)

        # Découpe en fenêtres
        Q_win = self._window_partition(Q_map)   # (B*nW, d, ws, ws)
        K_win = self._window_partition(K_map)
        V_win = self._window_partition(V_map)

        nW = Q_win.shape[0]
        d  = Q_win.shape[1]

        Q_win = Q_win.view(nW, d, -1)   # (B*nW, d, ws²)
        K_win = K_win.view(nW, d, -1)
        V_win = V_win.view(nW, C, -1)

        scale    = d ** 0.5
        attn     = F.softmax(
            torch.bmm(Q_win.permute(0, 2, 1), K_win) / scale, dim=-1
        )
        out_win  = torch.bmm(V_win, attn.permute(0, 2, 1))
        out_win  = out_win.view(nW, C, ws, ws)

        # Reconstruction
        out = self._window_reverse(out_win, B, Hp, Wp)

        # Supprime le padding
        if pad_h > 0 or pad_w > 0:
            out    = out[:, :, :H, :W]
            feat_b = feat_b[:, :, :H, :W]

        out = self.proj(out)
        # V17 : tanh simple, cohérent avec module global
        gamma_c = torch.tanh(self.gamma)
        return self.norm(feat_b + gamma_c * out)


# ═══════════════════════════════════════════════════════════════════════════════
# U-Net V5 — architecture principale
# ═══════════════════════════════════════════════════════════════════════════════

# Mapping canaux d'entrée → branches
STATIC_CHANNELS = [0, 1, 2, 3]   # DEM, HAND, ACC_log, TWI
BRANCH_A_CH     = [7]             # S1_flood_A
BRANCH_B_CH     = [4, 5, 6]      # CYGNSS_B, Mean_CYGNSS, Delta_CYGNSS
BRANCH_C_CH     = [8, 9]         # S1_flood_C, Mean_VV


class UNet(nn.Module):
    """
    U-Net V5 — trois encodeurs partagés + cross-attention multi-niveau.

    Args:
        in_channels  : canaux d'entrée totaux (default: 10, non utilisé directement)
        out_channels : canaux de sortie (default: 1)
        features     : liste des features par niveau encodeur
        dropout      : dropout dans les ConvBlocks
        window_size  : taille de fenêtre pour l'attention locale
    """

    def __init__(self, in_channels=10, out_channels=1,
                 features=None, dropout=0.15, window_size=8):
        super().__init__()

        if features is None:
            features = [32, 64, 128, 256]

        self.features     = features
        self.window_size  = window_size
        n_levels          = len(features)

        # ── Projection des canaux statiques vers chaque branche ───────────────
        n_static = len(STATIC_CHANNELS)
        n_a      = len(BRANCH_A_CH)
        n_b      = len(BRANCH_B_CH)
        n_c      = len(BRANCH_C_CH)

        static_proj_dim = 2  # chaque branche reçoit 2 canaux statiques projetés
        self.static_proj_a = nn.Conv2d(n_static, static_proj_dim, kernel_size=1)
        self.static_proj_b = nn.Conv2d(n_static, static_proj_dim, kernel_size=1)
        self.static_proj_c = nn.Conv2d(n_static, static_proj_dim, kernel_size=1)

        in_a = n_a + static_proj_dim
        in_b = n_b + static_proj_dim
        in_c = n_c + static_proj_dim

        # ── Encodeur partagé (poids communs entre A, B, C) ───────────────────
        # Note : les trois branches ont des dimensions d'entrée différentes
        # (in_a=3, in_b=5, in_c=4), donc on ne peut pas partager la première
        # couche. On partage à partir du deuxième ConvBlock.
        self.enc_first_a = nn.ModuleList()
        self.enc_first_b = nn.ModuleList()
        self.enc_first_c = nn.ModuleList()
        self.enc_shared  = nn.ModuleList()  # partagé entre A, B, C (niveaux 2+)

        # Niveau 1 : couches d'entrée séparées (dimensions d'entrée différentes)
        self.enc_first_a.append(ConvBlock(in_a,      features[0], dropout))
        self.enc_first_b.append(ConvBlock(in_b,      features[0], dropout))
        self.enc_first_c.append(ConvBlock(in_c,      features[0], dropout))

        # Niveaux 2+ : encodeur partagé
        for i in range(1, n_levels):
            self.enc_shared.append(ConvBlock(features[i-1], features[i], dropout))

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # ── Bottleneck (partagé → fusionne après attention) ───────────────────
        bottleneck_in = features[-1] * 3  # concaténation A + B + C
        self.bottleneck_fusion = nn.Conv2d(bottleneck_in, features[-1] * 2,
                                           kernel_size=1)
        self.bottleneck = ConvBlock(features[-1] * 2, features[-1] * 2, dropout)

        # ── Attention temporelle par niveau ───────────────────────────────────
        # Niveaux profonds (2 derniers + bottleneck) : attention globale
        # Niveaux superficiels (2 premiers) : attention locale
        self.attention_modules = nn.ModuleList()
        for i, feat in enumerate(features):
            if i < n_levels - 2:
                # Niveaux superficiels → attention locale
                self.attention_modules.append(
                    TemporalCrossAttentionLocal(feat, window_size=window_size)
                )
            else:
                # Niveaux profonds → attention globale
                self.attention_modules.append(
                    TemporalCrossAttentionGlobal(feat)
                )

        # Attention au bottleneck (globale)
        self.bottleneck_attention = TemporalCrossAttentionGlobal(features[-1] * 2)

        # ── Décodeur ──────────────────────────────────────────────────────────
        self.upconvs       = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()

        for feat in reversed(features):
            self.upconvs.append(
                nn.ConvTranspose2d(feat * 2, feat, kernel_size=2, stride=2)
            )
            # skip = feat (après attention, on prend feat_B fusionné)
            self.decoder_blocks.append(ConvBlock(feat * 2, feat, dropout))

        # ── Sortie ────────────────────────────────────────────────────────────
        self.output_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def _encode_branch(self, x, first_block, level):
        """Encode une branche temporelle sur un niveau donné."""
        if level == 0:
            return first_block[0](x)
        else:
            return self.enc_shared[level - 1](x)

    def forward(self, x):
        # ── Séparation des canaux d'entrée ────────────────────────────────────
        x_static = x[:, STATIC_CHANNELS, :, :]
        x_a      = x[:, BRANCH_A_CH,     :, :]
        x_b      = x[:, BRANCH_B_CH,     :, :]
        x_c      = x[:, BRANCH_C_CH,     :, :]

        # Injection des canaux statiques dans chaque branche
        x_a = torch.cat([x_a, self.static_proj_a(x_static)], dim=1)
        x_b = torch.cat([x_b, self.static_proj_b(x_static)], dim=1)
        x_c = torch.cat([x_c, self.static_proj_c(x_static)], dim=1)

        # ── Encodeur multi-branches ───────────────────────────────────────────
        skips_a, skips_b, skips_c = [], [], []
        attn_skips = []  # skip connections après attention temporelle

        for level in range(len(self.features)):
            # Encodage niveau `level`
            x_a = self._encode_branch(x_a, self.enc_first_a, level)
            x_b = self._encode_branch(x_b, self.enc_first_b, level)
            x_c = self._encode_branch(x_c, self.enc_first_c, level)

            skips_a.append(x_a)
            skips_b.append(x_b)
            skips_c.append(x_c)

            # Attention temporelle sur les skip connections
            x_b_attn = self.attention_modules[level](x_a, x_b, x_c)
            attn_skips.append(x_b_attn)

            # Pooling (sauf dernier niveau)
            if level < len(self.features) - 1:
                x_a = self.pool(x_a)
                x_b = self.pool(x_b)
                x_c = self.pool(x_c)

        # ── Bottleneck ────────────────────────────────────────────────────────
        # Pooling du dernier niveau
        x_a = self.pool(x_a)
        x_b = self.pool(x_b)
        x_c = self.pool(x_c)

        # Fusion par concaténation puis projection
        x = self.bottleneck_fusion(torch.cat([x_a, x_b, x_c], dim=1))
        x = self.bottleneck(x)

        # Attention au bottleneck
        # Pour l'attention bottleneck : on crée des proxy A et C depuis x
        # (les branches sont déjà fusionnées, on utilise des projections)
        x = self.bottleneck_attention(x, x, x)

        # ── Décodeur ──────────────────────────────────────────────────────────
        attn_skips = attn_skips[::-1]  # du plus profond au plus superficiel

        for i, (upconv, decoder) in enumerate(
            zip(self.upconvs, self.decoder_blocks)
        ):
            x    = upconv(x)
            skip = attn_skips[i]  # skip après attention temporelle
            x    = torch.cat([skip, x], dim=1)
            x    = decoder(x)

        return self.output_conv(x)

    def get_gammas(self):
        """Retourne les gammas de tous les modules d'attention pour le logging."""
        gammas = {}
        for i, attn in enumerate(self.attention_modules):
            gammas[f"gamma_level_{i}"] = attn.gamma.item()
        gammas["gamma_bottleneck"] = self.bottleneck_attention.gamma.item()
        return gammas


# ═══════════════════════════════════════════════════════════════════════════════
# Test
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("Test UNet V5 — features=[32,64,128,256] (V12)")
    print("=" * 60)
    model_v12 = UNet(in_channels=10, out_channels=1,
                     features=[32, 64, 128, 256])
    total_v12 = sum(p.numel() for p in model_v12.parameters())
    print(f"Paramètres totaux : {total_v12:,} ({total_v12/1e6:.2f}M)")

    dummy = torch.randn(2, 10, 256, 256)
    out   = model_v12(dummy)
    print(f"Input  : {dummy.shape}")
    print(f"Output : {out.shape}")
    print(f"Gammas initiaux : {model_v12.get_gammas()}")

    print("\n" + "=" * 60)
    print("Test UNet V5 — features=[64,128,256,512] (V13)")
    print("=" * 60)
    model_v13 = UNet(in_channels=10, out_channels=1,
                     features=[64, 128, 256, 512])
    total_v13 = sum(p.numel() for p in model_v13.parameters())
    print(f"Paramètres totaux : {total_v13:,} ({total_v13/1e6:.2f}M)")
    out = model_v13(dummy)
    print(f"Output : {out.shape}")
