"""
example.py
==========
Quick-start example demonstrating flood prediction on a pre-packaged tile.

This example uses a pre-processed tile (tile_08695, Bangladesh, 2020-08-23)
that already contains all required inputs in the correct format, allowing
you to test the model without downloading raw satellite data.

For inference on your own data, see inference/predict.py.
"""

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from model.unet import UNet
from model.diffusion import DiffusionUNet, cosine_beta_schedule, precompute_schedule
from inference.predict import load_norm_stats, normalize
from huggingface_hub import hf_hub_download

HF_REPO_ID   = "gauthiermalandrin/cygnss-sar-flood-mapping"
TILE_PATH    = Path("data/example/tile_08695.npz")
OUTPUT_DIR   = Path("data/example")
THRESHOLD    = 0.5
N_SAMPLES    = 5    # reduced for speed; use 20 for paper-quality results
T, S, T_START = 1000, 50, 200
COND_CHANNELS = [4, 0, 1, 2, 3, 7, 8, 6]


def ddim_sample(model, c, schedule, T, S, device, x_init=None, t_start=200):
    import torch.nn.functional as F
    timesteps = torch.linspace(t_start, 0, S, dtype=torch.long, device=device)
    B, _, H, W = c.shape
    if x_init is not None:
        sqrt_ab    = schedule['sqrt_ab'][t_start].to(device).view(1,1,1,1)
        sqrt_1m_ab = schedule['sqrt_1m_ab'][t_start].to(device).view(1,1,1,1)
        x = sqrt_ab * x_init + sqrt_1m_ab * torch.randn_like(x_init)
    else:
        x = torch.randn(B, 1, H, W, device=device)
    model.eval()
    with torch.no_grad():
        for i, t_val in enumerate(timesteps):
            eps     = model(x, t_val.expand(B), c)
            ab_t    = schedule['alpha_bar'][t_val].to(device)
            ab_prev = schedule['alpha_bar'][timesteps[i+1]].to(device) \
                      if i + 1 < S else torch.tensor(1.0, device=device)
            x0_pred = (x - torch.sqrt(1 - ab_t) * eps) / torch.sqrt(ab_t)
            x0_pred = x0_pred.clamp(-1, 1)
            x       = torch.sqrt(ab_prev) * x0_pred + \
                      torch.sqrt(1 - ab_prev) * eps
    return x0_pred


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── Load tile ─────────────────────────────────────────────────────────────
    print(f"\nLoading tile: {TILE_PATH}")
    d       = np.load(TILE_PATH)
    x       = d['input'].astype(np.float32)   # (10, 256, 256)
    target  = d['target'].astype(np.float32)  # (256, 256)
    v10_pred = d['v10_pred'].astype(np.float32) if 'v10_pred' in d else None

    # ── Download weights ──────────────────────────────────────────────────────
    print("\nDownloading model weights from HuggingFace...")
    unet_ckpt = hf_hub_download(HF_REPO_ID,
                                "checkpoints/best_model_final_vv.pth")
    diff_ckpt = hf_hub_download(HF_REPO_ID,
                                "checkpoints/best_model_diffusion.pth")
    stats_path = hf_hub_download(HF_REPO_ID, "model/norm_stats.json")

    # ── Load U-Net ────────────────────────────────────────────────────────────
    print("Loading U-Net...")
    unet = UNet(in_channels=10, out_channels=1,
                features=[32, 64, 128, 256], dropout=0.0).to(device)
    ckpt = torch.load(unet_ckpt, map_location=device, weights_only=False)
    unet.load_state_dict(ckpt.get('model_state_dict', ckpt))
    unet.eval()

    # ── Normalize and predict ─────────────────────────────────────────────────
    means, stds = load_norm_stats(stats_path)
    x_norm = normalize(x, means, stds)

    with torch.no_grad():
        inp  = torch.from_numpy(x_norm).unsqueeze(0).to(device)
        prob = torch.sigmoid(unet(inp)).squeeze().cpu().numpy()

    pred_bin = (prob > THRESHOLD).astype(np.uint8)
    tgt_bin  = (target > THRESHOLD)

    tp    = np.logical_and(pred_bin, tgt_bin).sum()
    union = np.logical_or(pred_bin,  tgt_bin).sum()
    iou   = tp / union if union > 0 else 0
    print(f"\nU-Net IoU = {iou:.3f}")

    # ── Load diffusion model ──────────────────────────────────────────────────
    print(f"\nGenerating diffusion ensemble ({N_SAMPLES} samples)...")
    ckpt_d = torch.load(diff_ckpt, map_location=device, weights_only=False)
    diff   = DiffusionUNet(
        n_cond=ckpt_d.get('n_cond', 9),
        features=ckpt_d.get('features', [16, 32, 64, 128])
    ).to(device)
    diff.load_state_dict(ckpt_d['model_state_dict'])
    diff.eval()

    betas    = cosine_beta_schedule(T)
    schedule = precompute_schedule(betas)

    # Build condition vector
    unet_pred_input = v10_pred if v10_pred is not None else prob
    cond = np.concatenate(
        [unet_pred_input[np.newaxis], x_norm[COND_CHANNELS]], axis=0
    )
    c_t    = torch.from_numpy(cond).unsqueeze(0).to(device)
    x_init = torch.from_numpy(
        (unet_pred_input[np.newaxis, np.newaxis] * 2.0 - 1.0)
        .clip(-1, 1).astype(np.float32)
    ).to(device)

    samples = []
    for i in range(N_SAMPLES):
        print(f"  sample {i+1}/{N_SAMPLES}", flush=True)
        x0  = ddim_sample(diff, c_t, schedule, T, S, device,
                           x_init=x_init, t_start=T_START)
        samples.append((x0.squeeze().cpu().numpy() + 1.0) / 2.0)

    samples     = np.stack(samples)
    diff_mean   = samples.mean(axis=0)
    uncertainty = samples.std(axis=0)

    # ── Visualize ─────────────────────────────────────────────────────────────
    print("\nGenerating output figure...")
    CYGNSS_CMAP = mcolors.ListedColormap(["white", "#94c4df", "#08306b"])
    cygnss_z    = x[4]
    cygnss_mapped = np.where(cygnss_z > 0.5, 1.0,
                    np.where(cygnss_z < -0.5, 0.0, 0.5))

    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))
    fig.patch.set_facecolor('white')

    panels = [
        (cygnss_mapped, CYGNSS_CMAP, "CYGNSS watermask B\n~1 km · daily",        0, 1),
        (pred_bin,      "Blues",      "U-Net prediction\n90 m · daily",           0, 1),
        (tgt_bin,       "Blues",      "Ground truth (Sentinel-1)\n~90 m · 12-day",0, 1),
        (diff_mean,     "Blues",      f"Diffusion ensemble mean\n(N={N_SAMPLES} samples)", 0, 1),
        (uncertainty,   "YlOrRd",     "Uncertainty (σ)\npixel-wise std",          0, uncertainty.max()),
    ]

    for ax, (img, cmap, title, vmin, vmax) in zip(axes, panels):
        ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
        ax.set_title(title, fontsize=9, fontweight='bold', pad=6)
        ax.axis('off')

    fig.suptitle(f"Bangladesh — 2020-08-23   |   U-Net IoU = {iou:.3f}",
                 fontsize=11, y=1.02)
    plt.tight_layout()

    out_path = OUTPUT_DIR / "example_output.png"
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nOutput saved → {out_path}")
    print("Done!")


if __name__ == "__main__":
    main()