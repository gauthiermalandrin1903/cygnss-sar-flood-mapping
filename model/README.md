# Model files

## Architecture files

### `unet.py` — Temporal U-Net (10.55M parameters)

The main flood mapping model. Takes a 10-channel input tensor and produces
a spatially explicit flood probability map at 90 m resolution.

Key design choices:
- **Three parallel encoding branches** (one per temporal acquisition A, B, C)
  that share weights from level 2 onwards
- **Learnable cross-attention gates** at every skip connection
  (window attention at shallow levels, global attention at deep levels)
- **Static topographic channels** (DEM, HAND, ACC, TWI) injected into each
  branch via a learned 1×1 projection
- **Gate initialization at zero** — the model starts as a standard U-Net
  and progressively learns to exploit temporal cross-attention

### `diffusion.py` — Conditional Diffusion Model (1.1M parameters)

A lightweight conditional U-Net used as the denoising backbone for
probabilistic flood mapping. Conditioned on 9 channels (U-Net prediction +
CYGNSS_B + DEM + HAND + ACC_log + TWI + SAR_A + SAR_C + Delta_CYGNSS).

Inference uses DDIM sampling (50 steps) with SDEdit initialization
(T_start = 200) from the deterministic U-Net prediction.

## Data files

### `norm_stats.json`

Z-score normalization statistics (mean and standard deviation per channel)
computed over 2,000 training tiles from the final dataset. Applied to all
10 input channels before inference.

```json
{
  "channel_names": ["DEM", "HAND", "ACC_log", "TWI", "CYGNSS_B",
                    "Mean_CYGNSS", "Delta_CYGNSS", "S1_flood_A",
                    "S1_flood_C", "Mean_VV"],
  "means": [...],
  "stds":  [...]
}
```

## Pre-trained weights

Weights are hosted on HuggingFace and downloaded automatically on first use:

| File | Size | Description |
|------|------|-------------|
| `checkpoints/best_model_final_vv.pth` | 121 MB | U-Net (Mean_VV variant) |
| `checkpoints/best_model_diffusion.pth` | 27 MB | Conditional diffusion model |

```python
from inference.predict import FloodPredictor
predictor = FloodPredictor.from_pretrained()  # auto-downloads weights
```