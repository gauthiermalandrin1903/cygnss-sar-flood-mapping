"""
predict_ensemble.py
===================
Probabilistic flood mapping via conditional diffusion model ensemble.

Extends FloodPredictor with uncertainty quantification: generates N
diffusion samples and returns both the ensemble mean flood map and a
pixel-wise uncertainty map (standard deviation across samples).

Usage:
    from inference.predict_ensemble import EnsemblePredictor

    predictor = EnsemblePredictor.from_pretrained()
    flood_map, uncertainty = predictor.predict(
        sar_a="path/to/s1_A.tif",
        cygnss_b="path/to/cygnss_B.nc",
        sar_c="path/to/s1_C.tif",
        merit_dir="path/to/merit_hydro/",
        region_bbox=(lon_min, lat_min, lon_max, lat_max),
        n_samples=20
    )
"""

import numpy as np
from pathlib import Path
import torch
from huggingface_hub import hf_hub_download

from inference.predict import FloodPredictor, load_norm_stats

HF_REPO_ID = "gauthiermalandrin/cygnss-sar-flood-mapping"

# Diffusion parameters (must match training configuration)
T       = 1000   # total diffusion steps
S       = 50     # DDIM steps at inference
T_START = 200    # SDEdit noise level

# Condition channel indices from the 10-channel input tensor
COND_CHANNELS = [4, 0, 1, 2, 3, 7, 8, 6]
# order: CYGNSS_B, DEM, HAND, ACC_log, TWI, SAR_A, SAR_C, Delta_CYGNSS
# + U-Net prediction = 9 channels total
N_COND = len(COND_CHANNELS) + 1


def ddim_sample(model, c, schedule, T, S, device, x_init=None, t_start=None):
    """
    DDIM reverse diffusion sampler with optional SDEdit initialization.

    Args:
        model    : DiffusionUNet denoising network
        c        : conditioning tensor (B, N_COND, H, W)
        schedule : precomputed noise schedule dict
        T        : total diffusion timesteps
        S        : number of DDIM sampling steps
        device   : torch device
        x_init   : optional initialization image for SDEdit (B, 1, H, W)
        t_start  : noise level for SDEdit initialization

    Returns:
        x0_pred : denoised prediction in [-1, 1], shape (B, 1, H, W)
    """
    if t_start is None:
        t_start = T - 1

    timesteps = torch.linspace(t_start, 0, S, dtype=torch.long, device=device)
    B, _, H, W = c.shape

    if x_init is not None:
        t_tensor   = torch.tensor([t_start], device=device)
        sqrt_ab    = schedule['sqrt_ab'][t_tensor.cpu()].to(device).view(1,1,1,1)
        sqrt_1m_ab = schedule['sqrt_1m_ab'][t_tensor.cpu()].to(device).view(1,1,1,1)
        noise = torch.randn_like(x_init)
        x = sqrt_ab * x_init + sqrt_1m_ab * noise
    else:
        x = torch.randn(B, 1, H, W, device=device)

    model.eval()
    with torch.no_grad():
        for i, t_val in enumerate(timesteps):
            t_batch = t_val.expand(B)
            eps     = model(x, t_batch, c)
            ab_t    = schedule['alpha_bar'][t_val].to(device)
            ab_prev = schedule['alpha_bar'][timesteps[i+1]].to(device) \
                      if i + 1 < S else torch.tensor(1.0, device=device)
            x0_pred = (x - torch.sqrt(1 - ab_t) * eps) / torch.sqrt(ab_t)
            x0_pred = x0_pred.clamp(-1, 1)
            x       = torch.sqrt(ab_prev) * x0_pred + \
                      torch.sqrt(1 - ab_prev) * eps

    return x0_pred


class EnsemblePredictor:
    """
    Probabilistic flood predictor using diffusion ensemble.

    Generates N independent flood map samples via conditional diffusion,
    returning the ensemble mean as the flood prediction and the pixel-wise
    standard deviation as the uncertainty map.

    Example:
        predictor = EnsemblePredictor.from_pretrained()
        flood_map, uncertainty = predictor.predict(
            sar_a="s1_2020-08-18.tif",
            cygnss_b="cygnss_2020-08-23.nc",
            sar_c="s1_2020-08-30.tif",
            merit_dir="merit_hydro/",
            region_bbox=(88.0, 22.0, 92.0, 25.0),
            n_samples=20
        )
    """

    def __init__(self, unet_predictor, diff_model, schedule, device):
        self.unet     = unet_predictor
        self.diff     = diff_model
        self.schedule = schedule
        self.device   = device

    @classmethod
    def from_pretrained(cls, device=None):
        """Load both U-Net and diffusion model weights from HuggingFace Hub."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from model.diffusion import DiffusionUNet, cosine_beta_schedule, precompute_schedule

        device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

        # Load U-Net predictor
        unet_predictor = FloodPredictor.from_pretrained(device=device)

        # Download and load diffusion model
        diff_ckpt_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename="checkpoints/best_model_diffusion.pth"
        )
        ckpt = torch.load(diff_ckpt_path, map_location=device, weights_only=False)
        diff_model = DiffusionUNet(
            n_cond=ckpt.get('n_cond', N_COND),
            features=ckpt.get('features', [16, 32, 64, 128])
        ).to(device)
        diff_model.load_state_dict(ckpt['model_state_dict'])
        diff_model.eval()

        # Precompute noise schedule
        betas    = cosine_beta_schedule(T)
        schedule = precompute_schedule(betas)

        return cls(unet_predictor, diff_model, schedule, device)

    def predict(self, sar_a, cygnss_b, sar_c, merit_dir, region_bbox,
                mean_cygnss=None, mean_vv=None,
                n_samples=20, threshold=0.5):
        """
        Generate an ensemble of flood maps and derive uncertainty estimates.

        Args:
            sar_a, cygnss_b, sar_c, merit_dir, region_bbox :
                same as FloodPredictor.predict()
            mean_cygnss  : optional long-term mean CYGNSS watermask
            mean_vv      : optional long-term mean VV backscatter
            n_samples    : number of diffusion samples (default: 20)
            threshold    : binarization threshold (default: 0.5)

        Returns:
            flood_map   : np.ndarray (H, W), binary flood mask {0, 1}
                          derived from ensemble mean > threshold
            uncertainty : np.ndarray (H, W), pixel-wise std across ensemble
                          high values indicate ambiguous flood boundaries
        """
        # Step 1: deterministic U-Net prediction
        _, prob_map = self.unet.predict(
            sar_a, cygnss_b, sar_c, merit_dir, region_bbox,
            mean_cygnss=mean_cygnss, mean_vv=mean_vv, threshold=threshold
        )

        # Step 2: build diffusion condition vector (9 channels)
        # Re-load the normalized input tensor
        from inference.predict import (
            load_sar_flood_mask, load_cygnss_watermask,
            load_merit_hydro, normalize, TILE_SIZE
        )
        target_shape = (TILE_SIZE, TILE_SIZE)
        flood_a = load_sar_flood_mask(sar_a, region_bbox, target_shape)
        cygnss  = load_cygnss_watermask(cygnss_b, region_bbox, target_shape)
        flood_c = load_sar_flood_mask(sar_c, region_bbox, target_shape)
        merit   = load_merit_hydro(merit_dir, region_bbox, target_shape)

        mean_cyg  = mean_cygnss if mean_cygnss is not None \
                    else np.zeros(target_shape, dtype=np.float32)
        delta_cyg = np.where(np.isnan(cygnss - mean_cyg), 0.0,
                             cygnss - mean_cyg).astype(np.float32)
        cygnss_clean = np.where(np.isnan(cygnss), 0.0, cygnss).astype(np.float32)

        x_full = np.stack([
            merit['DEM'], merit['HAND'], merit['ACC_log'], merit['TWI'],
            cygnss_clean, mean_cyg, delta_cyg, flood_a, flood_c,
            mean_vv if mean_vv is not None else np.zeros(target_shape, np.float32)
        ], axis=0).astype(np.float32)
        x_norm = normalize(x_full, self.unet.means, self.unet.stds)

        # Condition: [unet_pred] + selected channels
        cond_channels = x_norm[COND_CHANNELS]
        cond = np.concatenate(
            [prob_map[np.newaxis], cond_channels], axis=0
        )  # (9, H, W)
        c_t = torch.from_numpy(cond).unsqueeze(0).to(self.device)

        # SDEdit initialization from U-Net prediction
        x_init = torch.from_numpy(
            (prob_map[np.newaxis, np.newaxis] * 2.0 - 1.0).clip(-1, 1).astype(np.float32)
        ).to(self.device)

        # Step 3: generate N diffusion samples
        samples = []
        for _ in range(n_samples):
            x0 = ddim_sample(
                self.diff, c_t, self.schedule,
                T, S, self.device,
                x_init=x_init, t_start=T_START
            )
            prob = (x0.squeeze().cpu().numpy() + 1.0) / 2.0
            samples.append(prob)

        samples     = np.stack(samples)          # (N, H, W)
        ensemble_mean = samples.mean(axis=0)     # (H, W)
        uncertainty   = samples.std(axis=0)      # (H, W)

        flood_map = (ensemble_mean > threshold).astype(np.uint8)
        return flood_map, uncertainty