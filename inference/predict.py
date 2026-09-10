"""
predict.py
==========
Inference script for the CYGNSS–SAR flood mapping framework.

Given a temporal triplet (SAR anchor A, CYGNSS watermask B, SAR anchor C)
and MERIT Hydro topographic variables, produces a daily flood probability
map at 90 m resolution.

Usage:
    from inference.predict import FloodPredictor

    predictor = FloodPredictor.from_pretrained()
    flood_map, prob_map = predictor.predict(
        sar_a="path/to/s1_A.tif",
        cygnss_b="path/to/cygnss_B.nc",
        sar_c="path/to/s1_C.tif",
        merit_dir="path/to/merit_hydro/",
        region_bbox=(lon_min, lat_min, lon_max, lat_max)
    )
"""

import json
import numpy as np
from pathlib import Path
import torch
import rasterio
from rasterio.windows import from_bounds
from rasterio.enums import Resampling
import netCDF4 as nc
from huggingface_hub import hf_hub_download

# Channel indices in the 10-channel input tensor
# 0:DEM  1:HAND  2:ACC_log  3:TWI  4:CYGNSS_B  5:Mean_CYGNSS
# 6:Delta_CYGNSS  7:S1_flood_A  8:S1_flood_C  9:Mean_VV
CHANNEL_NAMES = [
    'DEM', 'HAND', 'ACC_log', 'TWI',
    'CYGNSS_B', 'Mean_CYGNSS', 'Delta_CYGNSS',
    'S1_flood_A', 'S1_flood_C', 'Mean_VV'
]

HF_REPO_ID  = "gauthiermalandrin/cygnss-sar-flood-mapping"
SAR_THRESHOLD_DB = -16.0  # dB threshold for Sentinel-1 VV open water detection
TILE_SIZE   = 256


# ── Data loading utilities ────────────────────────────────────────────────────

def load_sar_flood_mask(tif_path, bbox, target_shape, threshold_db=SAR_THRESHOLD_DB):
    """
    Load a Sentinel-1 GRD VV GeoTIFF and threshold it to a binary flood mask.

    Args:
        tif_path     : path to Sentinel-1 GeoTIFF (VV backscatter in dB)
        bbox         : (lon_min, lat_min, lon_max, lat_max)
        target_shape : (H, W) output shape in pixels
        threshold_db : backscatter threshold for water detection (default: -16 dB)

    Returns:
        flood_mask : np.ndarray of shape (H, W), dtype float32, values in {0, 1}
    """
    lon_min, lat_min, lon_max, lat_max = bbox
    with rasterio.open(tif_path) as src:
        window = from_bounds(lon_min, lat_min, lon_max, lat_max, src.transform)
        data   = src.read(1, window=window,
                          out_shape=target_shape,
                          resampling=Resampling.bilinear).astype(np.float32)
    # NaN → land (0)
    data = np.where(np.isnan(data), 1.0, data)  # NaN = no data → treat as land
    flood_mask = (data < threshold_db).astype(np.float32)
    return flood_mask


def load_cygnss_watermask(nc_path, bbox, target_shape):
    """
    Load a CYGNSS Berkeley-RWAWC watermask NetCDF file.

    Args:
        nc_path      : path to CYGNSS .nc file
        bbox         : (lon_min, lat_min, lon_max, lat_max)
        target_shape : (H, W) output shape in pixels

    Returns:
        watermask : np.ndarray of shape (H, W), dtype float32
                    values: 1=water, 0=land, NaN=no data (upsampled to target_shape)
    """
    lon_min, lat_min, lon_max, lat_max = bbox
    with nc.Dataset(nc_path) as ds:
        lats = np.array(ds.variables['lat'][:])
        lons = np.array(ds.variables['lon'][:])
        wm   = ds.variables['watermask'][:].data.astype(np.float32)

    lat_idx = np.where((lats >= lat_min) & (lats <= lat_max))[0]
    lon_idx = np.where((lons >= lon_min) & (lons <= lon_max))[0]
    patch   = wm[np.ix_(lat_idx, lon_idx)]

    # Replace nodata (-99) with NaN
    patch[patch == -99] = np.nan

    # Upsample to target resolution using nearest-neighbor
    from scipy.ndimage import zoom
    zoom_h = target_shape[0] / patch.shape[0]
    zoom_w = target_shape[1] / patch.shape[1]
    patch_up = zoom(patch, (zoom_h, zoom_w), order=0, prefilter=False)

    return patch_up.astype(np.float32)


def load_merit_hydro(merit_dir, bbox, target_shape):
    """
    Load MERIT Hydro topographic variables (DEM, HAND, ACC, TWI).

    Args:
        merit_dir    : directory containing merit GeoTIFF files:
                       dem.tif, hand.tif, acc.tif, twi.tif
        bbox         : (lon_min, lat_min, lon_max, lat_max)
        target_shape : (H, W) output shape in pixels

    Returns:
        dict with keys 'DEM', 'HAND', 'ACC_log', 'TWI',
        each an np.ndarray of shape (H, W), dtype float32
    """
    merit_dir = Path(merit_dir)
    files = {
        'DEM':     merit_dir / 'dem.tif',
        'HAND':    merit_dir / 'hand.tif',
        'ACC_log': merit_dir / 'acc.tif',
        'TWI':     merit_dir / 'twi.tif',
    }
    lon_min, lat_min, lon_max, lat_max = bbox

    result = {}
    for name, fpath in files.items():
        with rasterio.open(fpath) as src:
            window = from_bounds(lon_min, lat_min, lon_max, lat_max, src.transform)
            data   = src.read(1, window=window,
                              out_shape=target_shape,
                              resampling=Resampling.bilinear).astype(np.float32)
        if name == 'ACC_log':
            data = np.log1p(np.clip(data, 0, None))
        result[name] = data

    return result


# ── Normalisation ─────────────────────────────────────────────────────────────

def load_norm_stats(stats_path=None):
    """Load z-score normalization statistics."""
    if stats_path is None:
        stats_path = Path(__file__).parent.parent / 'model' / 'norm_stats.json'
    with open(stats_path) as f:
        stats = json.load(f)
    means = np.array(stats['means'], dtype=np.float32)
    stds  = np.array(stats['stds'],  dtype=np.float32)
    return means, stds


def normalize(tensor_10ch, means, stds):
    """Apply z-score normalization channel-wise."""
    return (tensor_10ch - means[:, None, None]) / stds[:, None, None]


# ── Main predictor class ──────────────────────────────────────────────────────

class FloodPredictor:
    """
    End-to-end flood predictor from raw satellite inputs.

    Example:
        predictor = FloodPredictor.from_pretrained()
        flood_map, prob_map = predictor.predict(
            sar_a="s1_2020-08-18.tif",
            cygnss_b="cygnss_2020-08-23.nc",
            sar_c="s1_2020-08-30.tif",
            merit_dir="merit_hydro/",
            region_bbox=(88.0, 22.0, 92.0, 25.0)
        )
    """

    def __init__(self, model, means, stds, device=None):
        self.model  = model
        self.means  = means
        self.stds   = stds
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def from_pretrained(cls, device=None):
        """
        Load model weights from HuggingFace Hub.
        Weights are downloaded automatically on first call and cached locally.
        """
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from model.unet import UNet

        # Download weights
        ckpt_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename="checkpoints/best_model_final_vv.pth"
        )
        stats_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename="model/norm_stats.json"
        )

        # Load model
        device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        model = UNet(in_channels=10, out_channels=1,
                     features=[32, 64, 128, 256], dropout=0.0)
        ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt.get('model_state_dict', ckpt))

        # Load norm stats
        means, stds = load_norm_stats(stats_path)

        return cls(model, means, stds, device)

    def predict(self, sar_a, cygnss_b, sar_c, merit_dir, region_bbox,
                mean_cygnss=None, mean_vv=None, threshold=0.5):
        """
        Predict a flood map from a temporal triplet.

        Args:
            sar_a        : path to Sentinel-1 GeoTIFF at date A (5-7 days before B)
            cygnss_b     : path to CYGNSS NetCDF file at target date B
            sar_c        : path to Sentinel-1 GeoTIFF at date C (5-7 days after B)
            merit_dir    : path to MERIT Hydro directory (dem.tif, hand.tif, acc.tif, twi.tif)
            region_bbox  : (lon_min, lat_min, lon_max, lat_max)
            mean_cygnss  : optional long-term mean CYGNSS (np.ndarray, same shape as tile)
                           if None, Delta_CYGNSS channel is set to zero
            mean_vv      : optional long-term mean VV backscatter (np.ndarray)
                           if None, Mean_VV channel is set to zero
            threshold    : binarization threshold (default: 0.5)

        Returns:
            flood_map : np.ndarray (H, W), binary flood mask {0, 1}
            prob_map  : np.ndarray (H, W), flood probability in [0, 1]
        """
        target_shape = (TILE_SIZE, TILE_SIZE)

        # Load inputs
        flood_a  = load_sar_flood_mask(sar_a,    region_bbox, target_shape)
        cygnss   = load_cygnss_watermask(cygnss_b, region_bbox, target_shape)
        flood_c  = load_sar_flood_mask(sar_c,    region_bbox, target_shape)
        merit    = load_merit_hydro(merit_dir,   region_bbox, target_shape)

        # Handle optional channels
        mean_cyg = mean_cygnss if mean_cygnss is not None else np.zeros(target_shape, dtype=np.float32)
        delta_cyg = cygnss - mean_cyg
        mvv       = mean_vv   if mean_vv      is not None else np.zeros(target_shape, dtype=np.float32)

        # Replace NaN in CYGNSS with 0 (training-set mean after z-score)
        cygnss_clean = np.where(np.isnan(cygnss), 0.0, cygnss)
        delta_clean  = np.where(np.isnan(delta_cyg), 0.0, delta_cyg)

        # Stack 10-channel tensor
        # Order: DEM(0) HAND(1) ACC_log(2) TWI(3) CYGNSS_B(4)
        #        Mean_CYGNSS(5) Delta_CYGNSS(6) S1_A(7) S1_C(8) Mean_VV(9)
        x = np.stack([
            merit['DEM'], merit['HAND'], merit['ACC_log'], merit['TWI'],
            cygnss_clean, mean_cyg, delta_clean,
            flood_a, flood_c, mvv
        ], axis=0).astype(np.float32)  # (10, H, W)

        # Normalize
        x = normalize(x, self.means, self.stds)

        # Inference
        with torch.no_grad():
            inp  = torch.from_numpy(x).unsqueeze(0).to(self.device)
            prob = torch.sigmoid(self.model(inp)).squeeze().cpu().numpy()

        flood_map = (prob > threshold).astype(np.uint8)
        return flood_map, prob