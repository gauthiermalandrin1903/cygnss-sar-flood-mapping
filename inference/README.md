# Inference scripts

## `predict.py` — Deterministic flood mapping

End-to-end inference from raw satellite inputs to a 90 m flood probability map.

```python
from inference.predict import FloodPredictor

predictor = FloodPredictor.from_pretrained()

flood_map, prob_map = predictor.predict(
    sar_a="path/to/s1_A.tif",        # Sentinel-1 GRD VV at date A
    cygnss_b="path/to/cygnss_B.nc",  # CYGNSS Berkeley-RWAWC at date B
    sar_c="path/to/s1_C.tif",        # Sentinel-1 GRD VV at date C
    merit_dir="path/to/merit/",       # directory with dem.tif, hand.tif, acc.tif, twi.tif
    region_bbox=(lon_min, lat_min, lon_max, lat_max)
)
# flood_map : np.ndarray (256, 256), binary {0, 1}
# prob_map  : np.ndarray (256, 256), float in [0, 1]
```

### Input requirements

| Input | Format | Notes |
|-------|--------|-------|
| `sar_a`, `sar_c` | GeoTIFF (.tif) | Sentinel-1 GRD VV backscatter in dB, 5–7 days before/after B |
| `cygnss_b` | NetCDF (.nc) | Berkeley-RWAWC watermask, available at [HydroShare](https://www.hydroshare.org/) |
| `merit_dir` | directory | Must contain `dem.tif`, `hand.tif`, `acc.tif`, `twi.tif` at 90 m resolution |
| `region_bbox` | tuple | `(lon_min, lat_min, lon_max, lat_max)` in WGS84 decimal degrees |

### Optional inputs

```python
flood_map, prob_map = predictor.predict(
    ...,
    mean_cygnss=mean_cygnss_array,  # long-term mean CYGNSS (np.ndarray, same shape)
                                     # if None, Delta_CYGNSS channel is set to zero
    mean_vv=mean_vv_array,          # long-term mean VV backscatter (np.ndarray)
                                     # if None, Mean_VV channel is set to zero
    threshold=0.5                   # binarization threshold (default: 0.5)
)
```

Providing `mean_cygnss` enables the climatological anomaly channel
(Delta_CYGNSS = CYGNSS_B − Mean_CYGNSS), which is the primary driver of
CYGNSS contribution according to our ablation study.

---

## `predict_ensemble.py` — Probabilistic flood mapping with uncertainty

Extends the deterministic predictor with a conditional diffusion model
ensemble, producing spatially explicit uncertainty estimates alongside
the flood prediction.

```python
from inference.predict_ensemble import EnsemblePredictor

predictor = EnsemblePredictor.from_pretrained()

flood_map, uncertainty = predictor.predict(
    sar_a="path/to/s1_A.tif",
    cygnss_b="path/to/cygnss_B.nc",
    sar_c="path/to/s1_C.tif",
    merit_dir="path/to/merit/",
    region_bbox=(lon_min, lat_min, lon_max, lat_max),
    n_samples=20   # number of diffusion samples (default: 20, use 5 for speed)
)
# flood_map   : np.ndarray (256, 256), binary {0, 1} from ensemble mean
# uncertainty : np.ndarray (256, 256), pixel-wise std across samples
#               high values = ambiguous flood boundaries
```

### When to use ensemble vs deterministic

| Use case | Recommended |
|----------|-------------|
| Fast operational mapping | `FloodPredictor` (deterministic) |
| Risk assessment, decision support | `EnsemblePredictor` (ensemble) |
| Flagging uncertain pixels for review | `EnsemblePredictor` (ensemble) |

### Computational cost

| Model | Device | Time per tile (256×256) |
|-------|--------|------------------------|
| U-Net (deterministic) | CPU | ~0.5 s |
| U-Net (deterministic) | GPU | ~0.05 s |
| Diffusion ensemble (N=20, S=50) | CPU | ~5 min |
| Diffusion ensemble (N=20, S=50) | GPU | ~15 s |

---

## Pre-processed tile inference

If you have a pre-processed `.npz` tile (same format as the training data),
you can run inference directly without loading raw satellite files:

```python
import numpy as np, torch
from model.unet import UNet
from inference.predict import load_norm_stats, normalize

d = np.load("data/example/tile_08695.npz")
x = d['input'].astype('float32')

means, stds = load_norm_stats()
x_norm = normalize(x, means, stds)

model = UNet(in_channels=10, out_channels=1, features=[32,64,128,256], dropout=0.0)
# load weights...

with torch.no_grad():
    prob = torch.sigmoid(model(torch.from_numpy(x_norm).unsqueeze(0))).squeeze().numpy()
```