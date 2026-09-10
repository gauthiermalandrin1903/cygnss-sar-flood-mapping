# Example tile

`tile_08695.npz` — Bangladesh, 2020-08-23

Pre-processed 256×256 tile at 90 m resolution containing all 10 input
channels required by the model. Use this tile to test the model without
downloading raw satellite data (see `example.py` in the root directory).

## Contents

| Key | Shape | dtype | Description |
|-----|-------|-------|-------------|
| `input` | (10, 256, 256) | float32 | Normalized 10-channel input tensor |
| `target` | (256, 256) | float32 | Sentinel-1 binary flood mask (ground truth) |
| `v10_pred` | (256, 256) | float32 | U-Net flood probability (required by diffusion model) |

## Channel order (`input` tensor)

| Index | Name | Description |
|-------|------|-------------|
| 0 | DEM | Digital Elevation Model (MERIT Hydro) |
| 1 | HAND | Height Above Nearest Drainage (MERIT Hydro) |
| 2 | ACC_log | Log-transformed upstream flow accumulation (MERIT Hydro) |
| 3 | TWI | Topographic Wetness Index (MERIT Hydro) |
| 4 | CYGNSS_B | CYGNSS Berkeley-RWAWC watermask at target date B |
| 5 | Mean_CYGNSS | Long-term regional mean CYGNSS watermask |
| 6 | Delta_CYGNSS | CYGNSS anomaly (CYGNSS_B − Mean_CYGNSS) |
| 7 | S1_flood_A | Sentinel-1 binary flood mask at date A (5–7 days before B) |
| 8 | S1_flood_C | Sentinel-1 binary flood mask at date C (5–7 days after B) |
| 9 | Mean_VV | Long-term mean Sentinel-1 VV backscatter |

All channels are z-score normalized using statistics computed over the
training set (see `model/norm_stats.json`).

## Geographic context

- **Region:** Bangladesh (Brahmaputra-Jamuna floodplain)
- **Date B (target):** 2020-08-23
- **Date A (SAR anchor):** 2020-08-18
- **Date C (SAR anchor):** 2020-08-30
- **Bounding box:** ~22.7°N–22.9°N, 90.5°E–90.7°E
- **Resolution:** 90 m (256×256 pixels ≈ 23×23 km)