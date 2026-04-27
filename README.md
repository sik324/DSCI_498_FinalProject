# 🌀 Hurricane Catastrophe Modeling with cGAN Super-Resolution

**CAT-402 + CSC-498 Final Project | Lehigh University | Spring 2026**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

---

## Project Description

This project applies **Conditional Generative Adversarial Networks (cGAN)** to hurricane catastrophe modeling for insurance loss estimation. We implement a complete four-module catastrophe model (Hazard, Exposure, Vulnerability, Loss) enhanced with a **Pix2Pix cGAN** that performs **10× wind field super-resolution** (0.05° → 0.005°).

- **Case study:** Hurricane Ian (2022), Lee County, Florida (FIPS 12071)
- **Key result:** cGAN identifies **$242M in additional losses** missed by the coarse Holland model

---

## Key Results

| Metric | Value |
|--------|-------|
| cGAN Correlation | **0.9965** vs Holland fine reference |
| MAE | **1.48 mph** (baseline bilinear: 1.51 mph) |
| Peak wind | **158.6 mph** (Holland: 158.7 mph) |
| Tracts reclassified | **+15 tracts** moved to 150-160 mph bin |
| TIV reclassified | **+$431M** to highest wind zone |
| Holland total loss | **$18.620B** |
| cGAN total loss | **$18.863B** |
| Difference | **+$242.3M (+1.30%)** |

---

## Project Modules

### Module 1 — Hazard
- Holland (1980) parametric wind field model
- IBTrACS v04r00 storm track data (317 Gulf Coast storms)
- Peak gust: 163 mph (raster) / 156 mph (tract max)

### Module 2 — Exposure
- Census ACS 2022 housing units
- HAZUS MH v4.0 building type distribution
- 311,512 buildings / $50.23B TIV / 221 land tracts

### Module 3 — Vulnerability
- HAZUS lognormal fragility curves
- 5 building types: W1, MH, M1, C1, S1
- Mean MDR: Holland 57.9% / cGAN 58.6%

### Module 4 — Loss
- Expected loss = TIV × MDR
- Holland: $18.620B / cGAN: $18.863B
- Difference: $242.3M (+1.30%)

---

## cGAN Architecture (Pix2Pix Framework)

```
Input:  Holland coarse wind field (22×21) + condition (Vmax, RMW, Pmin, lat)
Output: High-resolution wind field (201×201)
```

| Component | Details |
|-----------|---------|
| Generator | U-Net with skip connections — 3.9M parameters |
| Discriminator | PatchGAN 23×23 patch scores — 2.76M parameters |
| Loss | BCEWithLogitsLoss + L1×100 |
| Optimizer | Adam lr=2×10⁻⁵, β₁=0.5, gradient clipping=1.0 |
| Training | 100 epochs / GPU / Best epoch: 61 |
| Val loss | 0.0050 (95% reduction from 0.1546) |
| Training data | 2,500 balanced synthetic pairs (500 per Cat1-5) |

---

## Data Sources

1. **IBTrACS v04r00** — NOAA best-track storm data  
   https://www.ncei.noaa.gov/products/international-best-track-archive

2. **Census TIGER 2022** — Lee County census tracts  
   https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html

3. **HAZUS MH v4.0** — Building inventory + fragility parameters  
   https://www.fema.gov/flood-maps/tools-resources/flood-map-products/hazus

See `data/readme_data.txt` for detailed download instructions.

---

## File Structure

```
DSCI_498_FinalProject/
├── data/
│   └── readme_data.txt      # Data download instructions
├── app.py                   # Streamlit web app
├── main.py                  # Main pipeline runner
├── wind_field.py            # Holland wind model (Module 1)
├── gust_grid.py             # Grid computation utilities
├── ibtracs_ingest.py        # IBTrACS data processing
├── land_mask.py             # Land/water masking
├── exposure.py              # HAZUS exposure module (Module 2)
├── generator.py             # U-Net Generator (cGAN)
├── discriminator.py         # PatchGAN Discriminator (cGAN)
├── train.py                 # cGAN training loop
├── dataset.py               # WindDataset PyTorch class
├── storm_generator.py       # Synthetic storm generator v4
├── vulnerability.py         # HAZUS fragility + loss (Module 3+4)
├── requirements.txt         # Python packages
└── README.md                # This file
```

---

## Packages Required

```bash
pip install -r requirements.txt
```

Key packages:
```
numpy pandas scipy
geopandas rasterio shapely
torch torchvision
matplotlib streamlit plotly
```

For GPU support:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## How to Run

### Full pipeline:
```bash
python main.py
```

### Individual modules:
```bash
python main.py --module hazard      # Module 1 — wind field
python main.py --module exposure    # Module 2 — building inventory
python main.py --module cgan        # cGAN training
python main.py --module loss        # Loss computation
```

### Use pre-trained model (skip training):
```bash
python main.py --use-pretrained
```

### Launch Streamlit app:
```bash
streamlit run app.py
```

---

## Training Notes

| Hardware | Training time (100 epochs) |
|----------|---------------------------|
| CPU | ~6-8 hours |
| GPU (CUDA) | ~45 minutes |

**Balanced training strategy:**  
Historical Gulf storms are imbalanced (52% weak, <14% Cat 3+). We generate 2,500 synthetic storms using the Holland model — 500 per category (Cat1-Cat5) — ensuring the cGAN learns extreme Cat4-5 wind patterns. This improved peak wind prediction from 138 mph → 158.6 mph.

---

## References

- Holland, G.J. (1980). Monthly Weather Review, 108(8), 1212-1218.
- Willoughby & Rahn (2004). Monthly Weather Review, 132(12), 3033-3048.
- Isola et al. (2017). Pix2Pix. CVPR 2017.
- Ronneberger et al. (2015). U-Net. MICCAI.
- Stengel et al. (2020). Adversarial super-resolution of wind data. PNAS, 117(29).
- FEMA (2012). HAZUS-MH Hurricane Model Technical Manual.
- Knapp et al. (2010). IBTrACS. BAMS, 91(3).
- NHC (2023). Tropical Cyclone Report: Hurricane Ian.

---

## Contact

Lehigh University | CSC-498 | Spring 2026
