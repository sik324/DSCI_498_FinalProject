=======================================================
Hurricane Catastrophe Modeling with cGAN Super-Resolution
CSC-498 Final Project | Lehigh University | Spring 2026
=======================================================

DESCRIPTION
-----------
This project applies conditional Generative Adversarial
Networks (cGAN) to hurricane catastrophe modeling for
insurance loss estimation. We implement a complete
four-module catastrophe model (Hazard, Exposure,
Vulnerability, Loss) enhanced with a Pix2Pix cGAN
that performs 10x wind field super-resolution
(0.05 degrees to 0.005 degrees).

Case study : Hurricane Ian (2022), Lee County, Florida
Key result : cGAN identifies $242M in additional losses
             missed by the coarse Holland model

MODULES
-------
Module 1 - Hazard:
  Holland (1980) parametric wind field model
  IBTrACS v04r00 storm track data
  Peak gust: 163 mph (raster) / 156 mph (tract max)

Module 2 - Exposure:
  Census ACS 2022 housing units
  HAZUS MH v4.0 building type distribution
  311,512 buildings / $50.23B TIV / 221 land tracts

Module 3 - Vulnerability:
  HAZUS lognormal fragility curves
  5 building types: W1, MH, M1, C1, S1
  Mean MDR: Holland 57.9% / cGAN 58.6%

Module 4 - Loss:
  Expected loss = TIV x MDR
  Holland: $18.620B / cGAN: $18.863B
  Difference: $242.3M (+1.30%)

cGAN Architecture (Pix2Pix framework):
  Generator    : U-Net with skip connections (3.9M params)
  Discriminator: PatchGAN 23x23 patch scores (2.76M params)
  Training     : 100 epochs / Best epoch: 61 / Val loss: 0.0050
  Correlation  : 0.9965 vs Holland fine reference
  Training data: 2,500 balanced synthetic storm pairs
                 500 per Cat1-Cat5

DATA SOURCES
------------
1. IBTrACS v04r00 - NOAA storm tracks
   https://www.ncei.noaa.gov/products/international-best-track-archive

2. Census TIGER 2022 - Lee County census tracts
   https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html

3. HAZUS MH v4.0 - Building inventory + fragility parameters
   https://www.fema.gov/flood-maps/tools-resources/flood-map-products/hazus

See data/readme_data.txt for detailed download instructions.

PACKAGES REQUIRED
-----------------
Python 3.9+

Install all packages:
  pip install -r requirements.txt

Or install individually:
  pip install numpy pandas scipy
  pip install geopandas rasterio shapely fiona pyproj
  pip install torch torchvision
  pip install matplotlib requests

For GPU support (recommended):
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

FILE STRUCTURE
--------------
HurricaneCatModelcGAN/
├── data/
│   └── readme_data.txt      data download instructions
├── ReadMe.txt               this file
├── main.py                  main pipeline runner
├── wind_field.py            Holland wind model (Module 1)
├── gust_grid.py             grid computation utilities
├── ibtracs_ingest.py        IBTrACS data processing
├── land_mask.py             land/water masking
├── exposure.py              HAZUS exposure module (Module 2)
├── generator.py             U-Net Generator (cGAN)
├── discriminator.py         PatchGAN Discriminator (cGAN)
├── train.py                 cGAN training loop
├── dataset.py               WindDataset PyTorch class
├── storm_generator.py       synthetic storm generator v4
└── vulnerability.py         HAZUS fragility + loss (Module 3+4)

HOW TO RUN
----------
Step 1 - Download data (see data/readme_data.txt):
  Place IBTrACS CSV in:    data/raw/ibtracs/
  Place TIGER shapefiles:  data/raw/tiger/

Step 2 - Install packages:
  pip install -r requirements.txt

Step 3 - Run complete pipeline:
  python main.py

  This will:
  a) Process IBTrACS tracks (Module 1)
  b) Compute Holland wind field (Module 1)
  c) Build HAZUS exposure inventory (Module 2)
  d) Generate 2,500 balanced synthetic training pairs
  e) Train cGAN for 100 epochs
  f) Generate high-resolution cGAN wind field
  g) Compute HAZUS vulnerability and loss (Module 3+4)
  h) Compare Holland vs cGAN loss estimates
  i) Save all outputs to outputs/ folder

Step 4 - Run individual modules:
  python main.py --module hazard      Module 1 only
  python main.py --module exposure    Module 2 only
  python main.py --module cgan        cGAN training only
  python main.py --module loss        Loss computation only

Step 5 - Use pre-trained model (skip training):
  python main.py --use-pretrained
  Requires: outputs/cgan/generator_balanced_best.pth

NOTE ON TRAINING TIME
---------------------
CPU: approximately 6-8 hours for 100 epochs
GPU: approximately 45 minutes for 100 epochs

To use pre-trained weights (recommended):
  Download generator_balanced_best.pth from Google Drive
  Place in: outputs/cgan/generator_balanced_best.pth
  Then run: python main.py --use-pretrained

KEY RESULTS
-----------
  Correlation        : 0.9965
  MAE                : 1.48 mph (baseline: 1.51 mph)
  Peak wind          : 158.6 mph (Holland: 158.7 mph)
  Tracts reclassified: +15 tracts to 150-160 mph bin
  Holland total loss : $18.620B
  cGAN total loss    : $18.863B
  Difference         : +$242.3M (+1.30%)

STREAMLIT
----------

For Streamlit web app only:
  pip install -r requirements.txt

For full pipeline (Holland + cGAN training):
  pip install -r requirements_full.txt

REFERENCES
----------
Holland (1980) MWR | Isola et al. (2017) CVPR (Pix2Pix)
Ronneberger et al. (2015) MICCAI (U-Net)
Stengel et al. (2020) PNAS | FEMA HAZUS (2012)
Knapp et al. (2010) BAMS | NHC Ian Report (2023)

CONTACT
-------
Lehigh University | CSC-498 | Spring 2026
