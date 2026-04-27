=======================================================
DATA SOURCES — Hurricane CatModel cGAN
CSC-498 Final Project | Lehigh University | Spring 2026
=======================================================

This project requires 3 external datasets.
Download each dataset and place in the correct folder
as described below.

-------------------------------------------------------
DATASET 1 — IBTrACS Storm Track Data
-------------------------------------------------------
Source  : NOAA National Centers for Environmental Information
URL     : https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r00/access/csv/
File    : ibtracs.NA.list.v04r00.csv
Size    : ~55 MB

Download steps:
1. Go to the URL above
2. Download ibtracs.NA.list.v04r00.csv
3. Place file here:
   data/raw/ibtracs/ibtracs.NA.list.v04r00.csv

Citation:
Knapp, K.R. et al. (2010). The International Best Track
Archive for Climate Stewardship (IBTrACS).
Bulletin of the American Meteorological Society, 91(3), 363-376.

-------------------------------------------------------
DATASET 2 — US Census TIGER Shapefiles
-------------------------------------------------------
Source  : US Census Bureau TIGER/Line Shapefiles
URL     : https://www2.census.gov/geo/tiger/TIGER2022/TRACT/
File    : tl_2022_12_tract.zip (Florida census tracts)
Size    : ~22 MB

Download steps:
1. Go to URL above
2. Download tl_2022_12_tract.zip
3. Unzip the file
4. Place all files (.shp, .dbf, .shx, .prj) here:
   data/raw/tiger/

Also download county shapefile:
URL  : https://www2.census.gov/geo/tiger/TIGER2022/COUNTY/
File : tl_2022_us_county.zip
Place in: data/raw/tiger/

-------------------------------------------------------
DATASET 3 — Census ACS Housing Units
-------------------------------------------------------
Source  : US Census Bureau American Community Survey 2022
URL     : https://data.census.gov
Table   : B25001 (Housing Units)
FIPS    : 12071 (Lee County, Florida)

Note: The exposure module fetches this automatically
using the HAZUS published parameters. No manual
download required — the module uses pre-defined
HAZUS MBT distributions:
  W1: 52%, MH: 30%, M1: 10%, C1: 5%, S1: 3%
Total units: 311,522 (Lee County 2022)

-------------------------------------------------------
FOLDER STRUCTURE AFTER DOWNLOAD
-------------------------------------------------------
data/
├── raw/
│   ├── ibtracs/
│   │   └── ibtracs.NA.list.v04r00.csv
│   └── tiger/
│       ├── tl_2022_12_tract.shp
│       ├── tl_2022_12_tract.dbf
│       ├── tl_2022_12_tract.shx
│       ├── tl_2022_12_tract.prj
│       ├── tl_2022_us_county.shp
│       ├── tl_2022_us_county.dbf
│       ├── tl_2022_us_county.shx
│       └── tl_2022_us_county.prj
└── readme_data.txt

-------------------------------------------------------
NOTE ON SYNTHETIC TRAINING DATA
-------------------------------------------------------
The cGAN training data (2,500 synthetic wind field
pairs) is generated automatically by main.py using
the Holland parametric wind model. No additional
download is required for training data.

Generated files are saved to:
data/processed/balanced_data/
  X_balanced.npy  — coarse wind fields (2500, 22, 21)
  Y_balanced.npy  — fine wind fields   (2500, 201, 201)
  C_balanced.npy  — condition vectors  (2500, 4)
  X_max_bal.npy   — normalization factor

Generation takes approximately 15-20 minutes.
Pre-generated data available on Google Drive:
https://drive.google.com/[YOUR_DRIVE_LINK]

-------------------------------------------------------
HARDWARE REQUIREMENTS
-------------------------------------------------------
Minimum:
  RAM    : 16 GB
  Storage: 5 GB free
  GPU    : Optional (CPU training supported)

Recommended:
  GPU    : NVIDIA with CUDA support
  RAM    : 32 GB
  Storage: 10 GB free

Training time:
  CPU: ~6-8 hours for 100 epochs
  GPU: ~45 minutes for 100 epochs
