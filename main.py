"""
Hurricane Catastrophe Modeling with cGAN Super-Resolution
CSC-498 Final Project | Lehigh University | Spring 2026

Main pipeline runner — runs all 4 modules + cGAN
Usage:
  python main.py                    # Full pipeline
  python main.py --module hazard    # Module 1 only
  python main.py --module exposure  # Module 2 only
  python main.py --module cgan      # cGAN training
  python main.py --module loss      # Loss computation
  python main.py --use-pretrained   # Skip training
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(BASE_DIR, 'data')
RAW_DIR      = os.path.join(DATA_DIR, 'raw')
PROC_DIR     = os.path.join(DATA_DIR, 'processed')
OUT_DIR      = os.path.join(BASE_DIR, 'outputs')
HAZARD_DIR   = os.path.join(OUT_DIR,  'hazard')
EXPOSURE_DIR = os.path.join(OUT_DIR,  'exposure')
CGAN_DIR     = os.path.join(OUT_DIR,  'cgan')

# Create output directories
for d in [HAZARD_DIR, EXPOSURE_DIR, CGAN_DIR,
          os.path.join(PROC_DIR, 'tracks'),
          os.path.join(PROC_DIR, 'balanced_data')]:
    os.makedirs(d, exist_ok=True)

# Add hazard module to path
sys.path.insert(0, os.path.join(BASE_DIR, 'hazard'))
sys.path.insert(0, os.path.join(BASE_DIR, 'exposure'))
sys.path.insert(0, os.path.join(BASE_DIR, 'cgan'))
sys.path.insert(0, os.path.join(BASE_DIR, 'loss'))


def run_hazard():
    """Module 1 — Holland wind field computation"""
    print("\n" + "="*55)
    print("MODULE 1 — HAZARD")
    print("="*55)

    from ibtracs_ingest import (load_ibtracs, _best_wind_pres,
                                 filter_gulf_coast, _compute_forward_motion)
    from wind_field import compute_track_peak_gust
    from gust_grid import build_grid, save_geotiff

    ibtracs_path = os.path.join(RAW_DIR, 'ibtracs',
                                'ibtracs.NA.list.v04r00.csv')
    if not os.path.exists(ibtracs_path):
        raise FileNotFoundError(
            f"IBTrACS not found at {ibtracs_path}\n"
            "See data/readme_data.txt for download instructions."
        )

    print("Loading IBTrACS data...")
    df = load_ibtracs(ibtracs_path)
    df['ISO_TIME'] = pd.to_datetime(df['ISO_TIME'], errors='coerce')
    df = _best_wind_pres(df)
    df = filter_gulf_coast(df)
    df = (df.sort_values(['SID', 'ISO_TIME'])
            .groupby('SID', group_keys=False)
            .apply(_compute_forward_motion))
    df = df.dropna(subset=['LAT', 'LON', 'VMAX_kt'])

    keep = ['SID', 'NAME', 'SEASON', 'ISO_TIME', 'LAT', 'LON',
            'VMAX_kt', 'PMIN_mb', 'RMW_nmile', 'PENV_mb',
            'STORM_SPEED_kt', 'STORM_DIR_deg', 'USA_SSHS']
    df = df[[c for c in keep if c in df.columns]].reset_index(drop=True)

    tracks_path = os.path.join(PROC_DIR, 'tracks',
                               'gulf_tracks_processed.csv')
    df.to_csv(tracks_path, index=False)
    print(f"Gulf tracks saved: {len(df):,} records, "
          f"{df['SID'].nunique()} storms")

    # Ian 2022
    ian_2022 = df[(df['NAME'].str.upper() == 'IAN') &
                  (df['SEASON'] == 2022)].copy()
    print(f"Ian 2022: {len(ian_2022)} records, "
          f"max Vmax={ian_2022['VMAX_kt'].max():.0f} kt")

    # Florida domain wind field
    FLORIDA_DOMAIN = {
        'lat_min': 24.0, 'lat_max': 31.5,
        'lon_min': -87.0, 'lon_max': -79.0
    }
    lat_full, lon_full = build_grid(FLORIDA_DOMAIN, res_deg=0.02)
    print("Computing Florida wind field...")
    peak_gust_full = compute_track_peak_gust(ian_2022, lat_full, lon_full)
    print(f"Peak gust: {peak_gust_full.max()*2.237:.1f} mph")

    save_geotiff(peak_gust_full * 2.237, lat_full, lon_full,
                 os.path.join(HAZARD_DIR, 'ian_florida_gust_full.tif'))

    # Lee County domain
    LEE_DOMAIN = {
        'lat_min': 25.8, 'lat_max': 26.8,
        'lon_min': -82.3, 'lon_max': -81.3
    }
    lat_lee, lon_lee = build_grid(LEE_DOMAIN, res_deg=0.05)
    peak_gust_lee = compute_track_peak_gust(ian_2022, lat_lee, lon_lee)

    save_geotiff(peak_gust_lee * 2.237, lat_lee, lon_lee,
                 os.path.join(HAZARD_DIR, 'ian_lee_county_gust.tif'))

    print("Hazard module complete!")
    return df, ian_2022, peak_gust_full, lat_full, lon_full


def run_exposure(ian_2022, peak_gust_full, lat_full, lon_full):
    """Module 2 — HAZUS building exposure"""
    print("\n" + "="*55)
    print("MODULE 2 — EXPOSURE")
    print("="*55)

    from exposure import build_exposure

    tiger_dir = os.path.join(RAW_DIR, 'tiger')
    exposure_df, tracts = build_exposure(
        state_fips='12', county_fips='071',
        peak_gust=peak_gust_full,
        lat_grid=lat_full, lon_grid=lon_full,
        data_dir=tiger_dir,
        out_dir=EXPOSURE_DIR
    )

    total_tiv  = exposure_df['TIV_total'].sum()
    total_bldg = exposure_df.groupby('tract_id')['n_buildings'].sum().sum()
    print(f"Total buildings: {total_bldg:,}")
    print(f"Total TIV      : ${total_tiv/1e9:.2f}B")
    print("Exposure module complete!")
    return exposure_df, tracts


def run_cgan(tracks, use_pretrained=False):
    """CSC-498 — cGAN super-resolution"""
    print("\n" + "="*55)
    print("cGAN — WIND FIELD SUPER-RESOLUTION")
    print("="*55)

    import torch
    from generator import Generator
    from discriminator import Discriminator
    from train import train_cgan
    from storm_generator import generate_balanced_data
    from dataset import WindDataset
    from torch.utils.data import DataLoader

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    from gust_grid import build_grid
    LEE_DOMAIN = {
        'lat_min': 25.8, 'lat_max': 26.8,
        'lon_min': -82.3, 'lon_max': -81.3
    }
    lat_coarse, lon_coarse = build_grid(LEE_DOMAIN, res_deg=0.05)
    lat_fine,   lon_fine   = build_grid(LEE_DOMAIN, res_deg=0.005)

    # Load or generate training data
    bal_dir   = os.path.join(PROC_DIR, 'balanced_data')
    x_path    = os.path.join(bal_dir, 'X_balanced.npy')
    gen_path  = os.path.join(CGAN_DIR, 'generator_balanced_best.pth')

    if os.path.exists(x_path):
        print("Loading saved balanced training data...")
        X = np.load(os.path.join(bal_dir, 'X_balanced.npy'))
        Y = np.load(os.path.join(bal_dir, 'Y_balanced.npy'))
        C = np.load(os.path.join(bal_dir, 'C_balanced.npy'))
        X_max = float(np.load(os.path.join(bal_dir, 'X_max_bal.npy')))
    else:
        print("Generating 2,500 balanced synthetic training pairs...")
        from wind_field import compute_track_peak_gust
        X, Y, C, X_max = generate_balanced_data(
            tracks, lat_coarse, lon_coarse,
            lat_fine, lon_fine,
            n_per_cat=500, save_dir=bal_dir
        )

    print(f"Training pairs: {len(X)}")
    print(f"X_max: {X_max:.4f} m/s = {X_max*2.237:.1f} mph")

    # Build model
    G = Generator(condition_dim=4).to(device)
    D = Discriminator().to(device)

    if use_pretrained and os.path.exists(gen_path):
        print(f"Loading pre-trained model from {gen_path}")
        G.load_state_dict(torch.load(gen_path, map_location=device))
    else:
        # Train
        G, history = train_cgan(
            G, D, X, Y, C, X_max,
            device=device,
            n_epochs=100,
            batch_size=16,
            save_dir=CGAN_DIR
        )
        pd.DataFrame(history).to_csv(
            os.path.join(CGAN_DIR, 'training_loss_balanced.csv'),
            index=False
        )
        print("Training complete!")

    print("cGAN module complete!")
    return G, X_max, lat_coarse, lon_coarse, lat_fine, lon_fine, device


def run_loss(exposure_df, tracts, G, X_max,
             ian_2022, lat_coarse, lon_coarse,
             lat_fine, lon_fine, device):
    """Module 3+4 — Vulnerability and Loss"""
    print("\n" + "="*55)
    print("MODULE 3+4 — VULNERABILITY AND LOSS")
    print("="*55)

    import torch
    import rasterio
    from rasterio.transform import from_bounds
    from wind_field import compute_track_peak_gust
    from vulnerability import compute_mdr, FRAGILITY

    # Generate cGAN high-res wind field
    ian_coarse = compute_track_peak_gust(
        ian_2022, lat_coarse, lon_coarse)
    ian_fine   = compute_track_peak_gust(
        ian_2022, lat_fine, lon_fine)

    peak_row = ian_2022.loc[ian_2022['VMAX_kt'].idxmax()]
    ian_cond = np.array([[
        float(peak_row['VMAX_kt'])   / 150.0,
        float(peak_row['RMW_nmile']) / 100.0
        if pd.notna(peak_row['RMW_nmile']) else 25.0/100.0,
        float(peak_row['PMIN_mb'])   / 1013.0
        if pd.notna(peak_row['PMIN_mb'])  else 980.0/1013.0,
        float(abs(peak_row['LAT']))  / 35.0,
    ]])

    G.eval()
    with torch.no_grad():
        inp  = torch.FloatTensor(ian_coarse/X_max).unsqueeze(0).unsqueeze(0).to(device)
        cond = torch.FloatTensor(ian_cond).to(device)
        out  = G(inp, cond)

    ian_highres_mph = out.cpu().numpy()[0,0] * X_max * 2.237
    ian_coarse_mph  = ian_coarse * 2.237
    ian_fine_mph    = ian_fine   * 2.237

    print(f"Holland coarse peak : {ian_coarse_mph.max():.1f} mph")
    print(f"Holland fine peak   : {ian_fine_mph.max():.1f} mph")
    print(f"cGAN output peak    : {ian_highres_mph.max():.1f} mph")

    # Save GeoTIFFs
    def save_tiff(array, lat_grid, lon_grid, path):
        nrows, ncols = array.shape
        transform = from_bounds(
            lon_grid.min(), lat_grid.min(),
            lon_grid.max(), lat_grid.max(),
            ncols, nrows
        )
        with rasterio.open(
            path, 'w', driver='GTiff', dtype='float32',
            width=ncols, height=nrows, count=1,
            crs='EPSG:4326', transform=transform,
            nodata=-9999, compress='lzw'
        ) as dst:
            dst.write(
                np.flipud(array).astype(np.float32)[np.newaxis,:,:]
            )

    save_tiff(ian_coarse_mph, lat_coarse, lon_coarse,
              os.path.join(CGAN_DIR, 'ian_lee_coarse_input.tif'))
    save_tiff(ian_highres_mph, lat_fine, lon_fine,
              os.path.join(CGAN_DIR, 'ian_lee_cgan_highres.tif'))
    save_tiff(ian_fine_mph, lat_fine, lon_fine,
              os.path.join(CGAN_DIR, 'ian_lee_holland_fine.tif'))

    # Sample cGAN wind at tract centroids
    with rasterio.open(
        os.path.join(CGAN_DIR, 'ian_lee_cgan_highres.tif')
    ) as src:
        cgan_data      = src.read(1)
        cgan_transform = src.transform

    cgan_wind = {}
    for _, row in tracts.iterrows():
        centroid = row.geometry.centroid
        r_px, c_px = rasterio.transform.rowcol(
            cgan_transform, centroid.x, centroid.y)
        if (0 <= r_px < cgan_data.shape[0] and
                0 <= c_px < cgan_data.shape[1]):
            w = float(cgan_data[r_px, c_px])
            cgan_wind[row['tract_id']] = w if w > 0 else np.nan
        else:
            cgan_wind[row['tract_id']] = np.nan

    exposure_df = exposure_df.copy()
    exposure_df['tract_id'] = exposure_df['tract_id'].astype(str)
    tracts_wind = tracts[['tract_id', 'peak_gust_mph']].copy()
    tracts_wind['tract_id'] = tracts_wind['tract_id'].astype(str)
    tracts_wind['cgan_gust_mph'] = tracts_wind['tract_id'].map(cgan_wind)
    tracts_wind['cgan_gust_mph'] = tracts_wind['cgan_gust_mph'].fillna(
        tracts_wind['peak_gust_mph'])

    exposure_df = exposure_df.drop(
        columns=['peak_gust_mph'], errors='ignore')
    exposure_df = exposure_df.merge(
        tracts_wind, on='tract_id', how='left')

    # Compute losses
    rows = []
    for _, row in exposure_df.iterrows():
        mbt      = row['MBT']
        mdr_hol  = compute_mdr(float(row['peak_gust_mph']), mbt)
        mdr_cgan = compute_mdr(float(row['cgan_gust_mph']), mbt)
        rows.append({
            'tract_id':  str(row['tract_id']),
            'MBT':       mbt,
            'TIV_total': row['TIV_total'],
            'wind_hol':  row['peak_gust_mph'],
            'wind_cgan': row['cgan_gust_mph'],
            'loss_hol':  row['TIV_total'] * mdr_hol,
            'loss_cgan': row['TIV_total'] * mdr_cgan,
        })

    loss_df    = pd.DataFrame(rows)
    total_hol  = loss_df['loss_hol'].sum()
    total_cgan = loss_df['loss_cgan'].sum()
    diff       = total_cgan - total_hol

    loss_df.to_csv(
        os.path.join(OUT_DIR, 'loss_comparison.csv'), index=False)

    print(f"\n{'='*55}")
    print("FINAL RESULTS")
    print(f"{'='*55}")
    print(f"Holland total loss : ${total_hol/1e9:.3f}B")
    print(f"cGAN    total loss : ${total_cgan/1e9:.3f}B")
    print(f"Difference         : ${diff/1e6:.1f}M "
          f"(+{diff/total_hol*100:.2f}%)")
    print("\nLoss module complete!")
    return loss_df


def main():
    parser = argparse.ArgumentParser(
        description='Hurricane CatModel cGAN Pipeline'
    )
    parser.add_argument('--module', type=str, default='all',
                        choices=['all','hazard','exposure','cgan','loss'],
                        help='Which module to run')
    parser.add_argument('--use-pretrained', action='store_true',
                        help='Use pre-trained cGAN model')
    args = parser.parse_args()

    print("\n" + "="*55)
    print("Hurricane CatModel — cGAN Super-Resolution")
    print("CSC-498 | Lehigh University | Spring 2026")
    print("="*55)

    if args.module in ['all', 'hazard']:
        tracks, ian_2022, peak_gust_full, lat_full, lon_full = run_hazard()
    else:
        # Load existing tracks
        tracks_path = os.path.join(PROC_DIR, 'tracks',
                                   'gulf_tracks_processed.csv')
        tracks  = pd.read_csv(tracks_path, parse_dates=['ISO_TIME'])
        ian_2022 = tracks[(tracks['NAME'].str.upper() == 'IAN') &
                          (tracks['SEASON'] == 2022)].copy()
        from gust_grid import build_grid
        from wind_field import compute_track_peak_gust
        FLORIDA_DOMAIN = {
            'lat_min': 24.0, 'lat_max': 31.5,
            'lon_min': -87.0, 'lon_max': -79.0
        }
        lat_full, lon_full = build_grid(FLORIDA_DOMAIN, res_deg=0.02)
        peak_gust_full = compute_track_peak_gust(
            ian_2022, lat_full, lon_full)

    if args.module in ['all', 'exposure']:
        exposure_df, tracts = run_exposure(
            ian_2022, peak_gust_full, lat_full, lon_full)
    else:
        import geopandas as gpd
        exposure_df = pd.read_csv(
            os.path.join(EXPOSURE_DIR, 'exposure_12071_lee.csv'))
        tracts = gpd.read_file(
            os.path.join(EXPOSURE_DIR, 'lee_county_tracts.shp'))
        tracts['tract_id'] = (tracts['STATEFP'] +
                               tracts['COUNTYFP'] +
                               tracts['TRACTCE'])

    if args.module in ['all', 'cgan']:
        G, X_max, lat_coarse, lon_coarse, lat_fine, lon_fine, device = \
            run_cgan(tracks, use_pretrained=args.use_pretrained)
    else:
        import torch
        from gust_grid import build_grid
        from generator import Generator
        LEE_DOMAIN = {
            'lat_min': 25.8, 'lat_max': 26.8,
            'lon_min': -82.3, 'lon_max': -81.3
        }
        lat_coarse, lon_coarse = build_grid(LEE_DOMAIN, res_deg=0.05)
        lat_fine,   lon_fine   = build_grid(LEE_DOMAIN, res_deg=0.005)
        device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        G = Generator(condition_dim=4).to(device)
        G.load_state_dict(torch.load(
            os.path.join(CGAN_DIR, 'generator_balanced_best.pth'),
            map_location=device))
        X_max = float(np.load(
            os.path.join(PROC_DIR, 'balanced_data', 'X_max_bal.npy')))

    if args.module in ['all', 'loss']:
        loss_df = run_loss(
            exposure_df, tracts, G, X_max,
            ian_2022, lat_coarse, lon_coarse,
            lat_fine, lon_fine, device
        )

    print("\n" + "="*55)
    print("Pipeline complete! Check outputs/ folder.")
    print("="*55)


if __name__ == '__main__':
    main()
