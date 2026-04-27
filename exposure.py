
import requests, zipfile, io, numpy as np, pandas as pd
from pathlib import Path
import geopandas as gpd

# ── HAZUS Florida MBT distribution for residential buildings ─────────────────
# Source: HAZUS-MH Hurricane Technical Manual Table 4.1
# Lee County FL — coastal county, high mobile home fraction
HAZUS_MBT_DIST = {
    "W1":  0.52,   # Wood frame single family
    "MH":  0.30,   # Mobile home / manufactured housing
    "M1":  0.10,   # Unreinforced masonry
    "C1":  0.05,   # Concrete frame
    "S1":  0.03,   # Steel frame
}

# HAZUS default replacement costs ($/sq ft) Florida 2022
HAZUS_UNIT_COST = {
    "W1":  110.0,
    "MH":   55.0,
    "M1":  120.0,
    "C1":  140.0,
    "S1":  155.0,
}

# Average floor area (sq ft) by MBT
HAZUS_AVG_SQFT = {
    "W1":  1800.0,
    "MH":   900.0,
    "M1":  1600.0,
    "C1":  2000.0,
    "S1":  2200.0,
}

CENSUS_TRACT_URL = ("https://www2.census.gov/geo/tiger/TIGER2022/TRACT/"
                    "tl_2022_12_tract.zip")


def download_tracts(data_dir):
    """Download Florida census tract shapefile."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    shp_path = data_dir / "tl_2022_12_tract.shp"
    if shp_path.exists():
        print(f"[exposure] Using cached tracts: {shp_path}")
        return shp_path
    print("[exposure] Downloading Florida census tracts (~8 MB)...")
    r = requests.get(CENSUS_TRACT_URL, timeout=120)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        zf.extractall(data_dir)
    print(f"[exposure] Tracts extracted to {data_dir}")
    return shp_path


def fetch_housing_units(state_fips, county_fips):
    """Fetch housing unit counts per tract from Census ACS 5-year."""
    url = (f"https://api.census.gov/data/2022/acs/acs5"
           f"?get=NAME,B25001_001E"
           f"&for=tract:*"
           f"&in=state:{state_fips}%20county:{county_fips}")
    print(f"[exposure] Fetching ACS housing units from Census API...")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    data = r.json()
    df = pd.DataFrame(data[1:], columns=data[0])
    df["tract_id"]  = df["state"] + df["county"] + df["tract"]
    df["n_units"]   = pd.to_numeric(df["B25001_001E"], errors="coerce").fillna(0).astype(int)
    df["county_fips"] = df["state"] + df["county"]
    print(f"[exposure] {len(df)} tracts, {df['n_units'].sum():,} total housing units")
    return df[["tract_id", "county_fips", "n_units"]]


def classify_mbt(units_df):
    """Expand each tract into one row per MBT using HAZUS distribution."""
    rows = []
    for _, row in units_df.iterrows():
        for mbt, pct in HAZUS_MBT_DIST.items():
            n_bldg = max(1, round(row["n_units"] * pct))
            tiv_per_unit = HAZUS_UNIT_COST[mbt] * HAZUS_AVG_SQFT[mbt]
            rows.append({
                "tract_id":     row["tract_id"],
                "county_fips":  row["county_fips"],
                "n_units":      row["n_units"],
                "MBT":          mbt,
                "pct_share":    pct,
                "n_buildings":  n_bldg,
                "avg_sq_ft":    HAZUS_AVG_SQFT[mbt],
                "TIV_per_unit": tiv_per_unit,
                "TIV_total":    n_bldg * tiv_per_unit,
            })
    return pd.DataFrame(rows)


def join_gust_to_tracts(exposure_df, tracts_gdf, peak_gust, lat_grid, lon_grid):
    """Sample peak gust at each tract centroid."""
    centroids = tracts_gdf.copy()
    centroids["centroid_lat"] = tracts_gdf.geometry.centroid.y
    centroids["centroid_lon"] = tracts_gdf.geometry.centroid.x
    centroids["tract_id"] = (centroids["STATEFP"] +
                              centroids["COUNTYFP"] +
                              centroids["TRACTCE"])
    centroid_gust = {}
    for _, row in centroids.iterrows():
        lat_idx = np.abs(lat_grid[:, 0] - row["centroid_lat"]).argmin()
        lon_idx = np.abs(lon_grid[0, :] - row["centroid_lon"]).argmin()
        gust_ms  = float(peak_gust[lat_idx, lon_idx])
        centroid_gust[row["tract_id"]] = round(gust_ms * 2.23694, 1)
    exposure_df["peak_gust_mph"] = exposure_df["tract_id"].map(centroid_gust)
    return exposure_df


def build_exposure(state_fips, county_fips, peak_gust,
                   lat_grid, lon_grid, data_dir, out_dir):
    """Full exposure pipeline for one county."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[exposure] Building exposure for {state_fips}{county_fips}...")

    # Step 1 — tract boundaries
    shp_path = download_tracts(data_dir)
    tracts   = gpd.read_file(str(shp_path))
    tracts   = tracts[tracts["COUNTYFP"] == county_fips].to_crs("EPSG:4326")
    print(f"[exposure] Census tracts in county: {len(tracts)}")

    # Step 2 — housing units from Census ACS
    units_df = fetch_housing_units(state_fips, county_fips)

    # Step 3 — MBT classification + TIV
    exposure_df = classify_mbt(units_df)

    # Step 4 — spatial join to wind field
    exposure_df = join_gust_to_tracts(
        exposure_df, tracts, peak_gust, lat_grid, lon_grid
    )

    # Step 5 — save outputs
    out_path = out_dir / f"exposure_{state_fips}{county_fips}_lee.csv"
    exposure_df.to_csv(out_path, index=False)

    # Save tract shapefile for ArcGIS
    tracts.to_file(str(out_dir / "lee_county_tracts.shp"))

    # Summary
    total_tiv   = exposure_df["TIV_total"].sum()
    total_bldgs = exposure_df.groupby("tract_id")["n_buildings"].sum().sum()
    print(f"[exposure] Tracts          : {exposure_df['tract_id'].nunique()}")
    print(f"[exposure] Total buildings : {total_bldgs:,}")
    print(f"[exposure] Total TIV       : ${total_tiv/1e9:.2f}B")
    print(f"[exposure] Saved → {out_path}")
    return exposure_df, tracts
