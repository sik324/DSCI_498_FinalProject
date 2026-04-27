import requests, numpy as np, pandas as pd
from pathlib import Path

GULF_BBOX = {"lat_min":17.0,"lat_max":32.0,"lon_min":-100.0,"lon_max":-80.0}
MIN_VMAX_KT = 34.0
IBTRACS_URL = ("https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs"
               "/v04r00/access/csv/ibtracs.NA.list.v04r00.csv")
REQUIRED_COLS = ["SID","SEASON","NAME","ISO_TIME","LAT","LON",
                 "WMO_WIND","WMO_PRES","USA_WIND","USA_PRES","USA_SSHS"]
OPTIONAL_COLS = ["NUMBER","BASIN","SUBBASIN","NATURE",
                 "USA_RMW","USA_PENV","REUNION_RMW","BOM_RMW","TOKYO_RMW"]
RMW_CANDIDATE_COLS = ["USA_RMW","REUNION_RMW","BOM_RMW","TOKYO_RMW"]

def download_ibtracs(data_dir):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    out_path = data_dir / "ibtracs.NA.list.v04r00.csv"
    if out_path.exists():
        print(f"[ibtracs] Using cached file: {out_path}")
        return out_path
    print("[ibtracs] Downloading IBTrACS NA (~20 MB)...")
    r = requests.get(IBTRACS_URL, stream=True, timeout=120)
    r.raise_for_status()
    with open(out_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"[ibtracs] Saved: {out_path}")
    return out_path

def load_ibtracs(csv_path):
    csv_path = Path(csv_path)
    header_df = pd.read_csv(csv_path, nrows=0)
    available = set(header_df.columns.str.strip())
    load_cols = [c for c in (REQUIRED_COLS + OPTIONAL_COLS) if c in available]
    missing = set(REQUIRED_COLS) - available
    if missing:
        raise ValueError(f"Required columns missing: {missing}")
    df = pd.read_csv(csv_path, skiprows=[1], low_memory=False,
                     na_values=[" ","","NaN","-999","-9999","9999"," -999"," -9999"],
                     usecols=load_cols)
    for col in OPTIONAL_COLS:
        if col not in df.columns:
            df[col] = np.nan
    return df

def _compute_forward_motion(group):
    R = 3440.065
    lat = np.radians(group["LAT"].values)
    lon = np.radians(group["LON"].values)
    dlat, dlon = np.diff(lat), np.diff(lon)
    a = np.sin(dlat/2)**2 + np.cos(lat[:-1])*np.cos(lat[1:])*np.sin(dlon/2)**2
    dist_nm = 2*R*np.arcsin(np.sqrt(a))
    dt_h = group["ISO_TIME"].diff().dt.total_seconds().values[1:]/3600.0
    dt_h = np.where(dt_h <= 0, 6.0, dt_h)
    y = np.sin(dlon)*np.cos(lat[1:])
    x = np.cos(lat[:-1])*np.sin(lat[1:]) - np.sin(lat[:-1])*np.cos(lat[1:])*np.cos(dlon)
    group = group.copy()
    group["STORM_SPEED_kt"] = np.concatenate([[np.nan], dist_nm/dt_h])
    group["STORM_DIR_deg"]  = np.concatenate([[np.nan], np.degrees(np.arctan2(y,x))%360])
    return group

def _best_wind_pres(df):
    df = df.copy()
    df["VMAX_kt"] = pd.to_numeric(df["USA_WIND"], errors="coerce")
    mask = df["VMAX_kt"].isna()
    df.loc[mask,"VMAX_kt"] = pd.to_numeric(df.loc[mask,"WMO_WIND"], errors="coerce")
    df["PMIN_mb"] = pd.to_numeric(df["USA_PRES"], errors="coerce")
    mask = df["PMIN_mb"].isna()
    df.loc[mask,"PMIN_mb"] = pd.to_numeric(df.loc[mask,"WMO_PRES"], errors="coerce")
    df["RMW_nmile"] = np.nan
    for col in RMW_CANDIDATE_COLS:
        if col in df.columns:
            cand = pd.to_numeric(df[col], errors="coerce")
            miss = df["RMW_nmile"].isna()
            df.loc[miss,"RMW_nmile"] = cand[miss]
    miss = df["RMW_nmile"].isna()
    if miss.any():
        v  = df.loc[miss,"VMAX_kt"].fillna(50)
        la = df.loc[miss,"LAT"].fillna(25)
        df.loc[miss,"RMW_nmile"] = (51.6*np.exp(-0.0223*v+0.0281*la)).clip(5,150)
    df["PENV_mb"] = pd.to_numeric(df["USA_PENV"], errors="coerce").fillna(1013.0) if "USA_PENV" in df.columns else 1013.0
    return df

def filter_gulf_coast(df, bbox=GULF_BBOX):
    df = df.copy()
    df["LAT"] = pd.to_numeric(df["LAT"], errors="coerce")
    df["LON"] = pd.to_numeric(df["LON"], errors="coerce")
    in_box = (df["LAT"].between(bbox["lat_min"],bbox["lat_max"]) &
              df["LON"].between(bbox["lon_min"],bbox["lon_max"]) &
              (df["VMAX_kt"] >= MIN_VMAX_KT))
    gulf_sids = df.loc[in_box,"SID"].unique()
    print(f"[ibtracs] Gulf Coast storms (>={MIN_VMAX_KT} kt): {len(gulf_sids)}")
    return df[df["SID"].isin(gulf_sids)].copy()

def process_tracks(raw_csv, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("[ibtracs] Loading raw CSV...")
    df = load_ibtracs(raw_csv)
    df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"], errors="coerce")
    df = _best_wind_pres(df)
    df = filter_gulf_coast(df)
    print("[ibtracs] Computing forward motion...")
    df = (df.sort_values(["SID","ISO_TIME"])
            .groupby("SID", group_keys=False)
            .apply(_compute_forward_motion))
    df = df.dropna(subset=["LAT","LON","VMAX_kt"])
    keep = ["SID","NAME","SEASON","ISO_TIME","LAT","LON",
            "VMAX_kt","PMIN_mb","RMW_nmile","PENV_mb",
            "STORM_SPEED_kt","STORM_DIR_deg","USA_SSHS"]
    df = df[[c for c in keep if c in df.columns]].reset_index(drop=True)
    out_path = out_dir / "gulf_tracks_processed.csv"
    df.to_csv(out_path, index=False)
    print(f"[ibtracs] Saved → {out_path} ({len(df):,} records)")
    return df

def track_summary(df):
    return (df.groupby(["SID","NAME","SEASON"])
              .agg(peak_vmax_kt=("VMAX_kt","max"),
                   min_pmin_mb=("PMIN_mb","min"),
                   n_records=("LAT","count"))
              .reset_index()
              .sort_values(["SEASON","peak_vmax_kt"], ascending=[True,False]))
