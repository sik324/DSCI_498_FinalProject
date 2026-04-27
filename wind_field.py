import numpy as np
from dataclasses import dataclass
from typing import Optional

RHO_AIR=1.15; OMEGA=7.2921e-5; KT_TO_MS=0.514444; NM_TO_M=1852.0; MB_TO_PA=100.0
REDUCTION_WATER=0.90; REDUCTION_LAND=0.80; GUST_FACTOR_1MIN=1.11

@dataclass
class StormRecord:
    lat:float; lon:float; vmax_kt:float; pmin_mb:float
    penv_mb:float=1013.0; rmw_nmile:float=25.0
    storm_speed_kt:float=10.0; storm_dir_deg:float=0.0

def holland_B(vmax_ms, dp_pa, rho=RHO_AIR):
    if dp_pa <= 0: return 1.5
    return float(np.clip(rho*np.e*vmax_ms**2/dp_pa, 1.0, 2.5))

def coriolis(lat_deg):
    return 2.0*OMEGA*np.sin(np.radians(lat_deg))

def haversine_dist_bearing(lat1, lon1, lat_grid, lon_grid):
    p1=np.radians(lat1); p2=np.radians(lat_grid)
    dp=p2-p1; dl=np.radians(lon_grid-lon1)
    a=np.sin(dp/2)**2+np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2
    dist_m=2*6_371_000*np.arcsin(np.sqrt(np.clip(a,0,1)))
    y=np.sin(dl)*np.cos(p2)
    x=np.cos(p1)*np.sin(p2)-np.sin(p1)*np.cos(p2)*np.cos(dl)
    return dist_m, np.degrees(np.arctan2(y,x))%360.0

def gradient_wind(r_m, rmw_m, vmax_ms, B, f):
    r_safe=np.where(r_m<1.0,1.0,r_m)
    ratio=(rmw_m/r_safe)**B
    dp_pa=RHO_AIR*np.e*vmax_ms**2/B
    Vgr=np.sqrt((B/RHO_AIR)*ratio*dp_pa*np.exp(-ratio)+(r_safe*f/2)**2)-r_safe*abs(f)/2
    return np.maximum(np.where(r_m<1.0,0.0,Vgr),0.0)

def asymmetry_correction(Vgr, bearing_deg, storm_speed_ms, storm_dir_deg):
    return np.maximum(Vgr+0.5*storm_speed_ms*np.cos(np.radians(bearing_deg-storm_dir_deg)),0.0)

def compute_wind_field(storm, lat_grid, lon_grid, over_water=None):
    vmax_ms=storm.vmax_kt*KT_TO_MS; rmw_m=storm.rmw_nmile*NM_TO_M
    dp_pa=(storm.penv_mb-storm.pmin_mb)*MB_TO_PA
    storm_spd_ms=(storm.storm_speed_kt or 0.0)*KT_TO_MS
    B=holland_B(vmax_ms,dp_pa); f=coriolis(storm.lat)
    dist_m,bearing_deg=haversine_dist_bearing(storm.lat,storm.lon,lat_grid,lon_grid)
    Vgr=gradient_wind(dist_m,rmw_m,vmax_ms,B,f)
    Vgr=asymmetry_correction(Vgr,bearing_deg,storm_spd_ms,storm.storm_dir_deg or 0.0)
    if over_water is None: over_water=np.ones_like(lat_grid,dtype=bool)
    Vsurf=Vgr*np.where(over_water,REDUCTION_WATER,REDUCTION_LAND)
    return {"vgust_ms":Vsurf*GUST_FACTOR_1MIN,"vsurf_ms":Vsurf,"dist_m":dist_m}

def compute_track_peak_gust(track_df, lat_grid, lon_grid, over_water=None, min_vmax_kt=34.0):
    peak_gust=np.zeros_like(lat_grid,dtype=np.float32)
    for _,row in track_df.iterrows():
        if row["VMAX_kt"]<min_vmax_kt: continue
        storm=StormRecord(
            lat=row["LAT"],lon=row["LON"],vmax_kt=row["VMAX_kt"],
            pmin_mb=row["PMIN_mb"] if not np.isnan(row["PMIN_mb"]) else 950.0,
            penv_mb=row["PENV_mb"] if not np.isnan(row["PENV_mb"]) else 1013.0,
            rmw_nmile=row["RMW_nmile"],
            storm_speed_kt=row["STORM_SPEED_kt"] if not np.isnan(row["STORM_SPEED_kt"]) else 10.0,
            storm_dir_deg=row["STORM_DIR_deg"] if not np.isnan(row["STORM_DIR_deg"]) else 0.0)
        result=compute_wind_field(storm,lat_grid,lon_grid,over_water)
        peak_gust=np.maximum(peak_gust,result["vgust_ms"])
    return peak_gust.astype(np.float32)
