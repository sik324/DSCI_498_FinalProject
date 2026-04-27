import numpy as np, pandas as pd
from pathlib import Path

try:
    import rasterio
    from rasterio.transform import from_bounds
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    print("[gust_grid] WARNING: rasterio not installed.")

from wind_field import compute_track_peak_gust

DOMAIN={"lat_min":24.0,"lat_max":31.5,"lon_min":-98.0,"lon_max":-80.5}
DEFAULT_RES_DEG=0.05; NODATA=-9999.0

def build_grid(domain=DOMAIN, res_deg=DEFAULT_RES_DEG):
    lats=np.arange(domain["lat_min"],domain["lat_max"]+res_deg,res_deg)
    lons=np.arange(domain["lon_min"],domain["lon_max"]+res_deg,res_deg)
    lon_grid,lat_grid=np.meshgrid(lons,lats)
    return lat_grid,lon_grid

def save_geotiff(array, lat_grid, lon_grid, out_path, nodata=NODATA):
    out_path=Path(out_path); out_path.parent.mkdir(parents=True,exist_ok=True)
    arr=array.astype(np.float32); arr=np.where(arr==0,nodata,arr)
    if HAS_RASTERIO:
        nrows,ncols=arr.shape
        transform=from_bounds(lon_grid.min(),lat_grid.min(),
                              lon_grid.max(),lat_grid.max(),ncols,nrows)
        with rasterio.open(out_path,"w",driver="GTiff",dtype="float32",
                           width=ncols,height=nrows,count=1,crs="EPSG:4326",
                           transform=transform,nodata=nodata,compress="lzw") as dst:
            dst.write(np.flipud(arr)[np.newaxis,:,:])
        print(f"[gust_grid] Saved GeoTIFF → {out_path}")
    else:
        npy=out_path.with_suffix(".npy"); np.save(npy,arr)
        print(f"[gust_grid] Saved → {npy}")
