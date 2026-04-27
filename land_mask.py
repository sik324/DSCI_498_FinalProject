import sys, zipfile, io, requests, numpy as np
from pathlib import Path
import geopandas as gpd
from rasterio.transform import from_bounds
from rasterio.features import rasterize
import rasterio

GULF_STATE_FIPS={"48":"Texas","22":"Louisiana","28":"Mississippi","01":"Alabama","12":"Florida"}
TIGER_URL="https://www2.census.gov/geo/tiger/TIGER2022/COUNTY/tl_2022_us_county.zip"
DOMAIN={"lat_min":24.0,"lat_max":31.5,"lon_min":-98.0,"lon_max":-79.0}
DEFAULT_RES_DEG=0.05

def download_tiger_counties(data_dir):
    data_dir=Path(data_dir); data_dir.mkdir(parents=True,exist_ok=True)
    shp_path=data_dir/"tl_2022_us_county.shp"
    if shp_path.exists():
        print(f"[land_mask] Using cached TIGER: {shp_path}"); return shp_path
    print("[land_mask] Downloading Census TIGER counties (~13 MB)...")
    r=requests.get(TIGER_URL,timeout=120); r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf: zf.extractall(data_dir)
    print(f"[land_mask] Extracted to {data_dir}"); return shp_path

def load_gulf_counties(shp_path, out_gpkg):
    out_gpkg=Path(out_gpkg); out_gpkg.parent.mkdir(parents=True,exist_ok=True)
    if out_gpkg.exists():
        print(f"[land_mask] Using cached counties: {out_gpkg}")
        return gpd.read_file(out_gpkg)
    gdf=gpd.read_file(shp_path)
    gulf=gdf[gdf["STATEFP"].isin(GULF_STATE_FIPS.keys())].copy().to_crs("EPSG:4326")
    gulf["STATE_NAME"]=gulf["STATEFP"].map(GULF_STATE_FIPS)
    gulf["COUNTY_NAME"]=gulf["NAME"]
    gulf["FIPS_INT"]=gulf["GEOID"].astype(int)
    gulf=gulf[["GEOID","FIPS_INT","STATE_NAME","COUNTY_NAME","geometry"]]
    from shapely.geometry import box
    bbox=box(DOMAIN["lon_min"]-0.5,DOMAIN["lat_min"]-0.5,
             DOMAIN["lon_max"]+0.5,DOMAIN["lat_max"]+0.5)
    gulf=gulf[gulf.intersects(bbox)].reset_index(drop=True)
    gulf.to_file(out_gpkg,driver="GPKG")
    print(f"[land_mask] {len(gulf)} counties saved → {out_gpkg}")
    return gulf

def build_grid(domain=DOMAIN, res_deg=DEFAULT_RES_DEG):
    lats=np.arange(domain["lat_min"],domain["lat_max"]+res_deg,res_deg)
    lons=np.arange(domain["lon_min"],domain["lon_max"]+res_deg,res_deg)
    lon_grid,lat_grid=np.meshgrid(lons,lats)
    return lat_grid,lon_grid

def build_land_mask(data_dir, out_dir, res_deg=0.05):
    shp=download_tiger_counties(data_dir)
    counties=load_gulf_counties(shp, Path(out_dir).parent/"gulf_counties.gpkg")
    lat_grid,lon_grid=build_grid(DOMAIN,res_deg)
    out_dir=Path(out_dir); out_dir.mkdir(parents=True,exist_ok=True)
    nrows,ncols=lat_grid.shape
    transform=from_bounds(lon_grid.min(),lat_grid.min(),
                          lon_grid.max(),lat_grid.max(),ncols,nrows)
    shapes=[(geom,fips) for geom,fips in zip(counties.geometry,counties["FIPS_INT"])
            if geom is not None and not geom.is_empty]
    fips_raster=rasterize(shapes=shapes,out_shape=(nrows,ncols),
                          transform=transform,fill=0,dtype="int32",all_touched=False)
    land_raster=(fips_raster>0).astype(np.uint8)
    crs="EPSG:4326"
    with rasterio.open(out_dir/"land_mask.tif","w",driver="GTiff",dtype="uint8",
                       width=ncols,height=nrows,count=1,crs=crs,
                       transform=transform,nodata=255,compress="lzw") as dst:
        dst.write(np.flipud(land_raster)[np.newaxis,:,:])
    with rasterio.open(out_dir/"county_fips.tif","w",driver="GTiff",dtype="int32",
                       width=ncols,height=nrows,count=1,crs=crs,
                       transform=transform,nodata=0,compress="lzw") as dst:
        dst.write(np.flipud(fips_raster)[np.newaxis,:,:])
    n=int(land_raster.sum()); t=nrows*ncols
    print(f"[land_mask] Land: {n:,}/{t:,} cells ({100*n/t:.1f}%)")
    return land_raster, fips_raster

def apply_land_mask_to_gust(peak_gust, land_mask):
    masked=peak_gust.copy(); masked[~land_mask.astype(bool)]=0.0; return masked
