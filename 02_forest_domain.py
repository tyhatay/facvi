"""
FACVI Pipeline — Step 02: Forest Domain Mask
=============================================
Downloads ESA WorldCover 2021 (10 m) tiles and builds a 1 km binary
forest mask for the study area, then clips the 1 km grid to the
forest-access domain (forest cells plus a buffer zone).

Merged from:
  00c_forest_mask.py  — ESA WorldCover tile download + mosaic
  02c_forest_filter.py — grid clipping and buffer

Outputs
-------
  data/rasters/forest_mask_1km.tif        binary raster (1=forest)
  data/{CC}_grid_forest.gpkg              grid clipped to forest domain
"""

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.crs import CRS
from affine import Affine
from pathlib import Path
import geopandas as gpd
import logging

from config import (
    DATA_DIR, RASTER_DIR, LOG_DIR,
    BBOX, CRS_METRIC, ESA_TREE_CLASS, FOREST_BUFFER_M,
    COUNTRY_CODE, GRID_1KM_FILE,
)

LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "02_forest_domain.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

FOREST_MASK_FILE  = RASTER_DIR / "forest_mask_1km.tif"
GRID_FOREST_FILE  = DATA_DIR / f"{COUNTRY_CODE}_grid_forest.gpkg"

TARGET_RES = 0.009    # ~1 km in degrees (equatorial equivalent)

ESA_BASE_URL = (
    "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
    "ESA_WorldCover_10m_2021_v200_N{lat:02d}E{lon:03d}_Map.tif"
)


# ── Tile enumeration ──────────────────────────────────────────────────────────

def tile_urls_for_bbox(west, south, east, north):
    """Return ESA WorldCover tile (lat, lon, url) tuples covering the bounding box."""
    urls = []
    lat_start = (int(south) // 3) * 3
    lon_start = (int(west)  // 3) * 3
    lat_end   = (int(north) // 3) * 3
    lon_end   = (int(east)  // 3) * 3
    for lat in range(lat_start, lat_end + 1, 3):
        for lon in range(lon_start, lon_end + 1, 3):
            urls.append((lat, lon, ESA_BASE_URL.format(lat=lat, lon=lon)))
    return urls


# ── Single tile read ──────────────────────────────────────────────────────────

def _read_tile(url: str):
    """
    Stream one ESA WorldCover 3° tile at ~1 km resolution via /vsicurl.
    Returns (forest_array, transform, crs) or None on failure.
    """
    try:
        env = rasterio.Env(
            GDAL_HTTP_UNSAFESSL="YES",
            GDAL_HTTP_TIMEOUT="30",
            CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
        )
        with env:
            with rasterio.open(f"/vsicurl/{url}") as src:
                n_cols = int(round(3.0 / TARGET_RES))
                n_rows = int(round(3.0 / TARGET_RES))
                data = src.read(
                    1,
                    out_shape=(n_rows, n_cols),
                    resampling=Resampling.mode,
                )
                scale_x = src.width  / n_cols
                scale_y = src.height / n_rows
                new_transform = src.transform * src.transform.scale(scale_x, scale_y)
                forest = (data == ESA_TREE_CLASS).astype(np.uint8)
                return forest, new_transform, src.crs
    except Exception as e:
        log.warning(f"  Tile failed: {url.split('/')[-1]}  ({e})")
        return None


# ── Mosaic assembly ───────────────────────────────────────────────────────────

def build_mosaic(tile_results):
    west, south, east, north = BBOX["west"], BBOX["south"], BBOX["east"], BBOX["north"]
    n_cols = int(round((east  - west)  / TARGET_RES)) + 1
    n_rows = int(round((north - south) / TARGET_RES)) + 1

    dst_transform = Affine(TARGET_RES, 0, west, 0, -TARGET_RES, north)
    dst = np.zeros((n_rows, n_cols), dtype=np.uint8)
    dst_crs = CRS.from_epsg(4326)

    for item in tile_results:
        if item is None:
            continue
        tile_arr, src_transform, src_crs = item
        tmp = np.zeros((n_rows, n_cols), dtype=np.uint8)
        reproject(
            source=tile_arr,
            destination=tmp,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.mode,
        )
        dst = np.maximum(dst, tmp)

    return dst, dst_transform


# ── Step A: forest mask ───────────────────────────────────────────────────────

def build_forest_mask() -> Path:
    if FOREST_MASK_FILE.exists():
        with rasterio.open(FOREST_MASK_FILE) as src:
            d = src.read(1)
        log.info(f"[CACHED] Forest mask: {(d==1).sum():,} forest pixels "
                 f"({(d==1).mean()*100:.1f}%)")
        return FOREST_MASK_FILE

    tiles = tile_urls_for_bbox(
        BBOX["west"], BBOX["south"], BBOX["east"], BBOX["north"]
    )
    log.info(f"[ESA] Downloading {len(tiles)} WorldCover tiles ...")
    results, n_ok = [], 0
    for lat, lon, url in tiles:
        log.info(f"  N{lat:02d}E{lon:03d} ...")
        r = _read_tile(url)
        results.append(r)
        if r is not None:
            n_ok += 1

    log.info(f"  {n_ok}/{len(tiles)} tiles OK")
    if n_ok == 0:
        raise RuntimeError(
            "No ESA WorldCover tiles could be downloaded. "
            "Check internet access and the tile URL pattern."
        )

    log.info("[MOSAIC] Assembling ...")
    mosaic, transform = build_mosaic(results)

    forest_pct = mosaic.mean() * 100
    log.info(f"  Mosaic size: {mosaic.shape[0]}×{mosaic.shape[1]}  "
             f"Forest: {forest_pct:.1f}%")

    FOREST_MASK_FILE.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff", "dtype": "uint8",
        "width": mosaic.shape[1], "height": mosaic.shape[0], "count": 1,
        "crs": CRS.from_epsg(4326), "transform": transform,
        "nodata": 255, "compress": "lzw",
    }
    with rasterio.open(FOREST_MASK_FILE, "w", **profile) as dst:
        dst.write(mosaic, 1)

    log.info(f"  Forest mask saved: {FOREST_MASK_FILE.name} "
             f"({FOREST_MASK_FILE.stat().st_size / 1024 / 1024:.1f} MB)")
    return FOREST_MASK_FILE


# ── Step B: clip grid to forest domain ───────────────────────────────────────

def clip_grid_to_forest(grid_path: Path, mask_path: Path) -> gpd.GeoDataFrame:
    if GRID_FOREST_FILE.exists():
        gdf = gpd.read_file(GRID_FOREST_FILE)
        log.info(f"[CACHED] Forest grid: {len(gdf):,} cells")
        return gdf

    log.info(f"[LOAD] Grid: {grid_path.name}")
    grid = gpd.read_file(grid_path)
    log.info(f"  {len(grid):,} cells")

    log.info("[FOREST] Sampling forest mask at grid centroids ...")
    grid_wgs84 = grid.to_crs("EPSG:4326")
    lons = grid_wgs84.geometry.centroid.x.values
    lats = grid_wgs84.geometry.centroid.y.values
    coords = list(zip(lons, lats))

    with rasterio.open(mask_path) as src:
        forest_vals = np.array(
            [v[0] for v in src.sample(coords, indexes=1)], dtype=np.uint8
        )

    # Keep cells where centroid is forest OR within buffer of forest
    forest_cells = grid[forest_vals == 1].copy()
    log.info(f"  Forest centroid cells: {len(forest_cells):,}")

    # Buffer approach: cells within FOREST_BUFFER_M of any forest cell
    grid_m = grid.to_crs(CRS_METRIC)
    forest_m = grid_m[forest_vals == 1]

    from shapely.ops import unary_union
    log.info(f"  Building {FOREST_BUFFER_M / 1000:.0f} km buffer ...")
    forest_union = unary_union(forest_m.geometry)
    forest_buffer = forest_union.buffer(FOREST_BUFFER_M)

    centroids_m = grid_m.geometry.centroid
    in_buffer = centroids_m.within(forest_buffer)

    grid_forest = grid[in_buffer].copy().reset_index(drop=True)
    log.info(f"  Forest-access domain (forest + {FOREST_BUFFER_M}m buffer): "
             f"{len(grid_forest):,} cells "
             f"({len(grid_forest)/len(grid)*100:.1f}% of total)")

    grid_forest.to_file(GRID_FOREST_FILE, driver="GPKG")
    log.info(f"  Saved: {GRID_FOREST_FILE.name}")
    return grid_forest


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("FACVI Step 02: Forest Domain")
    log.info("=" * 60)

    mask_path = build_forest_mask()

    if not GRID_1KM_FILE.exists():
        raise FileNotFoundError(
            f"1 km grid not found: {GRID_1KM_FILE}\n"
            "Run 05_road_grid.py first."
        )

    clip_grid_to_forest(GRID_1KM_FILE, mask_path)
    log.info("Step 02 complete.")


if __name__ == "__main__":
    main()
