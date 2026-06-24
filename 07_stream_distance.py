"""
FACVI Pipeline — Step 07: Stream Distance (N3)
===============================================
Computes the distance from each grid cell centroid to the nearest
river segment using HydroRIVERS v1.0.

Adds column `dist_stream_m` to the physical grid in-place.

Inputs:
  data/{CC}_grid_physical.gpkg
  data/rasters/hydrosheds/HydroRIVERS_v10_{region}_shp/*.shp
    (downloaded by 01_download_data.py)
"""

import geopandas as gpd
import numpy as np
import logging
from pathlib import Path

from config import (
    DATA_DIR, RASTER_DIR, LOG_DIR,
    BBOX, COUNTRY_CODE, CRS_METRIC,
    HYDRORIVERS_REGION, GRID_PHYSICAL_FILE,
)

LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "07_stream_distance.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

HYDRO_DIR = (
    RASTER_DIR / "hydrosheds" /
    f"HydroRIVERS_v10_{HYDRORIVERS_REGION}_shp"
)


def compute_dist_stream(grid_path: Path) -> None:
    if not grid_path.exists():
        raise FileNotFoundError(f"Grid file not found: {grid_path}")

    log.info(f"[LOAD] {grid_path.name}")
    grid = gpd.read_file(grid_path)
    log.info(f"  {len(grid):,} cells")

    if "dist_stream_m" in grid.columns and grid["dist_stream_m"].notna().sum() > 1000:
        log.info("[CACHED] dist_stream_m already present — skipping.")
        return

    # Locate HydroRIVERS shapefile
    shp_files = sorted(HYDRO_DIR.glob("*.shp"))
    if not shp_files:
        log.error(f"[ERROR] HydroRIVERS shapefile not found in: {HYDRO_DIR}")
        log.info(
            "  Download from: https://www.hydrosheds.org/products/hydrorivers\n"
            "  Or run: python 01_download_data.py --only hydrorivers"
        )
        return

    shp_path = shp_files[0]
    log.info(f"[HYDRO] Loading: {shp_path.name}")

    study_bbox = (BBOX["west"], BBOX["south"], BBOX["east"], BBOX["north"])
    hydro = gpd.read_file(shp_path, bbox=study_bbox)
    log.info(f"  {len(hydro):,} river segments in study bbox")

    if hydro.empty:
        log.error("[ERROR] No river segments found within study bounding box.")
        return

    log.info(f"[CRS] Projecting to {CRS_METRIC} ...")
    grid_m    = grid.to_crs(CRS_METRIC)
    hydro_m   = hydro.to_crs(CRS_METRIC)

    centroids = grid_m.copy()
    centroids.geometry = grid_m.geometry.centroid

    log.info(f"[DISTANCE] sjoin_nearest ({len(centroids):,} cells, "
             f"{len(hydro_m):,} river segments) ...")
    joined = centroids[["geometry"]].sjoin_nearest(
        hydro_m[["geometry"]],
        how="left",
        distance_col="dist_stream_m",
    )
    joined = joined[~joined.index.duplicated(keep="first")]

    grid["dist_stream_m"] = joined["dist_stream_m"].values

    n_nan = int(grid["dist_stream_m"].isna().sum())
    if n_nan > 0:
        med = float(grid["dist_stream_m"].median())
        grid["dist_stream_m"] = grid["dist_stream_m"].fillna(med)
        log.info(f"  {n_nan:,} NaN → median ({med:.0f} m)")

    log.info(f"  Mean distance: {grid['dist_stream_m'].mean():.0f} m")
    log.info(f"  Range: {grid['dist_stream_m'].min():.0f} – "
             f"{grid['dist_stream_m'].max():.0f} m")

    grid.to_file(grid_path, driver="GPKG")
    log.info(f"[SAVED] dist_stream_m appended to {grid_path.name}")


def main():
    log.info("=" * 60)
    log.info("FACVI Step 07: Stream Distance (N3)")
    log.info("=" * 60)

    compute_dist_stream(GRID_PHYSICAL_FILE)
    log.info("Step 07 complete.")


if __name__ == "__main__":
    main()
