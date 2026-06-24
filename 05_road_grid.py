"""
FACVI Pipeline — Step 05: 1 km Grid and Road Density
=====================================================
Creates a 1 km × 1 km grid covering the study country, then
calculates per-cell road density (N1) and surface quality (N2)
from the OSM road segments extracted in step 04.

Outputs
-------
  data/{CC}_grid_1km.gpkg    — full 1 km grid with N1, N2 columns
"""

import geopandas as gpd
import numpy as np
import logging
from shapely.geometry import box
from pathlib import Path

from config import (
    DATA_DIR, LOG_DIR,
    COUNTRY_NAME, COUNTRY_CODE,
    CRS_METRIC, CELL_SIZE_M,
    GRID_1KM_FILE, ROADS_FILE, BOUNDARY_FILE,
)

LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "05_road_grid.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

CRS_GEO = "EPSG:4326"


# ── Country boundary ──────────────────────────────────────────────────────────

def get_boundary() -> gpd.GeoDataFrame:
    if BOUNDARY_FILE.exists():
        log.info(f"[CACHED] Boundary: {BOUNDARY_FILE.name}")
        return gpd.read_file(BOUNDARY_FILE)

    log.info(f"[OSM] Downloading boundary for '{COUNTRY_NAME}' ...")
    import osmnx as ox
    boundary = ox.geocode_to_gdf(COUNTRY_NAME)
    boundary = boundary.to_crs(CRS_GEO)
    boundary.to_file(BOUNDARY_FILE, driver="GPKG")
    log.info(f"  Boundary saved: {BOUNDARY_FILE.name}")
    return boundary


# ── Grid creation ─────────────────────────────────────────────────────────────

def create_grid(boundary: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    boundary_proj = boundary.to_crs(CRS_METRIC)
    minx, miny, maxx, maxy = boundary_proj.total_bounds

    log.info(f"[GRID] Extent: {(maxx-minx)/1000:.0f} km × {(maxy-miny)/1000:.0f} km")

    x_coords = np.arange(minx, maxx, CELL_SIZE_M)
    y_coords = np.arange(miny, maxy, CELL_SIZE_M)
    log.info(f"[GRID] Candidate cells: {len(x_coords) * len(y_coords):,}")

    cells = []
    for x in x_coords:
        for y in y_coords:
            cells.append({
                "geometry":   box(x, y, x + CELL_SIZE_M, y + CELL_SIZE_M),
                "cell_id":    f"{int(x)}_{int(y)}",
                "centroid_x": x + CELL_SIZE_M / 2,
                "centroid_y": y + CELL_SIZE_M / 2,
            })

    grid = gpd.GeoDataFrame(cells, crs=CRS_METRIC)

    log.info("[GRID] Clipping to country boundary ...")
    country_union = boundary_proj.union_all()
    grid = grid[grid.geometry.intersects(country_union)].copy()
    grid = grid.reset_index(drop=True)

    log.info(f"[GRID] Cells within country: {len(grid):,}")
    return grid


# ── Road density ──────────────────────────────────────────────────────────────

def calculate_road_density(grid: gpd.GeoDataFrame,
                           roads: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Compute N1 (road density m/km²) and N2 (weighted surface quality) per cell.
    Uses a spatial overlay to split road segments at cell boundaries.
    """
    if roads.crs.to_string() != CRS_METRIC:
        roads = roads.to_crs(CRS_METRIC)
    if grid.crs.to_string() != CRS_METRIC:
        grid = grid.to_crs(CRS_METRIC)

    log.info("[DENSITY] Computing spatial intersection ...")
    roads_clip = gpd.overlay(
        roads[["geometry", "surface_score"]],
        grid[["geometry", "cell_id"]],
        how="intersection",
    )
    roads_clip["seg_length_m"] = roads_clip.geometry.length
    log.info(f"  Intersection segments: {len(roads_clip):,}")

    agg = roads_clip.groupby("cell_id").agg(
        total_length_m=("seg_length_m", "sum"),
        weighted_surface=("seg_length_m", lambda x: (
            np.average(roads_clip.loc[x.index, "surface_score"], weights=x)
            if x.sum() > 0 else np.nan
        )),
    ).reset_index()

    grid = grid.merge(agg, on="cell_id", how="left")
    grid["total_length_m"]   = grid["total_length_m"].fillna(0)
    grid["weighted_surface"] = grid["weighted_surface"].fillna(0)

    # N1: m/km² (cell = 1 km² so density = total length)
    grid["road_density_m_km2"] = grid["total_length_m"]
    # N2: surface quality score
    grid["surface_quality"]    = grid["weighted_surface"]

    has_roads = grid["road_density_m_km2"] > 0
    log.info(f"  Cells with roads: {has_roads.sum():,} ({has_roads.mean()*100:.1f}%)")
    log.info(f"  Mean density (road cells): "
             f"{grid.loc[has_roads, 'road_density_m_km2'].mean():.0f} m/km²")
    log.info(f"  Max density: {grid['road_density_m_km2'].max():.0f} m/km²")
    return grid


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if GRID_1KM_FILE.exists():
        grid = gpd.read_file(GRID_1KM_FILE)
        log.info(f"[CACHED] Grid: {len(grid):,} cells in {GRID_1KM_FILE.name}")
        return

    log.info("=" * 60)
    log.info("FACVI Step 05: 1 km Grid + Road Density")
    log.info("=" * 60)

    boundary = get_boundary()
    grid     = create_grid(boundary)

    if not ROADS_FILE.exists():
        raise FileNotFoundError(
            f"Road network not found: {ROADS_FILE}\n"
            "Run 04_osm_roads.py first."
        )
    log.info(f"[LOAD] Roads: {ROADS_FILE.name}")
    roads = gpd.read_file(ROADS_FILE)
    log.info(f"  {len(roads):,} segments")

    grid = calculate_road_density(grid, roads)
    grid_geo = grid.to_crs(CRS_GEO)
    grid_geo.to_file(GRID_1KM_FILE, driver="GPKG")
    log.info(f"[SAVED] {GRID_1KM_FILE.name}  ({len(grid_geo):,} cells)")
    log.info("Step 05 complete.")


if __name__ == "__main__":
    main()
