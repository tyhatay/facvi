"""
FACVI Pipeline — Step 06: Physical Vulnerability Variables (S1–S5 partial)
===========================================================================
Computes per-cell S1–S4 from raster data:

  S1 — slope_deg      SRTM 30m, GDAL Horn algorithm
  S2 — twi            SRTM 30m, pysheds D8 (UTM projection for accuracy)
  S3 — k_factor       SoilGrids v2 EPIC K-factor (from step 01)
  S4 — ndvi_mean      MODIS MOD13A3 annual mean (from step 01)

  S5 (snow_days_mean) is added by 08_snow_cover.py.
  N3 (dist_stream_m)  is added by 07_stream_distance.py.

TWI computation method (pysheds D8):
  1. Resample SRTM mosaic to 250 m via gdal.Warp in UTM metric CRS
  2. pysheds fill_pits → fill_depressions → flowdir → accumulation
  3. SCA = acc × cell_area / cell_width  (m)
  4. TWI = ln(SCA / tan(slope))  where slope is from pysheds cell_slopes
  5. Reproject result to 1 km WGS84 via rasterio.reproject

Inputs  (produced by 01_download_data.py):
  data/rasters/srtm_{CC}_raw.tif
  data/rasters/k_factor_{CC}.tif
  data/rasters/ndvi_{CC}.tif

Output: data/{CC}_grid_physical.gpkg
"""

import numpy as np
import logging
import warnings
from pathlib import Path

import geopandas as gpd
import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.crs import CRS

from config import (
    DATA_DIR, RASTER_DIR, LOG_DIR,
    COUNTRY_CODE, CRS_METRIC,
    GRID_1KM_FILE, GRID_PHYSICAL_FILE,
)

warnings.filterwarnings("ignore")

LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "06_physical.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# Raster file names (auto-derived from COUNTRY_CODE)
SRTM_RAW   = RASTER_DIR / f"srtm_{COUNTRY_CODE}_raw.tif"
SRTM_250M  = RASTER_DIR / f"_srtm_{COUNTRY_CODE}_250m_utm.tif"   # intermediate
TWI_250M   = RASTER_DIR / f"_twi_{COUNTRY_CODE}_250m_utm.tif"    # intermediate
SLOPE_1KM  = RASTER_DIR / f"srtm_slope_{COUNTRY_CODE}.tif"
TWI_1KM    = RASTER_DIR / f"srtm_twi_{COUNTRY_CODE}.tif"
K_FACTOR   = RASTER_DIR / f"k_factor_{COUNTRY_CODE}.tif"
NDVI       = RASTER_DIR / f"ndvi_{COUNTRY_CODE}.tif"


# ── Step A: GDAL slope ────────────────────────────────────────────────────────

def compute_slope():
    if SLOPE_1KM.exists():
        log.info(f"[CACHED] Slope: {SLOPE_1KM.name}")
        return

    if not SRTM_RAW.exists():
        log.warning(f"[SKIP] SRTM mosaic not found: {SRTM_RAW.name}")
        return

    log.info("[SLOPE] Computing via GDAL DEMProcessing (Horn) ...")
    from osgeo import gdal
    gdal.UseExceptions()
    opts = gdal.DEMProcessingOptions(
        slopeFormat="degree",
        format="GTiff",
        creationOptions=["COMPRESS=LZW", "TILED=YES", "BIGTIFF=YES"],
    )
    ds = gdal.DEMProcessing(str(SLOPE_1KM), str(SRTM_RAW), "slope", options=opts)
    ds = None
    log.info(f"  Slope saved: {SLOPE_1KM.name} "
             f"({SLOPE_1KM.stat().st_size / 1e6:.1f} MB)")


# ── Step B: pysheds TWI ───────────────────────────────────────────────────────

def compute_twi():
    if TWI_1KM.exists():
        log.info(f"[CACHED] TWI: {TWI_1KM.name}")
        return

    if not SRTM_RAW.exists():
        log.warning(f"[SKIP] SRTM mosaic not found: {SRTM_RAW.name}")
        return

    try:
        from pysheds.grid import Grid as PGrid
    except ImportError:
        log.error("[TWI] pysheds not installed.  pip install pysheds")
        return

    from osgeo import gdal
    gdal.UseExceptions()

    # 1 — Reproject to metric UTM at 250 m
    if not SRTM_250M.exists():
        log.info(f"[TWI] Reprojecting DEM to {CRS_METRIC} at 250 m ...")
        ds = gdal.Warp(
            str(SRTM_250M), str(SRTM_RAW),
            dstSRS=CRS_METRIC,
            xRes=250, yRes=250,
            resampleAlg=gdal.GRA_Bilinear,
            dstNodata=-9999,
            creationOptions=["COMPRESS=LZW", "TILED=YES",
                             "BLOCKXSIZE=256", "BLOCKYSIZE=256"],
        )
        ds = None
        log.info(f"  UTM 250m DEM: {SRTM_250M.stat().st_size / 1e6:.1f} MB")

    # 2 — pysheds D8 flow analysis
    log.info("[TWI] Running pysheds D8 (fill → flowdir → accumulation) ...")
    pgrid   = PGrid.from_raster(str(SRTM_250M))
    dem     = pgrid.read_raster(str(SRTM_250M))

    pit_filled = pgrid.fill_pits(dem)
    flooded    = pgrid.fill_depressions(pit_filled)
    fdir       = pgrid.flowdir(flooded)
    acc        = pgrid.accumulation(fdir)
    slope_mm   = pgrid.cell_slopes(flooded, fdir)   # m/m in metric CRS

    with rasterio.open(SRTM_250M) as src:
        res_m   = abs(src.transform.a)
        profile = src.profile.copy()
        profile.update(dtype="float32", nodata=-9999.0, count=1,
                       compress="LZW", tiled=True, blockxsize=256, blockysize=256)

    log.info(f"  Cell size: {res_m:.1f} m")

    # SCA = acc * cell_area / cell_width  (m)
    sca         = acc.astype(np.float64) * (res_m * res_m) / res_m
    slope_safe  = np.where(slope_mm < 0.001, 0.001, slope_mm.astype(np.float64))
    twi         = np.log(sca / slope_safe)
    valid_mask  = np.isfinite(twi) & (np.asarray(acc) > 0)
    twi_out     = np.where(valid_mask, twi, -9999.0).astype(np.float32)

    valid = twi[valid_mask]
    log.info(f"  Valid pixels: {valid_mask.sum():,}")
    log.info(f"  TWI range: {valid.min():.2f} – {valid.max():.2f}  "
             f"mean={valid.mean():.2f}")

    with rasterio.open(TWI_250M, "w", **profile) as dst:
        dst.write(twi_out, 1)

    # 3 — Reproject to WGS84 1 km
    log.info("[TWI] Reprojecting to 1 km WGS84 ...")
    dst_crs = CRS.from_epsg(4326)
    with rasterio.open(TWI_250M) as src:
        tf_dst, width, height = calculate_default_transform(
            src.crs, dst_crs,
            src.width, src.height,
            left=src.bounds.left, bottom=src.bounds.bottom,
            right=src.bounds.right, top=src.bounds.top,
            resolution=0.008983,
        )
        out = np.empty((height, width), dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=out,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=tf_dst,
            dst_crs=dst_crs,
            resampling=Resampling.average,
            src_nodata=-9999.0,
            dst_nodata=-9999.0,
        )

    prof1km = {"driver": "GTiff", "dtype": "float32", "nodata": -9999.0,
               "width": width, "height": height, "count": 1,
               "crs": dst_crs, "transform": tf_dst,
               "compress": "LZW", "tiled": True, "blockxsize": 256, "blockysize": 256}
    with rasterio.open(TWI_1KM, "w", **prof1km) as dst:
        dst.write(out, 1)

    log.info(f"  1 km TWI saved: {TWI_1KM.name}")

    # Clean up intermediates
    SRTM_250M.unlink(missing_ok=True)
    TWI_250M.unlink(missing_ok=True)


# ── Raster sampling helpers ───────────────────────────────────────────────────

def _centroids_wgs84(grid: gpd.GeoDataFrame) -> np.ndarray:
    geo = grid.to_crs("EPSG:4326")
    return np.column_stack([geo.geometry.centroid.x.values,
                            geo.geometry.centroid.y.values])


def _sample_raster(path: Path, coords_wgs84: np.ndarray) -> np.ndarray:
    """
    Sample a raster at WGS84 centroid coordinates.
    Downsample to ~1 km first if the native resolution is finer than 0.005°.
    """
    with rasterio.open(path) as src:
        res_deg = max(src.res)
        nd      = src.nodata if src.nodata is not None else -9999

    if res_deg < 0.005:
        from osgeo import gdal
        gdal.UseExceptions()
        tmp = RASTER_DIR / f"_{path.stem}_1km_tmp.tif"
        if not tmp.exists():
            ds = gdal.Warp(str(tmp), str(path),
                           xRes=0.009, yRes=0.009,
                           resampleAlg=gdal.GRA_Average,
                           dstNodata=nd,
                           creationOptions=["COMPRESS=LZW"])
            ds = None
        read_path = tmp
    else:
        read_path = path
        tmp       = None

    with rasterio.open(read_path) as src:
        nd = src.nodata if src.nodata is not None else -9999
        if src.crs.to_epsg() != 4326:
            from pyproj import Transformer
            tr = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
            xs, ys = tr.transform(coords_wgs84[:, 0], coords_wgs84[:, 1])
            coords = np.column_stack([xs, ys])
        else:
            coords = coords_wgs84
        vals = np.array([v[0] for v in src.sample(coords, indexes=1)],
                        dtype=np.float64)

    if tmp and tmp.exists():
        tmp.unlink(missing_ok=True)

    return np.where(vals == nd, np.nan, vals)


def _fill_nan_with_median(grid: gpd.GeoDataFrame, col: str) -> gpd.GeoDataFrame:
    n_nan = int(grid[col].isna().sum())
    if n_nan > 0:
        med = float(grid[col].median())
        grid[col] = grid[col].fillna(med)
        log.info(f"    {n_nan:,} NaN → median ({med:.3f})")
    return grid


# ── Synthetic fallback ────────────────────────────────────────────────────────

def generate_synthetic(grid: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Produce spatially-structured synthetic values for demonstration when
    real rasters are absent.  Values are NOT suitable for publication.
    """
    log.warning("[SYNTHETIC] Real rasters not found — generating demo values.")
    log.warning("  For real analysis: run 01_download_data.py first.")

    n   = len(grid)
    rng = np.random.default_rng(seed=42)
    geo = grid.to_crs("EPSG:4326")
    lon = geo.geometry.centroid.x.values
    lat = geo.geometry.centroid.y.values

    slope = np.clip(
        10 * (lat - lat.min()) / (lat.max() - lat.min()) +
        15 * (lon - lon.min()) / (lon.max() - lon.min()) +
        rng.normal(0, 5, n),
        0, 60,
    )
    grid["slope_deg"] = slope
    grid["twi"]       = np.clip(12 - 0.15 * slope + rng.normal(0, 2, n), 2, 20)
    grid["k_factor"]  = np.clip(0.030 + rng.normal(0, 0.008, n), 0.005, 0.070)
    grid["ndvi_mean"] = np.clip(0.4 + rng.normal(0, 0.1, n), 0.05, 0.85)
    return grid


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if GRID_PHYSICAL_FILE.exists():
        gdf = gpd.read_file(GRID_PHYSICAL_FILE)
        log.info(f"[CACHED] Physical grid: {len(gdf):,} cells in "
                 f"{GRID_PHYSICAL_FILE.name}")
        return

    log.info("=" * 60)
    log.info("FACVI Step 06: Physical Variables (S1–S4)")
    log.info("=" * 60)

    # Pre-compute slope and TWI from SRTM if not already done
    compute_slope()
    compute_twi()

    if not GRID_1KM_FILE.exists():
        raise FileNotFoundError(
            f"1 km grid not found: {GRID_1KM_FILE}\n"
            "Run 05_road_grid.py first."
        )

    log.info(f"[LOAD] {GRID_1KM_FILE.name}")
    grid = gpd.read_file(GRID_1KM_FILE)
    log.info(f"  {len(grid):,} cells")

    raster_map = {
        "slope_deg": SLOPE_1KM,
        "twi":       TWI_1KM,
        "ndvi_mean": NDVI,
        "k_factor":  K_FACTOR,
    }

    missing_required = [k for k in ("slope_deg", "twi", "ndvi_mean")
                        if not raster_map[k].exists()]

    if missing_required:
        log.warning(f"  Required rasters missing: {missing_required}  "
                    f"→ using synthetic values")
        grid = generate_synthetic(grid)
    else:
        coords = _centroids_wgs84(grid)
        for col, path in raster_map.items():
            if not path.exists():
                log.warning(f"  [{col}] not found, skipping.")
                continue
            log.info(f"  [{col}] Sampling {path.name} ...")
            vals = _sample_raster(path, coords)
            log.info(f"    mean={np.nanmean(vals):.3f}  "
                     f"std={np.nanstd(vals):.3f}  "
                     f"NaN={np.isnan(vals).sum():,}")
            grid[col] = vals
            grid = _fill_nan_with_median(grid, col)

        # If any required column is still missing, fall back to synthetic
        for col in ("slope_deg", "twi", "ndvi_mean", "k_factor"):
            if col not in grid.columns:
                log.warning(f"  [{col}] still missing → synthetic")
                grid = generate_synthetic(grid)
                break

    # Final NaN fill
    for col in ("slope_deg", "twi", "k_factor", "ndvi_mean"):
        if col in grid.columns:
            grid = _fill_nan_with_median(grid, col)

    grid.to_file(GRID_PHYSICAL_FILE, driver="GPKG")
    log.info(f"[SAVED] {GRID_PHYSICAL_FILE.name}  ({len(grid):,} cells)")

    log.info("\n  Variable summary:")
    for col in ("slope_deg", "twi", "k_factor", "ndvi_mean"):
        if col in grid.columns:
            log.info(f"    {col:15s}  mean={grid[col].mean():.3f}  "
                     f"std={grid[col].std():.3f}  "
                     f"min={grid[col].min():.3f}  "
                     f"max={grid[col].max():.3f}")

    log.info("Step 06 complete.")


if __name__ == "__main__":
    main()
