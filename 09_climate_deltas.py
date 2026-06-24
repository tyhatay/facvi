"""
FACVI Pipeline — Step 09: WorldClim CMIP6 Climate Deltas (C1–C4)
=================================================================
Computes per-cell future minus baseline climate change (Δ) for
four scenarios × two periods from WorldClim v2.1 CMIP6 data.

Variables:
  C1 — delta_bio1   ΔT mean annual (°C)          BIO01
  C2 — delta_bio12  ΔP annual (mm/yr)             BIO12
  C3 — delta_bio5   ΔT max warmest month (°C)     BIO05
  C4 — delta_bio15  ΔP seasonality (coefficient)  BIO15

WorldClim CMIP6 files are multi-band GeoTIFFs (19 bands = BIO01–BIO19).
Band index = BIO number (band 1 = BIO01, band 12 = BIO12, etc.)

Multiple GCMs are ensemble-averaged for each scenario × period.

Inputs:
  data/worldclim/baseline/         wc2.1_2.5m_bio{N}.tif
  data/worldclim/ssp245_50/        wc2.1_2.5m_bioc_{MODEL}_ssp245_2041-2060.tif
  data/worldclim/ssp245_90/        wc2.1_2.5m_bioc_{MODEL}_ssp245_2081-2100.tif
  data/worldclim/ssp585_50/
  data/worldclim/ssp585_90/

Output: data/{CC}_grid_climate.gpkg
"""

import numpy as np
import logging
import warnings
import geopandas as gpd
from pathlib import Path

warnings.filterwarnings("ignore")

from config import (
    DATA_DIR, WC_DIR, LOG_DIR,
    COUNTRY_CODE, SCENARIOS, CMIP6_MODELS,
    GRID_PHYSICAL_FILE, GRID_CLIMATE_FILE,
)

LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "09_climate_deltas.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# Map FACVI variable code → (BIO number, description)
BIO_VARS = {
    "C1": (1,  "ΔT mean annual (°C)"),
    "C2": (12, "ΔP annual (mm)"),
    "C3": (5,  "ΔT max warmest month (°C)"),
    "C4": (15, "ΔP seasonality (coefficient)"),
}


# ── Centroid coordinates ──────────────────────────────────────────────────────

def _centroids_wgs84(grid: gpd.GeoDataFrame) -> np.ndarray:
    geo = grid.to_crs("EPSG:4326")
    return np.column_stack([geo.geometry.centroid.x.values,
                            geo.geometry.centroid.y.values])


# ── Raster sampling ───────────────────────────────────────────────────────────

def sample_raster(path: Path, coords: np.ndarray, band: int = 1) -> np.ndarray:
    """Sample a raster at WGS84 (lon, lat) coordinates, returning band values."""
    import rasterio
    with rasterio.open(path) as src:
        if src.crs.to_epsg() != 4326:
            from pyproj import Transformer
            tr = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
            xs, ys = tr.transform(coords[:, 0], coords[:, 1])
            pts = list(zip(xs, ys))
        else:
            pts = list(zip(coords[:, 0], coords[:, 1]))

        nd   = src.nodata if src.nodata is not None else -9999
        band = min(band, src.count)
        vals = np.array([v[0] for v in src.sample(pts, indexes=band)],
                        dtype=np.float64)

    return np.where(vals == nd, np.nan, vals)


def _ensemble_mean(scenario_dir: Path, bio_num: int,
                   coords: np.ndarray) -> np.ndarray | None:
    """
    Average WorldClim CMIP6 values across all configured GCMs.
    Accepts both single-variable and multi-band (19-band) files.
    """
    arrays = []
    for model in CMIP6_MODELS:
        # Single-variable filename
        single = scenario_dir / f"wc2.1_2.5m_bio{bio_num:02d}_{model}.tif"
        # Multi-band filename patterns used by WorldClim CMIP6
        candidates = [single] + sorted(
            scenario_dir.glob(f"*{model}*.tif")
        )
        found = next((p for p in candidates if p.exists()), None)
        if found is None:
            log.debug(f"    [{model}] not found in {scenario_dir.name}")
            continue
        try:
            arr = sample_raster(found, coords, band=bio_num)
            arrays.append(arr)
        except Exception as e:
            log.warning(f"    [{model}] read error: {e}")

    if not arrays:
        return None
    log.info(f"    Ensemble: {len(arrays)}/{len(CMIP6_MODELS)} GCMs")
    return np.nanmean(np.stack(arrays), axis=0)


# ── Synthetic fallback ────────────────────────────────────────────────────────

def _synthetic_deltas(grid: gpd.GeoDataFrame, scenario_key: str) -> dict[str, np.ndarray]:
    """
    Produce spatially-structured synthetic climate deltas when WorldClim
    files are absent.  Values are NOT suitable for publication.
    Based on approximate IPCC AR6 regional patterns.
    """
    ssp    = scenario_key.split("_")[0]
    period = scenario_key.split("_")[1]

    n   = len(grid)
    rng = np.random.default_rng(seed=hash(scenario_key) % 2**32)
    geo = grid.to_crs("EPSG:4326")
    lon = geo.geometry.centroid.x.values
    lat = geo.geometry.centroid.y.values

    ef = {"ssp245": 1.0, "ssp585": 2.2}[ssp]
    pf = {"2050": 1.0, "2100": 1.8}[period]
    sc = ef * pf

    deltas: dict[str, np.ndarray] = {}

    # C1: ΔT mean annual
    deltas["C1"] = sc * (1.5 + 0.5 * (lat - lat.min()) / (lat.max() - lat.min()) +
                         rng.normal(0, 0.3, n))

    # C2: ΔP annual (north gains, south loses)
    deltas["C2"] = sc * ((lat - 39) * 10 + rng.normal(0, 8, n))

    # C3: ΔT max warmest month
    deltas["C3"] = sc * (2.0 + 0.4 * (lon - lon.min()) / (lon.max() - lon.min()) +
                         rng.normal(0, 0.4, n))

    # C4: ΔP seasonality (increases with warming)
    inland = np.clip(1 - np.abs(lon - lon.mean()) / 10, 0, 1)
    deltas["C4"] = sc * (2.0 + 3.0 * inland + rng.normal(0, 0.8, n))

    return deltas


# ── Per-scenario processing ───────────────────────────────────────────────────

def process_scenario(grid: gpd.GeoDataFrame,
                     scenario_key: str,
                     scenario_cfg: dict,
                     baseline: dict[str, np.ndarray],
                     coords: np.ndarray,
                     use_real: bool) -> gpd.GeoDataFrame:
    log.info(f"\n[SCENARIO] {scenario_key}")
    folder = scenario_cfg["folder"]
    scenario_dir = WC_DIR / folder

    if use_real and scenario_dir.exists() and any(scenario_dir.glob("*.tif")):
        all_ok = True
        deltas: dict[str, np.ndarray] = {}
        for code, (bio_num, desc) in BIO_VARS.items():
            log.info(f"  [{code}] BIO{bio_num:02d}: {desc}")
            proj = _ensemble_mean(scenario_dir, bio_num, coords)
            if proj is None:
                log.warning(f"  [{code}] No GCM files found — falling back to synthetic.")
                all_ok = False
                break
            delta = proj - baseline[code]
            log.info(f"    delta mean={np.nanmean(delta):.3f}  std={np.nanstd(delta):.3f}")
            deltas[code] = delta

        if not all_ok:
            deltas = _synthetic_deltas(grid, scenario_key)
    else:
        log.warning(f"  WorldClim folder not found: {folder}  → using synthetic values")
        deltas = _synthetic_deltas(grid, scenario_key)

    for code, arr in deltas.items():
        grid[f"{code}_{scenario_key}"] = arr

    return grid


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if GRID_CLIMATE_FILE.exists():
        gdf = gpd.read_file(GRID_CLIMATE_FILE)
        log.info(f"[CACHED] Climate grid: {len(gdf):,} cells in {GRID_CLIMATE_FILE.name}")
        return

    log.info("=" * 60)
    log.info("FACVI Step 09: Climate Deltas (C1–C4)")
    log.info("=" * 60)

    if not GRID_PHYSICAL_FILE.exists():
        raise FileNotFoundError(
            f"Physical grid not found: {GRID_PHYSICAL_FILE}\n"
            "Run 06_physical_variables.py (and 07/08) first."
        )

    grid = gpd.read_file(GRID_PHYSICAL_FILE)
    log.info(f"[LOAD] {GRID_PHYSICAL_FILE.name}  ({len(grid):,} cells)")

    coords = _centroids_wgs84(grid)

    # Check WorldClim availability
    baseline_dir = WC_DIR / "baseline"
    use_real = baseline_dir.exists() and any(baseline_dir.glob("*.tif"))
    if use_real:
        log.info("[WORLDCLIM] Baseline data found.")
    else:
        log.warning("[WORLDCLIM] No baseline data — synthetic deltas will be used.")
        log.warning(f"  Download from: https://worldclim.org/data/worldclim21.html")

    # Read baseline
    baseline: dict[str, np.ndarray] = {}
    if use_real:
        log.info("[BASELINE] Reading WorldClim 1970-2000 ...")
        for code, (bio_num, desc) in BIO_VARS.items():
            # Accept both wc2.1_2.5m_bio01.tif and wc2.1_2.5m_bio1.tif
            candidates = [
                baseline_dir / f"wc2.1_2.5m_bio{bio_num:02d}.tif",
                baseline_dir / f"wc2.1_2.5m_bio{bio_num}.tif",
            ] + sorted(baseline_dir.glob(f"*bio{bio_num:02d}*.tif"))
            path = next((p for p in candidates if p.exists()), None)
            if path is None:
                log.warning(f"  [{code}] Baseline BIO{bio_num:02d} not found → synthetic")
                use_real = False
                baseline.clear()
                break
            vals = sample_raster(path, coords, band=1)
            baseline[code] = vals
            log.info(f"  [{code}] BIO{bio_num:02d} baseline: mean={np.nanmean(vals):.3f}")

    # Process each scenario
    for scenario_key, scenario_cfg in SCENARIOS.items():
        grid = process_scenario(
            grid, scenario_key, scenario_cfg, baseline, coords, use_real
        )

    grid.to_file(GRID_CLIMATE_FILE, driver="GPKG")
    log.info(f"[SAVED] {GRID_CLIMATE_FILE.name}  ({len(grid):,} cells)")
    log.info("Step 09 complete.")


if __name__ == "__main__":
    main()
