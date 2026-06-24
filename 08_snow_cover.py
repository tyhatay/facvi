"""
FACVI Pipeline — Step 08: MODIS Snow Cover (S5)
================================================
Downloads MOD10CM v6.1 monthly snow cover (0.05°, ~5 km) from
NASA Earthdata and computes mean annual snow-cover days per
grid cell → column `snow_days_mean` (S5).

Product  : MOD10CM.061  (Monthly CMG, global)
Period   : SNOW_START_YEAR – SNOW_END_YEAR (from config.py)
Layer    : Snow_Cover_Monthly_CMG  (0–100 % monthly snow cover)
CMG grid : 3600 rows (90° → -90°) × 7200 cols (-180° → 180°), 0.05°

Credentials — NEVER hard-coded here:
  Set env vars before running:
    export EARTHDATA_USERNAME=your_username
    export EARTHDATA_PASSWORD=your_password
  Or the script will prompt interactively.

HDF files are cached in data/modis_cache/.
Requires: pyhdf   (pip install pyhdf)

Output: snow_days_mean column appended to data/{CC}_grid_physical.gpkg
"""

import os
import getpass
import logging
import warnings
import numpy as np
import geopandas as gpd
from pathlib import Path

warnings.filterwarnings("ignore")

from config import (
    DATA_DIR, LOG_DIR,
    BBOX,
    SNOW_START_YEAR, SNOW_END_YEAR,
    COUNTRY_CODE,
    GRID_PHYSICAL_FILE,
)

LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "08_snow_cover.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

CACHE_DIR = DATA_DIR / "modis_cache"
MONTH_DAYS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]


# ── Credentials ───────────────────────────────────────────────────────────────

def get_credentials() -> tuple[str, str]:
    user = os.environ.get("EARTHDATA_USERNAME", "")
    pwd  = os.environ.get("EARTHDATA_PASSWORD", "")
    if not user:
        user = input("NASA Earthdata username: ").strip()
    if not pwd:
        pwd = getpass.getpass("NASA Earthdata password: ")
    if not user or not pwd:
        raise RuntimeError("NASA Earthdata credentials are required.")
    return user, pwd


def earthdata_session(user: str, pwd: str):
    import requests
    from requests.auth import HTTPBasicAuth
    token_url = "https://urs.earthdata.nasa.gov/api/users/tokens"
    r = requests.get(token_url, auth=HTTPBasicAuth(user, pwd), timeout=20)
    r.raise_for_status()
    tokens = r.json()
    if tokens:
        token = tokens[0]["access_token"]
        log.info("  Using existing Bearer token.")
    else:
        r2 = requests.post("https://urs.earthdata.nasa.gov/api/users/token",
                           auth=HTTPBasicAuth(user, pwd), timeout=20)
        r2.raise_for_status()
        token = r2.json()["access_token"]
        log.info("  New Bearer token created.")
    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {token}"})
    return session


# ── CMR granule lookup ────────────────────────────────────────────────────────

def get_granule_urls(year: int, month: int, session) -> list[str]:
    date_start = f"{year}-{month:02d}-01T00:00:00Z"
    date_end   = (f"{year}-{month+1:02d}-01T00:00:00Z" if month < 12
                  else f"{year+1}-01-01T00:00:00Z")
    try:
        r = session.get(
            "https://cmr.earthdata.nasa.gov/search/granules.json",
            params={"short_name": "MOD10CM",
                    "temporal[]": f"{date_start},{date_end}",
                    "page_size":  5},
            timeout=30,
        )
        r.raise_for_status()
        entries = r.json().get("feed", {}).get("entry", [])
        urls = []
        for e in entries:
            for link in e.get("links", []):
                if "data#" in link.get("rel", "") and link.get("href", "").endswith(".hdf"):
                    urls.append(link["href"])
            if not urls:
                for link in e.get("links", []):
                    if link.get("href", "").endswith(".hdf"):
                        urls.append(link["href"])
        return list(dict.fromkeys(urls))
    except Exception as ex:
        log.warning(f"  CMR query failed ({year}-{month:02d}): {ex}")
        return []


# ── HDF4 reading ──────────────────────────────────────────────────────────────

def read_snow_cover_hdf(path: Path) -> np.ndarray | None:
    """
    Read Snow_Cover_Monthly_CMG from a MOD10CM HDF4 file.
    Returns (3600, 7200) float32 array; invalid pixels = NaN.
    """
    try:
        from pyhdf.SD import SD, SDC
        hdf  = SD(str(path), SDC.READ)
        ds   = hdf.select("Snow_Cover_Monthly_CMG")
        data = ds[:].astype(np.float32)
        fill = float(ds.attributes().get("_FillValue", 255))
        hdf.end()
        data[data == fill] = np.nan
        data[data > 100]   = np.nan
        data[data < 0]     = np.nan
        return data
    except Exception as e:
        log.warning(f"  HDF4 read error ({path.name}): {e}")
        return None


# ── CMG coordinate mapping ────────────────────────────────────────────────────

def lonlat_to_cmg_rowcol(lons: np.ndarray, lats: np.ndarray,
                          n_rows: int = 3600, n_cols: int = 7200):
    pixel = 180.0 / n_rows     # 0.05°
    cols  = np.clip(((lons + 180.0) / pixel).astype(int), 0, n_cols - 1)
    rows  = np.clip(((90.0 - lats)  / pixel).astype(int), 0, n_rows - 1)
    return rows, cols


def _sample_snow(snow_arr: np.ndarray, lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
    rows, cols = lonlat_to_cmg_rowcol(lons, lats,
                                       snow_arr.shape[0], snow_arr.shape[1])
    return snow_arr[rows, cols]


# ── Download helper ───────────────────────────────────────────────────────────

def _download(url: str, out: Path, session) -> bool:
    if out.exists() and out.stat().st_size > 1_000_000:
        return True
    try:
        r = session.get(url, stream=True, timeout=300)
        r.raise_for_status()
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "wb") as f:
            for chunk in r.iter_content(65536):
                f.write(chunk)
        log.info(f"    Downloaded {out.name} ({out.stat().st_size/1e6:.1f} MB)")
        return True
    except Exception as e:
        log.warning(f"    Download failed ({out.name}): {e}")
        if out.exists():
            out.unlink()
        return False


# ── Main computation ──────────────────────────────────────────────────────────

def compute_s5(grid: gpd.GeoDataFrame, session) -> gpd.GeoDataFrame:
    if ("snow_days_mean" in grid.columns and
            grid["snow_days_mean"].notna().sum() > 1000):
        log.info("[S5] snow_days_mean already present.")
        return grid

    n = len(grid)
    geo = grid.to_crs("EPSG:4326")
    lons = geo.geometry.centroid.x.values
    lats = geo.geometry.centroid.y.values

    annual_totals = []

    for year in range(SNOW_START_YEAR, SNOW_END_YEAR + 1):
        log.info(f"  [MODIS] {year} ...")
        yearly = np.zeros(n, dtype=np.float32)
        months_ok = 0

        for month in range(1, 13):
            cache_f = CACHE_DIR / f"MOD10CM_{year}_{month:02d}.hdf"
            if not cache_f.exists():
                urls = get_granule_urls(year, month, session)
                if not urls or not _download(urls[0], cache_f, session):
                    continue

            snow_arr = read_snow_cover_hdf(cache_f)
            if snow_arr is None:
                continue

            # Days in month (leap-year aware)
            dim = MONTH_DAYS[month - 1]
            if month == 2 and year % 4 == 0 and (year % 100 != 0 or year % 400 == 0):
                dim = 29

            pct = _sample_snow(snow_arr, lons, lats)
            yearly += np.where(np.isnan(pct), 0.0, pct * dim / 100.0)
            months_ok += 1

        if months_ok > 0:
            if months_ok < 12:
                yearly *= 12.0 / months_ok   # scale up for missing months
            annual_totals.append(yearly)
            log.info(f"    {year}: {months_ok}/12 months  "
                     f"mean snow days = {yearly.mean():.1f}")
        else:
            log.warning(f"    {year}: no data, skipped.")

    if not annual_totals:
        raise RuntimeError("No MOD10CM data could be processed.")

    snow_mean = np.mean(np.stack(annual_totals), axis=0)
    grid["snow_days_mean"] = snow_mean

    log.info(f"\n[S5] snow_days_mean ({len(annual_totals)} years):")
    log.info(f"  Mean={snow_mean.mean():.1f}  Max={snow_mean.max():.1f}  "
             f"P95={np.percentile(snow_mean, 95):.1f} days/yr")
    log.info(f"  Zero-snow cells: {(snow_mean == 0).sum():,} "
             f"({(snow_mean == 0).mean()*100:.1f}%)")
    return grid


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("FACVI Step 08: MODIS Snow Cover (S5)")
    log.info("=" * 60)

    if not GRID_PHYSICAL_FILE.exists():
        raise FileNotFoundError(
            f"Physical grid not found: {GRID_PHYSICAL_FILE}\n"
            "Run 06_physical_variables.py first."
        )

    grid = gpd.read_file(GRID_PHYSICAL_FILE)
    log.info(f"[LOAD] {GRID_PHYSICAL_FILE.name}  ({len(grid):,} cells)")

    log.info("[AUTH] NASA Earthdata credentials required.")
    log.info("  Set EARTHDATA_USERNAME and EARTHDATA_PASSWORD env vars, or enter below.")
    user, pwd = get_credentials()
    session   = earthdata_session(user, pwd)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    grid = compute_s5(grid, session)
    grid.to_file(GRID_PHYSICAL_FILE, driver="GPKG")
    log.info(f"[SAVED] snow_days_mean (S5) appended to {GRID_PHYSICAL_FILE.name}")
    log.info("Step 08 complete.")


if __name__ == "__main__":
    main()
