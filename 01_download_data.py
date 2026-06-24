"""
FACVI Pipeline — Step 01: Data Download
=========================================
Downloads all required input datasets:

  [1] WorldClim v2.1 Baseline (1970-2000)          — no login required
  [2] WorldClim CMIP6 SSP projections               — no login required
      (4 scenarios × 4 GCMs; 2041-2060 and 2081-2100)
  [3] SRTM 30m DEM (study-area tiles)              — NASA Earthdata login
  [4] HydroRIVERS drainage network                 — no login required
  [5] MODIS MOD13A3 NDVI (monthly 1 km)            — NASA Earthdata login
  [6] SoilGrids v2.0 (clay/silt/sand/SOC → K-factor) — no login required

NASA Earthdata account (free): https://urs.earthdata.nasa.gov/users/new
Set credentials as environment variables to avoid interactive prompts:
  export EARTHDATA_USER=your_username
  export EARTHDATA_PASS=your_password

Usage:
  python 01_download_data.py               # download everything
  python 01_download_data.py --skip-nasa   # skip SRTM and MODIS (no login needed)
  python 01_download_data.py --only worldclim
  python 01_download_data.py --check       # report existing files and exit
"""

import os
import sys
import time
import shutil
import zipfile
import argparse
import logging
import requests
from pathlib import Path
from tqdm import tqdm

from config import (
    BASE_DIR, DATA_DIR, RASTER_DIR, WC_DIR, LOG_DIR,
    BBOX, CMIP6_MODELS, SCENARIOS,
    SRTM_LAT_RANGE, SRTM_LON_RANGE,
    HYDRORIVERS_URL, NDVI_START, NDVI_END,
    COUNTRY_CODE,
)

LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "01_download.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

WC_BASE_URL = "https://geodata.ucdavis.edu/climate/worldclim/2_1"

WC_BASELINE_VARS = {
    "bio1":  "wc2.1_2.5m_bio_1.tif",
    "bio5":  "wc2.1_2.5m_bio_5.tif",
    "bio12": "wc2.1_2.5m_bio_12.tif",
    "bio15": "wc2.1_2.5m_bio_15.tif",
}

CMIP6_VARS = ["bio1", "bio5", "bio12", "bio15"]

# SRTM tile names derived from config bounding box
SRTM_TILES = [
    f"N{lat:02d}E0{lon:03d}"
    for lat in range(*SRTM_LAT_RANGE)
    for lon in range(*SRTM_LON_RANGE)
]


# ── Generic download helper ───────────────────────────────────────────────────

def download_file(url: str, dest_path: Path, desc: str = "",
                  chunk_size: int = 1024 * 1024, auth=None,
                  timeout: int = 60) -> bool:
    """Download a file with resume support."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if dest_path.exists() and dest_path.stat().st_size > 1000:
        log.info(f"  [CACHED] {dest_path.name}")
        return True

    temp_path = dest_path.with_suffix(dest_path.suffix + ".part")
    downloaded = temp_path.stat().st_size if temp_path.exists() else 0

    headers = {}
    if downloaded > 0:
        headers["Range"] = f"bytes={downloaded}-"
        log.info(f"  [RESUME] from {downloaded / 1024 / 1024:.1f} MB")

    try:
        response = requests.get(
            url, stream=True, headers=headers,
            auth=auth, timeout=timeout, allow_redirects=True,
        )
        if response.status_code == 416:
            temp_path.rename(dest_path)
            return True
        if response.status_code not in (200, 206):
            log.error(f"  HTTP {response.status_code}: {url}")
            return False

        total = int(response.headers.get("content-length", 0)) + downloaded
        mode = "ab" if downloaded > 0 else "wb"

        with open(temp_path, mode) as f, tqdm(
            total=total, initial=downloaded, unit="B", unit_scale=True,
            desc=desc[:40] if desc else dest_path.name[:40], leave=False,
        ) as bar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))

        temp_path.rename(dest_path)
        log.info(f"  OK {dest_path.name} ({dest_path.stat().st_size / 1024 / 1024:.1f} MB)")
        return True

    except requests.exceptions.Timeout:
        log.error(f"  Timeout: {url}")
        return False
    except requests.exceptions.ConnectionError as e:
        log.error(f"  Connection error: {e}")
        return False
    except Exception as e:
        log.error(f"  Download error: {e}")
        return False


def extract_zip(zip_path: Path, extract_to: Path, remove_zip: bool = True):
    log.info(f"  [ZIP] Extracting {zip_path.name} -> {extract_to}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)
    if remove_zip:
        zip_path.unlink()


# ── 1. WorldClim Baseline ─────────────────────────────────────────────────────

def download_worldclim_baseline() -> bool:
    log.info("\n[1/5] WorldClim v2.1 Baseline (1970-2000)")
    out_dir = WC_DIR / "baseline"
    out_dir.mkdir(parents=True, exist_ok=True)

    tif_files = [out_dir / f"wc2.1_2.5m_bio{v}.tif" for v in ["1", "5", "12", "15"]]
    if all(p.exists() and p.stat().st_size > 1_000_000 for p in tif_files):
        log.info("  [CACHED] Baseline TIF files present.")
        return True

    zip_url  = f"{WC_BASE_URL}/base/wc2.1_2.5m_bio.zip"
    zip_path = out_dir / "wc2.1_2.5m_bio.zip"

    if not download_file(zip_url, zip_path, "WorldClim Baseline BIO"):
        return False

    log.info("  Extracting BIO1, BIO5, BIO12, BIO15 ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        all_files = zf.namelist()
        target = [f for f in all_files
                  if any(f"bio_{v}" in f or f"bio{v}" in f
                         for v in ["1", "5", "12", "15"])]
        if not target:
            log.warning("  Target files not found by name — extracting all.")
            zf.extractall(out_dir)
        else:
            for fname in target:
                zf.extract(fname, out_dir)
                log.info(f"    -> {fname}")

    zip_path.unlink(missing_ok=True)

    for bio_num, standard in [("1", "wc2.1_2.5m_bio1.tif"), ("5", "wc2.1_2.5m_bio5.tif"),
                               ("12", "wc2.1_2.5m_bio12.tif"), ("15", "wc2.1_2.5m_bio15.tif")]:
        for c in list(out_dir.rglob(f"*bio*{bio_num}*.tif")) + list(out_dir.rglob(f"*bio_{bio_num}.tif")):
            target = out_dir / standard
            if c.name != standard and not target.exists():
                c.rename(target)
                log.info(f"  Renamed: {c.name} -> {standard}")

    log.info("  WorldClim Baseline done.")
    return True


# ── 2. WorldClim CMIP6 ───────────────────────────────────────────────────────

def download_worldclim_cmip6() -> bool:
    log.info("\n[2/5] WorldClim CMIP6 SSP projections")

    # Build scenario list: from config SCENARIOS dict, derive (ssp, period, folder)
    scenario_configs = {}
    for key, cfg in SCENARIOS.items():
        scenario_configs[key] = (cfg["ssp"], _period_label(cfg["folder"]), cfg["folder"])

    # Add 2081-2100 period if not already present
    extra_90 = {}
    for key, cfg in SCENARIOS.items():
        if cfg["folder"].endswith("_90"):
            extra_90[key] = (cfg["ssp"], "2081-2100", cfg["folder"])
    if extra_90:
        scenario_configs.update(extra_90)

    total = len(CMIP6_MODELS) * len(scenario_configs)
    done, failed = 0, []

    for scenario_key, (ssp, period, folder) in scenario_configs.items():
        out_dir = WC_DIR / folder
        out_dir.mkdir(parents=True, exist_ok=True)

        for model in CMIP6_MODELS:
            existing = list(out_dir.glob(f"*{model}*bio12*.tif")) + \
                       list(out_dir.glob(f"*{model}*bio_12*.tif"))
            if existing:
                log.info(f"  [CACHED] {scenario_key}/{model}")
                done += 1
                continue

            zip_name = f"wc2.1_2.5m_bioc_{model}_{ssp}_{period}.zip"
            url      = f"{WC_BASE_URL}/fut/2.5m/{zip_name}"
            zip_path = out_dir / zip_name

            log.info(f"  [{scenario_key}] {model} ...")
            if not download_file(url, zip_path, f"{model} {ssp} {period}"):
                failed.append(f"{model}/{scenario_key}")
                continue

            try:
                with zipfile.ZipFile(zip_path, "r") as zf:
                    bio_files = [f for f in zf.namelist()
                                 if any(f.endswith(f"bio{v}.tif") or f.endswith(f"bio_{v}.tif")
                                        for v in ["1", "5", "12", "15"])]
                    if bio_files:
                        for f in bio_files:
                            zf.extract(f, out_dir)
                    else:
                        zf.extractall(out_dir)
                zip_path.unlink()
                for tif in out_dir.rglob("*.tif"):
                    if tif.parent != out_dir:
                        dest = out_dir / tif.name
                        if not dest.exists():
                            tif.rename(dest)
                for sub in sorted(out_dir.iterdir()):
                    if sub.is_dir():
                        try:
                            sub.rmdir()
                        except OSError:
                            pass
                done += 1
                log.info(f"    OK {done}/{total}")
            except zipfile.BadZipFile:
                log.error(f"  Corrupt ZIP: {zip_name}")
                zip_path.unlink(missing_ok=True)
                failed.append(f"{model}/{scenario_key}")

    if failed:
        log.warning(f"  Failed: {failed}")
    log.info(f"  WorldClim CMIP6 done: {done}/{total}")
    return done > 0


def _period_label(folder: str) -> str:
    """Map folder suffix to WorldClim period string."""
    suffix = folder.split("_")[-1]
    return {"50": "2041-2060", "70": "2061-2080", "90": "2081-2100"}.get(suffix, "2041-2060")


# ── 3. SRTM DEM ──────────────────────────────────────────────────────────────

def _earthaccess_login():
    import earthaccess
    user = (os.environ.get("EARTHDATA_USERNAME") or
            os.environ.get("EARTHDATA_USER", "")).strip()
    pw   = (os.environ.get("EARTHDATA_PASSWORD") or
            os.environ.get("EARTHDATA_PASS", "")).strip()
    if user and pw:
        os.environ["EARTHDATA_USERNAME"] = user
        os.environ["EARTHDATA_PASSWORD"] = pw
        return earthaccess.login(strategy="environment")
    return earthaccess.login(strategy="interactive")


def download_srtm() -> bool:
    log.info("\n[3/5] SRTM 30m DEM (NASA LP DAAC)")
    srtm_dir = RASTER_DIR / "srtm_tiles"
    srtm_dir.mkdir(parents=True, exist_ok=True)

    try:
        import earthaccess
        _earthaccess_login()
    except Exception as e:
        log.warning(f"  earthaccess login failed: {e}")
        log.info("  Manual download: https://dwtkns.com/srtm30m/")
        return False

    try:
        results = earthaccess.search_data(
            short_name="SRTMGL1", version="003",
            bounding_box=(BBOX["west"], BBOX["south"], BBOX["east"], BBOX["north"]),
        )
        log.info(f"  {len(results)} SRTM tiles found.")
        earthaccess.download(results, srtm_dir)
        _merge_srtm_tiles(srtm_dir)
        return True
    except Exception as e:
        log.error(f"  SRTM download error: {e}")
        log.info("  Manual download: https://dwtkns.com/srtm30m/")
        return False


def _merge_srtm_tiles(srtm_dir: Path):
    try:
        import rasterio
        from rasterio.merge import merge

        zip_files = list(srtm_dir.glob("*.zip"))
        if zip_files:
            for zp in zip_files:
                try:
                    with zipfile.ZipFile(zp, "r") as zf:
                        zf.extractall(srtm_dir)
                    zp.unlink()
                except Exception as e:
                    log.warning(f"  Cannot open ZIP {zp.name}: {e}")

        tif_files = list(srtm_dir.glob("*.hgt")) + list(srtm_dir.glob("*.tif"))
        if not tif_files:
            log.warning("  No tile files found; skipping mosaic.")
            return

        log.info(f"  Merging {len(tif_files)} tiles ...")
        datasets = [rasterio.open(f) for f in tif_files]
        mosaic, transform = merge(datasets)

        meta = datasets[0].meta.copy()
        meta.update({"driver": "GTiff", "height": mosaic.shape[1],
                     "width": mosaic.shape[2], "transform": transform,
                     "crs": "EPSG:4326", "compress": "lzw"})

        out_path = RASTER_DIR / f"srtm_{COUNTRY_CODE}_raw.tif"
        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(mosaic)
        for ds in datasets:
            ds.close()

        log.info(f"  Mosaic saved: {out_path}")
        log.info("  Compute slope and TWI in SAGA GIS:")
        log.info("    Terrain Analysis -> Morphometry -> Slope")
        log.info("    Terrain Analysis -> Hydrology -> SAGA Wetness Index")
        log.info(f"  Save outputs to:")
        log.info(f"    {RASTER_DIR}/srtm_slope_{COUNTRY_CODE}.tif")
        log.info(f"    {RASTER_DIR}/srtm_twi_{COUNTRY_CODE}.tif")
    except Exception as e:
        log.error(f"  Mosaic error: {e}")


# ── 4. HydroRIVERS ───────────────────────────────────────────────────────────

def download_hydrorivers() -> bool:
    log.info("\n[4/5] HydroRIVERS v1.0")
    hydro_dir = RASTER_DIR / "hydrosheds"
    hydro_dir.mkdir(parents=True, exist_ok=True)

    if list(hydro_dir.glob("*.shp")):
        log.info("  [CACHED] HydroRIVERS shapefile present.")
        return True

    log.info("  HydroSHEDS terms of use: https://www.hydrosheds.org/page/license")
    zip_path = hydro_dir / Path(HYDRORIVERS_URL).name

    success = download_file(HYDRORIVERS_URL, zip_path, "HydroRIVERS", timeout=120)
    if not success:
        log.warning("  HydroRIVERS download failed.")
        log.info("  Manual download: https://www.hydrosheds.org/products/hydrorivers")
        return False

    extract_zip(zip_path, hydro_dir, remove_zip=False)
    zip_path.unlink(missing_ok=True)
    log.info(f"  HydroRIVERS saved to: {hydro_dir}")
    return True


# ── 5. MODIS NDVI ────────────────────────────────────────────────────────────

def download_modis_ndvi() -> bool:
    log.info(f"\n[5/5] MODIS MOD13A3 NDVI ({NDVI_START} to {NDVI_END})")
    ndvi_dir = RASTER_DIR / "modis_ndvi"
    ndvi_dir.mkdir(parents=True, exist_ok=True)

    try:
        import earthaccess
        _earthaccess_login()
    except Exception as e:
        log.warning(f"  earthaccess login failed: {e}")
        return False

    try:
        results = earthaccess.search_data(
            short_name="MOD13A3", version="061",
            bounding_box=(BBOX["west"], BBOX["south"], BBOX["east"], BBOX["north"]),
            temporal=(NDVI_START, NDVI_END),
        )
        log.info(f"  {len(results)} MODIS granules found.")
        if len(results) > 200:
            results = results[:120]
        earthaccess.download(results, ndvi_dir)
        _compute_mean_ndvi(ndvi_dir)
        return True
    except Exception as e:
        log.error(f"  MODIS download error: {e}")
        log.info("  Alternative: use NASA AppEEARS — https://appeears.earthdatacloud.nasa.gov/")
        return False


def _compute_mean_ndvi(ndvi_dir: Path):
    import re
    import numpy as np
    import rasterio
    from rasterio.crs import CRS
    from rasterio.transform import from_origin
    from osgeo import gdal
    gdal.UseExceptions()

    TILE_SIZE = 1200
    PIX_SIZE  = 926.625433055833
    TILE_M    = TILE_SIZE * PIX_SIZE
    X0 = -20015109.354
    Y0 =  10007554.677
    SINU_PROJ4 = "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +a=6371007.181 +b=6371007.181 +units=m +no_defs"
    crs_sinu = CRS.from_proj4(SINU_PROJ4)

    hdf_files = sorted(list(ndvi_dir.glob("*.hdf")) + list(ndvi_dir.glob("*.HDF")))
    if not hdf_files:
        log.warning("  No HDF files found.")
        return

    log.info(f"  Processing NDVI from {len(hdf_files)} HDF files ...")

    try:
        from pyhdf.SD import SD, SDC
    except ImportError:
        log.error("  pyhdf not installed. Run: pip install pyhdf")
        return

    tiles: dict = {}
    for hdf in hdf_files:
        m = re.search(r"h(\d{2})v(\d{2})", hdf.stem)
        if not m:
            continue
        h, v = int(m.group(1)), int(m.group(2))
        try:
            sd = SD(str(hdf), SDC.READ)
            ndvi_ds  = sd.select("1 km monthly NDVI")
            ndvi_raw = ndvi_ds[:].astype(float)
            attrs    = ndvi_ds.attributes()
            scale    = float(attrs.get("scale_factor", 0.0001))
            fill     = float(attrs.get("_FillValue", -28672))
            offset   = float(attrs.get("add_offset", 0.0))
            sd.end()
        except Exception as e:
            log.debug(f"  pyhdf read error {hdf.name}: {e}")
            continue

        ndvi = np.where(ndvi_raw == fill, np.nan, ndvi_raw / scale + offset)
        ndvi = np.where(ndvi < -1, np.nan, ndvi)
        tiles.setdefault((h, v), []).append(ndvi)

    if not tiles:
        log.warning("  No NDVI data could be read.")
        return

    log.info(f"  {len(tiles)} unique tiles; computing temporal means ...")
    tmp_tifs = []
    for (h, v), arrays in tiles.items():
        mean_ndvi = np.nanmean(np.stack(arrays, axis=0), axis=0).astype("float32")
        x_min = X0 + h * TILE_M
        y_max = Y0 - v * TILE_M
        transform = from_origin(x_min, y_max, PIX_SIZE, PIX_SIZE)

        tmp_path = ndvi_dir / f"_tmp_tile_h{h:02d}v{v:02d}.tif"
        with rasterio.open(tmp_path, "w", driver="GTiff", dtype="float32",
                           count=1, crs=crs_sinu, transform=transform,
                           height=TILE_SIZE, width=TILE_SIZE,
                           nodata=-9999, compress="lzw") as dst:
            dst.write(np.where(np.isnan(mean_ndvi), -9999, mean_ndvi), 1)
        tmp_tifs.append(tmp_path)

    out_path = RASTER_DIR / f"ndvi_{COUNTRY_CODE}.tif"
    warp_opts = gdal.WarpOptions(
        format="GTiff",
        dstSRS="EPSG:4326",
        outputBounds=(BBOX["west"], BBOX["south"], BBOX["east"], BBOX["north"]),
        outputBoundsSRS="EPSG:4326",
        xRes=0.009009, yRes=0.009009,
        resampleAlg=gdal.GRA_Average,
        srcNodata=-9999, dstNodata=-9999,
        creationOptions=["COMPRESS=LZW"],
    )
    gdal.Warp(str(out_path), [str(p) for p in tmp_tifs], options=warp_opts)
    for p in tmp_tifs:
        p.unlink(missing_ok=True)

    if out_path.exists():
        log.info(f"  NDVI saved: {out_path.name} ({out_path.stat().st_size / 1024 / 1024:.1f} MB)")
    else:
        log.error("  NDVI mosaic failed (gdal.Warp returned no output).")


# ── SoilGrids → K-factor ─────────────────────────────────────────────────────

SOILGRIDS_LAYERS = {
    "clay": {"scale": 0.1,  "unit": "g/kg", "desc": "Clay content 0-30cm (%)"},
    "silt": {"scale": 0.1,  "unit": "g/kg", "desc": "Silt content 0-30cm (%)"},
    "sand": {"scale": 0.1,  "unit": "g/kg", "desc": "Sand content 0-30cm (%)"},
    "soc":  {"scale": 0.01, "unit": "dg/kg", "desc": "SOC 0-30cm (%)"},
}


def download_soilgrids() -> bool:
    log.info("\n[6/6] SoilGrids v2.0 — soil variables (K-factor)")
    k_out = RASTER_DIR / f"k_factor_{COUNTRY_CODE}.tif"
    if k_out.exists() and k_out.stat().st_size > 100_000:
        log.info("  [CACHED] K-factor raster present.")
        return True

    try:
        from osgeo import gdal
        import rasterio
        import numpy as np
        gdal.UseExceptions()
        gdal.SetConfigOption("GDAL_HTTP_TIMEOUT", "120")
        gdal.SetConfigOption("CPL_VSIL_CURL_CACHE_SIZE", "128000000")
    except ImportError as e:
        log.error(f"  GDAL or rasterio not found: {e}")
        return False

    soil_dir = RASTER_DIR / "soilgrids"
    soil_dir.mkdir(parents=True, exist_ok=True)

    ISRIC_BASE = "https://files.isric.org/soilgrids/latest/data"
    DEPTHS = [("0-5cm", 5), ("5-15cm", 10), ("15-30cm", 15)]

    downloaded = {}
    for var_name, cfg in SOILGRIDS_LAYERS.items():
        out_path = soil_dir / f"soilgrids_{var_name}_{COUNTRY_CODE}.tif"
        if out_path.exists() and out_path.stat().st_size > 10_000:
            log.info(f"  [CACHED] {var_name}")
            downloaded[var_name] = out_path
            continue

        log.info(f"  [{var_name}] {cfg['desc']}")
        depth_arrays, meta = [], None

        for depth, weight in DEPTHS:
            vrt_url = f"/vsicurl/{ISRIC_BASE}/{var_name}/{var_name}_{depth}_mean.vrt"
            tmp_path = str(soil_dir / f"_{var_name}_{depth}.tif")
            try:
                ds = gdal.Warp(
                    tmp_path, vrt_url, format="GTiff", dstSRS="EPSG:4326",
                    outputBounds=(BBOX["west"], BBOX["south"], BBOX["east"], BBOX["north"]),
                    outputBoundsSRS="EPSG:4326",
                    xRes=0.002083333, yRes=0.002083333,
                    resampleAlg=gdal.GRA_Bilinear,
                    creationOptions=["COMPRESS=LZW", "TILED=YES"],
                )
                ds = None
                with rasterio.open(tmp_path) as src:
                    arr = src.read(1).astype(float)
                    nd  = src.nodata if src.nodata is not None else -32768
                    arr = np.where(arr == nd, np.nan, arr)
                    depth_arrays.append(arr * weight)
                    if meta is None:
                        meta = src.meta.copy()
                Path(tmp_path).unlink(missing_ok=True)
            except Exception as e:
                log.error(f"    [{depth}] Error: {e}")
                Path(tmp_path).unlink(missing_ok=True)

        if not depth_arrays:
            log.warning(f"  [{var_name}] No depth layers downloaded.")
            continue

        actual_w = sum(w for _, w in DEPTHS[:len(depth_arrays)])
        mean_arr = np.nansum(depth_arrays, axis=0) / actual_w
        meta.update({"driver": "GTiff", "dtype": "float32", "nodata": -9999, "compress": "lzw"})
        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(np.where(np.isnan(mean_arr), -9999, mean_arr).astype("float32"), 1)
        downloaded[var_name] = out_path

    if downloaded:
        _compute_k_factor(downloaded, soil_dir)
        return True
    return False


def _compute_k_factor(downloaded: dict, soil_dir: Path):
    """Compute EPIC K-factor from SoilGrids texture variables."""
    log.info("  [K-FACTOR] Computing via EPIC formula (Williams 1990)...")
    try:
        import rasterio
        import numpy as np
        from rasterio.warp import reproject, Resampling

        ref_var = next(iter(downloaded))
        with rasterio.open(downloaded[ref_var]) as ref:
            meta   = ref.meta.copy()
            shape  = (ref.height, ref.width)
            transform = ref.transform
            crs    = ref.crs

        arrays = {}
        for var_name, path in downloaded.items():
            with rasterio.open(path) as src:
                arr = src.read(1).astype(float)
                nd  = src.nodata if src.nodata else -32768
                arr = np.where(arr == nd, np.nan, arr) * SOILGRIDS_LAYERS[var_name]["scale"]
                if arr.shape != shape:
                    arr_r = np.full(shape, np.nan, dtype=float)
                    reproject(source=arr, destination=arr_r,
                              src_transform=src.transform, src_crs=src.crs,
                              dst_transform=transform, dst_crs=crs,
                              resampling=Resampling.bilinear)
                    arr = arr_r
                arrays[var_name] = arr

        defaults = {"clay": 25.0, "silt": 30.0, "sand": 45.0, "soc": 1.2}
        for var in defaults:
            if var not in arrays:
                log.warning(f"  [{var}] missing — using default: {defaults[var]}")
                arrays[var] = np.full(shape, defaults[var], dtype=float)

        clay, silt, sand, soc = arrays["clay"], arrays["silt"], arrays["sand"], arrays["soc"]

        f1 = 0.2 + 0.3 * np.exp(-0.0256 * sand * (1 - silt / 100))
        denom_f2 = np.where(clay + silt == 0, 0.001, clay + silt)
        f2 = (silt / denom_f2) ** 0.3
        f3 = 1 - (0.25 * soc) / (soc + np.exp(3.72 - 2.95 * soc))
        SN1 = (100 - sand) / 100
        f4 = 1 - (0.7 * SN1) / (SN1 + np.exp(-5.51 + 22.9 * SN1))

        K = np.clip(f1 * f2 * f3 * f4, 0.01, 0.65)
        nan_mask = np.isnan(clay) | np.isnan(silt) | np.isnan(sand) | np.isnan(soc)
        K = np.where(nan_mask, np.nan, K)

        log.info(f"  K-factor: mean={np.nanmean(K):.4f}, "
                 f"std={np.nanstd(K):.4f}, range=[{np.nanmin(K):.4f}, {np.nanmax(K):.4f}]")

        out_path = RASTER_DIR / f"k_factor_{COUNTRY_CODE}.tif"
        meta.update({"driver": "GTiff", "dtype": "float32", "count": 1,
                     "nodata": -9999, "compress": "lzw"})
        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(np.where(np.isnan(K), -9999, K).astype("float32"), 1)
        log.info(f"  K-factor saved: {out_path.name}")

    except Exception as e:
        import traceback
        log.error(f"  K-factor computation error: {e}\n{traceback.format_exc()}")


# ── Status check ──────────────────────────────────────────────────────────────

def check_status():
    print(f"\n{'='*60}\nDATA STATUS REPORT\n{'='*60}")
    checks = [
        ("WorldClim Baseline BIO1",  WC_DIR / "baseline/wc2.1_2.5m_bio1.tif",  True),
        ("WorldClim Baseline BIO12", WC_DIR / "baseline/wc2.1_2.5m_bio12.tif", True),
        ("WorldClim SSP245/2050",    WC_DIR / "ssp245_50",                       True),
        ("WorldClim SSP585/2100",    WC_DIR / "ssp585_90",                       True),
        ("SRTM Slope",               RASTER_DIR / f"srtm_slope_{COUNTRY_CODE}.tif", False),
        ("SRTM TWI",                 RASTER_DIR / f"srtm_twi_{COUNTRY_CODE}.tif",   False),
        ("MODIS NDVI",               RASTER_DIR / f"ndvi_{COUNTRY_CODE}.tif",        False),
        ("HydroRIVERS",              RASTER_DIR / "hydrosheds",                       False),
        ("SoilGrids Clay",           RASTER_DIR / f"soilgrids/soilgrids_clay_{COUNTRY_CODE}.tif", False),
        ("K-factor",                 RASTER_DIR / f"k_factor_{COUNTRY_CODE}.tif",     False),
    ]
    ready = 0
    for desc, path, required in checks:
        if path.exists():
            info = (f"{len(list(path.glob('*.tif')))+len(list(path.glob('*.shp')))} files"
                    if path.is_dir() else f"{path.stat().st_size / 1024 / 1024:.1f} MB")
            print(f"  [OK] {desc:40s} {info}")
            ready += 1
        else:
            tag = "(required)" if required else "(optional)"
            print(f"  [--] {desc:40s} MISSING {tag}")
    print(f"\n  Ready: {ready}/{len(checks)}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="FACVI Data Download")
    parser.add_argument("--skip-nasa", action="store_true",
                        help="Skip SRTM and MODIS (no NASA login required)")
    parser.add_argument("--only", choices=["worldclim", "srtm", "hydrorivers", "modis", "soilgrids"])
    parser.add_argument("--check", action="store_true", help="Show file status and exit")
    args = parser.parse_args()

    if args.check:
        check_status()
        return

    log.info("=" * 60)
    log.info("FACVI Data Download")
    log.info("=" * 60)
    check_status()

    results = {}
    if args.only in ("worldclim", None):
        results["WorldClim Baseline"] = download_worldclim_baseline()
        results["WorldClim CMIP6"]    = download_worldclim_cmip6()
    if args.only in ("hydrorivers", None):
        results["HydroRIVERS"] = download_hydrorivers()
    if args.only in ("soilgrids", None):
        results["SoilGrids + K-factor"] = download_soilgrids()
    if not args.skip_nasa:
        if args.only in ("srtm", None):
            results["SRTM"] = download_srtm()
        if args.only in ("modis", None):
            results["MODIS NDVI"] = download_modis_ndvi()
    else:
        log.info("  --skip-nasa: SRTM and MODIS skipped.")

    log.info("\n" + "=" * 60 + "\nDOWNLOAD SUMMARY\n" + "=" * 60)
    for name, success in results.items():
        log.info(f"  {name:30s}: {'OK' if success else 'FAILED / SKIPPED'}")

    check_status()
    log.info("Download complete. Run the pipeline next:")
    log.info("  python run_pipeline.py")


if __name__ == "__main__":
    main()
