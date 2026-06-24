"""
FACVI Pipeline — Step 04: OSM Road Network
===========================================
Downloads the country OSM PBF from Geofabrik and extracts
track / unclassified / tertiary highway segments.

Uses ogr2ogr (bundled with GDAL/QGIS) to read the PBF without
hitting Overpass API rate limits.

ogr2ogr discovery order:
  1. GDAL_OGR2OGR env var (set this if auto-detection fails)
  2. System PATH  (works on Linux/Mac after 'conda install gdal')
  3. Common Windows QGIS / Anaconda paths

Output: data/osm_forest_roads_{CC}.gpkg
"""

import geopandas as gpd
import subprocess
import shutil
import logging
import os
from pathlib import Path

from config import (
    DATA_DIR, LOG_DIR,
    GEOFABRIK_URL, OSM_HIGHWAY_TYPES,
    COUNTRY_CODE, ROADS_FILE,
)

LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "04_osm_roads.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

PBF_FILE = DATA_DIR / f"{COUNTRY_CODE}-latest.osm.pbf"
OSMCONF  = DATA_DIR / "osmconf.ini"


# ── ogr2ogr discovery ─────────────────────────────────────────────────────────

def _find_ogr2ogr() -> str:
    """Return path to ogr2ogr executable or raise RuntimeError."""
    # 1. Explicit env var
    env_path = os.environ.get("GDAL_OGR2OGR", "")
    if env_path and Path(env_path).is_file():
        return env_path

    # 2. System PATH
    found = shutil.which("ogr2ogr")
    if found:
        return found

    # 3. Common Windows locations (QGIS, Anaconda)
    candidates = [
        Path(os.environ.get("CONDA_PREFIX", "")) / "Library/bin/ogr2ogr.exe",
        Path("C:/Program Files/QGIS 3.28/bin/ogr2ogr.exe"),
        Path("C:/Program Files/QGIS 3.34/bin/ogr2ogr.exe"),
        Path("C:/OSGeo4W/bin/ogr2ogr.exe"),
    ]
    for c in candidates:
        if c.is_file():
            return str(c)

    raise RuntimeError(
        "ogr2ogr not found.\n"
        "Options:\n"
        "  • Set env var: GDAL_OGR2OGR=/path/to/ogr2ogr\n"
        "  • Install GDAL via conda:  conda install -c conda-forge gdal\n"
        "  • Install QGIS (includes ogr2ogr in its bin/ directory)\n"
    )


# ── osmconf.ini ───────────────────────────────────────────────────────────────

def _write_osmconf():
    OSMCONF.write_text("""\
[general]
report_all_tags=yes
report_all_nodes=no
report_all_ways=yes
report_all_multipolygons=no

[lines]
osm_id=yes
attributes=highway,surface,width,name,access

[multilinestrings]
osm_id=yes
attributes=highway,surface,width,name

[points]
osm_id=yes
attributes=

[multipolygons]
osm_id=yes
attributes=

[other_relations]
osm_id=yes
attributes=
""")


# ── PBF download ─────────────────────────────────────────────────────────────

def download_pbf():
    if PBF_FILE.exists() and PBF_FILE.stat().st_size > 50_000_000:
        log.info(f"[CACHED] PBF: {PBF_FILE.stat().st_size / 1e6:.0f} MB")
        return

    import requests
    log.info(f"[DOWNLOAD] {GEOFABRIK_URL}")

    headers, mode = {}, "wb"
    existing = PBF_FILE.stat().st_size if PBF_FILE.exists() else 0
    if existing > 0:
        headers["Range"] = f"bytes={existing}-"
        mode = "ab"
        log.info(f"  Resuming from {existing / 1e6:.0f} MB ...")

    with requests.get(GEOFABRIK_URL, headers=headers, stream=True, timeout=600) as r:
        r.raise_for_status()
        total     = int(r.headers.get("Content-Length", 0)) + existing
        downloaded = existing
        with open(PBF_FILE, mode) as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                downloaded += len(chunk)
                if total and downloaded % (50 * 1024 * 1024) < 1024 * 1024:
                    log.info(f"  {downloaded / 1e6:.0f}/{total / 1e6:.0f} MB "
                             f"({downloaded / total * 100:.0f}%)")

    log.info(f"  PBF downloaded: {PBF_FILE.stat().st_size / 1e6:.0f} MB")


# ── PBF → GeoPackage ──────────────────────────────────────────────────────────

SURFACE_SCORE = {
    "asphalt": 1.0, "concrete": 0.9, "paved": 0.9,
    "compacted": 0.6, "gravel": 0.5, "fine_gravel": 0.5,
    "pebblestone": 0.4, "ground": 0.2, "dirt": 0.2,
    "mud": 0.1, "grass": 0.1, "unpaved": 0.2,
}

HIGHWAY_SCORE = {"tertiary": 0.8, "unclassified": 0.5, "track": 0.3}


def extract_roads() -> gpd.GeoDataFrame:
    ogr = _find_ogr2ogr()
    log.info(f"[OGR] Using: {ogr}")

    _write_osmconf()

    tmp_gpkg = DATA_DIR / "_roads_tmp.gpkg"
    tmp_gpkg.unlink(missing_ok=True)

    highway_sql = ", ".join(f"'{h}'" for h in OSM_HIGHWAY_TYPES)
    cmd = [
        ogr,
        "-f", "GPKG",
        str(tmp_gpkg),
        str(PBF_FILE),
        "-oo", f"CONFIG_FILE={OSMCONF.resolve()}",
        "-where", f"highway IN ({highway_sql})",
        "lines",
        "--config", "OSM_MAX_TMPFILE_SIZE", "2000",
        "--config", "OSM_USE_CUSTOM_INDEXING", "NO",
    ]

    log.info(f"[OGR] Extracting {OSM_HIGHWAY_TYPES} roads ...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if result.returncode != 0:
        log.error(result.stderr[:500])
        raise RuntimeError(f"ogr2ogr failed (exit {result.returncode})")

    gdf = gpd.read_file(tmp_gpkg)
    tmp_gpkg.unlink(missing_ok=True)

    if gdf.empty:
        raise RuntimeError("ogr2ogr produced an empty output.")

    # Standardise columns
    drop_cols = [c for c in ["OGC_FID", "other_tags", "z_order", "way_area"] if c in gdf.columns]
    gdf = gdf.drop(columns=drop_cols)
    if "osm_id" in gdf.columns:
        gdf = gdf.rename(columns={"osm_id": "osmid"})

    for col in ["highway", "surface", "width", "name", "osmid"]:
        if col not in gdf.columns:
            gdf[col] = ""
        gdf[col] = gdf[col].fillna("").astype(str)

    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    else:
        gdf = gdf.to_crs("EPSG:4326")

    # Surface quality and length
    gdf["surface_score"] = (
        gdf["surface"].str.lower().str.strip().map(SURFACE_SCORE).fillna(0.2)
    )
    gdf["highway_score"] = (
        gdf["highway"].str.lower().map(HIGHWAY_SCORE).fillna(0.3)
    )
    gdf_proj = gdf.to_crs("EPSG:32636")
    gdf["length_m"] = gdf_proj.geometry.length
    gdf = gdf[gdf["length_m"] >= 10].copy()

    log.info(f"  Segments: {len(gdf):,}")
    log.info(f"  Surface data: {(gdf['surface_score'] > 0.2).sum():,} "
             f"({(gdf['surface_score'] > 0.2).mean()*100:.1f}%)")
    log.info(f"  Total length: {gdf['length_m'].sum() / 1000:,.0f} km")
    return gdf


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if ROADS_FILE.exists():
        gdf = gpd.read_file(ROADS_FILE)
        log.info(f"[CACHED] Roads: {len(gdf):,} segments in {ROADS_FILE.name}")
        return

    log.info("=" * 60)
    log.info("FACVI Step 04: OSM Road Network")
    log.info("=" * 60)

    download_pbf()
    gdf = extract_roads()
    gdf.to_file(ROADS_FILE, driver="GPKG")
    log.info(f"[SAVED] {ROADS_FILE.name}")
    log.info("Step 04 complete.")


if __name__ == "__main__":
    main()
