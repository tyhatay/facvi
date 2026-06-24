"""
FACVI Pipeline — Step 03: Auxiliary Data
==========================================
Downloads RESOLVE Ecoregions 2017 clipped to the study area.
Used later for within-ecoregion analysis and map figures.

Source: Dinerstein et al. (2017) BioScience 67(6):534-545
        https://doi.org/10.1093/biosci/bix014

The script tries four endpoints in order:
  1. ArcGIS Living Atlas REST API
  2. UNEP-WCMC ArcGIS REST API
  3. Local shapefile (data/Ecoregions2017.shp)
  4. Manual download instructions printed to console

Output: data/{CC}_ecoregions.gpkg
"""

import geopandas as gpd
import requests
import json
import logging
from pathlib import Path
from shapely.geometry import box

from config import DATA_DIR, LOG_DIR, BBOX, COUNTRY_CODE, ECOREGIONS_FILE

LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "03_auxiliary.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

ARCGIS_URL = (
    "https://services.arcgis.com/P3ePLMYs2RVChkJx/arcgis/rest/services/"
    "Resolve_Ecoregions/FeatureServer/0/query"
)
UNEP_URL = (
    "https://data-gis.unep-wcmc.org/server/rest/services/Bio-geographicalRegions/"
    "Resolve_Ecoregions/FeatureServer/0/query"
)

# Ecoregion name → short label for figures.
# Replace or extend this dict for other countries.
ECO_SHORT: dict[str, str] = {
    "Aegean and Western Turkey sclerophyllous and mixed forests": "Aegean-W. Turkey",
    "Anatolian conifer and deciduous mixed forests":              "Anatolian Mixed",
    "Azerbaijan shrub desert and steppe":                        "Azerbaijan Shrub",
    "Balkan mixed forests":                                      "Balkan Mixed",
    "Caucasus mixed forests":                                    "Caucasus Mixed",
    "Central Anatolian steppe":                                  "C. Anatolian Steppe",
    "Central Anatolian steppe and woodlands":                    "C. Anatolian Steppe-Wood",
    "Cyprus Mediterranean forests":                              "Cyprus Med.",
    "Eastern Anatolian deciduous forests":                       "E. Anatolian Decid.",
    "Eastern Anatolian montane steppe":                          "E. Anatolian Steppe",
    "Eastern Mediterranean conifer-broadleaf forests":           "E. Mediterranean",
    "Euxine-Colchic broadleaf forests":                         "Euxine-Colchic",
    "Mesopotamian shrub desert":                                 "Mesopotamian Desert",
    "Northern Anatolian conifer and deciduous forests":          "N. Anatolian Conifer",
    "Rodope montane mixed forests":                              "Rhodope Mixed",
    "Southern Anatolian montane conifer and deciduous forests":  "S. Anatolian Montane",
    "Syrian xeric grasslands and shrublands":                    "Syrian Xeric",
    "Zagros Mountains forest steppe":                            "Zagros F. Steppe",
}


# ── ArcGIS REST query ────────────────────────────────────────────────────────

def _query_arcgis(service_url: str, bbox: tuple) -> gpd.GeoDataFrame | None:
    envelope = json.dumps({
        "xmin": bbox[0], "ymin": bbox[1],
        "xmax": bbox[2], "ymax": bbox[3],
        "spatialReference": {"wkid": 4326},
    })
    params = {
        "where":             "1=1",
        "geometry":          envelope,
        "geometryType":      "esriGeometryEnvelope",
        "spatialRel":        "esriSpatialRelIntersects",
        "outFields":         "ECO_NAME,BIOME_NAME,BIOME_NUM,REALM,ECO_NUM",
        "returnGeometry":    "true",
        "resultRecordCount": "200",
        "f":                 "geojson",
    }
    try:
        log.info(f"  Querying: {service_url.split('/arcgis')[0]} ...")
        r = requests.get(service_url, params=params, timeout=90)
        r.raise_for_status()
        data = r.json()
        if "error" in data:
            raise ValueError(data["error"].get("message", str(data["error"])))
        features = data.get("features", [])
        if not features:
            log.warning("  Query returned no features.")
            return None
        gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
        log.info(f"  {len(gdf)} ecoregions received.")
        return gdf
    except Exception as e:
        log.warning(f"  Failed: {e}")
        return None


# ── Local shapefile fallback ─────────────────────────────────────────────────

def _try_local(bbox: tuple) -> gpd.GeoDataFrame | None:
    candidates = [
        DATA_DIR / "Ecoregions2017.shp",
        DATA_DIR / "Ecoregions2017" / "Ecoregions2017.shp",
        DATA_DIR / "resolve_ecoregions" / "Ecoregions2017.shp",
        DATA_DIR / "ecoregions" / "Ecoregions2017.shp",
    ]
    for p in candidates:
        if p.exists():
            log.info(f"  Local file found: {p}")
            gdf = gpd.read_file(p, bbox=bbox)
            log.info(f"  {len(gdf)} ecoregions loaded.")
            return gdf
    return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if ECOREGIONS_FILE.exists():
        gdf = gpd.read_file(ECOREGIONS_FILE)
        log.info(f"[CACHED] Ecoregions: {len(gdf)} polygons in {ECOREGIONS_FILE.name}")
        return

    log.info(f"[RESOLVE] Downloading ecoregions for {COUNTRY_CODE} ...")
    study_bbox = (BBOX["west"], BBOX["south"], BBOX["east"], BBOX["north"])

    gdf = _query_arcgis(ARCGIS_URL, study_bbox)
    if gdf is None:
        log.info("  Trying UNEP-WCMC endpoint ...")
        gdf = _query_arcgis(UNEP_URL, study_bbox)
    if gdf is None:
        log.info("  Trying local shapefile ...")
        gdf = _try_local(study_bbox)
    if gdf is None:
        raise RuntimeError(
            "Could not obtain RESOLVE Ecoregions data.\n\n"
            "Manual download:\n"
            "  1. Visit https://ecoregions.appspot.com/\n"
            "  2. Click 'Shapefile Download' (~150 MB)\n"
            "  3. Extract all files into the data/ directory\n"
            "  4. Re-run this script.\n"
        )

    # Ensure WGS84
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")

    # Clip to study bounding box
    study_box = box(*study_bbox)
    gdf = gpd.clip(gdf, study_box)
    gdf = gdf[gdf.geometry.is_valid & ~gdf.geometry.is_empty].reset_index(drop=True)

    # Add short labels
    gdf["ECO_SHORT"] = gdf["ECO_NAME"].map(ECO_SHORT).fillna(
        gdf["ECO_NAME"].str[:25]
    )

    gdf.to_file(ECOREGIONS_FILE, driver="GPKG")
    log.info(f"[SAVED] {ECOREGIONS_FILE.name}  ({len(gdf)} ecoregions)")

    log.info("\n  Ecoregions in study area:")
    for _, row in gdf.sort_values("ECO_NAME").iterrows():
        log.info(f"    {row['ECO_NAME']}")


if __name__ == "__main__":
    main()
