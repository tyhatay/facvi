"""
FACVI Pipeline — Configuration
================================
All country-specific and methodology-specific settings live here.
To adapt this pipeline to a different country, update the values in
the STUDY AREA and DATA SOURCES sections below. The methodology
parameters (FACVI METHODOLOGY) should only be changed if you have
a theoretical reason to do so.
"""

from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent   # repo root (the facvi/ folder itself)
DATA_DIR   = BASE_DIR / "data"
RASTER_DIR = DATA_DIR / "rasters"
WC_DIR     = DATA_DIR / "worldclim"
LOG_DIR    = BASE_DIR / "logs"

# ── STUDY AREA ────────────────────────────────────────────────────────────────
# Change these values to adapt the pipeline to a different country.

COUNTRY_NAME = "Turkey"  # Geocoding string used by osmnx (English name)
COUNTRY_CODE = "tur"     # Lowercase ISO 3166 alpha-3; used in output filenames

# WGS84 bounding box (west, south, east, north)
BBOX = {
    "west":  25.5,
    "south": 35.8,
    "east":  44.9,
    "north": 42.2,
}

# Projected CRS in metres — used for area and distance calculations.
# Turkey: EPSG:32636 (UTM Zone 36N).
# Examples:
#   Luxembourg: EPSG:32632 (UTM Zone 32N)
#   Iran:        EPSG:32638 (UTM Zone 38N)
#   Brazil:      EPSG:32722 (UTM Zone 22S)
CRS_METRIC = "EPSG:32636"

# ── DATA SOURCES ──────────────────────────────────────────────────────────────
# OSM road network — Geofabrik download page:
#   https://download.geofabrik.de/
GEOFABRIK_URL = "https://download.geofabrik.de/europe/turkey-latest.osm.pbf"

# HydroRIVERS package — choose the regional package that covers your study area.
#   eu = Europe        as = Asia        na = North America
#   sa = South America af = Africa      au = Australia & Oceania
# Download page: https://www.hydrosheds.org/products/hydrorivers
HYDRORIVERS_REGION = "eu"
HYDRORIVERS_URL = (
    f"https://data.hydrosheds.org/file/HydroRIVERS/"
    f"HydroRIVERS_v10_{HYDRORIVERS_REGION}_shp.zip"
)

# SRTM 30m tile bounding box — integer degree ranges covering the study area.
# Turkey spans N36–N42, E026–E044.
SRTM_LAT_RANGE = (36, 43)   # range(start, stop): N36–N42
SRTM_LON_RANGE = (26, 45)   # range(start, stop): E026–E044

# ── LUXEMBOURG (test) — restore these values to run Luxembourg analysis ───────
# COUNTRY_NAME = "Luxembourg"
# COUNTRY_CODE = "lux"
# BBOX = {"west": 5.7, "south": 49.4, "east": 6.6, "north": 50.2}
# CRS_METRIC = "EPSG:32632"
# GEOFABRIK_URL = "https://download.geofabrik.de/europe/luxembourg-latest.osm.pbf"
# HYDRORIVERS_REGION = "eu"
# SRTM_LAT_RANGE = (49, 50)
# SRTM_LON_RANGE = (5, 7)

# MODIS NDVI period (MOD13A3, monthly 1 km)
NDVI_START = "2015-01-01"
NDVI_END   = "2020-12-31"

# MODIS snow cover period (MOD10CM, monthly 0.05°)
SNOW_START_YEAR = 2001
SNOW_END_YEAR   = 2023

# ── FOREST DOMAIN ─────────────────────────────────────────────────────────────
# CORINE Land Cover 2018 classes included in the forest-access domain.
# 311–313 = forest classes; 321–323 = adjacent semi-natural vegetation.
CORINE_CLASSES  = [311, 312, 313, 321, 322, 323]
FOREST_BUFFER_M = 2000   # 2 km adjacency buffer around forest mask

# ESA WorldCover 2021 class for tree cover (used as an alternative forest mask)
ESA_TREE_CLASS = 10

# ── FACVI METHODOLOGY ─────────────────────────────────────────────────────────
# Grid resolution in metres (1 km²)
CELL_SIZE_M = 1000

# Literature-informed dimension-level prior weights (must sum to 1.0)
DIM_WEIGHTS = {
    "Climate":  0.40,
    "Physical": 0.35,
    "Network":  0.25,
}

# Within-dimension entropy weight cap (0.50 = 50%)
# Prevents a single highly differentiated variable from dominating its dimension.
DIM_CAP = 0.50

# CMIP6 scenario–horizon combinations
# 'folder' must match the subdirectory name under data/worldclim/
SCENARIOS = {
    "ssp245_2050": {"ssp": "ssp245", "period": "2050", "folder": "ssp245_50"},
    "ssp245_2100": {"ssp": "ssp245", "period": "2100", "folder": "ssp245_90"},
    "ssp585_2050": {"ssp": "ssp585", "period": "2050", "folder": "ssp585_50"},
    "ssp585_2100": {"ssp": "ssp585", "period": "2100", "folder": "ssp585_90"},
}

# GCMs used for ensemble averaging
CMIP6_MODELS = [
    "BCC-CSM2-MR",
    "CNRM-CM6-1",
    "IPSL-CM6A-LR",
    "MRI-ESM2-0",
]

# FACVI variable definitions
# Each entry: code → (column_name, invert, description)
#   invert=True  means raw higher value = lower vulnerability → invert before normalisation
#   invert=False means raw higher value = higher vulnerability → no inversion
VARIABLE_MAP = {
    # Climate (scenario-dependent; columns added by 09_climate_deltas.py)
    "C1": ("delta_bio1",  False, "Delta mean annual temperature (°C)"),
    "C2": ("delta_bio12", False, "Delta annual precipitation (mm)"),
    "C3": ("delta_bio5",  False, "Delta max-month temperature (°C)"),
    "C4": ("delta_bio15", False, "Delta precipitation seasonality (BIO15)"),
    # Physical (static)
    "S1": ("slope_deg",          False, "Slope (°)"),
    "S2": ("twi",                False, "Topographic Wetness Index"),
    "S3": ("k_factor",           False, "Soil erodibility K-factor"),
    "S4": ("ndvi_mean",          True,  "Mean annual NDVI (inverted)"),
    "S5": ("snow_days_mean",     False, "Mean annual snow cover days"),
    # Network (static)
    "N1": ("road_density_m_km2", True,  "Road density m/km² (inverted)"),
    "N2": ("surface_quality",    True,  "Road surface quality score (inverted)"),
    "N3": ("dist_stream_m",      True,  "Distance to nearest stream (inverted)"),
    "N4": ("dist_watchtower_m",  False, "Distance to nearest fire watchtower"),
    "N5": ("dist_fmc_m",         False, "Distance to nearest fire management centre"),
}

DIMENSIONS = {
    "Climate":  ["C1", "C2", "C3", "C4"],
    "Physical": ["S1", "S2", "S3", "S4", "S5"],
    "Network":  ["N1", "N2", "N3", "N4", "N5"],
}

# Road types extracted from OSM
OSM_HIGHWAY_TYPES = {"track", "unclassified", "tertiary"}

# ── OUTPUT FILE NAMES ─────────────────────────────────────────────────────────
# Derived from COUNTRY_CODE so they update automatically when you change the country.
def grid_path(suffix: str) -> Path:
    """Return a standard data-layer path, e.g. grid_path('1km') → data/tur_grid_1km.gpkg"""
    return DATA_DIR / f"{COUNTRY_CODE}_{suffix}.gpkg"

GRID_1KM_FILE      = grid_path("grid_1km")
GRID_PHYSICAL_FILE = grid_path("grid_physical")
GRID_CLIMATE_FILE  = grid_path("grid_climate")
GRID_FOREST_FILE   = grid_path("grid_forest_climate")
GRID_RVI_FILE      = grid_path("rvi_all_scenarios")
ROADS_FILE         = DATA_DIR / f"osm_forest_roads_{COUNTRY_CODE}.gpkg"
BOUNDARY_FILE      = DATA_DIR / f"{COUNTRY_CODE}_boundary.gpkg"
ECOREGIONS_FILE    = DATA_DIR / f"ecoregions_{COUNTRY_CODE}.gpkg"
