# FACVI — Forest Access Climate Vulnerability Index

A reproducible, open-data pipeline that computes a composite Forest Access Climate Vulnerability Index (FACVI) for any country using only publicly available datasets.

## Overview

FACVI combines 14 variables across three dimensions:

| Dimension | Code | Variable | Source |
|-----------|------|----------|--------|
| Climate (40%) | C1 | Delta mean annual temperature | WorldClim v2.1 CMIP6 |
| | C2 | Delta annual precipitation | WorldClim v2.1 CMIP6 |
| | C3 | Delta max-month temperature | WorldClim v2.1 CMIP6 |
| | C4 | Delta precipitation seasonality | WorldClim v2.1 CMIP6 |
| Physical (35%) | S1 | Terrain slope | SRTM 30m |
| | S2 | Topographic Wetness Index | SRTM 30m (pysheds D8) |
| | S3 | Soil erodibility K-factor | SoilGrids v2.0 |
| | S4 | Vegetation cover / NDVI (inv.) | MODIS MOD13A3 |
| | S5 | Snow cover days | MODIS MOD10CM |
| Network (25%) | N1 | Road density (inv.) | OpenStreetMap |
| | N2 | Road surface quality (inv.) | OpenStreetMap |
| | N3 | Stream proximity (inv.) | HydroRIVERS v1.0 |
| | N4 | Watchtower distance | User-provided (optional) |
| | N5 | Fire management centre distance | User-provided (optional) |

Weights are computed via Shannon entropy with dimension-level priors (OECD 2008). Four CMIP6 scenarios are processed: SSP2-4.5 and SSP5-8.5 at 2050 and 2100.

## Critical note on N4 and N5 (fire infrastructure variables)

N4 (fire watchtower distance) and N5 (fire management centre distance) carried **24.3% of total FACVI weight** in the published Turkey analysis (N4 = 12.5%, N5 = 11.8%). By contrast, N1+N2+N3 together accounted for only 0.7%. The Network dimension is therefore dominated by fire suppression infrastructure proximity, not road density.

**If you run the pipeline without N4/N5 data, the resulting FACVI values will differ substantially from the published results.** The index will still run (N4 and N5 are treated as optional and gracefully excluded), but the Network dimension will collapse to near-zero weight, and the 25% Network prior will be effectively unrepresented.

To reproduce Turkey results or to produce a comparable index for another country:
- **Turkey**: the GDF (General Directorate of Forestry) fire watchtower and fire management centre coordinates are included in GDF's publicly released annual reports and spatial data portals.
- **Other countries**: identify equivalent national fire agency infrastructure layers (watchtowers, dispatch centres, ranger stations) and provide them as point GeoPackages. Compute Euclidean distances to grid cell centroids and add columns `dist_watchtower_m` and `dist_fmc_m` to the grid file before running step 10.

A future utility script (`utility_compute_distances.py`) can automate this step once your point layers are in place.

## Country adaptation

All country-specific settings live in `config.py`. To run the pipeline for a different country, change only these fields:

```python
COUNTRY_NAME = "Germany"     # English name used by osmnx
COUNTRY_CODE = "deu"         # ISO 3166 alpha-3 (lowercase)
BBOX = {"west": 6.0, "south": 47.3, "east": 15.0, "north": 55.1}
CRS_METRIC   = "EPSG:32632"  # UTM zone for the study area
GEOFABRIK_URL = "https://download.geofabrik.de/europe/germany-latest.osm.pbf"
HYDRORIVERS_REGION = "eu"
SRTM_LAT_RANGE = (47, 56)
SRTM_LON_RANGE = (6, 16)
```

No other file needs to change. Output filenames are derived from `COUNTRY_CODE` automatically.

## Installation

Conda is recommended (handles GDAL binary builds):

```bash
conda create -n facvi python=3.11
conda activate facvi
conda install -c conda-forge gdal pyproj rasterio geopandas shapely pysheds \
    libpysal esda osmnx scikit-learn matplotlib numpy pandas pyhdf pyogrio
pip install shap
```

Or with pip (requires GDAL already installed on the system):

```bash
pip install -r requirements.txt
```

`ogr2ogr` (GDAL CLI) must be discoverable on `PATH`, or set the environment variable:

```bash
export GDAL_OGR2OGR=/path/to/ogr2ogr   # Linux/macOS
set GDAL_OGR2OGR=C:\path\to\ogr2ogr.exe  # Windows
```

## MODIS credentials

Steps 01 and 08 download MODIS data from NASA Earthdata. Credentials are read from environment variables only and must never be stored in any file:

```bash
export EARTHDATA_USERNAME=your_username
export EARTHDATA_PASSWORD=your_password
```

Register for a free account at https://urs.earthdata.nasa.gov/

## Pipeline steps

```
facvi/
  config.py                  Country settings and methodology parameters
  01_download_data.py        Download WorldClim, SRTM, HydroRIVERS, NDVI, SoilGrids
  02_forest_domain.py        ESA WorldCover forest mask, 1km grid creation
  03_auxiliary_data.py       RESOLVE Ecoregions 2017
  04_osm_roads.py            OSM road download and surface scoring
  05_road_grid.py            Road density per 1km cell (N1, N2)
  06_physical_variables.py   Slope, TWI (pysheds D8), K-factor, NDVI (S1-S4)
  07_stream_distance.py      HydroRIVERS stream distance (N3)
  08_snow_cover.py           MODIS snow cover days (S5)
  09_climate_deltas.py       WorldClim CMIP6 climate deltas (C1-C4)
  10_facvi.py                Shannon entropy weights and FACVI scores
  11_spatial_analysis.py     Moran's I, LISA, scenario comparison maps
  12_sensitivity.py          Monte Carlo and leave-one-out sensitivity
  13_fire_validation.py *    FACVI validation against fire records
  14_compound_hazard.py *    Compound hazard: high FACVI x landslide inventory
  15_ml_comparison.py *      Random Forest + SHAP vs. entropy weights
  run_pipeline.py            Orchestrator
```

`*` Steps 13-15 require external validation data (see below).

## Run the full pipeline

```bash
# Full run
python run_pipeline.py

# From step 09 onwards (if earlier outputs exist)
python run_pipeline.py --from-step 09

# With validation data
python run_pipeline.py \
  --fire-data data/fire_inventory.gpkg \
  --landslide data/landslide_inventory.gpkg

# Stop on first failure
python run_pipeline.py --stop-on-error
```

## Validation data formats

### Fire inventory (steps 13 and 15)

GeoPackage or CSV with columns:

| Column | Type | Description |
|--------|------|-------------|
| geometry | Point | Fire location (any CRS) |
| year | int | Calendar year |
| burned_area_ha | float | Total burned area (ha) |
| response_time_min | float | First response time in minutes (optional) |

Public sources:
- EFFIS (Europe): https://effis.jrc.ec.europa.eu/
- NASA FIRMS: https://firms.modaps.eosdis.nasa.gov/
- GWIS: https://gwis.jrc.ec.europa.eu/

### Landslide inventory (step 14)

GeoPackage or Shapefile with geometry (Point or Polygon, any CRS). No other columns required. Presence of a record within a grid cell is treated as hazard signal.

Public sources:
- NASA Global Landslide Catalog: https://gpm.nasa.gov/landslides/
- Froude & Petley (2018): https://doi.org/10.5194/nhess-18-2161-2018

## Outputs

After a full run, outputs are in:

```
data/
  {cc}_grid_1km.gpkg           1km grid with road density
  {cc}_grid_physical.gpkg      + slope, TWI, K-factor, NDVI, snow, stream dist.
  {cc}_grid_climate.gpkg       + climate deltas (C1-C4, all scenarios)
  {cc}_grid_forest_climate.gpkg  forest-filtered grid
  {cc}_rvi_all_scenarios.gpkg  + FACVI scores and classes
  entropy_weights.csv

outputs/
  fig_facvi_maps.png
  fig_entropy_weights.png
  fig_scenario_comparison.png
  fig_lisa_*.png
  fig_sensitivity.png
  moran_results.csv
  sensitivity_mc.csv
  sensitivity_exclusion.csv
  (validation outputs if steps 13-15 were run)
```

## Data directory structure

Create the following before running:

```
data/
  worldclim/
    baseline/        wc2.1_2.5m_bio01.tif ... bio19.tif
    ssp245_50/       wc2.1_2.5m_bioc_{MODEL}_ssp245_2041-2060.tif
    ssp245_90/
    ssp585_50/
    ssp585_90/
  rasters/           (auto-populated by 01_download_data.py)
```

If WorldClim files are absent, steps 09 and 10 fall back to spatially structured synthetic deltas (suitable for testing, not for publication).

## Differences from the published paper

The following methodological differences exist between the GitHub pipeline and the published analysis. They are intentional: the GitHub version removes all proprietary dependencies and SAGA/Windows-only tools to make the pipeline globally reproducible.

| Aspect | Published paper | This pipeline | Impact |
|--------|----------------|---------------|--------|
| TWI method | SAGA Wetness Index (Conrad et al., 2015), with plan curvature correction | pysheds D8 flow accumulation | TWI cell values differ; S2 entropy weight slightly affected |
| Forest domain source | CORINE Land Cover 2018 (classes 311-313, 321-323) | ESA WorldCover 2021 (class 10: Tree Cover) | Grid cell selection marginally different |
| CRS (Turkey) | EPSG:5255 (ITRF96/TM30) | EPSG:32636 (UTM Zone 36N) | Sub-metre distance differences at national scale |
| N4/N5 data | GDF fire infrastructure (Turkey-specific) | User-provided or absent | If absent: Network dimension loses 24% weight (see note above) |
| Fire validation data | GDF incident database (33,116 records) | User-provided standard GeoPackage | Results in step 13 depend on data quality |
| Landslide data | MTA national inventory (Turkey) | User-provided GeoPackage | Results in step 14 depend on inventory completeness |

None of these differences affect the core FACVI formula, entropy weighting logic, or dimension structure.

## Methodology reference

Shannon entropy weighting:
> E_j = -(1/ln n) * sum(p_ij * ln(p_ij))

Hybrid weights:
> w_j = DIM_WEIGHT[dim(j)] * (within-dim entropy weight of j)
> capped at DIM_CAP=50% within each dimension

FACVI score:
> FACVI_i = sum(w_j * x*_ij)   where x* is min-max normalised to [0,1]

Risk classes: quintile-based (1 = Very Low, 5 = Very High).

## License

Code: MIT License.
Data: each source dataset carries its own license (WorldClim CC BY 4.0, OpenStreetMap ODbL, HydroRIVERS CC BY 4.0, MODIS and SRTM public domain via NASA).
