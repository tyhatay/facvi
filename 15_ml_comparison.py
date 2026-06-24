"""
FACVI Pipeline — Step 15: Random Forest + SHAP vs. Shannon Entropy Weights
===========================================================================
Data-driven validation of FACVI variable selection and weighting.

Two RF models using SSP2-4.5/2050 as the representative scenario:
  Model A (classifier): Large fire occurrence (>= LARGE_THR ha) — AUC
  Model B (regressor):  First response time (cell median) — R², Spearman rho
                        Skipped if response_time_min column is absent.

SHAP feature importances are compared with the Shannon entropy weights
produced in Step 10.

This is the public version of the original ML comparison step.
OGM-proprietary fire data replaced with a STANDARD GeoPackage
(same format accepted by 13_fire_validation.py).

Fire inventory columns required:
  geometry            Point (WGS84 or any CRS)
  year                int
  burned_area_ha      float
  response_time_min   float  (optional — enables RF-B if present)

CLI:
  python 15_ml_comparison.py --fire-data data/fire_inventory.gpkg

Outputs:
  outputs/fig_ml_shap_comparison.png
  outputs/ml_shap_results.csv
  outputs/ml_meta.csv
"""

import matplotlib
matplotlib.use("Agg")

import argparse
import geopandas as gpd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
import logging
import warnings
warnings.filterwarnings("ignore")

from config import (
    DATA_DIR, LOG_DIR, COUNTRY_CODE, SCENARIOS,
    GRID_FOREST_FILE, GRID_CLIMATE_FILE, GRID_RVI_FILE,
)

LOG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "15_ml_comparison.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

WEIGHTS_FILE = DATA_DIR / "entropy_weights.csv"
SCENARIO     = "ssp245_2050"
LARGE_THR    = 5      # ha — threshold for "large fire"
RT_MIN_FIRES = 2      # cells with >= this many fires qualify for RF-B
SHAP_SAMPLE  = 5_000  # subsample for SHAP speed
RF_SEED      = 42

# ── Variable definitions ──────────────────────────────────────────────────────
# N3 uses dist_stream_m (public HydroRIVERS; NOT OGM drainage column)
# N4, N5 are optional: included if column present, skipped otherwise

FEATURES_REQUIRED = {
    "C1": f"C1_{SCENARIO}",
    "C2": f"C2_{SCENARIO}",
    "C3": f"C3_{SCENARIO}",
    "C4": f"C4_{SCENARIO}",
    "S1": "slope_deg",
    "S2": "twi",
    "S3": "k_factor",
    "S4": "ndvi_mean",
    "S5": "snow_days_mean",
    "N1": "road_density_m_km2",
    "N2": "weighted_surface",
    "N3": "dist_stream_m",
}

FEATURES_OPTIONAL = {
    "N4": "dist_watchtower_m",
    "N5": "dist_fmc_m",
}

FEATURE_LABELS = {
    "C1": "DeltaTemp Mean\n(C1)",
    "C2": "DeltaAnnual Precip.\n(C2)",
    "C3": "DeltaMax-Month Temp.\n(C3)",
    "C4": "DeltaPrecip Season.\n(C4)",
    "S1": "Terrain Slope\n(S1)",
    "S2": "Topographic Wetness\n(S2)",
    "S3": "Soil Erodibility\n(S3)",
    "S4": "Veg. Cover / NDVI inv.\n(S4)",
    "S5": "Snow Cover Days\n(S5)",
    "N1": "Road Density\n(N1)",
    "N2": "Road Surface Quality\n(N2)",
    "N3": "Stream Proximity\n(N3)",
    "N4": "Fire Watchtower Dist.\n(N4)",
    "N5": "Fire Mgmt. Centre Dist.\n(N5)",
}

DIM_COLORS = {
    "Climate":  "#2196F3",
    "Physical": "#4CAF50",
    "Network":  "#FF9800",
}

DIM_MAP = {
    **{v: "Climate"  for v in ["C1", "C2", "C3", "C4"]},
    **{v: "Physical" for v in ["S1", "S2", "S3", "S4", "S5"]},
    **{v: "Network"  for v in ["N1", "N2", "N3", "N4", "N5"]},
}


# ── Load grid ─────────────────────────────────────────────────────────────────

def load_grid() -> gpd.GeoDataFrame:
    candidates = [GRID_RVI_FILE, GRID_FOREST_FILE, GRID_CLIMATE_FILE]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise FileNotFoundError(
            "Grid not found. Run steps 09 and 10 first.\n"
            f"Expected one of: {[str(p) for p in candidates]}"
        )
    log.info(f"[LOAD] {path.name}")
    grid = gpd.read_file(path)
    log.info(f"  {len(grid):,} cells  ({len(grid.columns)} columns)")

    if "ndvi_mean" in grid.columns and grid["ndvi_mean"].max() > 10:
        grid["ndvi_mean"] = grid["ndvi_mean"] / 1e8
        log.info("  [NDVI] Rescaled from raw (x1e8) to [0,1]")

    # Merge FACVI columns if not already present
    if f"FACVI_{SCENARIO}" not in grid.columns and GRID_RVI_FILE.exists() and path != GRID_RVI_FILE:
        log.info(f"  [MERGE] Adding FACVI columns from {GRID_RVI_FILE.name}")
        rvi = gpd.read_file(GRID_RVI_FILE,
                            columns=[f"FACVI_{SCENARIO}",
                                     f"FACVI_class_{SCENARIO}"])
        for col in [f"FACVI_{SCENARIO}", f"FACVI_class_{SCENARIO}"]:
            if col in rvi.columns:
                grid[col] = rvi[col].values

    return grid


# ── Load fire inventory ───────────────────────────────────────────────────────

def load_fire_inventory(fire_path: Path, target_crs) -> gpd.GeoDataFrame:
    if not fire_path.exists():
        raise FileNotFoundError(
            f"Fire inventory not found: {fire_path}\n\n"
            "Provide a GeoPackage or CSV with columns:\n"
            "  geometry, year (int), burned_area_ha (float),\n"
            "  response_time_min (float, optional)\n\n"
            "Public data sources:\n"
            "  EFFIS: https://effis.jrc.ec.europa.eu/\n"
            "  FIRMS: https://firms.modaps.eosdis.nasa.gov/"
        )

    log.info(f"[LOAD] Fire inventory: {fire_path}")
    if fire_path.suffix.lower() == ".csv":
        df = pd.read_csv(fire_path)
        if "longitude" in df.columns and "latitude" in df.columns:
            from shapely.geometry import Point
            gdf = gpd.GeoDataFrame(
                df,
                geometry=[Point(x, y) for x, y in zip(df["longitude"], df["latitude"])],
                crs="EPSG:4326",
            )
        else:
            raise ValueError("CSV must have 'longitude' and 'latitude' columns.")
    else:
        gdf = gpd.read_file(fire_path)

    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")

    if "burned_area_ha" not in gdf.columns:
        raise ValueError("Fire data must have 'burned_area_ha' column.")

    gdf = gdf[gdf.geometry.notna()].to_crs(target_crs)
    log.info(f"  {len(gdf):,} fire records")
    return gdf


# ── Derive fire outcomes per cell ─────────────────────────────────────────────

def derive_fire_outcomes(grid: gpd.GeoDataFrame,
                          fires: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    grid = grid.copy()
    grid["_idx"] = np.arange(len(grid))

    fires_sub = fires.copy()
    fires_sub["ba"]       = pd.to_numeric(fires_sub["burned_area_ha"], errors="coerce").fillna(0)
    fires_sub["is_large"] = (fires_sub["ba"] >= LARGE_THR).astype(int)

    joined_f = gpd.sjoin(
        fires_sub[["ba", "is_large", "geometry"]].reset_index(drop=True),
        grid[["_idx", "geometry"]], how="left", predicate="within",
    ).dropna(subset=["_idx"])
    joined_f["_idx"] = joined_f["_idx"].astype(int)

    cell_f = (joined_f.groupby("_idx")
              .agg(fire_count=("ba", "count"),
                   sum_ba=("ba", "sum"),
                   large_count=("is_large", "sum"))
              .reindex(np.arange(len(grid)), fill_value=0))

    grid["has_fire"]  = (cell_f["fire_count"].values > 0).astype(int)
    grid["has_large"] = (cell_f["large_count"].values > 0).astype(int)
    grid["sum_ba"]    = cell_f["sum_ba"].values
    log.info(f"  has_large: {grid['has_large'].sum():,} cells "
             f"({grid['has_large'].mean()*100:.2f}%)")

    # Response time (optional)
    has_rt = "response_time_min" in fires.columns
    grid["median_rt"] = np.nan

    if has_rt:
        rt_fires = fires_sub[fires_sub["response_time_min"].notna()].copy()
        rt_fires["rt"] = pd.to_numeric(rt_fires["response_time_min"], errors="coerce")
        rt_fires = rt_fires[(rt_fires["rt"] >= 1) & (rt_fires["rt"] <= 480)]

        if not rt_fires.empty:
            joined_r = gpd.sjoin(
                rt_fires[["rt", "geometry"]].reset_index(drop=True),
                grid[["_idx", "geometry"]], how="left", predicate="within",
            ).dropna(subset=["_idx"])
            joined_r["_idx"] = joined_r["_idx"].astype(int)

            cell_rt = (joined_r.groupby("_idx")["rt"]
                       .agg(median_rt="median", n_fires="count"))
            cell_rt = cell_rt[cell_rt["n_fires"] >= RT_MIN_FIRES]

            grid.loc[cell_rt.index, "median_rt"] = cell_rt["median_rt"].values
            log.info(f"  Response time cells: {len(cell_rt):,}")
        else:
            log.warning("  No valid response times after filtering (1–480 min).")

    return grid


# ── Feature matrix ────────────────────────────────────────────────────────────

def build_X(grid: gpd.GeoDataFrame) -> tuple[np.ndarray, list[str]]:
    codes, cols = [], []
    for code, col in FEATURES_REQUIRED.items():
        if col in grid.columns:
            codes.append(code)
            cols.append(col)
        else:
            log.warning(f"  [SKIP] Required feature {code} ({col}) not in grid.")

    for code, col in FEATURES_OPTIONAL.items():
        if col in grid.columns:
            codes.append(code)
            cols.append(col)
            log.info(f"  [OPT] {code} ({col}) included")
        else:
            log.info(f"  [OPT] {code} ({col}) absent — excluded from RF")

    if not codes:
        raise ValueError("No feature columns found in grid.")

    X = grid[cols].values.astype(np.float64)
    for j in range(X.shape[1]):
        mask = np.isnan(X[:, j])
        if mask.any():
            X[mask, j] = float(np.nanmedian(X[:, j]))

    log.info(f"[FEATURES] X.shape={X.shape}  NaN remaining={int(np.isnan(X).sum())}")
    return X, codes


# ── RF Model A: large fire occurrence (classifier) ───────────────────────────

def train_fire_model(X: np.ndarray, y: np.ndarray, codes: list[str]) -> dict:
    log.info(f"\n[RF-A] Large fire model  (n={len(y):,}  pos={int(y.sum()):,})")

    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    sample_neg = min(len(neg_idx), len(pos_idx) * 20)
    rng     = np.random.default_rng(RF_SEED)
    neg_sel = rng.choice(neg_idx, size=sample_neg, replace=False)
    sel_idx = np.sort(np.concatenate([pos_idx, neg_sel]))

    Xs, ys = X[sel_idx], y[sel_idx]
    log.info(f"  Subsample: {len(ys):,}  (pos={int(ys.sum()):,}  neg={int(len(ys)-ys.sum()):,})")

    rf = RandomForestClassifier(
        n_estimators=300, max_depth=10, min_samples_leaf=20,
        class_weight="balanced", n_jobs=1, random_state=RF_SEED,
    )

    cv   = StratifiedKFold(n_splits=5, shuffle=True, random_state=RF_SEED)
    aucs = cross_val_score(rf, Xs, ys, cv=cv, scoring="roc_auc", n_jobs=1)
    log.info(f"  5-fold AUC = {aucs.mean():.4f} +/- {aucs.std():.4f}")

    rf.fit(Xs, ys)

    shap_imp = _compute_shap(rf, Xs, is_classifier=True)

    return dict(auc=aucs.mean(), auc_std=aucs.std(),
                mdi_imp=rf.feature_importances_, shap_imp=shap_imp,
                codes=codes)


# ── RF Model B: response time (regressor) ────────────────────────────────────

def train_rt_model(X: np.ndarray, grid: gpd.GeoDataFrame,
                   codes: list[str]) -> dict | None:
    rt_mask = ~np.isnan(grid["median_rt"].values)
    n_rt    = int(rt_mask.sum())

    if n_rt < 30:
        log.warning(f"[RF-B] Only {n_rt} cells with response time — skipping RF-B.")
        return None

    Xr = X[rt_mask]
    yr = grid.loc[rt_mask, "median_rt"].values
    log.info(f"\n[RF-B] Response time model  (n={len(yr):,}  "
             f"median={float(np.median(yr)):.1f} min)")

    rf = RandomForestRegressor(
        n_estimators=300, max_depth=10, min_samples_leaf=5,
        n_jobs=1, random_state=RF_SEED,
    )

    cv  = KFold(n_splits=5, shuffle=True, random_state=RF_SEED)
    r2s = cross_val_score(rf, Xr, yr, cv=cv, scoring="r2", n_jobs=1)
    log.info(f"  5-fold R² = {r2s.mean():.4f} +/- {r2s.std():.4f}")

    rf.fit(Xr, yr)
    y_pred = rf.predict(Xr)
    rho, p = spearmanr(yr, y_pred)
    log.info(f"  Train-set Spearman rho = {rho:.4f}  p={p:.2e}")

    shap_imp = _compute_shap(rf, Xr, is_classifier=False)

    return dict(r2=r2s.mean(), r2_std=r2s.std(),
                oob_spear=rho, oob_spear_p=p,
                mdi_imp=rf.feature_importances_, shap_imp=shap_imp,
                codes=codes)


# ── SHAP ─────────────────────────────────────────────────────────────────────

def _compute_shap(rf, X: np.ndarray, is_classifier: bool) -> np.ndarray | None:
    try:
        import shap
        rng  = np.random.default_rng(RF_SEED)
        idx  = rng.choice(len(X), size=min(SHAP_SAMPLE, len(X)), replace=False)
        Xs   = X[idx]
        expl = shap.TreeExplainer(rf)
        sv   = expl.shap_values(Xs)

        if isinstance(sv, list):
            # old shap: list per class — take class 1 for classifiers
            sv = sv[1] if is_classifier else sv[0]
        if sv.ndim == 3:
            # new shap (>=0.46): shape (n_samples, n_features, n_classes)
            sv = sv[:, :, 1] if is_classifier else sv[:, :, 0]

        imp = np.abs(sv).mean(axis=0)
        log.info(f"  SHAP computed ({len(Xs):,} samples)")
        return imp
    except Exception as e:
        log.warning(f"  SHAP failed ({type(e).__name__}: {e}) — using MDI")
        return None


# ── Entropy weights ───────────────────────────────────────────────────────────

def load_entropy_weights(codes: list[str]) -> np.ndarray:
    if not WEIGHTS_FILE.exists():
        raise FileNotFoundError(
            f"Entropy weights not found: {WEIGHTS_FILE}\n"
            "Run 10_facvi.py first."
        )
    df = pd.read_csv(WEIGHTS_FILE)
    if "scenario" in df.columns:
        df = df[df["scenario"] == SCENARIO]
    w_series = df.set_index("variable")["weight"]
    weights = np.array([float(w_series.get(c, 0.0)) for c in codes])
    log.info("[WEIGHTS] Entropy weights loaded:")
    for c, w in zip(codes, weights):
        log.info(f"  {c}: {w*100:.2f}%")
    return weights


# ── Figure ────────────────────────────────────────────────────────────────────

def make_figure(fire_res: dict, rt_res: dict | None, entropy_w: np.ndarray):
    codes  = fire_res["codes"]
    labels = [FEATURE_LABELS.get(c, c) for c in codes]
    dim_c  = [DIM_COLORS[DIM_MAP.get(c, "Network")] for c in codes]

    def norm(arr):
        s = arr.sum()
        return arr / s if s > 0 else arr

    ew       = norm(entropy_w)
    fire_imp = norm(fire_res["shap_imp"] if fire_res["shap_imp"] is not None
                    else fire_res["mdi_imp"])

    n_panels = 4 if rt_res is not None else 3
    fig, axes = plt.subplots(
        1, n_panels, figsize=(5.5 * n_panels, 7.5),
        gridspec_kw={"wspace": 0.38},
    )
    if n_panels == 3:
        ax_fire, ax_ew, ax_scatter = axes
        ax_rt = None
    else:
        ax_rt, ax_fire, ax_ew, ax_scatter = axes

    def bar_h(ax, vals, title, xlabel):
        vals  = np.asarray(vals, dtype=float).flatten()
        order = np.argsort(vals).tolist()
        ax.barh([labels[i] for i in order],
                vals[order],
                color=[dim_c[i] for i in order],
                edgecolor="white", linewidth=0.5)
        for v, yi in zip(vals[order], range(len(order))):
            ax.text(v + vals.max() * 0.01, yi, f"{v*100:.1f}%",
                    va="center", fontsize=7.5)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(0, vals.max() * 1.30)
        ax.tick_params(axis="y", labelsize=7.5)

    # RF-A: fire occurrence
    fire_imp_type = "SHAP Mean |val|" if fire_res["shap_imp"] is not None else "MDI"
    bar_h(ax_fire, fire_imp,
          f"RF-A: Large Fire Occurrence\n"
          f"{fire_imp_type}  |  AUC = {fire_res['auc']:.3f} +/- {fire_res['auc_std']:.3f}",
          "Normalized Importance")

    # RF-B: response time (if available)
    if rt_res is not None:
        rt_imp      = norm(rt_res["shap_imp"] if rt_res["shap_imp"] is not None
                           else rt_res["mdi_imp"])
        rt_imp_type = "SHAP Mean |val|" if rt_res["shap_imp"] is not None else "MDI"
        bar_h(ax_rt, rt_imp,
              f"RF-B: First Response Time\n"
              f"{rt_imp_type}  |  R² = {rt_res['r2']:.3f}  rho = {rt_res['oob_spear']:.3f}",
              "Normalized Importance")

    # Entropy weights
    bar_h(ax_ew, ew,
          "Shannon Entropy Weights\nFACVI  (dimension prior x within-dim entropy)",
          "Weight")

    # Scatter: entropy vs. fire importance
    ref_imp      = (norm(rt_res["shap_imp"] if rt_res["shap_imp"] is not None
                         else rt_res["mdi_imp"]) if rt_res is not None else fire_imp)
    ref_imp_type = ("RT " + ("SHAP" if rt_res is not None and rt_res["shap_imp"] is not None
                              else "MDI"))
    for c, xi, yi in zip(codes, ew, ref_imp):
        ax_scatter.scatter(xi, yi, color=DIM_COLORS[DIM_MAP.get(c, "Network")],
                           s=80, zorder=3, edgecolors="white", linewidths=0.5)
        ax_scatter.annotate(c, (xi, yi), textcoords="offset points",
                            xytext=(4, 3), fontsize=8.5)

    rho_sc, p_sc = spearmanr(ew, ref_imp)
    p_str = f"= {p_sc:.3f}" if p_sc >= 0.001 else "< 0.001"
    ax_scatter.set_xlabel("Shannon Entropy Weight", fontsize=9)
    ax_scatter.set_ylabel(f"RF Importance ({ref_imp_type})", fontsize=9)
    ax_scatter.set_title(
        f"Entropy Weight vs. RF Importance\n"
        f"Spearman rho = {rho_sc:.3f}  p {p_str}",
        fontsize=10, fontweight="bold", pad=6,
    )
    ax_scatter.spines[["top", "right"]].set_visible(False)
    ax_scatter.grid(alpha=0.15)

    patches = [mpatches.Patch(color=v, label=k) for k, v in DIM_COLORS.items()]
    fig.legend(handles=patches, loc="lower center", ncol=3, fontsize=9,
               framealpha=0.8, bbox_to_anchor=(0.5, -0.03))

    fig.suptitle(
        "Random Forest vs. Shannon Entropy: Variable Importance Comparison\n"
        f"FACVI {len(codes)} variables  |  {SCENARIO}",
        fontsize=12, fontweight="bold",
    )
    out = OUTPUT_DIR / "fig_ml_shap_comparison.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    log.info(f"[FIG] {out.name}")


# ── Save CSV ──────────────────────────────────────────────────────────────────

def save_csv(fire_res: dict, rt_res: dict | None, entropy_w: np.ndarray):
    codes = fire_res["codes"]

    def norm(arr): s = arr.sum(); return arr / s if s > 0 else arr

    ew       = norm(entropy_w)
    fire_imp = norm(fire_res["shap_imp"] if fire_res["shap_imp"] is not None
                    else fire_res["mdi_imp"])
    fire_mdi = norm(fire_res["mdi_imp"])

    row_data: dict = {
        "variable":        codes,
        "dimension":       [DIM_MAP.get(c, "Network") for c in codes],
        "entropy_weight":  [round(float(v), 6) for v in ew],
        "fire_importance": [round(float(v), 6) for v in fire_imp],
        "fire_mdi":        [round(float(v), 6) for v in fire_mdi],
    }

    if rt_res is not None:
        rt_imp = norm(rt_res["shap_imp"] if rt_res["shap_imp"] is not None
                      else rt_res["mdi_imp"])
        row_data["rt_importance"] = [round(float(v), 6) for v in rt_imp]
        row_data["rt_mdi"]        = [round(float(v), 6) for v in norm(rt_res["mdi_imp"])]

    df = pd.DataFrame(row_data)
    df.to_csv(OUTPUT_DIR / "ml_shap_results.csv", index=False, encoding="utf-8-sig")
    log.info("[SAVED] ml_shap_results.csv")

    rho_fire, p_fire = spearmanr(ew, fire_imp)

    meta_rows = [
        {"metric": "rf_a_auc_fire",           "value": fire_res["auc"],
         "std": fire_res["auc_std"]},
        {"metric": "spearman_entropy_vs_fire", "value": rho_fire, "std": p_fire},
    ]
    if rt_res is not None:
        rt_imp   = norm(rt_res["shap_imp"] if rt_res["shap_imp"] is not None
                        else rt_res["mdi_imp"])
        rho_rt, p_rt = spearmanr(ew, rt_imp)
        meta_rows += [
            {"metric": "rf_b_r2_rt",             "value": rt_res["r2"],
             "std": rt_res["r2_std"]},
            {"metric": "spearman_entropy_vs_rt",  "value": rho_rt, "std": p_rt},
        ]

    pd.DataFrame(meta_rows).to_csv(OUTPUT_DIR / "ml_meta.csv",
                                   index=False, encoding="utf-8-sig")
    log.info("[SAVED] ml_meta.csv")
    log.info(f"  Spearman rho (entropy vs. fire importance): {rho_fire:.3f}  p={p_fire:.2e}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="FACVI ML comparison (Step 15)"
    )
    parser.add_argument(
        "--fire-data", type=Path,
        default=DATA_DIR / "fire_inventory.gpkg",
        help="Path to fire GeoPackage or CSV with columns: "
             "geometry, year, burned_area_ha [, response_time_min]",
    )
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("FACVI Step 15: Random Forest + SHAP vs. Entropy Weights")
    log.info("=" * 60)

    grid = load_grid()

    fires = load_fire_inventory(args.fire_data, grid.crs)
    grid  = derive_fire_outcomes(grid, fires)

    log.info("\n[FEATURES] Building feature matrix ...")
    X, codes = build_X(grid)

    entropy_w = load_entropy_weights(codes)

    log.info("\n[RF] Training models ...")
    y_fire   = grid["has_large"].values.astype(int)
    fire_res = train_fire_model(X, y_fire, codes)

    rt_res = train_rt_model(X, grid, codes)

    log.info("\n[FIGURE] Generating ...")
    make_figure(fire_res, rt_res, entropy_w)
    save_csv(fire_res, rt_res, entropy_w)

    log.info(f"\n  RF-A (large fire) AUC = {fire_res['auc']:.4f}")
    if rt_res is not None:
        log.info(f"  RF-B (response time) R² = {rt_res['r2']:.4f}")
    log.info("Step 15 complete.")


if __name__ == "__main__":
    main()
