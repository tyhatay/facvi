"""
FACVI Pipeline — Step 13: Fire Record Validation
=================================================
Validates FACVI scores against fire occurrence records.

This is the public version of the original fire validation step.
It accepts a STANDARD fire GeoPackage / CSV instead of the OGM
database (which is proprietary and cannot be distributed).

Required fire data format (GeoPackage or CSV with geometry):
  Column            Type     Description
  -------           ------   -----------
  geometry          Point    Fire location (WGS84)
  year              int      Calendar year (used for time filtering)
  burned_area_ha    float    Total burned area in hectares (0+ ; NaN allowed)
  response_time_min float    First response time in minutes (optional)

Where to get public fire data:
  - EFFIS (Europe):       https://effis.jrc.ec.europa.eu/applications/data-and-services
  - GWIS global:          https://gwis.jrc.ec.europa.eu/
  - NASA FIRMS:           https://firms.modaps.eosdis.nasa.gov/
  - National forest agencies (check for open data portals)

Three validation metrics:
  1. Binary fire occurrence (all / large ≥LARGE_THR ha) → ROC-AUC
  2. Burned area per cell → Spearman ρ
  3. First response time → Spearman ρ  (if response_time_min column present)

Output: outputs/fig_validation_fire.png, outputs/validation_fire.csv
"""

import matplotlib
matplotlib.use("Agg")

import argparse
import geopandas as gpd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, roc_curve
import logging
import warnings
warnings.filterwarnings("ignore")

from config import (
    DATA_DIR, LOG_DIR,
    COUNTRY_CODE, GRID_RVI_FILE,
)

LOG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "13_fire_validation.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

SCENARIO    = "ssp245_2050"
YEAR_START  = 2015
YEAR_END    = 2022
LARGE_FIRE_THR = 5       # ha
RT_MIN      = 1
RT_MAX      = 480        # minutes

CLASS_LABELS = {1: "Very\nLow", 2: "Low", 3: "Moderate",
                4: "High", 5: "Very\nHigh"}


# ── Load fire data ────────────────────────────────────────────────────────────

def load_fires(fire_path: Path, target_crs) -> gpd.GeoDataFrame:
    if not fire_path.exists():
        raise FileNotFoundError(
            f"Fire data file not found: {fire_path}\n\n"
            "Provide a GeoPackage or CSV with columns:\n"
            "  geometry (Point), year (int), burned_area_ha (float),\n"
            "  response_time_min (float, optional)\n\n"
            "Public data sources:\n"
            "  EFFIS:  https://effis.jrc.ec.europa.eu/\n"
            "  FIRMS:  https://firms.modaps.eosdis.nasa.gov/\n"
            "  GWIS:   https://gwis.jrc.ec.europa.eu/"
        )

    log.info(f"[LOAD] Fire data: {fire_path}")
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
            raise ValueError("CSV must have 'longitude' and 'latitude' columns, "
                             "or provide a GeoPackage with geometry column.")
    else:
        gdf = gpd.read_file(fire_path)

    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")

    # Validate required columns
    for col in ("year", "burned_area_ha"):
        if col not in gdf.columns:
            raise ValueError(
                f"Required column '{col}' not found in fire data.\n"
                f"Available columns: {list(gdf.columns)}"
            )

    # Filter by year
    gdf["year"] = pd.to_numeric(gdf["year"], errors="coerce")
    fires = gdf[(gdf["year"] >= YEAR_START) & (gdf["year"] <= YEAR_END)].copy()
    fires = fires[fires.geometry.notna()].copy()
    fires = fires.to_crs(target_crs)

    log.info(f"  {len(fires):,} fire records ({YEAR_START}–{YEAR_END})")
    return fires


# ── Spatial join fires → grid ─────────────────────────────────────────────────

def assign_fires_to_grid(grid: gpd.GeoDataFrame,
                          fires: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    grid = grid.copy()
    grid["_idx"] = np.arange(len(grid))

    p99_ba = float(np.percentile(fires["burned_area_ha"].dropna().clip(lower=0), 99))
    log.info(f"  Burned area P99 cap: {p99_ba:.1f} ha")

    fires_sub = fires[["burned_area_ha", "geometry"]].copy()
    fires_sub["ba_capped"] = fires_sub["burned_area_ha"].fillna(0).clip(upper=p99_ba)
    fires_sub["is_large"]  = (fires_sub["burned_area_ha"].fillna(0) >= LARGE_FIRE_THR).astype(int)

    joined = gpd.sjoin(
        fires_sub[["ba_capped", "is_large", "geometry"]].reset_index(drop=True),
        grid[["_idx", "geometry"]],
        how="left", predicate="within",
    ).dropna(subset=["_idx"])
    joined["_idx"] = joined["_idx"].astype(int)

    stats = (joined.groupby("_idx")
             .agg(fire_count=("ba_capped", "count"),
                  sum_ba_ha=("ba_capped", "sum"),
                  large_count=("is_large", "sum"))
             .reindex(np.arange(len(grid)), fill_value=0))

    grid["fire_count"]  = stats["fire_count"].values
    grid["has_fire"]    = (grid["fire_count"] > 0).astype(int)
    grid["sum_ba_ha"]   = stats["sum_ba_ha"].values
    grid["large_count"] = stats["large_count"].values
    grid["has_large"]   = (grid["large_count"] > 0).astype(int)

    log.info(f"  Cells with fires:   {grid['has_fire'].sum():,} "
             f"({grid['has_fire'].mean()*100:.1f}%)")
    log.info(f"  Large fire cells:   {grid['has_large'].sum():,} "
             f"({grid['has_large'].mean()*100:.1f}%)")
    return grid, p99_ba


# ── Response time (optional) ──────────────────────────────────────────────────

def compute_response_stats(grid: gpd.GeoDataFrame,
                            fires: gpd.GeoDataFrame) -> dict | None:
    if "response_time_min" not in fires.columns:
        log.info("  [RT] response_time_min column not present — skipping.")
        return None

    rt = fires[["response_time_min", "geometry"]].copy()
    rt["rt"] = pd.to_numeric(rt["response_time_min"], errors="coerce")
    rt = rt[(rt["rt"] >= RT_MIN) & (rt["rt"] <= RT_MAX)].copy()

    if rt.empty:
        log.warning("  [RT] No valid response times after filtering.")
        return None

    grid2 = grid.copy()
    grid2["_idx2"] = np.arange(len(grid2))

    joined = gpd.sjoin(
        rt[["rt", "geometry"]].reset_index(drop=True),
        grid2[["_idx2", "geometry"]],
        how="left", predicate="within",
    ).dropna(subset=["_idx2"])
    joined["_idx2"] = joined["_idx2"].astype(int)

    cell_rt = (joined.groupby("_idx2")["rt"]
               .agg(median_rt="median", n_fires="count"))
    cell_rt = cell_rt[cell_rt["n_fires"] >= 2]

    idx        = cell_rt.index.values
    rt_vals    = cell_rt["median_rt"].values
    facvi_vals = grid2.loc[idx, f"FACVI_{SCENARIO}"].values

    rho, p = spearmanr(facvi_vals, rt_vals)
    log.info(f"  [RT] {len(idx):,} cells (≥2 fires)  ρ={rho:.4f}  p={p:.2e}")

    class_col = f"FACVI_class_{SCENARIO}"
    classes   = sorted(grid2[class_col].dropna().unique())
    grid2["median_rt"] = np.nan
    grid2.loc[idx, "median_rt"] = rt_vals

    class_mrt, class_n = [], []
    for cls in classes:
        mask = (grid2[class_col] == cls) & grid2["median_rt"].notna()
        vals = grid2.loc[mask, "median_rt"].values
        class_mrt.append(float(np.median(vals)) if len(vals) > 0 else np.nan)
        class_n.append(int(len(vals)))

    return dict(rho=rho, p=p, classes=classes,
                class_mrt=class_mrt, class_n=class_n, n_cells_rt=len(idx))


# ── Stats ─────────────────────────────────────────────────────────────────────

def compute_stats(grid: gpd.GeoDataFrame) -> dict:
    facvi    = grid[f"FACVI_{SCENARIO}"].values
    has_fire = grid["has_fire"].values
    has_lg   = grid["has_large"].values
    sum_ba   = grid["sum_ba_ha"].values

    rho_all, p_all = spearmanr(facvi, grid["fire_count"].values)
    rho_ba,  p_ba  = spearmanr(facvi, sum_ba)
    auc_all  = roc_auc_score(has_fire, facvi)
    auc_lg   = roc_auc_score(has_lg,   facvi)
    fpr_all, tpr_all, _ = roc_curve(has_fire, facvi)
    fpr_lg,  tpr_lg,  _ = roc_curve(has_lg,   facvi)

    log.info(f"\n  All fires   — ρ={rho_all:.4f} p={p_all:.2e}  AUC={auc_all:.4f}")
    log.info(f"  Large ≥{LARGE_FIRE_THR} ha — AUC={auc_lg:.4f}")
    log.info(f"  Burned area — ρ={rho_ba:.4f}  p={p_ba:.2e}")

    class_col = f"FACVI_class_{SCENARIO}"
    classes = sorted(grid[class_col].dropna().unique())
    fire_rates, lg_rates, mean_ba, n_cells = [], [], [], []
    for cls in classes:
        m = (grid[class_col] == cls).values
        fire_rates.append(float(has_fire[m].mean() * 100))
        lg_rates.append(float(has_lg[m].mean() * 100))
        mean_ba.append(float(sum_ba[m].mean()))
        n_cells.append(int(m.sum()))

    return dict(
        rho_all=rho_all, p_all=p_all, auc_all=auc_all,
        rho_ba=rho_ba, p_ba=p_ba, auc_lg=auc_lg,
        fpr_all=fpr_all, tpr_all=tpr_all,
        fpr_lg=fpr_lg, tpr_lg=tpr_lg,
        classes=classes, fire_rates=fire_rates, lg_rates=lg_rates,
        mean_ba=mean_ba, n_cells=n_cells,
        n_fire=int(has_fire.sum()), n_large=int(has_lg.sum()), n_total=len(grid),
    )


# ── Figure ────────────────────────────────────────────────────────────────────

def make_figure(st: dict, ba_cap: float, rt_stats: dict | None):
    fig = plt.figure(figsize=(15, 9))
    gs  = fig.add_gridspec(2, 3, hspace=0.44, wspace=0.38)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])

    classes  = st["classes"]
    xlabels  = [CLASS_LABELS.get(int(c), str(c)) for c in classes]
    x        = np.arange(len(classes))
    col_all  = plt.colormaps["YlOrRd"](np.linspace(0.20, 0.90, len(classes)))
    col_lg   = plt.colormaps["Oranges"](np.linspace(0.25, 0.92, len(classes)))
    col_ba   = plt.colormaps["YlOrBr"](np.linspace(0.20, 0.90, len(classes)))

    def bar_panel(ax, values, colors, ylabel, title, baseline=None, fmt=".1f"):
        bars = ax.bar(x, values, color=colors, edgecolor="white", linewidth=0.7, width=0.65)
        if baseline is not None:
            ax.axhline(baseline, color="#555", linestyle="--", linewidth=1.1,
                       label=f"Avg ({baseline:{fmt}})")
        for bar, val in zip(bars, values):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + max(v for v in values if not np.isnan(v))*0.03,
                        f"{val:{fmt}}", ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, fontsize=8.5)
        ax.set_ylabel(ylabel, fontsize=8.5)
        ax.set_title(title, fontsize=9.5, fontweight="bold", pad=6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.25, linewidth=0.5)
        if baseline is not None:
            ax.legend(fontsize=8, framealpha=0.6)
        valid = [v for v in values if not np.isnan(v)]
        if valid:
            ax.set_ylim(0, max(valid) * 1.32)

    # (a) Response time
    if rt_stats is not None:
        col_rt = plt.colormaps["RdYlGn_r"](np.linspace(0.15, 0.90, len(rt_stats["classes"])))
        rt_p = ("p < 0.0001" if rt_stats["p"] < 0.0001 else f"p = {rt_stats['p']:.4f}")
        bar_panel(ax1, rt_stats["class_mrt"], col_rt,
                  "Median First Response Time (min)",
                  f"(a) Response Time by FACVI Class\nSpearman ρ = {rt_stats['rho']:.3f}  {rt_p}")
    else:
        ax1.text(0.5, 0.5, "Response time\ndata not available",
                 ha="center", va="center", transform=ax1.transAxes, color="gray")
        ax1.set_title("(a) First Response Time", fontsize=9.5, fontweight="bold")

    # (b) Fire rate
    p_str = "p < 0.0001" if st["p_all"] < 0.0001 else f"p = {st['p_all']:.4f}"
    bar_panel(ax2, st["fire_rates"], col_all,
              f"Fire Rate (%, {YEAR_START}–{YEAR_END})",
              f"(b) All Fires: Rate by FACVI Class\nSpearman ρ = {st['rho_all']:.3f}  {p_str}",
              baseline=st["n_fire"] / st["n_total"] * 100)

    # (c) ROC curves
    ax3.plot(st["fpr_all"], st["tpr_all"], color="#e74c3c", linewidth=2,
             label=f"All fires (AUC = {st['auc_all']:.3f})")
    ax3.plot(st["fpr_lg"],  st["tpr_lg"],  color="#c0392b", linewidth=2, linestyle="--",
             label=f"Large ≥{LARGE_FIRE_THR} ha (AUC = {st['auc_lg']:.3f})")
    ax3.plot([0, 1], [0, 1], "k:", linewidth=1, alpha=0.5)
    ax3.fill_between(st["fpr_all"], st["tpr_all"], alpha=0.07, color="#e74c3c")
    ax3.set_xlabel("False Positive Rate", fontsize=8.5)
    ax3.set_ylabel("True Positive Rate", fontsize=8.5)
    ax3.set_title("(c) ROC Curves\nFACVI vs. Fire Occurrence", fontsize=9.5, fontweight="bold")
    ax3.legend(fontsize=8, loc="lower right")
    ax3.set_xlim(0, 1); ax3.set_ylim(0, 1)
    ax3.spines[["top", "right"]].set_visible(False)
    ax3.grid(alpha=0.15)

    # (d) Burned area
    p_ba = "p < 0.0001" if st["p_ba"] < 0.0001 else f"p = {st['p_ba']:.4f}"
    bar_panel(ax4, st["mean_ba"], col_ba,
              f"Mean Burned Area / cell (ha)\n[P99 cap: {ba_cap:.0f} ha]",
              f"(d) Burned Area by FACVI Class\nSpearman ρ = {st['rho_ba']:.3f}  {p_ba}",
              fmt=".3f")

    # (e) Large fire rate
    bar_panel(ax5, st["lg_rates"], col_lg,
              f"Large Fire Rate (%, ≥{LARGE_FIRE_THR} ha)",
              f"(e) Large Fires by FACVI Class\nAUC = {st['auc_lg']:.3f}",
              baseline=st["n_large"] / st["n_total"] * 100,
              fmt=".2f")

    # (f) Class cell counts
    ax6.bar(x, st["n_cells"], color=plt.colormaps["Blues"](np.linspace(0.3, 0.8, len(classes))),
            edgecolor="white", linewidth=0.7)
    ax6.set_xticks(x)
    ax6.set_xticklabels(xlabels, fontsize=8.5)
    ax6.set_ylabel("Number of forest-access cells", fontsize=8.5)
    ax6.set_title("(f) Cells per FACVI Class\n(quintile boundaries)",
                  fontsize=9.5, fontweight="bold")
    ax6.spines[["top", "right"]].set_visible(False)
    ax6.grid(axis="y", alpha=0.25)

    plt.suptitle(
        f"FACVI Validation Against Fire Records ({YEAR_START}–{YEAR_END})",
        fontsize=11.5, fontweight="bold", y=1.01,
    )
    out = OUTPUT_DIR / "fig_validation_fire.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    log.info(f"[FIG] {out.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="FACVI fire validation (Step 13)"
    )
    parser.add_argument(
        "--fire-data", type=Path, required=True,
        help="Path to fire GeoPackage or CSV with columns: "
             "geometry, year, burned_area_ha [, response_time_min]",
    )
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("FACVI Step 13: Fire Validation")
    log.info("=" * 60)

    if not GRID_RVI_FILE.exists():
        raise FileNotFoundError(
            f"FACVI grid not found: {GRID_RVI_FILE}\n"
            "Run 10_facvi.py first."
        )

    grid = gpd.read_file(GRID_RVI_FILE)
    log.info(f"[LOAD] {GRID_RVI_FILE.name}  ({len(grid):,} cells)")

    fires = load_fires(args.fire_data, grid.crs)
    grid, ba_cap = assign_fires_to_grid(grid, fires)

    log.info("\n[STATS] Computing validation metrics ...")
    st = compute_stats(grid)

    log.info("\n[RT] Response time analysis ...")
    rt_stats = compute_response_stats(grid, fires)

    make_figure(st, ba_cap, rt_stats)

    # Save CSV
    rows: dict = {
        "facvi_class":    st["classes"],
        "n_cells":        st["n_cells"],
        "fire_rate_pct":  [round(v, 3) for v in st["fire_rates"]],
        "large_rate_pct": [round(v, 3) for v in st["lg_rates"]],
        "mean_ba_ha":     [round(v, 4) for v in st["mean_ba"]],
        "spearman_rho_count": [st["rho_all"]] * len(st["classes"]),
        "auc_all_fires":  [st["auc_all"]] * len(st["classes"]),
        "auc_large_fires":[st["auc_lg"]]  * len(st["classes"]),
    }
    if rt_stats is not None:
        rows["median_rt_min"] = [round(v, 1) if not np.isnan(v) else None
                                 for v in rt_stats["class_mrt"]]
        rows["spearman_rho_rt"] = [round(rt_stats["rho"], 4)] * len(st["classes"])
    out_csv = OUTPUT_DIR / "validation_fire.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8-sig")
    log.info(f"[SAVED] {out_csv.name}")

    log.info(f"\n  AUC (all fires):   {st['auc_all']:.4f}")
    log.info(f"  AUC (large ≥{LARGE_FIRE_THR} ha): {st['auc_lg']:.4f}")
    log.info(f"  ρ (burned area):   {st['rho_ba']:.4f}")
    if rt_stats:
        log.info(f"  ρ (response time): {rt_stats['rho']:.4f}")
    log.info("Step 13 complete.")


if __name__ == "__main__":
    main()
