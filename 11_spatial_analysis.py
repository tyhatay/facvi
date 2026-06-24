"""
FACVI Pipeline — Step 11: Spatial Autocorrelation and Figures
=============================================================
Computes global Moran's I and LISA cluster maps, then produces
publication-quality figures.

Outputs (in outputs/):
  moran_results.csv
  lisa_ssp585_2100.gpkg
  fig_facvi_maps.png         4-scenario map panel
  fig_weights.png            Entropy weight lollipop chart
  fig_scenario_comparison.png  Violin + delta raster + risk-tier bars
"""

import matplotlib
matplotlib.use("Agg")

import geopandas as gpd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")
import logging
from pathlib import Path

from config import (
    DATA_DIR, LOG_DIR,
    BBOX, SCENARIOS, DIMENSIONS, COUNTRY_CODE,
    GRID_RVI_FILE, ECOREGIONS_FILE,
)

LOG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "11_spatial_analysis.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

WEIGHTS_FILE = DATA_DIR / "entropy_weights.csv"
MIN_ECO_CELLS = 500

SCENARIO_LABELS = {
    "ssp245_2050": "SSP2-4.5 / 2050",
    "ssp245_2100": "SSP2-4.5 / 2100",
    "ssp585_2050": "SSP5-8.5 / 2050",
    "ssp585_2100": "SSP5-8.5 / 2100",
}
SC_SHORT = {k: v.replace(" / ", "\n") for k, v in SCENARIO_LABELS.items()}
SC_COLORS = {
    "ssp245_2050": "#5b9bd5", "ssp245_2100": "#1a5276",
    "ssp585_2050": "#e57373", "ssp585_2100": "#b71c1c",
}

BBOX_LIST = [BBOX["west"], BBOX["east"], BBOX["south"], BBOX["north"]]


# ── Moran's I ─────────────────────────────────────────────────────────────────

def compute_global_moran(grid: gpd.GeoDataFrame, scenario: str) -> dict:
    try:
        from libpysal.weights import KNN
        from esda.moran import Moran

        col = f"FACVI_{scenario}"
        if col not in grid.columns:
            return {"scenario": scenario, "error": f"column {col} not found"}

        grid_proj = grid.to_crs("EPSG:32636")

        if len(grid_proj) > 100_000:
            log.info(f"  [{scenario}] Sampling 10,000 cells for Moran's I ...")
            rng = np.random.default_rng(42)
            idx = rng.choice(len(grid_proj), 10_000, replace=False)
            sample = grid_proj.iloc[idx].reset_index(drop=True)
        else:
            sample = grid_proj.copy()

        w = KNN.from_dataframe(sample, k=8, silence_warnings=True)
        w.transform = "R"
        mi = Moran(sample[col].values, w)

        result = {
            "scenario":     scenario,
            "moran_I":      float(mi.I),
            "expected_I":   float(mi.EI),
            "z_score":      float(mi.z_norm),
            "p_value":      float(mi.p_norm),
            "n_sample":     len(sample),
        }
        log.info(f"  [{scenario}] I={mi.I:.4f}  z={mi.z_norm:.2f}  p={mi.p_norm:.4f}")
        return result
    except Exception as e:
        log.error(f"  [{scenario}] Moran error: {e}")
        return {"scenario": scenario, "error": str(e)}


def compute_lisa(grid: gpd.GeoDataFrame, scenario: str) -> gpd.GeoDataFrame:
    try:
        from libpysal.weights import KNN
        from esda.moran import Moran_Local

        col = f"FACVI_{scenario}"
        grid_proj = grid.to_crs("EPSG:32636")

        n_lisa = min(len(grid_proj), 15_000)
        if len(grid_proj) > n_lisa:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(grid_proj), n_lisa, replace=False)
            grid_proj = grid_proj.iloc[idx].reset_index(drop=True)

        w = KNN.from_dataframe(grid_proj, k=8, silence_warnings=True)
        w.transform = "R"
        lisa = Moran_Local(grid_proj[col].values, w, permutations=99, seed=42)

        sig = lisa.p_sim < 0.05
        cluster_map = {1: "HH", 2: "LH", 3: "LL", 4: "HL"}
        grid_proj[f"lisa_cluster_{scenario}"] = np.where(
            sig, pd.Series(lisa.q).map(cluster_map), "NS"
        )
        return grid_proj

    except Exception as e:
        log.error(f"  LISA error: {e}")
        return grid


# ── Grid → raster ─────────────────────────────────────────────────────────────

def _grid_to_raster(grid: gpd.GeoDataFrame, col: str, res: float = 0.009):
    centroids = grid.geometry.centroid
    lons, lats = centroids.x.values, centroids.y.values
    vals = grid[col].values.astype(float)

    lon_min, lon_max = lons.min(), lons.max()
    lat_min, lat_max = lats.min(), lats.max()
    n_cols = int(np.round((lon_max - lon_min) / res)) + 1
    n_rows = int(np.round((lat_max - lat_min) / res)) + 1

    col_i = np.clip(np.round((lons - lon_min) / res).astype(int), 0, n_cols - 1)
    row_i = np.clip(np.round((lat_max - lats) / res).astype(int), 0, n_rows - 1)

    raster = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
    raster[row_i, col_i] = vals
    extent = [lon_min - res/2, lon_max + res/2, lat_min - res/2, lat_max + res/2]
    return raster, extent


# ── FACVI map panel ───────────────────────────────────────────────────────────

def plot_facvi_maps(grid: gpd.GeoDataFrame, eco: gpd.GeoDataFrame | None):
    rasters = {}
    for sc in SCENARIOS:
        col = f"FACVI_{sc}"
        if col in grid.columns:
            rasters[sc] = _grid_to_raster(grid, col)

    ref = rasters.get("ssp585_2100", (None, None))[0]
    global_p80 = float(np.nanpercentile(ref, 80)) if ref is not None else 0.6

    cmap = plt.colormaps["YlOrRd"].copy()
    cmap.set_bad(color="#d4d4d4")

    fig, axes = plt.subplots(2, 2, figsize=(16, 11),
                             gridspec_kw={"hspace": 0.28, "wspace": 0.1})
    axes = axes.flatten()

    for i, sc in enumerate(SCENARIOS):
        ax = axes[i]
        if sc not in rasters:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            continue
        raster, extent = rasters[sc]
        ax.imshow(raster, cmap=cmap, vmin=0, vmax=0.5,
                  extent=extent, origin="upper", aspect="auto",
                  interpolation="nearest")
        if eco is not None:
            eco.boundary.plot(ax=ax, color="white", linewidth=0.9,
                              alpha=0.55, zorder=3)
        ax.text(0.015, 0.975, f"({chr(ord('a') + i)})",
                transform=ax.transAxes, fontsize=11, fontweight="bold",
                va="top", color="white")
        ax.set_title(SCENARIO_LABELS[sc], fontsize=11, fontweight="bold", pad=6)
        if i >= 2:
            ax.set_xlabel("Longitude (°)", fontsize=8)
        if i % 2 == 0:
            ax.set_ylabel("Latitude (°)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_xlim(BBOX["west"], BBOX["east"])
        ax.set_ylim(BBOX["south"], BBOX["north"])
        fvals = raster[np.isfinite(raster)]
        mean_v = float(np.nanmean(raster))
        hi_pct = float((fvals > global_p80).mean() * 100)
        ax.text(0.015, 0.035,
                f"Mean FACVI: {mean_v:.3f}\nAbove P80: {hi_pct:.1f}%",
                transform=ax.transAxes, fontsize=7.5, va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="none", alpha=0.80), zorder=6)

    fig.subplots_adjust(bottom=0.12, hspace=0.28, wspace=0.1)
    cbar_ax = fig.add_axes([0.25, 0.05, 0.50, 0.018])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 0.5))
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal",
                      label="FACVI score")
    cb.ax.tick_params(labelsize=8)
    fig.suptitle(
        "Forest Access Climate Vulnerability Index (FACVI) across CMIP6 SSP Scenarios",
        fontsize=13, fontweight="bold", y=1.005,
    )
    out = OUTPUT_DIR / "fig_facvi_maps.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    log.info(f"[FIG] {out.name}")


# ── Weight chart ──────────────────────────────────────────────────────────────

def plot_entropy_weights(weights_df: pd.DataFrame):
    VAR_LABELS = {
        "C1": "ΔMean Temperature (C1)",
        "C2": "ΔAnnual Precipitation (C2)",
        "C3": "ΔMax-Month Temperature (C3)",
        "C4": "ΔPrecip. Seasonality (C4)",
        "S1": "Terrain Slope (S1)",
        "S2": "Topographic Wetness Index (S2)",
        "S3": "Soil Erodibility K-factor (S3)",
        "S4": "Veg. Cover / NDVI inv. (S4)",
        "S5": "Snow Cover Days (S5)",
        "N1": "Road Density (N1)",
        "N2": "Road Surface Quality (N2)",
        "N3": "Stream Proximity (N3)",
        "N4": "Fire Watchtower Dist. (N4)",
        "N5": "Fire Mgmt. Centre Dist. (N5)",
    }
    DIM_COLORS = {"Climate": "#2196F3", "Physical": "#4CAF50", "Network": "#FF9800"}
    DIM_MAP: dict[str, str] = {}
    for dim, codes in DIMENSIONS.items():
        for c in codes:
            DIM_MAP[c] = dim

    weights_df = weights_df.copy()
    weights_df["dimension"] = weights_df["variable"].map(DIM_MAP)

    pivot = weights_df.pivot_table(index="variable", columns="scenario",
                                   values="weight_pct")
    for sc in SCENARIOS:
        if sc not in pivot.columns:
            pivot[sc] = np.nan
    pivot["mean"] = pivot[list(SCENARIOS)].mean(axis=1)
    pivot["lo"]   = pivot[list(SCENARIOS)].min(axis=1)
    pivot["hi"]   = pivot[list(SCENARIOS)].max(axis=1)
    pivot = pivot.sort_values("mean", ascending=True)

    fig, (ax_lol, ax_dim) = plt.subplots(
        1, 2, figsize=(16, 5.5), gridspec_kw={"width_ratios": [2.2, 1]}
    )

    y_pos  = np.arange(len(pivot))
    clrs   = [DIM_COLORS.get(DIM_MAP.get(v, ""), "#888") for v in pivot.index]
    labels = [VAR_LABELS.get(v, v) for v in pivot.index]

    ax_lol.hlines(y_pos, pivot["lo"], pivot["hi"], color=clrs, linewidth=1.5, alpha=0.35)
    ax_lol.scatter(pivot["mean"], y_pos, color=clrs, s=80, zorder=4)
    ax_lol.hlines(y_pos, 0, pivot["lo"], color=clrs, linewidth=0.6, alpha=0.2, linestyle=":")
    for i, (_, row) in enumerate(pivot.iterrows()):
        lbl = "<0.1%" if row["mean"] < 0.05 else f"{row['mean']:.1f}%"
        ax_lol.text(row["hi"] + 0.3, i, lbl, va="center", fontsize=8, color="#333")
    ax_lol.set_yticks(y_pos)
    ax_lol.set_yticklabels(labels, fontsize=9)
    n_vars = len(pivot)
    ax_lol.axvline(x=100/n_vars, color="#888", linestyle="--", linewidth=1, alpha=0.6,
                   label=f"Equal weight ({100/n_vars:.1f}%)")
    ax_lol.set_xlabel("Entropy Weight (%)", fontsize=10)
    ax_lol.set_title("Variable Weights by Entropy Method\n"
                      "(dot = mean; bar = range across scenarios)",
                      fontsize=11, fontweight="bold")
    ax_lol.set_xlim(0, pivot["hi"].max() + 4)
    ax_lol.grid(axis="x", alpha=0.25)
    ax_lol.spines[["top", "right", "left"]].set_visible(False)
    patches = [mpatches.Patch(color=c, label=d) for d, c in DIM_COLORS.items()]
    ax_lol.legend(handles=patches, loc="lower right", fontsize=9,
                  framealpha=0.8, edgecolor="none")

    dim_sc = (weights_df.groupby(["scenario", "dimension"])["weight_pct"]
              .sum().unstack("dimension"))
    for dim in ["Climate", "Physical", "Network"]:
        if dim not in dim_sc.columns:
            dim_sc[dim] = 0.0
    dim_sc = dim_sc[["Climate", "Physical", "Network"]]
    dim_sc = dim_sc.reindex([sc for sc in SCENARIOS if sc in dim_sc.index])
    dim_sc.index = [SC_SHORT.get(s, s) for s in dim_sc.index]

    bottom = np.zeros(len(dim_sc))
    for dim in ["Climate", "Physical", "Network"]:
        vals = dim_sc[dim].values
        ax_dim.bar(range(len(dim_sc)), vals, bottom=bottom,
                   color=DIM_COLORS[dim], label=dim,
                   edgecolor="white", linewidth=0.8, width=0.6)
        for j, (v, b) in enumerate(zip(vals, bottom)):
            if v > 3:
                ax_dim.text(j, b + v/2, f"{v:.0f}%",
                            ha="center", va="center",
                            fontsize=8, color="white", fontweight="bold")
        bottom += vals

    ax_dim.set_xticks(range(len(dim_sc)))
    ax_dim.set_xticklabels(dim_sc.index, fontsize=9)
    ax_dim.set_ylabel("Cumulative Weight (%)", fontsize=10)
    ax_dim.set_ylim(0, 105)
    ax_dim.set_title("Dimension Contribution\nby Scenario-Horizon",
                     fontsize=11, fontweight="bold")
    ax_dim.legend(loc="upper right", fontsize=8, framealpha=0.8, edgecolor="none")
    ax_dim.spines[["top", "right"]].set_visible(False)
    ax_dim.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    out = OUTPUT_DIR / "fig_weights.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    log.info(f"[FIG] {out.name}")


# ── Scenario comparison ───────────────────────────────────────────────────────

def plot_scenario_comparison(grid: gpd.GeoDataFrame, eco: gpd.GeoDataFrame | None):
    sc_list  = list(SCENARIOS.keys()) if isinstance(SCENARIOS, dict) else list(SCENARIOS)
    col_ref  = f"FACVI_{sc_list[0]}"
    col_high = f"FACVI_{sc_list[-1]}"
    if col_ref not in grid.columns or col_high not in grid.columns:
        log.warning("  Skipping scenario comparison: columns not found.")
        return

    grid_tmp = grid.copy()
    grid_tmp["_delta"] = grid[col_high].values - grid[col_ref].values
    delta_raster, extent = _grid_to_raster(grid_tmp, "_delta")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6),
                             gridspec_kw={"width_ratios": [1.1, 1.6, 0.9]})
    ax_viol, ax_delta, ax_bar = axes

    # (a) Violin
    valid_sc  = [s for s in sc_list if f"FACVI_{s}" in grid.columns]
    rvi_arrs  = [grid[f"FACVI_{s}"].values for s in valid_sc]
    parts = ax_viol.violinplot(rvi_arrs, positions=range(len(valid_sc)),
                               widths=0.65, showmedians=True, showextrema=False)
    for pc, sc in zip(parts["bodies"], valid_sc):
        pc.set_facecolor(SC_COLORS.get(sc, "#888"))
        pc.set_alpha(0.75)
        pc.set_edgecolor("none")
    parts["cmedians"].set_color("white")
    parts["cmedians"].set_linewidth(2)
    p80 = float(np.nanpercentile(grid[col_high].values, 80))
    ax_viol.axhline(p80, color="#c0392b", linestyle="--", linewidth=1.2,
                    label=f"P80 threshold ({p80:.2f})", alpha=0.8)
    ax_viol.set_xticks(range(len(valid_sc)))
    ax_viol.set_xticklabels([SC_SHORT.get(s, s) for s in valid_sc], fontsize=9)
    ax_viol.set_ylabel("FACVI Score", fontsize=10)
    ax_viol.set_ylim(0, 0.6)
    ax_viol.set_title("(a) FACVI Distribution", fontsize=11, fontweight="bold")
    ax_viol.legend(fontsize=8, loc="upper left", framealpha=0.8, edgecolor="none")
    ax_viol.grid(axis="y", alpha=0.25)
    ax_viol.spines[["top", "right"]].set_visible(False)

    # (b) Delta raster
    abs_max = min(max(abs(float(np.nanpercentile(delta_raster, 2))),
                      abs(float(np.nanpercentile(delta_raster, 98)))), 0.4)
    cmap_div = plt.colormaps["RdYlBu_r"].copy()
    cmap_div.set_bad(color="#d4d4d4")
    im = ax_delta.imshow(delta_raster, cmap=cmap_div, vmin=-abs_max, vmax=abs_max,
                         extent=extent, origin="upper", aspect="auto",
                         interpolation="nearest")
    fig.colorbar(im, ax=ax_delta, shrink=0.6, pad=0.12, orientation="horizontal",
                 label=f"ΔFACVI ({sc_list[-1]} − {sc_list[0]})").ax.tick_params(labelsize=7)
    if eco is not None:
        eco.boundary.plot(ax=ax_delta, color="white", linewidth=0.7,
                          alpha=0.5, zorder=3)
    ax_delta.set_xlim(BBOX["west"], BBOX["east"])
    ax_delta.set_ylim(BBOX["south"], BBOX["north"])
    ax_delta.set_xlabel("Longitude (°)", fontsize=8)
    ax_delta.set_ylabel("Latitude (°)", fontsize=8)
    ax_delta.tick_params(labelsize=7)
    ax_delta.set_title("(b) ΔFACVI Change Map", fontsize=11, fontweight="bold")

    # (c) Risk-tier bars
    ref_arr = grid[col_ref].values
    t_lo = float(np.percentile(ref_arr, 33))
    t_hi = float(np.percentile(ref_arr, 67))
    tiers = [(0.0, t_lo), (t_lo, t_hi), (t_hi, 1.01)]
    tier_lbl = [f"Low", "Moderate", "High"]
    tier_clr = ["#5b9bd5", "#f0ad4e", "#c0392b"]
    bottoms = np.zeros(len(valid_sc))
    for (lo, hi), lbl, clr in zip(tiers, tier_lbl, tier_clr):
        vals = [float(np.sum((grid[f"FACVI_{s}"].values >= lo) &
                             (grid[f"FACVI_{s}"].values < hi)) / len(grid) * 100)
                for s in valid_sc]
        ax_bar.bar(range(len(valid_sc)), vals, bottom=bottoms, color=clr,
                   label=lbl, edgecolor="white", linewidth=0.8, width=0.6)
        for j, (v, b) in enumerate(zip(vals, bottoms)):
            if v > 5:
                ax_bar.text(j, b + v/2, f"{v:.0f}%", ha="center", va="center",
                            fontsize=8, color="white", fontweight="bold")
        bottoms += np.array(vals)
    ax_bar.set_xticks(range(len(valid_sc)))
    ax_bar.set_xticklabels([SC_SHORT.get(s, s) for s in valid_sc], fontsize=9)
    ax_bar.set_ylabel("Forest-access cells (%)", fontsize=10)
    ax_bar.set_ylim(0, 105)
    ax_bar.set_title("(c) Risk-Tier Composition", fontsize=11, fontweight="bold")
    ax_bar.legend(loc="upper left", fontsize=8, framealpha=0.8, edgecolor="none")
    ax_bar.spines[["top", "right"]].set_visible(False)
    ax_bar.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    out = OUTPUT_DIR / "fig_scenario_comparison.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    log.info(f"[FIG] {out.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("FACVI Step 11: Spatial Analysis")
    log.info("=" * 60)

    if not GRID_RVI_FILE.exists():
        raise FileNotFoundError(
            f"FACVI grid not found: {GRID_RVI_FILE}\n"
            "Run 10_facvi.py first."
        )

    log.info(f"[LOAD] {GRID_RVI_FILE.name}")
    grid = gpd.read_file(GRID_RVI_FILE)
    log.info(f"  {len(grid):,} cells")

    # Load ecoregions
    eco = None
    if ECOREGIONS_FILE.exists():
        eco = gpd.read_file(ECOREGIONS_FILE)
        log.info(f"  Ecoregions: {len(eco)} polygons")
    else:
        log.warning(f"  Ecoregions not found: {ECOREGIONS_FILE.name}  "
                    f"(run 03_auxiliary_data.py)")

    # Global Moran's I
    log.info("\n[MORAN] Global spatial autocorrelation ...")
    moran_rows = []
    for sc in SCENARIOS:
        result = compute_global_moran(grid, sc)
        moran_rows.append(result)
    moran_df = pd.DataFrame(moran_rows)
    moran_df.to_csv(OUTPUT_DIR / "moran_results.csv", index=False, encoding="utf-8-sig")
    log.info(f"[SAVED] moran_results.csv")

    # LISA for worst-case scenario
    worst_sc = list(SCENARIOS.keys())[-1] if isinstance(SCENARIOS, dict) else SCENARIOS[-1]
    log.info(f"\n[LISA] Local Moran's I for {worst_sc} ...")
    lisa_grid = compute_lisa(grid, worst_sc)
    out_lisa = OUTPUT_DIR / f"lisa_{worst_sc}.gpkg"
    lisa_grid.to_file(out_lisa, driver="GPKG")
    log.info(f"[SAVED] {out_lisa.name}")

    # Figures
    log.info("\n[FIGURES] Generating ...")
    plot_facvi_maps(grid, eco)
    plot_scenario_comparison(grid, eco)

    if WEIGHTS_FILE.exists():
        weights_df = pd.read_csv(WEIGHTS_FILE)
        plot_entropy_weights(weights_df)
    else:
        log.warning(f"  Weights file not found: {WEIGHTS_FILE.name}  (skip weight figure)")

    log.info("\nStep 11 complete. Outputs in: outputs/")


if __name__ == "__main__":
    main()
