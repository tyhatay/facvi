"""
FACVI Pipeline — Step 14: Compound Hazard Analysis
===================================================
Identifies grid cells with coincident high FACVI and landslide risk
(compound hazard) and analyses them within ecoregion strata.

Two analyses:
  A) Within-ecoregion FACVI vs. landslide presence (AUC, Spearman rho)
  B) Compound hazard map: High FACVI (class >= 4) AND landslide present

This is the public version of the original compound hazard step.
OGM-proprietary heyelan (landslide) data is replaced with a
STANDARD GEOPACKAGE supplied by the user.

Landslide inventory format (GeoPackage or shapefile):
  - geometry: Point or Polygon (any CRS; reprojected automatically)
  - No mandatory columns beyond geometry; presence = hazard signal

Where to get public landslide data:
  - NASA Global Landslide Catalog:
    https://gpm.nasa.gov/landslides/projects.html
  - Global Fatal Landslide Database (Froude & Petley 2018):
    https://doi.org/10.5194/nhess-18-2161-2018
  - ESA Landslide ECV:
    https://cci.esa.int/
  - National geological surveys (many have open portals)

CLI:
  python 14_compound_hazard.py --landslide data/landslide_inventory.gpkg

Outputs:
  outputs/ecoregion_landslide_analysis.csv
  outputs/compound_hazard_ecoregion.csv
  outputs/fig_ecoregion_landslide.png
  outputs/fig_compound_hazard.png
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
from sklearn.metrics import roc_auc_score
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
        logging.FileHandler(LOG_DIR / "14_compound_hazard.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

SCENARIO    = "ssp245_2050"
FACVI_HIGH  = 4       # class >= 4 → "High FACVI"
MIN_CELLS   = 500     # minimum cells per ecoregion for analysis
ECO_FILE    = DATA_DIR / f"{COUNTRY_CODE}_ecoregions.gpkg"


# ── Load and rasterise landslide inventory ────────────────────────────────────

def load_landslide_inventory(path: Path, target_crs) -> gpd.GeoDataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Landslide inventory not found: {path}\n\n"
            "Provide a GeoPackage or Shapefile with landslide locations.\n\n"
            "Public sources:\n"
            "  NASA GLC: https://gpm.nasa.gov/landslides/projects.html\n"
            "  Froude & Petley 2018: https://doi.org/10.5194/nhess-18-2161-2018\n"
            "  National geological surveys"
        )
    log.info(f"[LOAD] Landslide inventory: {path}")
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    gdf = gdf[gdf.geometry.notna()].to_crs(target_crs)
    log.info(f"  {len(gdf):,} landslide records")
    return gdf


def assign_landslide_to_grid(grid: gpd.GeoDataFrame,
                              landslides: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Binary landslide_present column: 1 if any landslide geometry within cell."""
    grid = grid.copy()
    grid["_idx"] = np.arange(len(grid))

    centroids = grid.copy()
    centroids["geometry"] = grid.geometry.centroid

    # Normalise all geometries to centroids for the join (Point inventory)
    ls_pts = landslides.copy()
    has_poly = ~ls_pts.geometry.geom_type.isin(["Point", "MultiPoint"])
    if has_poly.any():
        ls_pts.loc[has_poly, "geometry"] = ls_pts.loc[has_poly].geometry.centroid

    joined = gpd.sjoin(
        ls_pts[["geometry"]].reset_index(drop=True),
        grid[["_idx", "geometry"]],
        how="left", predicate="within",
    ).dropna(subset=["_idx"])
    joined["_idx"] = joined["_idx"].astype(int)

    hit_idx = joined["_idx"].unique()
    grid["landslide_present"] = 0
    grid.loc[grid["_idx"].isin(hit_idx), "landslide_present"] = 1
    grid = grid.drop(columns="_idx")

    n_ls = grid["landslide_present"].sum()
    log.info(f"  Grid cells with landslide: {n_ls:,} ({n_ls/len(grid)*100:.2f}%)")
    return grid


# ── Ecoregion assignment ──────────────────────────────────────────────────────

def assign_ecoregions(grid: gpd.GeoDataFrame,
                       eco: gpd.GeoDataFrame) -> pd.Series:
    if eco.crs != grid.crs:
        eco = eco.to_crs(grid.crs)

    centroids = grid.copy()
    centroids["geometry"] = grid.geometry.centroid

    joined = gpd.sjoin(
        centroids[["geometry"]],
        eco[["ECO_NAME", "geometry"]],
        how="left", predicate="within",
    )
    eco_names = joined["ECO_NAME"].copy()

    unmatched = eco_names.isna()
    if unmatched.any():
        nn = gpd.sjoin_nearest(
            centroids[unmatched][["geometry"]],
            eco[["ECO_NAME", "geometry"]],
            how="left",
        )
        eco_names.loc[unmatched] = nn["ECO_NAME"].values

    return eco_names.values


# ── Analysis A: within-ecoregion ─────────────────────────────────────────────

def ecoregion_landslide_analysis(grid: gpd.GeoDataFrame) -> pd.DataFrame:
    facvi_col = f"FACVI_{SCENARIO}"
    cls_col   = f"FACVI_class_{SCENARIO}"
    ls_col    = "landslide_present"

    rows = []
    for eco_name in sorted(grid["ECO_NAME"].dropna().unique()):
        sub = grid[grid["ECO_NAME"] == eco_name]
        n   = len(sub)
        if n < MIN_CELLS:
            continue

        facvi = sub[facvi_col].values
        ls    = sub[ls_col].values.astype(int)
        n_ls  = ls.sum()

        if n_ls < 50 or n_ls > n - 50:
            continue

        rho, pval = spearmanr(facvi, ls)
        try:
            auc = roc_auc_score(ls, facvi)
        except Exception:
            auc = np.nan

        cls_arr    = sub[cls_col].values
        class_rates = {}
        for cls in [1, 2, 3, 4, 5]:
            m = (cls_arr == cls)
            class_rates[cls] = float(ls[m].mean() * 100) if m.sum() > 0 else np.nan

        rows.append({
            "ecoregion":    eco_name,
            "n_cells":      n,
            "ls_rate_pct":  n_ls / n * 100,
            "n_landslide":  n_ls,
            "spearman_rho": rho,
            "spearman_p":   pval,
            "roc_auc":      auc,
            **{f"ls_rate_cls{c}": class_rates[c] for c in [1,2,3,4,5]},
        })

    df = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
    log.info(f"  {len(df)} ecoregions analysed (>= {MIN_CELLS} cells, >= 50 landslide hits)")
    return df


# ── Analysis B: compound hazard stats ────────────────────────────────────────

def compound_hazard_stats(grid: gpd.GeoDataFrame) -> pd.DataFrame:
    cls_col = f"FACVI_class_{SCENARIO}"
    ls_col  = "landslide_present"

    grid = grid.copy()
    grid["high_facvi"] = (grid[cls_col] >= FACVI_HIGH).astype(int)
    grid["compound"]   = ((grid[cls_col] >= FACVI_HIGH) &
                          (grid[ls_col] == 1)).astype(int)

    n_tot      = len(grid)
    n_hfacvi   = int(grid["high_facvi"].sum())
    n_ls       = int(grid[ls_col].sum())
    n_compound = int(grid["compound"].sum())
    expected   = n_hfacvi * n_ls / n_tot

    log.info(f"\n  National Compound Hazard Summary:")
    log.info(f"    High FACVI (class >= {FACVI_HIGH}): {n_hfacvi:,} ({n_hfacvi/n_tot*100:.1f}%)")
    log.info(f"    Landslide present:                {n_ls:,} ({n_ls/n_tot*100:.1f}%)")
    log.info(f"    Compound (both):                  {n_compound:,} ({n_compound/n_tot*100:.1f}%)")
    log.info(f"    Expected (independence):          {expected:.0f}")

    eco_rows = []
    for eco_name, sub in grid.groupby("ECO_NAME"):
        if len(sub) < 500:
            continue
        n  = len(sub)
        nc = int(sub["compound"].sum())
        nh = int(sub["high_facvi"].sum())
        nl = int(sub[ls_col].sum())
        eco_rows.append({
            "ecoregion":                   eco_name,
            "n_cells":                     n,
            "n_high_facvi":                nh,
            "n_landslide":                 nl,
            "n_compound":                  nc,
            "compound_pct":                nc / n * 100,
            "pct_of_highfacvi_with_ls":    nc / nh * 100 if nh > 0 else 0.0,
        })

    return (pd.DataFrame(eco_rows)
            .sort_values("n_compound", ascending=False))


# ── Figures ───────────────────────────────────────────────────────────────────

def fig_ecoregion_analysis(eco_df: pd.DataFrame):
    if eco_df.empty:
        log.warning("[FIG] No ecoregion data — skipping ecoregion figure.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    order   = eco_df.sort_values("roc_auc", ascending=True)
    n_eco   = len(order)
    aucs    = order["roc_auc"].values
    colors  = ["#c0392b" if a >= 0.5 else "#2980b9" for a in aucs]

    ax1.barh(range(n_eco), aucs, color=colors, edgecolor="white", linewidth=0.6, height=0.7)
    ax1.axvline(0.5, color="#333", linewidth=1.4, linestyle="--")

    for i, (auc, rho, p) in enumerate(
            zip(aucs, order["spearman_rho"], order["spearman_p"])):
        p_s  = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
        sign = "+" if rho >= 0 else "-"
        ax1.text(max(auc, 0.5) + 0.003, i,
                 f"AUC={auc:.3f}  ρ={sign}{abs(rho):.3f}{p_s}",
                 va="center", ha="left", fontsize=8)

    ax1.set_yticks(range(n_eco))
    ax1.set_yticklabels(
        [e[:30] for e in order["ecoregion"]], fontsize=8
    )
    ax1.set_xlabel("ROC-AUC (FACVI vs. Landslide Presence)", fontsize=10)
    ax1.set_title(
        f"(a) Within-Ecoregion FACVI vs. Landslide Inventory\n{SCENARIO}",
        fontsize=10.5, fontweight="bold"
    )
    ax1.set_xlim(0.25, min(1.0, aucs.max() + 0.18))
    ax1.spines[["top", "right"]].set_visible(False)
    red_p  = mpatches.Patch(color="#c0392b", label="AUC >= 0.50 (positive association)")
    blue_p = mpatches.Patch(color="#2980b9", label="AUC < 0.50 (inverse association)")
    ax1.legend(handles=[red_p, blue_p], fontsize=8.5, loc="lower right")

    # (b) Class-level landslide rates for top ecoregions
    cls_cols = [f"ls_rate_cls{c}" for c in [1,2,3,4,5]]
    top_eco  = eco_df.sort_values("n_landslide", ascending=False).head(4)
    palette  = ["#1a6faf", "#c0392b", "#27ae60", "#8e44ad"]
    x = np.arange(5)
    w = 0.18
    for i, (_, row) in enumerate(top_eco.iterrows()):
        rates  = [row[c] if not pd.isna(row[c]) else 0.0 for c in cls_cols]
        short  = row["ecoregion"][:22]
        offset = (i - 1.5) * w
        ax2.bar(x + offset, rates, width=w, color=palette[i],
                alpha=0.85, label=short, edgecolor="white", linewidth=0.5)

    ax2.set_xticks(x)
    ax2.set_xticklabels(["Very Low", "Low", "Medium", "High", "Very High"], fontsize=9)
    ax2.set_xlabel(f"FACVI Class ({SCENARIO})", fontsize=10)
    ax2.set_ylabel("Landslide Occurrence Rate (%)", fontsize=10)
    ax2.set_title("(b) Landslide Rate by FACVI Class — Top Ecoregions",
                  fontsize=10.5, fontweight="bold")
    ax2.legend(fontsize=8, loc="upper right")
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    out = OUTPUT_DIR / "fig_ecoregion_landslide.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    log.info(f"[FIG] {out.name}")


def fig_compound_map(grid: gpd.GeoDataFrame, comp_df: pd.DataFrame):
    cls_col = f"FACVI_class_{SCENARIO}"
    ls_col  = "landslide_present"

    grid = grid.copy()
    cat_colors = {1: "#d0d0d0", 2: "#3498db", 3: "#f39c12", 4: "#c0392b"}
    cat_labels = {
        1: "Low FACVI, no landslide",
        2: "Low FACVI + landslide",
        3: "High FACVI, no landslide",
        4: "Compound hazard (High FACVI + landslide)",
    }

    grid["cat"] = 0
    grid.loc[(grid[cls_col] < FACVI_HIGH)  & (grid[ls_col] == 0), "cat"] = 1
    grid.loc[(grid[cls_col] < FACVI_HIGH)  & (grid[ls_col] == 1), "cat"] = 2
    grid.loc[(grid[cls_col] >= FACVI_HIGH) & (grid[ls_col] == 0), "cat"] = 3
    grid.loc[(grid[cls_col] >= FACVI_HIGH) & (grid[ls_col] == 1), "cat"] = 4

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5),
                                    gridspec_kw={"width_ratios": [1.6, 1]})

    for cat in [1, 2, 3, 4]:
        sub = grid[grid["cat"] == cat]
        if sub.empty:
            continue
        sub.plot(ax=ax1, color=cat_colors[cat],
                 markersize=0.1, linewidth=0, alpha=0.85)

    n_compound = int((grid["cat"] == 4).sum())
    patches = [
        mpatches.Patch(color=cat_colors[c],
                       label=f"{cat_labels[c]}\n(n={int((grid['cat']==c).sum()):,})")
        for c in [4, 3, 2, 1]
    ]
    ax1.legend(handles=patches, fontsize=8, loc="lower right",
               framealpha=0.9, title="Category", title_fontsize=9)
    ax1.set_title(
        f"(a) Compound Hazard Map: High FACVI + Landslide Inventory\n"
        f"{SCENARIO}  |  Compound cells = {n_compound:,} "
        f"({n_compound/len(grid)*100:.1f}%)",
        fontsize=10.5, fontweight="bold"
    )
    ax1.axis("off")

    # (b) By ecoregion
    top = comp_df[comp_df["n_compound"] > 0].head(10).copy()
    if not top.empty:
        x = np.arange(len(top))
        ax2.bar(x, top["n_compound"], color="#c0392b", alpha=0.85,
                edgecolor="white", linewidth=0.6, label="Compound hazard cells")
        ax2b = ax2.twinx()
        ax2b.plot(x, top["pct_of_highfacvi_with_ls"],
                  color="#8e44ad", marker="o", linewidth=2, markersize=6,
                  label="% of High-FACVI cells\nalso with landslide")
        ax2b.set_ylabel("% of High-FACVI cells with landslide",
                        fontsize=9, color="#8e44ad")
        ax2b.tick_params(axis="y", labelcolor="#8e44ad")

        for bar, val in zip(ax2.patches, top["n_compound"]):
            ax2.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + top["n_compound"].max()*0.02,
                     f"{val:,}", ha="center", va="bottom", fontsize=7.5,
                     fontweight="bold")

        ax2.set_xticks(x)
        ax2.set_xticklabels([e[:18] for e in top["ecoregion"]],
                            fontsize=7.5, rotation=35, ha="right")
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2b.get_legend_handles_labels()
        ax2.legend(lines1+lines2, labels1+labels2, fontsize=8, loc="upper right")
    else:
        ax2.text(0.5, 0.5, "No compound hazard cells found",
                 ha="center", va="center", transform=ax2.transAxes, color="gray")

    ax2.set_ylabel("Number of compound hazard cells", fontsize=9)
    ax2.set_title("(b) Compound hazard by ecoregion",
                  fontsize=10.5, fontweight="bold")
    ax2.spines[["top"]].set_visible(False)

    plt.suptitle(
        "Compound Hazard Analysis: FACVI x Landslide Inventory",
        fontsize=11.5, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    out = OUTPUT_DIR / "fig_compound_hazard.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    log.info(f"[FIG] {out.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="FACVI compound hazard analysis (Step 14)"
    )
    parser.add_argument(
        "--landslide", type=Path,
        default=DATA_DIR / "landslide_inventory.gpkg",
        help="Path to landslide inventory GeoPackage or Shapefile "
             "(geometry: Point or Polygon, any CRS)",
    )
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("FACVI Step 14: Compound Hazard Analysis")
    log.info("=" * 60)

    if not GRID_RVI_FILE.exists():
        raise FileNotFoundError(
            f"FACVI grid not found: {GRID_RVI_FILE}\n"
            "Run 10_facvi.py first."
        )

    log.info(f"[LOAD] {GRID_RVI_FILE.name}")
    grid = gpd.read_file(GRID_RVI_FILE)
    log.info(f"  {len(grid):,} cells")

    # Assign landslide presence
    landslides = load_landslide_inventory(args.landslide, grid.crs)
    grid = assign_landslide_to_grid(grid, landslides)

    # Assign ecoregions if ecoregions file exists
    use_eco = ECO_FILE.exists()
    if use_eco:
        log.info(f"[ECO] Loading {ECO_FILE.name}")
        eco = gpd.read_file(ECO_FILE)
        grid["ECO_NAME"] = assign_ecoregions(grid, eco)
        n_assigned = pd.notna(grid["ECO_NAME"]).sum()
        log.info(f"  Assigned: {n_assigned:,} / {len(grid):,}")
    else:
        log.warning(f"[ECO] {ECO_FILE.name} not found — ecoregion analysis skipped.")
        log.warning(f"  Run 03_auxiliary_data.py to generate ecoregion data.")
        grid["ECO_NAME"] = "Unknown"

    # A: within-ecoregion analysis
    log.info("\n[A] Within-ecoregion FACVI vs. landslide analysis ...")
    eco_df = ecoregion_landslide_analysis(grid)
    eco_df.to_csv(OUTPUT_DIR / "ecoregion_landslide_analysis.csv",
                  index=False, encoding="utf-8-sig")
    log.info("[SAVED] ecoregion_landslide_analysis.csv")

    if not eco_df.empty:
        log.info(f"\n  Ecoregion                      N        ls%    rho     AUC")
        log.info(f"  {'-'*60}")
        for _, row in eco_df.iterrows():
            log.info(f"  {row['ecoregion'][:30]:30s}  "
                     f"{row['n_cells']:7,}  "
                     f"{row['ls_rate_pct']:5.1f}%  "
                     f"{row['spearman_rho']:+.3f}  "
                     f"{row['roc_auc']:.3f}")

    # B: compound hazard
    log.info("\n[B] Compound hazard statistics ...")
    comp_df = compound_hazard_stats(grid)
    comp_df.to_csv(OUTPUT_DIR / "compound_hazard_ecoregion.csv",
                   index=False, encoding="utf-8-sig")
    log.info("[SAVED] compound_hazard_ecoregion.csv")

    # Figures
    log.info("\n[FIG] Generating figures ...")
    fig_ecoregion_analysis(eco_df)
    fig_compound_map(grid, comp_df)

    n_compound = int(
        ((grid[f"FACVI_class_{SCENARIO}"] >= FACVI_HIGH) &
         (grid["landslide_present"] == 1)).sum()
    )
    log.info(f"\n  Compound hazard cells: {n_compound:,} "
             f"({n_compound/len(grid)*100:.1f}% of forest-access grid)")
    log.info("Step 14 complete.")


if __name__ == "__main__":
    main()
