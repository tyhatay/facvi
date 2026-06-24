"""
FACVI Pipeline — Step 12: Sensitivity Analysis
==============================================
Tests robustness of FACVI rankings via:

  1. Monte Carlo (n=1000): uniform ±30% weight perturbation;
     Spearman ρ between original and perturbed ranking
  2. Leave-one-out exclusion: each variable removed in turn;
     ρ drop = variable importance

Reference scenario: ssp245_2050 (baseline scenario)

Outputs:
  outputs/sensitivity_mc.csv
  outputs/sensitivity_exclusion.csv
  outputs/fig_sensitivity.png
"""

import matplotlib
matplotlib.use("Agg")

import geopandas as gpd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import spearmanr
import logging
from pathlib import Path

from config import (
    DATA_DIR, LOG_DIR,
    SCENARIOS, DIMENSIONS, VARIABLE_MAP,
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
        logging.FileHandler(LOG_DIR / "12_sensitivity.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

WEIGHTS_FILE = DATA_DIR / "entropy_weights.csv"
N_ITER       = 1000
PERTURB      = 0.30
RNG_SEED     = 42
REF_SCENARIO = list(SCENARIOS.keys())[0] if isinstance(SCENARIOS, dict) else SCENARIOS[0]

VAR_LABELS = {
    "C1": "ΔTemp Mean (C1)",        "C2": "ΔAnnual Precip. (C2)",
    "C3": "ΔMax-Month Temp. (C3)",  "C4": "ΔPrecip Seasonality (C4)",
    "S1": "Terrain Slope (S1)",     "S2": "Topographic Wetness (S2)",
    "S3": "Soil Erodibility (S3)",  "S4": "Veg. Cover / NDVI inv. (S4)",
    "S5": "Snow Cover Days (S5)",
    "N1": "Road Density (N1)",      "N2": "Road Surface Quality (N2)",
    "N3": "Stream Proximity (N3)",  "N4": "Fire Watchtower Dist. (N4)",
    "N5": "Fire Mgmt. Centre Dist. (N5)",
}
DIM_COLORS = {
    "C1": "#2196F3", "C2": "#2196F3", "C3": "#2196F3", "C4": "#2196F3",
    "S1": "#4CAF50", "S2": "#4CAF50", "S3": "#4CAF50",
    "S4": "#4CAF50", "S5": "#4CAF50",
    "N1": "#FF9800", "N2": "#FF9800", "N3": "#FF9800",
    "N4": "#FF9800", "N5": "#FF9800",
}


# ── Rebuild normalised matrix ─────────────────────────────────────────────────

def _minmax(arr: np.ndarray, lo=None, hi=None, eps=1e-8) -> np.ndarray:
    arr = arr.astype(float)
    lo = lo if lo is not None else float(np.nanmin(arr))
    hi = hi if hi is not None else float(np.nanmax(arr))
    if hi - lo < 1e-10:
        return np.full_like(arr, eps)
    return np.clip((arr - lo) / (hi - lo) + eps, eps, 1.0)


def build_X(grid: gpd.GeoDataFrame, scenario: str,
            climate_ranges: dict) -> tuple[np.ndarray, list[str]]:
    """Reproduce the same normalisation used in 10_facvi.py."""
    n = len(grid)
    norm: dict[str, np.ndarray | None] = {}

    # Climate (global range)
    for code in DIMENSIONS["Climate"]:
        col = f"{code}_{scenario}"
        if col not in grid.columns:
            norm[code] = np.full(n, 1e-8)
            continue
        raw = np.abs(grid[col].values.astype(float))
        lo, hi = climate_ranges.get(code, (0, 1))
        norm[code] = _minmax(raw, lo=lo, hi=hi)

    # Physical and Network (from VARIABLE_MAP)
    for dim in ("Physical", "Network"):
        for code in DIMENSIONS[dim]:
            meta = VARIABLE_MAP.get(code)
            if meta is None:
                norm[code] = None
                continue
            col_name, invert, _ = meta
            if col_name not in grid.columns:
                norm[code] = None
                continue
            raw = grid[col_name].values.astype(float)
            n_arr = _minmax(raw)
            norm[code] = (1.0 - n_arr + 1e-8) if invert else n_arr

    all_codes = (DIMENSIONS["Climate"] + DIMENSIONS["Physical"] + DIMENSIONS["Network"])
    var_order = [c for c in all_codes if norm.get(c) is not None]
    X = np.column_stack([norm[c] for c in var_order])

    for j in range(X.shape[1]):
        bad = np.isnan(X[:, j])
        if bad.any():
            X[bad, j] = float(np.nanmedian(X[:, j]))

    return X, var_order


# ── Monte Carlo ───────────────────────────────────────────────────────────────

def monte_carlo_sensitivity(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    rng         = np.random.default_rng(RNG_SEED)
    facvi_orig  = X @ w
    rho_vals    = np.zeros(N_ITER)

    for i in range(N_ITER):
        noise = rng.uniform(1 - PERTURB, 1 + PERTURB, len(w))
        wp    = np.clip(w * noise, 0, None)
        wp   /= wp.sum()
        rho_vals[i], _ = spearmanr(facvi_orig, X @ wp)
        if (i + 1) % 200 == 0:
            log.info(f"  [{i+1}/{N_ITER}] median ρ = {np.median(rho_vals[:i+1]):.4f}")

    return rho_vals


# ── Leave-one-out exclusion ───────────────────────────────────────────────────

def exclusion_sensitivity(X: np.ndarray, w: np.ndarray,
                          var_order: list[str]) -> pd.DataFrame:
    facvi_orig = X @ w
    rows = []
    for j, var in enumerate(var_order):
        we = w.copy()
        we[j] = 0.0
        if we.sum() > 0:
            we /= we.sum()
        rho, _ = spearmanr(facvi_orig, X @ we)
        rows.append({"variable": var, "rho_excluded": rho,
                     "rho_drop": 1.0 - rho, "weight_orig": w[j]})
        log.info(f"  [{var}] ρ={rho:.4f}  Δρ={1-rho:.4f}  w={w[j]*100:.2f}%")
    return pd.DataFrame(rows).sort_values("rho_drop", ascending=False)


# ── Figure ────────────────────────────────────────────────────────────────────

def plot_sensitivity(rho_vals: np.ndarray, excl_df: pd.DataFrame,
                     w_orig: np.ndarray, var_order: list[str]):
    fig = plt.figure(figsize=(14, 6), layout="constrained")
    gs  = fig.add_gridspec(1, 3, wspace=0.4)
    ax1, ax2, ax3 = [fig.add_subplot(gs[i]) for i in range(3)]

    # (a) Monte Carlo histogram
    q5, q50, q95 = np.percentile(rho_vals, [5, 50, 95])
    ax1.hist(rho_vals, bins=40, color="#2196F3", alpha=0.75, edgecolor="white", linewidth=0.4)
    ax1.axvline(q50, color="#c0392b", linewidth=1.8, label=f"Median ρ = {q50:.3f}")
    ax1.axvline(q5,  color="#c0392b", linewidth=1.2, linestyle="--",
                label=f"P5 ρ = {q5:.3f}")
    pct95 = (rho_vals >= 0.95).mean() * 100
    ax1.text(0.03, 0.96,
             f"N = {len(rho_vals):,} runs\nPerturbation: ±{int(PERTURB*100)}%\n"
             f"ρ ≥ 0.95: {pct95:.1f}%",
             transform=ax1.transAxes, fontsize=8.5, va="top",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    ax1.set_xlabel("Spearman ρ  (original vs perturbed ranking)", fontsize=9)
    ax1.set_ylabel("Frequency", fontsize=9)
    ax1.set_title(f"(a) Monte Carlo Weight Perturbation\n"
                  f"({N_ITER:,} iterations, ±{int(PERTURB*100)}% uniform)",
                  fontsize=10, fontweight="bold")
    ax1.legend(fontsize=7.5, loc="lower left")
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.set_xlim(max(0.5, rho_vals.min() - 0.02), 1.01)

    # (b) Exclusion bar
    excl_s = excl_df.sort_values("rho_drop", ascending=True)
    y_pos  = np.arange(len(excl_s))
    clrs   = [DIM_COLORS.get(v, "#888") for v in excl_s["variable"]]
    vals   = excl_s["rho_drop"].clip(lower=0) * 100
    bars   = ax2.barh(y_pos, vals, color=clrs, alpha=0.82, edgecolor="white", height=0.65)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([VAR_LABELS.get(v, v) for v in excl_s["variable"]], fontsize=7)
    ax2.set_xlabel("Rank Correlation Drop (Δρ × 100)", fontsize=9)
    ax2.set_title("(b) Variable Exclusion Sensitivity\n"
                  "(higher = more critical)", fontsize=10, fontweight="bold")
    for bar, v in zip(bars, excl_s["rho_drop"].clip(lower=0)):
        ax2.text(bar.get_width() + 0.0003, bar.get_y() + bar.get_height()/2,
                 f"{v*100:.2f}", va="center", fontsize=7.5, fontweight="bold")
    ax2.spines[["top", "right"]].set_visible(False)
    patches = [mpatches.Patch(color="#2196F3", alpha=0.82, label="Climate"),
               mpatches.Patch(color="#4CAF50", alpha=0.82, label="Physical"),
               mpatches.Patch(color="#FF9800", alpha=0.82, label="Network")]
    ax2.legend(handles=patches, fontsize=7.5, loc="lower right")

    # (c) Weight vs delta-rho scatter
    for _, row in excl_df.iterrows():
        ax3.scatter(row["weight_orig"] * 100, max(0, row["rho_drop"]) * 100,
                    color=DIM_COLORS.get(row["variable"], "#888"),
                    s=70, alpha=0.85, edgecolors="white", linewidths=0.5, zorder=3)
        if row["rho_drop"] > 0.005:
            ax3.annotate(row["variable"],
                         xy=(row["weight_orig"] * 100, max(0, row["rho_drop"]) * 100),
                         xytext=(3, 2), textcoords="offset points",
                         fontsize=7.5, color="#333")
    ax3.set_xlabel("Original Weight (%)", fontsize=9)
    ax3.set_ylabel("Rank Correlation Drop (Δρ × 100)", fontsize=9)
    ax3.set_title("(c) Weight vs. Importance\n(each point = one variable)",
                  fontsize=10, fontweight="bold")
    ax3.spines[["top", "right"]].set_visible(False)
    ax3.grid(alpha=0.25)
    ax3.legend(handles=patches, fontsize=7.5)

    fig.suptitle(
        f"FACVI Sensitivity Analysis  |  Reference: {REF_SCENARIO}",
        fontsize=11, fontweight="bold",
    )
    out = OUTPUT_DIR / "fig_sensitivity.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    log.info(f"[FIG] {out.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("FACVI Step 12: Sensitivity Analysis")
    log.info("=" * 60)

    # Load grid
    candidates = [GRID_FOREST_FILE, GRID_CLIMATE_FILE]
    grid_path  = next((p for p in candidates if p.exists()), None)
    if grid_path is None:
        raise FileNotFoundError(
            "Climate/forest grid not found.  Run 09_climate_deltas.py first."
        )

    log.info(f"[LOAD] {grid_path.name}")
    grid = gpd.read_file(grid_path)
    log.info(f"  {len(grid):,} cells")

    if "ndvi_mean" in grid.columns and grid["ndvi_mean"].max() > 10:
        grid["ndvi_mean"] = grid["ndvi_mean"] / 1e8

    # Global climate ranges
    log.info("[CLIMATE RANGES] ...")
    sc_list = list(SCENARIOS.keys()) if isinstance(SCENARIOS, dict) else list(SCENARIOS)
    climate_ranges: dict[str, tuple[float, float]] = {}
    for code in DIMENSIONS["Climate"]:
        all_abs = []
        for sc in sc_list:
            col = f"{code}_{sc}"
            if col in grid.columns:
                arr = np.abs(grid[col].values.astype(float))
                all_abs.append(arr[np.isfinite(arr)])
        if all_abs:
            combined = np.concatenate(all_abs)
            climate_ranges[code] = (float(combined.min()), float(combined.max()))

    # Build X matrix
    log.info(f"[MATRIX] Building for {REF_SCENARIO} ...")
    X, var_order = build_X(grid, REF_SCENARIO, climate_ranges)
    log.info(f"  X shape: {X.shape}")

    # Load original weights
    if not WEIGHTS_FILE.exists():
        raise FileNotFoundError(
            f"Weights file not found: {WEIGHTS_FILE}\n"
            "Run 10_facvi.py first."
        )
    w_df = pd.read_csv(WEIGHTS_FILE)
    w_sc = w_df[w_df["scenario"] == REF_SCENARIO].set_index("variable")["weight"]
    w_orig = np.array([float(w_sc.get(v, 0.0)) for v in var_order])
    if w_orig.sum() > 0:
        w_orig /= w_orig.sum()
    log.info("  Original weights:")
    for v, w in zip(var_order, w_orig):
        log.info(f"    {v}: {w*100:.2f}%")

    # Monte Carlo
    log.info(f"\n[MONTE CARLO] {N_ITER} iterations, ±{int(PERTURB*100)}% perturbation ...")
    rho_vals = monte_carlo_sensitivity(X, w_orig)
    log.info(f"\n  Median ρ = {np.median(rho_vals):.4f}")
    log.info(f"  P5 ρ = {np.percentile(rho_vals, 5):.4f}")
    log.info(f"  ρ ≥ 0.95: {(rho_vals >= 0.95).mean()*100:.1f}%")

    mc_df = pd.DataFrame({"iteration": np.arange(1, N_ITER+1), "spearman_rho": rho_vals})
    mc_df.to_csv(OUTPUT_DIR / "sensitivity_mc.csv", index=False, encoding="utf-8-sig")
    log.info("[SAVED] sensitivity_mc.csv")

    # Exclusion
    log.info("\n[EXCLUSION] Leave-one-out analysis ...")
    excl_df = exclusion_sensitivity(X, w_orig, var_order)
    excl_df.to_csv(OUTPUT_DIR / "sensitivity_exclusion.csv",
                   index=False, encoding="utf-8-sig")
    log.info("[SAVED] sensitivity_exclusion.csv")

    # Figure
    log.info("\n[FIGURE] Generating ...")
    plot_sensitivity(rho_vals, excl_df, w_orig, var_order)

    log.info("Step 12 complete.")


if __name__ == "__main__":
    main()
