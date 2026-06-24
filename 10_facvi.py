"""
FACVI Pipeline — Step 10: Shannon Entropy Weighting and FACVI Score
====================================================================
Computes the Forest Access Climate Vulnerability Index (FACVI) for
all four CMIP6 scenarios using a hybrid Shannon entropy + dimension
prior weighting scheme.

Method:
  1. Min-max normalise all variables to [0, 1]; high = high vulnerability
     (inverted variables: S4=1-NDVI, N1/N2/N3 inverted)
  2. Climate variables use a GLOBAL range (same across all scenarios) so
     cross-scenario comparisons are meaningful
  3. Shannon entropy weights: E_j = -(1/ln n) × Σ p_ij × ln(p_ij)
  4. Hybrid weight: literature dimension priors × within-dim entropy
     Dimension priors: Climate 40%, Physical 35%, Network 25%
     Within-dim cap: 50% max per variable (OECD 2008)
  5. FACVI_i = Σ w_j × x*_ij  → risk classes (quintile-based)

Input : data/{CC}_grid_forest_climate.gpkg   (preferred)
        data/{CC}_grid_climate.gpkg           (fallback)
Outputs:
        data/{CC}_rvi_all_scenarios.gpkg
        data/entropy_weights.csv
"""

import numpy as np
import pandas as pd
import logging
import geopandas as gpd
from pathlib import Path

from config import (
    DATA_DIR, LOG_DIR,
    COUNTRY_CODE, SCENARIOS, DIMENSIONS, VARIABLE_MAP,
    DIM_WEIGHTS, DIM_CAP,
    GRID_FOREST_FILE, GRID_CLIMATE_FILE, GRID_RVI_FILE,
)

LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "10_facvi.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

OUTPUT_WEIGHTS = DATA_DIR / "entropy_weights.csv"

# Prefer the forest-clipped climate grid; fall back to full climate grid
_INPUT_FILE = (GRID_FOREST_FILE if GRID_FOREST_FILE.exists() else GRID_CLIMATE_FILE)


# ── Normalisation ─────────────────────────────────────────────────────────────

def _minmax(arr: np.ndarray, lo: float = None, hi: float = None,
            eps: float = 1e-8) -> np.ndarray:
    arr = arr.astype(float)
    if lo is None:
        lo = float(np.nanmin(arr))
    if hi is None:
        hi = float(np.nanmax(arr))
    if hi - lo < 1e-10:
        return np.full_like(arr, eps)
    return np.clip((arr - lo) / (hi - lo) + eps, eps, 1.0)


# ── Shannon entropy weights ───────────────────────────────────────────────────

def entropy_weights(X: np.ndarray,
                    var_names: list[str]) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Hybrid entropy weighting:
      dimension priors (DIM_WEIGHTS) × within-dimension Shannon entropy
    Within-dimension cap: DIM_CAP (50%) so no single variable dominates.
    """
    n, p = X.shape
    log.info(f"[ENTROPY] {n:,} cells × {p} variables")

    # variable → dimension lookup
    var_to_dim: dict[str, str] = {}
    for dim, codes in DIMENSIONS.items():
        for c in codes:
            var_to_dim[c] = dim

    # Column-proportional matrix P
    col_sums = X.sum(axis=0, keepdims=True)
    col_sums = np.where(col_sums == 0, 1e-10, col_sums)
    P        = X / col_sums
    log_P    = np.where(P > 0, np.log(P), 0.0)
    entropy  = np.clip(-(1.0 / np.log(n)) * np.sum(P * log_P, axis=0), 0.0, 1.0)
    diverge  = 1.0 - entropy

    weights  = np.zeros(p)
    rows = []

    for dim_name, dim_w in DIM_WEIGHTS.items():
        idx = [i for i, v in enumerate(var_names) if var_to_dim.get(v) == dim_name]
        if not idx:
            continue

        div = diverge[idx]
        total = div.sum()
        within = div / total if total > 0 else np.full(len(idx), 1.0 / len(idx))

        # Apply within-dimension cap
        if len(idx) > 1 and within.max() > DIM_CAP:
            capped  = np.minimum(within, DIM_CAP)
            excess  = within.sum() - capped.sum()
            uncapped = within < DIM_CAP
            if uncapped.sum() > 0 and within[uncapped].sum() > 0:
                capped[uncapped] += excess * (within[uncapped] / within[uncapped].sum())
            within = capped / capped.sum()

        for k, i in enumerate(idx):
            weights[i] = dim_w * within[k]

        log.info(f"  [{dim_name}] prior={dim_w*100:.0f}%  "
                 f"realized={weights[idx].sum()*100:.1f}%")
        for k, i in enumerate(idx):
            rows.append({
                "variable":   var_names[i],
                "dimension":  dim_name,
                "entropy":    float(entropy[i]),
                "divergence": float(diverge[i]),
                "within_w":   float(within[k]),
                "weight":     float(weights[i]),
                "weight_pct": float(weights[i] * 100),
            })
            log.info(f"    {var_names[i]}: within={within[k]*100:.1f}%  "
                     f"final={weights[i]*100:.2f}%")

    if weights.sum() > 0:
        weights = weights / weights.sum()

    df = pd.DataFrame(rows).sort_values("weight", ascending=False).reset_index(drop=True)
    log.info(f"  Total weight: {weights.sum():.6f}")
    return weights, df


# ── FACVI per scenario ────────────────────────────────────────────────────────

def compute_facvi(grid: gpd.GeoDataFrame,
                  scenario: str,
                  climate_global_ranges: dict) -> tuple[np.ndarray, np.ndarray,
                                                        list, pd.DataFrame]:
    n = len(grid)
    norm: dict[str, np.ndarray] = {}

    # ── Climate variables (scenario-specific, global normalization) ──
    for code in DIMENSIONS["Climate"]:
        col_meta = VARIABLE_MAP.get(code)
        col = f"{code}_{scenario}"
        if col not in grid.columns:
            log.warning(f"  [{col}] missing — zeroed.")
            norm[code] = np.full(n, 1e-8)
            continue
        raw = np.abs(grid[col].values.astype(float))
        lo, hi = climate_global_ranges.get(code, (0, 1))
        norm[code] = _minmax(raw, lo=lo, hi=hi)
        log.info(f"  [{code}] range=[{lo:.2f},{hi:.2f}]  "
                 f"mean_norm={norm[code].mean():.3f}")

    # ── Physical variables (static) ──
    for code in DIMENSIONS["Physical"]:
        col_meta = VARIABLE_MAP.get(code)
        if col_meta is None:
            continue
        col_name, invert, _ = col_meta
        if col_name not in grid.columns:
            log.warning(f"  [{code}] column '{col_name}' missing — zeroed.")
            norm[code] = np.full(n, 1e-8)
            continue
        raw = grid[col_name].values.astype(float)
        n_arr = _minmax(raw)
        norm[code] = (1.0 - n_arr + 1e-8) if invert else n_arr
        log.info(f"  [{code}] col={col_name}  invert={invert}  "
                 f"mean={norm[code].mean():.3f}")

    # ── Network variables (static, with optionals) ──
    for code in DIMENSIONS["Network"]:
        col_meta = VARIABLE_MAP.get(code)
        if col_meta is None:
            continue
        col_name, invert, _ = col_meta
        if col_name not in grid.columns:
            log.warning(f"  [{code}] column '{col_name}' missing — excluded.")
            norm[code] = None
            continue
        raw = grid[col_name].values.astype(float)
        n_arr = _minmax(raw)
        norm[code] = (1.0 - n_arr + 1e-8) if invert else n_arr
        log.info(f"  [{code}] col={col_name}  invert={invert}  "
                 f"mean={norm[code].mean():.3f}")

    # Build matrix — skip None columns
    all_codes = (DIMENSIONS["Climate"] + DIMENSIONS["Physical"] + DIMENSIONS["Network"])
    var_order = [c for c in all_codes if norm.get(c) is not None]
    log.info(f"\n  Active variables ({len(var_order)}): {var_order}")
    X = np.column_stack([norm[c] for c in var_order])

    # Fill any remaining NaN with column median
    n_nan = int(np.isnan(X).sum())
    if n_nan > 0:
        for j in range(X.shape[1]):
            bad = np.isnan(X[:, j])
            if bad.any():
                X[bad, j] = float(np.nanmedian(X[:, j]))
        log.info(f"  {n_nan:,} NaN → column median")

    # Entropy weights
    weights, df_w = entropy_weights(X, var_order)
    df_w["scenario"] = scenario

    # FACVI score
    facvi = np.clip(X @ weights, 0, 1)

    # Risk classes (quintile)
    cuts      = np.nanpercentile(facvi, [20, 40, 60, 80])
    risk_cls  = np.digitize(facvi, cuts) + 1
    cls_names = {1: "Very Low", 2: "Low", 3: "Moderate",
                 4: "High", 5: "Very High"}

    log.info(f"\n  FACVI stats:")
    log.info(f"    mean={facvi.mean():.4f}  std={facvi.std():.4f}  "
             f"min={facvi.min():.4f}  max={facvi.max():.4f}")
    log.info("  Risk class distribution:")
    for cls in range(1, 6):
        cnt = (risk_cls == cls).sum()
        log.info(f"    {cls_names[cls]:12s}: {cnt:,} ({cnt/n*100:.1f}%)")

    return facvi, risk_cls, var_order, df_w


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if GRID_RVI_FILE.exists():
        gdf = gpd.read_file(GRID_RVI_FILE)
        log.info(f"[CACHED] FACVI: {len(gdf):,} cells in {GRID_RVI_FILE.name}")
        return

    log.info("=" * 60)
    log.info("FACVI Step 10: Shannon Entropy + FACVI Score")
    log.info("=" * 60)

    # Reload _INPUT_FILE at runtime in case it was created after import
    candidates = [GRID_FOREST_FILE, GRID_CLIMATE_FILE]
    input_file = next((p for p in candidates if p.exists()), None)
    if input_file is None:
        raise FileNotFoundError(
            f"Climate grid not found.\n"
            f"Run 09_climate_deltas.py first.\n"
            f"Expected: {GRID_CLIMATE_FILE}"
        )

    log.info(f"[LOAD] {input_file.name}")
    grid = gpd.read_file(input_file)
    log.info(f"  {len(grid):,} cells  columns={list(grid.columns[:8])} ...")

    # NDVI scale correction (some pipelines produce raw scaled values)
    if "ndvi_mean" in grid.columns and grid["ndvi_mean"].max() > 10:
        grid["ndvi_mean"] = grid["ndvi_mean"] / 1e8
        log.info("  [NDVI] Scaled from raw (×1e8) to [0,1]")

    # Global climate normalization ranges (shared across all scenarios)
    log.info("[GLOBAL RANGE] Computing cross-scenario climate ranges ...")
    climate_global_ranges: dict[str, tuple[float, float]] = {}
    for code in DIMENSIONS["Climate"]:
        all_abs = []
        for sc in SCENARIOS:
            col = f"{code}_{sc}"
            if col in grid.columns:
                arr = np.abs(grid[col].values.astype(float))
                all_abs.append(arr[np.isfinite(arr)])
        if all_abs:
            combined = np.concatenate(all_abs)
            climate_global_ranges[code] = (float(combined.min()),
                                           float(combined.max()))
            lo, hi = climate_global_ranges[code]
            log.info(f"  {code}: [{lo:.4f}, {hi:.4f}]")

    all_weights = []

    for scenario_key in SCENARIOS:
        log.info(f"\n{'='*60}")
        log.info(f"[SCENARIO] {scenario_key}")
        log.info(f"{'='*60}")
        facvi, rvi_cls, var_order, df_w = compute_facvi(
            grid, scenario_key, climate_global_ranges
        )
        grid[f"FACVI_{scenario_key}"]       = facvi
        grid[f"FACVI_class_{scenario_key}"] = rvi_cls
        all_weights.append(df_w)

    # Scenario comparison summary
    log.info("\n── Scenario Comparison (mean FACVI) ──")
    for sc in SCENARIOS:
        col = f"FACVI_{sc}"
        log.info(f"  {sc:20s}: {grid[col].mean():.4f} ± {grid[col].std():.4f}")

    # Save weights
    weights_all = pd.concat(all_weights, ignore_index=True)
    weights_all.to_csv(OUTPUT_WEIGHTS, index=False, encoding="utf-8-sig")
    log.info(f"[SAVED] {OUTPUT_WEIGHTS.name}")

    # Save FACVI grid
    grid.to_file(GRID_RVI_FILE, driver="GPKG")
    log.info(f"[SAVED] {GRID_RVI_FILE.name}  ({len(grid):,} cells)")
    log.info("Step 10 complete.")


if __name__ == "__main__":
    main()
