"""
FACVI Pipeline — Full Orchestrator
====================================
Runs all pipeline steps in sequence.

Usage:
  python run_pipeline.py [--steps 01-15] [options]

Full run:
  python run_pipeline.py

Partial run (e.g., re-run from step 09 onwards):
  python run_pipeline.py --from-step 09

Run specific steps only:
  python run_pipeline.py --only 04 05 06

Validation steps (require external data files):
  python run_pipeline.py --fire-data data/fire_inventory.gpkg
  python run_pipeline.py --landslide data/landslide_inventory.gpkg

Pass --help for the full option list.
"""

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "run_pipeline.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent

STEPS = {
    "01": ("01_download_data.py",    [],                       "Download WorldClim, SRTM, HydroRIVERS, NDVI, SoilGrids"),
    "02": ("02_forest_domain.py",    [],                       "ESA WorldCover forest mask and 1km grid clipping"),
    "03": ("03_auxiliary_data.py",   [],                       "RESOLVE Ecoregions 2017 download and clipping"),
    "04": ("04_osm_roads.py",        [],                       "OSM road download and surface scoring"),
    "05": ("05_road_grid.py",        [],                       "1km grid creation and road density calculation"),
    "06": ("06_physical_variables.py",[],                      "Slope, TWI, soil erodibility, NDVI sampling"),
    "07": ("07_stream_distance.py",  [],                       "HydroRIVERS stream distance (N3)"),
    "08": ("08_snow_cover.py",       [],                       "MODIS MOD10CM snow cover days (S5)"),
    "09": ("09_climate_deltas.py",   [],                       "WorldClim CMIP6 climate deltas (C1-C4)"),
    "10": ("10_facvi.py",            [],                       "Shannon entropy weighting and FACVI scores"),
    "11": ("11_spatial_analysis.py", [],                       "Global Moran's I, LISA, scenario comparison maps"),
    "12": ("12_sensitivity.py",      [],                       "Monte Carlo and leave-one-out sensitivity"),
    "13": ("13_fire_validation.py",  ["--fire-data"],          "FACVI validation against fire records"),
    "14": ("14_compound_hazard.py",  ["--landslide"],          "Compound hazard: high FACVI x landslide inventory"),
    "15": ("15_ml_comparison.py",    ["--fire-data"],          "Random Forest + SHAP vs. Shannon entropy weights"),
}

OPTIONAL_STEPS = {"13", "14", "15"}


def run_step(step_id: str, script: str, extra_args: list[str]) -> bool:
    path = SCRIPT_DIR / script
    if not path.exists():
        log.error(f"[{step_id}] Script not found: {path}")
        return False

    cmd = [sys.executable, str(path)] + extra_args
    log.info(f"\n{'='*60}")
    log.info(f"[STEP {step_id}] {script}")
    log.info(f"{'='*60}")
    log.info(f"  Command: {' '.join(cmd)}")

    t0 = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=str(SCRIPT_DIR.parent),
            capture_output=False,
        )
        elapsed = time.time() - t0
        if result.returncode == 0:
            log.info(f"  [OK] Step {step_id} completed in {elapsed:.1f}s")
            return True
        else:
            log.error(f"  [FAIL] Step {step_id} exited with code {result.returncode}")
            return False
    except Exception as e:
        log.error(f"  [ERROR] Step {step_id}: {e}")
        return False


def parse_args():
    p = argparse.ArgumentParser(
        description="FACVI full pipeline orchestrator"
    )
    p.add_argument("--from-step", metavar="N", type=str, default=None,
                   help="Run from step N onwards (e.g. --from-step 09)")
    p.add_argument("--only", nargs="+", metavar="N",
                   help="Run only these steps (e.g. --only 04 05 06)")
    p.add_argument("--skip", nargs="+", metavar="N", default=[],
                   help="Skip these steps")
    p.add_argument("--fire-data", type=Path, default=None,
                   help="Fire inventory GeoPackage (required for steps 13 and 15)")
    p.add_argument("--landslide", type=Path, default=None,
                   help="Landslide inventory GeoPackage (required for step 14)")
    p.add_argument("--stop-on-error", action="store_true",
                   help="Stop the pipeline on first failure (default: continue)")
    return p.parse_args()


def main():
    args = parse_args()

    all_step_ids = sorted(STEPS.keys())

    if args.only:
        selected = sorted(args.only)
    elif args.from_step:
        selected = [s for s in all_step_ids if s >= args.from_step.zfill(2)]
    else:
        selected = all_step_ids

    skip = {s.zfill(2) for s in (args.skip or [])}

    log.info("=" * 60)
    log.info("FACVI Pipeline")
    log.info(f"Selected steps: {selected}")
    log.info(f"Skipped steps:  {sorted(skip)}")
    log.info("=" * 60)

    passed, failed, skipped = [], [], []
    t_total = time.time()

    for step_id in selected:
        if step_id not in STEPS:
            log.warning(f"Unknown step: {step_id} — skipping.")
            skipped.append(step_id)
            continue

        if step_id in skip:
            log.info(f"[SKIP] Step {step_id}")
            skipped.append(step_id)
            continue

        script, extra_arg_keys, description = STEPS[step_id]

        # Build extra args for this step
        extra_args: list[str] = []
        skip_optional = False
        for key in extra_arg_keys:
            if key == "--fire-data":
                if args.fire_data is None:
                    if step_id in OPTIONAL_STEPS:
                        log.warning(f"[SKIP] Step {step_id}: --fire-data not provided.")
                        skip_optional = True
                    else:
                        extra_args += ["--fire-data", str(Path("data/fire_inventory.gpkg"))]
                else:
                    extra_args += ["--fire-data", str(args.fire_data)]
            elif key == "--landslide":
                if args.landslide is None:
                    if step_id in OPTIONAL_STEPS:
                        log.warning(f"[SKIP] Step {step_id}: --landslide not provided.")
                        skip_optional = True
                    else:
                        extra_args += ["--landslide", str(Path("data/landslide_inventory.gpkg"))]
                else:
                    extra_args += ["--landslide", str(args.landslide)]

        if skip_optional:
            skipped.append(step_id)
            continue

        ok = run_step(step_id, script, extra_args)
        if ok:
            passed.append(step_id)
        else:
            failed.append(step_id)
            if args.stop_on_error:
                log.error("Stopping pipeline due to failure (--stop-on-error).")
                break

    elapsed = time.time() - t_total
    log.info(f"\n{'='*60}")
    log.info("Pipeline Summary")
    log.info(f"{'='*60}")
    log.info(f"  Total time: {elapsed/60:.1f} min")
    log.info(f"  Passed:  {len(passed)}  {passed}")
    log.info(f"  Failed:  {len(failed)}  {failed}")
    log.info(f"  Skipped: {len(skipped)}  {skipped}")

    if failed:
        log.error("Some steps FAILED. Check the logs above.")
        sys.exit(1)
    else:
        log.info("All selected steps completed successfully.")


if __name__ == "__main__":
    main()
