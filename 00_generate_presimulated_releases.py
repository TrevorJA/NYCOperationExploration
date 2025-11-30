"""
00: Generate pre-simulated releases for trimmed model mode.

This script runs a single full model simulation with baseline NYC operations
and extracts reservoir releases for the independent STARFIT reservoirs.
These pre-simulated releases are then used by the trimmed model to significantly
reduce runtime during sensitivity analysis.

This is a ONE-TIME setup step that must be run before running the sensitivity
analysis with 02_run_simulations.py.

Usage:
    python 00_generate_presimulated_releases.py

The script will:
1. Build and run a full Pywr-DRB model with default NYC operations
2. Extract releases from 11 independent STARFIT reservoirs
3. Save releases to outputs/presim/presimulated_releases_mgd.csv
4. Save metadata for validation

Note:
    This step can take 10-30 minutes depending on the simulation period.
    It only needs to be run once per inflow_type/date range combination.
"""

import os
import sys
import time
from pathlib import Path

# Add methods to path
sys.path.insert(0, str(Path(__file__).parent))

import pywrdrb
from pywrdrb.post import generate_presimulated_releases

from config import (
    START_DATE,
    END_DATE,
    INFLOW_TYPE,
    PRESIM_DIR,
    PRESIM_FILE,
)


def main():
    """Generate pre-simulated releases for trimmed model mode."""

    print("=" * 70)
    print("GENERATE PRE-SIMULATED RELEASES FOR TRIMMED MODEL")
    print("=" * 70)

    print(f"\nSimulation Period: {START_DATE} to {END_DATE}")
    print(f"Inflow Type: {INFLOW_TYPE}")
    print(f"Output Directory: {PRESIM_DIR}")

    # Check if presim file already exists
    if PRESIM_FILE.exists():
        print(f"\nWARNING: Pre-simulated releases file already exists:")
        print(f"  {PRESIM_FILE}")
        response = input("Overwrite? (y/n): ").strip().lower()
        if response != 'y':
            print("Aborted.")
            return

    # Create output directory
    PRESIM_DIR.mkdir(parents=True, exist_ok=True)

    # Temporary files for full model run
    model_file = PRESIM_DIR / "full_model_baseline.json"
    output_file = PRESIM_DIR / "full_model_baseline_output.hdf5"

    # Step 1: Build full model with baseline NYC operations
    print("\n" + "-" * 70)
    print("Step 1: Building full model with baseline NYC operations...")
    print("-" * 70)

    t0 = time.perf_counter()

    mb = pywrdrb.ModelBuilder(
        inflow_type=INFLOW_TYPE,
        start_date=START_DATE,
        end_date=END_DATE,
        options={"nyc_nj_demand_source": "historical"}
    )
    mb.make_model()
    mb.write_model(str(model_file))

    build_time = time.perf_counter() - t0
    print(f"  Model built in {build_time:.1f} seconds")
    print(f"  Nodes: {len(mb.model_dict['nodes'])}")
    print(f"  Edges: {len(mb.model_dict['edges'])}")
    print(f"  Parameters: {len(mb.model_dict['parameters'])}")

    # Step 2: Run simulation
    print("\n" + "-" * 70)
    print("Step 2: Running full model simulation...")
    print("-" * 70)

    t0 = time.perf_counter()

    model = pywrdrb.Model.load(str(model_file))
    recorder = pywrdrb.OutputRecorder(
        model=model,
        output_filename=str(output_file)
    )
    model.run()

    run_time = time.perf_counter() - t0
    print(f"  Simulation completed in {run_time:.1f} seconds")

    # Step 3: Generate pre-simulated releases
    print("\n" + "-" * 70)
    print("Step 3: Extracting pre-simulated releases...")
    print("-" * 70)

    t0 = time.perf_counter()

    metadata = generate_presimulated_releases(
        output_filename=str(output_file),
        inflow_type=INFLOW_TYPE,
        output_dir=str(PRESIM_DIR),
    )

    extract_time = time.perf_counter() - t0
    print(f"  Extraction completed in {extract_time:.1f} seconds")

    # Step 4: Cleanup temporary files (optional - keep for debugging)
    print("\n" + "-" * 70)
    print("Step 4: Cleanup...")
    print("-" * 70)

    # Keep model and output files for reference/debugging
    print(f"  Keeping full model files for reference:")
    print(f"    Model: {model_file}")
    print(f"    Output: {output_file}")

    # Summary
    total_time = build_time + run_time + extract_time

    print("\n" + "=" * 70)
    print("PRE-SIMULATION COMPLETE")
    print("=" * 70)

    print(f"\nTotal time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"\nPre-simulated releases saved to:")
    print(f"  {metadata['output_file']}")
    print(f"\nMetadata saved to:")
    print(f"  {metadata['metadata_file']}")

    print(f"\nReservoirs included ({len(metadata['reservoirs'])}):")
    for res in metadata['reservoirs']:
        print(f"  - {res}")

    print(f"\nDate range: {metadata['start_date']} to {metadata['end_date']}")

    print("\n" + "=" * 70)
    print("You can now run sensitivity analysis with trimmed model:")
    print("  python 02_run_simulations.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
