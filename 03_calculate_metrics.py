"""
03: Calculate performance metrics from simulation outputs.

This script extracts performance metrics from each simulation output file
and saves them for sensitivity analysis.

Usage:
    python 03_calculate_metrics.py

Example:
    python 03_calculate_metrics.py
"""

import argparse
import sys
from pathlib import Path

# Add methods to path
sys.path.insert(0, str(Path(__file__).parent))

from config import METRICS_TO_CALCULATE, SIMULATIONS_DIR, METRICS_DIR
from methods.metrics import (
    calculate_all_metrics,
    save_metrics,
    list_available_metrics
)
from methods.simulation import load_simulation_results


def main(use_results_file: bool = True):
    """Calculate metrics for all completed simulations."""

    print("=" * 70)
    print("PERFORMANCE METRICS CALCULATION")
    print("=" * 70)

    # List available metrics
    print("\nConfigured metrics:")
    for m in METRICS_TO_CALCULATE:
        print(f"  - {m}")

    # Load simulation results if available
    if use_results_file:
        try:
            simulation_results = load_simulation_results()
            n_total = len(simulation_results)
            n_success = (simulation_results["status"] == "success").sum()
            print(f"\nLoaded simulation results:")
            print(f"  Total samples: {n_total}")
            print(f"  Successful: {n_success}")
        except FileNotFoundError:
            print("\nNo simulation results file found, scanning directory...")
            simulation_results = None
    else:
        simulation_results = None

    # Calculate metrics
    print("\nCalculating metrics...")
    metrics_df = calculate_all_metrics(simulation_results, metrics=METRICS_TO_CALCULATE)

    # Save metrics
    save_metrics(metrics_df)

    # Print summary statistics
    print("\n" + "=" * 70)
    print("METRICS SUMMARY")
    print("=" * 70)

    for col in metrics_df.columns:
        if col != "sample_id":
            data = metrics_df[col].dropna()
            print(f"\n{col}:")
            print(f"  Count: {len(data)}")
            print(f"  Mean:  {data.mean():.4f}")
            print(f"  Std:   {data.std():.4f}")
            print(f"  Min:   {data.min():.4f}")
            print(f"  Max:   {data.max():.4f}")

    print("\n" + "=" * 70)
    print("METRICS CALCULATION COMPLETE")
    print("=" * 70)
    print(f"\nMetrics saved to: {METRICS_DIR}")
    print(f"Next step: Run sensitivity analysis with 04_analyze_sensitivity.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate metrics from simulation outputs")
    parser.add_argument("--scan-dir", action="store_true",
                        help="Scan directory instead of using results file")

    args = parser.parse_args()
    main(use_results_file=not args.scan_dir)
