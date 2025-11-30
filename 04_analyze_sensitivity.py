"""
04: Compute Sobol sensitivity indices using SALib.

This script calculates first-order (S1), total-order (ST), and second-order (S2)
Sobol indices for each metric using the SALib library.

Usage:
    python 04_analyze_sensitivity.py
"""

import sys
from pathlib import Path

# Add methods to path
sys.path.insert(0, str(Path(__file__).parent))

from config import METRICS_TO_CALCULATE, ANALYSIS_DIR
from methods.sampling import load_samples
from methods.metrics import load_metrics
from methods.analysis import (
    calculate_sobol_indices,
    save_sobol_results,
    print_sobol_summary
)

# =============================================================================
# SCRIPT SETTINGS (modify these as needed)
# =============================================================================
# Set to True to analyze all metrics (not just those in METRICS_TO_CALCULATE)
ANALYZE_ALL_METRICS = False

# Set to False to skip second-order interaction indices (faster)
CALC_SECOND_ORDER = True


def main():
    """Compute Sobol indices for all metrics."""

    print("=" * 70)
    print("SOBOL SENSITIVITY ANALYSIS")
    print("=" * 70)

    # Load samples and metrics
    print("\nLoading data...")
    samples, problem = load_samples("sobol")
    metrics_df = load_metrics()

    n_samples = len(samples)
    n_metrics_loaded = len(metrics_df)
    n_params = problem["num_vars"]

    print(f"  Samples: {n_samples}")
    print(f"  Metrics records: {n_metrics_loaded}")
    print(f"  Parameters: {n_params}")

    # Validate sample count matches metrics count
    if n_samples != n_metrics_loaded:
        print(f"\n  WARNING: Sample count ({n_samples}) != metrics count ({n_metrics_loaded})")
        print("  Some simulations may have failed. Proceeding with available data...")

    # Select metrics
    if ANALYZE_ALL_METRICS:
        metric_columns = [c for c in metrics_df.columns if c != "sample_id"]
    else:
        metric_columns = [m for m in METRICS_TO_CALCULATE if m in metrics_df.columns]

    print(f"\nAnalyzing {len(metric_columns)} metrics:")
    for m in metric_columns:
        print(f"  - {m}")

    # Calculate Sobol indices
    print("\nCalculating Sobol indices...")
    print(f"  Second-order interactions: {CALC_SECOND_ORDER}")

    sobol_results = calculate_sobol_indices(
        samples,
        metrics_df,
        problem,
        metric_columns=metric_columns,
        calc_second_order=CALC_SECOND_ORDER,
        n_bootstrap=1000,
        confidence_level=0.95
    )

    # Save results
    print("\nSaving results...")
    save_sobol_results(sobol_results)

    # Print summary
    print_sobol_summary(sobol_results, top_n=5)

    print("\n" + "=" * 70)
    print("SENSITIVITY ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to: {ANALYSIS_DIR}")
    print(f"Next step: Generate visualizations with 05_visualize_results.py")


if __name__ == "__main__":
    main()
