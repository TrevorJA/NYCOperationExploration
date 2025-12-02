"""
05: Generate visualization figures for Sobol sensitivity analysis.

This script generates publication-quality figures including:
- Bar charts comparing S1 and ST indices
- Tornado plots showing parameter rankings
- Heatmaps comparing sensitivity across metrics
- Interaction matrices (if second-order indices available)

Usage:
    python 05_visualize_results.py
"""

import sys
from pathlib import Path

# Add methods to path
sys.path.insert(0, str(Path(__file__).parent))

from config import METRICS_TO_CALCULATE, FIGURES_DIR
from methods.analysis import load_sobol_results, print_sobol_summary
from methods.plotting import (
    plot_sobol_bars,
    plot_tornado,
    plot_multi_metric_heatmap,N
    plot_interaction_matrix,
    generate_all_figures
)

import matplotlib.pyplot as plt

# =============================================================================
# SCRIPT SETTINGS (modify these as needed)
# =============================================================================
# Set to a list of metric names to only plot specific metrics, or None for all
METRICS_TO_PLOT = None  # e.g., ["montague_min_flow_mgd", "nyc_min_storage_pct"]


def main():
    """Generate all visualization figures."""

    print("=" * 70)
    print("GENERATING SENSITIVITY ANALYSIS FIGURES")
    print("=" * 70)

    # Load results
    print("\nLoading Sobol results...")
    try:
        formatted, raw = load_sobol_results()
    except FileNotFoundError:
        print("ERROR: No Sobol results found.")
        print("Run 04_analyze_sensitivity.py first!")
        return

    # Determine which metrics have valid results
    valid_metrics = [m for m in raw.keys() if "error" not in raw.get(m, {"error": True})]
    print(f"  Found {len(valid_metrics)} metrics with valid results")

    # Filter to requested metrics if specified
    if METRICS_TO_PLOT is not None:
        valid_metrics = [m for m in METRICS_TO_PLOT if m in valid_metrics]

    print(f"\nGenerating figures for metrics:")
    for m in valid_metrics:
        print(f"  - {m}")

    # Generate all figures
    print("\n" + "-" * 50)
    generate_all_figures(raw, metrics=valid_metrics)

    # Close any remaining figures
    plt.close("all")

    print("\n" + "=" * 70)
    print("VISUALIZATION COMPLETE")
    print("=" * 70)
    print(f"\nFigures saved to: {FIGURES_DIR}")
    print("\nGenerated figures:")
    print("  - sobol_bars_*.png/pdf    : Bar charts for each metric")
    print("  - tornado_*.png/pdf       : Ranked parameter importance")
    print("  - heatmap_ST.png/pdf      : Multi-metric comparison (Total Order)")
    print("  - heatmap_S1.png/pdf      : Multi-metric comparison (First Order)")
    print("  - interactions_*.png/pdf  : Second-order interaction matrices")


if __name__ == "__main__":
    main()
