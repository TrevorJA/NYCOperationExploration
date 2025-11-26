"""
Visualization functions for Sobol sensitivity analysis.

This module provides:
- Bar charts of first and total order indices
- Tornado plots for parameter ranking
- Multi-metric comparison heatmaps
- Interaction matrix plots
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import FIGURES_DIR, PARAMETER_GROUPS

# Default plot settings
DPI_HIGH = 300
FIGSIZE_DEFAULT = (10, 6)


def get_parameter_colors():
    """Get color mapping for parameter groups."""
    colors = {
        "delivery": "#1f77b4",  # blue
        "mrf": "#2ca02c",       # green
        "flood": "#ff7f0e",     # orange
        "storage_zones": "#d62728",  # red
    }

    param_colors = {}
    for group_name, group in PARAMETER_GROUPS.items():
        for param_name in group["parameters"]:
            param_colors[param_name] = colors.get(group_name, "#7f7f7f")

    return param_colors


def plot_sobol_bars(sobol_results: dict, metric: str,
                    figsize=FIGSIZE_DEFAULT, save=True, filename=None):
    """
    Create bar chart comparing S1 and ST indices for all parameters.

    Parameters
    ----------
    sobol_results : dict
        Output from calculate_sobol_indices
    metric : str
        Metric name to plot
    figsize : tuple
        Figure size
    save : bool
        If True, save figure to file
    filename : str, optional
        Custom filename (without extension)

    Returns
    -------
    tuple : (fig, ax) or None if metric not found
    """
    if metric not in sobol_results or "error" in sobol_results[metric]:
        print(f"Cannot plot {metric}: results not available")
        return None

    indices = sobol_results[metric]
    params = indices["parameter_names"]
    n_params = len(params)

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(n_params)
    width = 0.35

    # First order indices
    bars1 = ax.bar(x - width/2, indices["S1"], width,
                   yerr=indices["S1_conf"], label="First Order (S1)",
                   color="steelblue", alpha=0.8, capsize=3)

    # Total order indices
    bars2 = ax.bar(x + width/2, indices["ST"], width,
                   yerr=indices["ST_conf"], label="Total Order (ST)",
                   color="darkorange", alpha=0.8, capsize=3)

    ax.set_xlabel("Parameter", fontsize=11)
    ax.set_ylabel("Sensitivity Index", fontsize=11)
    ax.set_title(f"Sobol Sensitivity Indices: {metric}", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(params, rotation=45, ha="right", fontsize=9)
    ax.legend(loc="upper right")
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if save:
        if filename is None:
            filename = f"sobol_bars_{metric}"
        plt.savefig(FIGURES_DIR / f"{filename}.png", dpi=DPI_HIGH, bbox_inches="tight")
        plt.savefig(FIGURES_DIR / f"{filename}.pdf", bbox_inches="tight")
        print(f"Saved: {FIGURES_DIR / filename}.png")

    return fig, ax


def plot_tornado(sobol_results: dict, metric: str,
                 index_type: str = "ST", top_n: int = 10,
                 figsize=(8, 6), save=True, filename=None):
    """
    Create tornado plot showing top N most influential parameters.

    Parameters
    ----------
    sobol_results : dict
        Output from calculate_sobol_indices
    metric : str
        Metric name to plot
    index_type : str
        "S1" or "ST" (default: "ST")
    top_n : int
        Number of top parameters to show
    figsize : tuple
        Figure size
    save : bool
        If True, save figure
    filename : str, optional
        Custom filename

    Returns
    -------
    tuple : (fig, ax)
    """
    from methods.analysis import rank_parameters

    ranked = rank_parameters(sobol_results, metric, index_type)
    ranked = ranked.head(top_n)

    # Get colors by parameter group
    param_colors = get_parameter_colors()
    colors = [param_colors.get(p, "#7f7f7f") for p in ranked["parameter"]]

    fig, ax = plt.subplots(figsize=figsize)

    y_pos = np.arange(len(ranked))

    ax.barh(y_pos, ranked["index"],
            xerr=ranked["conf"],
            align="center", color=colors, alpha=0.8, capsize=3)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(ranked["parameter"])
    ax.invert_yaxis()  # Top parameter at top
    ax.set_xlabel(f"{index_type} Index", fontsize=11)
    ax.set_title(f"Parameter Importance: {metric}", fontsize=12, fontweight="bold")
    ax.axvline(x=0, color="gray", linestyle="-", linewidth=0.5)
    ax.grid(axis="x", alpha=0.3)

    # Add legend for parameter groups
    from matplotlib.patches import Patch
    legend_elements = []
    for group_name, group in PARAMETER_GROUPS.items():
        if group["enabled"]:
            color = get_parameter_colors().get(list(group["parameters"].keys())[0], "#7f7f7f")
            legend_elements.append(Patch(facecolor=color, alpha=0.8, label=group_name))
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    plt.tight_layout()

    if save:
        if filename is None:
            filename = f"tornado_{metric}_{index_type}"
        plt.savefig(FIGURES_DIR / f"{filename}.png", dpi=DPI_HIGH, bbox_inches="tight")
        plt.savefig(FIGURES_DIR / f"{filename}.pdf", bbox_inches="tight")
        print(f"Saved: {FIGURES_DIR / filename}.png")

    return fig, ax


def plot_multi_metric_heatmap(sobol_results: dict,
                              metrics: list = None,
                              index_type: str = "ST",
                              figsize=(12, 8), save=True):
    """
    Create heatmap comparing parameter importance across multiple metrics.

    Parameters
    ----------
    sobol_results : dict
        Output from calculate_sobol_indices
    metrics : list, optional
        List of metrics to include
    index_type : str
        "S1" or "ST"
    figsize : tuple
        Figure size
    save : bool
        If True, save figure

    Returns
    -------
    tuple : (fig, ax)
    """
    from methods.analysis import format_sobol_results

    formatted = format_sobol_results(sobol_results)

    if metrics is not None:
        formatted = formatted[formatted["metric"].isin(metrics)]

    # Pivot to get parameter x metric matrix
    pivot = formatted.pivot(
        index="parameter",
        columns="metric",
        values=index_type
    )

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto", vmin=0)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)

    ax.set_xlabel("Metric", fontsize=11)
    ax.set_ylabel("Parameter", fontsize=11)
    ax.set_title(f"{index_type} Indices Across Metrics", fontsize=12, fontweight="bold")

    plt.colorbar(im, ax=ax, label=f"{index_type} Index")

    # Add value annotations
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                color = "white" if val > 0.3 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        color=color, fontsize=8)

    plt.tight_layout()

    if save:
        plt.savefig(FIGURES_DIR / f"heatmap_{index_type}.png",
                    dpi=DPI_HIGH, bbox_inches="tight")
        plt.savefig(FIGURES_DIR / f"heatmap_{index_type}.pdf",
                    bbox_inches="tight")
        print(f"Saved: {FIGURES_DIR / f'heatmap_{index_type}.png'}")

    return fig, ax


def plot_interaction_matrix(sobol_results: dict, metric: str,
                            figsize=(10, 8), save=True, filename=None):
    """
    Create heatmap of second-order interaction indices.

    Parameters
    ----------
    sobol_results : dict
        Output from calculate_sobol_indices (with calc_second_order=True)
    metric : str
        Metric name
    figsize : tuple
        Figure size
    save : bool
        If True, save figure
    filename : str, optional
        Custom filename

    Returns
    -------
    tuple : (fig, ax)
    """
    from methods.analysis import get_interaction_matrix

    try:
        S2_matrix = get_interaction_matrix(sobol_results, metric)
    except ValueError as e:
        print(f"Cannot plot interaction matrix: {e}")
        return None

    fig, ax = plt.subplots(figsize=figsize)

    # Mask diagonal (self-interactions are NaN)
    masked_data = S2_matrix.values.copy()
    np.fill_diagonal(masked_data, np.nan)

    im = ax.imshow(masked_data, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(range(len(S2_matrix.columns)))
    ax.set_xticklabels(S2_matrix.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(S2_matrix.index)))
    ax.set_yticklabels(S2_matrix.index, fontsize=9)

    ax.set_title(f"Parameter Interactions (S2): {metric}", fontsize=12, fontweight="bold")

    plt.colorbar(im, ax=ax, label="S2 Index")

    # Add value annotations for significant interactions
    for i in range(len(S2_matrix.index)):
        for j in range(len(S2_matrix.columns)):
            if i != j:
                val = S2_matrix.values[i, j]
                if not np.isnan(val) and abs(val) > 0.01:
                    color = "white" if val > 0.1 else "black"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            color=color, fontsize=7)

    plt.tight_layout()

    if save:
        if filename is None:
            filename = f"interactions_{metric}"
        plt.savefig(FIGURES_DIR / f"{filename}.png", dpi=DPI_HIGH, bbox_inches="tight")
        plt.savefig(FIGURES_DIR / f"{filename}.pdf", bbox_inches="tight")
        print(f"Saved: {FIGURES_DIR / filename}.png")

    return fig, ax


def plot_convergence(sobol_results_list: list, metric: str,
                     n_samples_list: list, index_type: str = "ST",
                     figsize=FIGSIZE_DEFAULT, save=True):
    """
    Plot convergence of sensitivity indices with increasing sample size.

    Parameters
    ----------
    sobol_results_list : list
        List of sobol_results dicts for increasing sample sizes
    metric : str
        Metric name
    n_samples_list : list
        List of sample sizes corresponding to each results dict
    index_type : str
        "S1" or "ST"
    figsize : tuple
        Figure size
    save : bool
        If True, save figure

    Returns
    -------
    tuple : (fig, ax)
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Get parameter names from first result
    param_names = sobol_results_list[0][metric]["parameter_names"]
    param_colors = get_parameter_colors()

    for i, param in enumerate(param_names):
        values = []
        conf_lower = []
        conf_upper = []

        for results in sobol_results_list:
            if metric in results and "error" not in results[metric]:
                val = results[metric][index_type][i]
                conf = results[metric][f"{index_type}_conf"][i]
                values.append(val)
                conf_lower.append(val - conf)
                conf_upper.append(val + conf)

        if values:
            color = param_colors.get(param, "#7f7f7f")
            ax.plot(n_samples_list[:len(values)], values, 'o-',
                    label=param, color=color, alpha=0.7)
            ax.fill_between(n_samples_list[:len(values)],
                            conf_lower, conf_upper,
                            color=color, alpha=0.2)

    ax.set_xlabel("Number of Samples (N)", fontsize=11)
    ax.set_ylabel(f"{index_type} Index", fontsize=11)
    ax.set_title(f"Convergence of Sensitivity Indices: {metric}",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save:
        plt.savefig(FIGURES_DIR / f"convergence_{metric}_{index_type}.png",
                    dpi=DPI_HIGH, bbox_inches="tight")
        print(f"Saved: {FIGURES_DIR / f'convergence_{metric}_{index_type}.png'}")

    return fig, ax


def generate_all_figures(sobol_results: dict, metrics: list = None):
    """
    Generate all standard figures for sensitivity analysis.

    Parameters
    ----------
    sobol_results : dict
        Output from calculate_sobol_indices
    metrics : list, optional
        List of metrics to plot
    """
    if metrics is None:
        metrics = [m for m in sobol_results.keys() if "error" not in sobol_results.get(m, {})]

    print(f"Generating figures for {len(metrics)} metrics...")

    # Bar charts for each metric
    print("\nGenerating bar charts...")
    for metric in metrics:
        plot_sobol_bars(sobol_results, metric, save=True)
        plt.close()

    # Tornado plots for each metric
    print("\nGenerating tornado plots...")
    for metric in metrics:
        plot_tornado(sobol_results, metric, index_type="ST", save=True)
        plt.close()

    # Multi-metric heatmap
    print("\nGenerating heatmaps...")
    plot_multi_metric_heatmap(sobol_results, metrics=metrics, index_type="ST", save=True)
    plt.close()

    plot_multi_metric_heatmap(sobol_results, metrics=metrics, index_type="S1", save=True)
    plt.close()

    # Interaction matrices (if available)
    print("\nGenerating interaction matrices...")
    for metric in metrics:
        if "S2" in sobol_results.get(metric, {}):
            plot_interaction_matrix(sobol_results, metric, save=True)
            plt.close()

    print(f"\nAll figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    # Test plotting with existing results
    from methods.analysis import load_sobol_results

    try:
        formatted, raw = load_sobol_results()
        print("Generating figures from existing results...")
        generate_all_figures(raw)
    except FileNotFoundError:
        print("No Sobol results found. Run 04_analyze_sensitivity.py first.")
