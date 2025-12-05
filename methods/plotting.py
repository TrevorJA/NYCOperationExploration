"""
Visualization functions for Sobol sensitivity analysis and Pareto analysis.

This module provides:
- Bar charts of first and total order indices
- Tornado plots for parameter ranking
- Multi-metric comparison heatmaps
- Interaction matrix plots
- Custom parallel coordinates plot with actual value annotations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colormaps, cm
from matplotlib.lines import Line2D
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import FIGURES_DIR, PARAMETER_GROUPS

# Default plot settings
DPI_HIGH = 300
FIGSIZE_DEFAULT = (10, 6)

# =============================================================================
# RADIAL SENSITIVITY PLOT SETTINGS
# =============================================================================

# Radial plot scaling constants
S1_MIN_SIZE = 50      # min marker area (points^2)
S1_MAX_SIZE = 800     # max marker area
RING_MIN_SIZE = 80    # min ring size
RING_MAX_SIZE = 1200  # max ring size
S2_MIN_WIDTH = 0.5    # min line width
S2_MAX_WIDTH = 6.0    # max line width

# Radial plot colors (matching reference figure)
RADIAL_MARKER_COLOR = '#800000'  # maroon
RADIAL_RING_COLOR = '#800000'    # maroon
RADIAL_LINE_COLOR = '#000080'    # navy blue

# Parameter name shorthand dictionary for radial plots
PARAM_SHORT_NAMES = {
    # Individual MRF
    'mrf_cannonsville': 'MRF\nCan',
    'mrf_pepacton': 'MRF\nPep',
    'mrf_neversink': 'MRF\nNev',
    # MRF Factor Profiles - shifts
    'mrf_summer_start_shift': 'Summer\nStart',
    'mrf_fall_start_shift': 'Fall\nStart',
    'mrf_winter_start_shift': 'Winter\nStart',
    'mrf_spring_start_shift': 'Spring\nStart',
    # MRF Factor Profiles - scales
    'mrf_summer_scale': 'Summer\nScale',
    'mrf_fall_scale': 'Fall\nScale',
    'mrf_winter_scale': 'Winter\nScale',
    'mrf_spring_scale': 'Spring\nScale',
    # Storage Zones - vertical shifts
    'zone_level1b_vertical_shift': 'L1b\nv-shift',
    'zone_level1c_vertical_shift': 'L1c\nv-shift',
    'zone_level2_vertical_shift': 'L2\nv-shift',
    'zone_level3_vertical_shift': 'L3\nv-shift',
    'zone_level4_vertical_shift': 'L4\nv-shift',
    'zone_level5_vertical_shift': 'L5\nv-shift',
    # Storage Zones - time shifts
    'zone_level1b_time_shift': 'L1b\nt-shift',
    'zone_level1c_time_shift': 'L1c\nt-shift',
    'zone_level2_time_shift': 'L2\nt-shift',
    'zone_level3_time_shift': 'L3\nt-shift',
    'zone_level4_time_shift': 'L4\nt-shift',
    'zone_level5_time_shift': 'L5\nt-shift',
}


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


# =============================================================================
# RADIAL SENSITIVITY PLOT HELPER FUNCTIONS
# =============================================================================

def get_short_param_name(name: str) -> str:
    """
    Get shortened parameter name for radial plot labels.

    Parameters
    ----------
    name : str
        Full parameter name

    Returns
    -------
    str
        Shortened name from PARAM_SHORT_NAMES or original if not found
    """
    return PARAM_SHORT_NAMES.get(name, name)


def _get_grouped_param_order(param_names: list) -> tuple:
    """
    Get parameter indices ordered by group for radial layout.

    Parameters
    ----------
    param_names : list
        List of parameter names

    Returns
    -------
    tuple : (ordered_indices, group_boundaries)
        - ordered_indices: list of indices into param_names in group order
        - group_boundaries: list of (start_idx, end_idx, group_name) tuples
    """
    group_order = ["individual_mrf", "mrf_factor_profiles", "storage_zones"]
    ordered_indices = []
    group_boundaries = []

    for group_name in group_order:
        if group_name in PARAMETER_GROUPS and PARAMETER_GROUPS[group_name]["enabled"]:
            group_params = list(PARAMETER_GROUPS[group_name]["parameters"].keys())
            start_idx = len(ordered_indices)
            for p in group_params:
                if p in param_names:
                    ordered_indices.append(param_names.index(p))
            if len(ordered_indices) > start_idx:
                group_boundaries.append((start_idx, len(ordered_indices), group_name))

    # Add any remaining parameters not in groups
    for i, p in enumerate(param_names):
        if i not in ordered_indices:
            ordered_indices.append(i)

    return ordered_indices, group_boundaries


def _calculate_radial_positions(n_params: int, radius: float = 1.0,
                                 start_angle: float = 90.0,
                                 group_boundaries: list = None,
                                 gap_angle: float = 15.0) -> tuple:
    """
    Calculate (x, y) positions for parameters on a circle.

    Parameters arranged counter-clockwise starting from top (90 degrees).
    Adds gaps between parameter groups if boundaries provided.

    Parameters
    ----------
    n_params : int
        Number of parameters
    radius : float
        Radius of the circle
    start_angle : float
        Starting angle in degrees (90 = top)
    group_boundaries : list, optional
        List of (start_idx, end_idx, group_name) tuples
    gap_angle : float
        Angle in degrees to add between groups

    Returns
    -------
    tuple : (x, y, angles)
        - x: array of x positions
        - y: array of y positions
        - angles: array of angles in degrees
    """
    if group_boundaries and len(group_boundaries) > 1:
        n_gaps = len(group_boundaries)
        total_gap = gap_angle * n_gaps
        available_arc = 360 - total_gap
        angle_per_param = available_arc / n_params
    else:
        angle_per_param = 360 / n_params
        group_boundaries = []

    angles = []
    current_angle = start_angle

    # Track which indices start a new group
    group_starts = {gb[0] for gb in group_boundaries}

    for i in range(n_params):
        # Add gap before group start (except first)
        if i in group_starts and i > 0:
            current_angle -= gap_angle

        angles.append(current_angle)
        current_angle -= angle_per_param

    angles = np.array(angles)
    angles_rad = np.deg2rad(angles)
    x = radius * np.cos(angles_rad)
    y = radius * np.sin(angles_rad)

    return x, y, angles


def _add_radial_legend(fig, ax, s1_min: float, s1_max: float,
                       st_min: float, st_max: float,
                       s2_min: float, s2_max: float):
    """
    Add legend panel below the radial plot showing scale for S1, ST, and S2.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure object
    ax : matplotlib.axes.Axes
        Main plot axes
    s1_min, s1_max : float
        Range of S1 values in data
    st_min, st_max : float
        Range of ST values in data
    s2_min, s2_max : float
        Range of S2 values in data (absolute)
    """
    from matplotlib.patches import Circle
    from matplotlib.lines import Line2D

    # Create legend axes at bottom of figure
    legend_ax = fig.add_axes([0.1, 0.02, 0.8, 0.15])
    legend_ax.set_xlim(0, 10)
    legend_ax.set_ylim(0, 3)
    legend_ax.axis('off')

    # === First-Order Sensitivities (S1) - filled circles ===
    legend_ax.text(1.5, 2.7, "First-Order Sensitivities", ha='center',
                   fontsize=10, fontweight='bold')

    # Show min and max values with sizes
    s1_legend_vals = [s1_min, (s1_min + s1_max) / 2, s1_max]
    s1_legend_vals = [v for v in s1_legend_vals if v > 0.001]  # Filter near-zero
    if not s1_legend_vals:
        s1_legend_vals = [0.01, 0.1]

    for i, val in enumerate(s1_legend_vals[:3]):
        norm_val = (val - s1_min) / (s1_max - s1_min) if s1_max > s1_min else 0.5
        size = S1_MIN_SIZE + norm_val * (S1_MAX_SIZE - S1_MIN_SIZE)
        x_pos = 0.5 + i * 1.0
        legend_ax.scatter(x_pos, 1.5, s=size, c=RADIAL_MARKER_COLOR, alpha=0.85)
        legend_ax.text(x_pos, 0.6, f"{val*100:.0f}%", ha='center', fontsize=8)

    # === Total-Order Sensitivities (ST) - rings ===
    legend_ax.text(5, 2.7, "Total-Order Sensitivities", ha='center',
                   fontsize=10, fontweight='bold')

    st_legend_vals = [st_min, (st_min + st_max) / 2, st_max]
    st_legend_vals = [v for v in st_legend_vals if v > 0.001]
    if not st_legend_vals:
        st_legend_vals = [0.01, 0.1]

    for i, val in enumerate(st_legend_vals[:3]):
        norm_val = (val - st_min) / (st_max - st_min) if st_max > st_min else 0.5
        size = RING_MIN_SIZE + norm_val * (RING_MAX_SIZE - RING_MIN_SIZE)
        x_pos = 4 + i * 1.0
        legend_ax.scatter(x_pos, 1.5, s=size, facecolors='none',
                         edgecolors=RADIAL_RING_COLOR, linewidths=2)
        legend_ax.text(x_pos, 0.6, f"{val*100:.0f}%", ha='center', fontsize=8)

    # === Second-Order Interactions (S2) - lines ===
    legend_ax.text(8.5, 2.7, "Second-Order Sensitivities", ha='center',
                   fontsize=10, fontweight='bold')

    s2_legend_vals = [s2_min, (s2_min + s2_max) / 2, s2_max]
    s2_legend_vals = [v for v in s2_legend_vals if v > 0.001]
    if not s2_legend_vals:
        s2_legend_vals = [0.01, 0.1]

    for i, val in enumerate(s2_legend_vals[:3]):
        norm_val = (val - s2_min) / (s2_max - s2_min) if s2_max > s2_min else 0.5
        width = S2_MIN_WIDTH + norm_val * (S2_MAX_WIDTH - S2_MIN_WIDTH)
        x_pos = 7.5 + i * 1.0
        legend_ax.plot([x_pos - 0.3, x_pos + 0.3], [1.5, 1.5],
                      color=RADIAL_LINE_COLOR, linewidth=width, alpha=0.7)
        legend_ax.text(x_pos, 0.6, f"{val*100:.0f}%", ha='center', fontsize=8)


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
        # plt.savefig(FIGURES_DIR / f"{filename}.pdf", bbox_inches="tight")
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
        # plt.savefig(FIGURES_DIR / f"{filename}.pdf", bbox_inches="tight")
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

    # Use divergent colormap centered on zero for S2 interactions
    vmax = np.nanmax(np.abs(masked_data))
    im = ax.imshow(masked_data, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)

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
                    # Use white text for extreme values (dark cells in divergent colormap)
                    color = "white" if abs(val) > 0.3 * vmax else "black"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            color=color, fontsize=7)

    plt.tight_layout()

    if save:
        if filename is None:
            filename = f"interactions_{metric}"
        plt.savefig(FIGURES_DIR / f"{filename}.png", dpi=DPI_HIGH, bbox_inches="tight")
        # plt.savefig(FIGURES_DIR / f"{filename}.pdf", bbox_inches="tight")
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

    # Radial sensitivity plots (if S2 available)
    print("\nGenerating radial sensitivity plots...")
    for metric in metrics:
        if "S2" in sobol_results.get(metric, {}):
            plot_radial_sensitivity(sobol_results, metric, save=True)
            plt.close()

    print(f"\nAll figures saved to {FIGURES_DIR}")


# =============================================================================
# RADIAL SENSITIVITY PLOT
# =============================================================================

def plot_radial_sensitivity(
    sobol_results: dict,
    metric: str,
    s2_threshold: float = 0.01,
    figsize: tuple = (12, 14),
    save: bool = True,
    filename: str = None,
    title: str = None
) -> tuple:
    """
    Create radial/chord-style sensitivity visualization.

    Parameters arranged in a circle with:
    - Filled marker size proportional to S1 (first-order sensitivity)
    - Outer ring radius proportional to ST (total-order sensitivity)
    - Line thickness between parameters proportional to |S2| (interactions)

    Parameters
    ----------
    sobol_results : dict
        Output from calculate_sobol_indices (raw format)
    metric : str
        Metric name to visualize
    s2_threshold : float
        Minimum |S2| value to draw interaction lines (default: 0.01)
    figsize : tuple
        Figure size (default: (12, 14) for main plot + legend)
    save : bool
        If True, save figure to FIGURES_DIR
    filename : str, optional
        Custom filename (without extension). Default: radial_{metric}
    title : str, optional
        Custom title. Default: metric name

    Returns
    -------
    tuple : (fig, ax) or None if metric not found/invalid
    """
    # === Data Extraction and Validation ===
    if metric not in sobol_results:
        print(f"Error: Metric '{metric}' not found in sobol_results")
        return None

    if "error" in sobol_results[metric]:
        print(f"Error: Results for '{metric}' contain errors")
        return None

    indices = sobol_results[metric]
    param_names = indices["parameter_names"]
    n_params = len(param_names)

    S1 = np.array(indices["S1"])
    ST = np.array(indices["ST"])
    S2 = np.array(indices.get("S2", np.zeros((n_params, n_params))))

    # Check for NaN values in S1 and ST - these indicate incomplete analysis
    if np.any(np.isnan(S1)):
        print(f"Error: S1 contains NaN values for metric '{metric}'. Cannot plot.")
        return None
    if np.any(np.isnan(ST)):
        print(f"Error: ST contains NaN values for metric '{metric}'. Cannot plot.")
        return None

    # S2 matrix from SALib has NaN values by design:
    # - Diagonal (S2[i,i]): NaN because self-interaction is meaningless
    # - Lower triangle: NaN because SALib only fills upper triangle (S2 is symmetric)
    # We treat these NaN values as 0 interaction for plotting purposes
    S2 = np.nan_to_num(S2, nan=0.0)

    # Ensure non-negative (clip small numerical artifacts)
    S1 = np.maximum(S1, 0)
    ST = np.maximum(ST, 0)

    # === Group and Order Parameters ===
    ordered_indices, group_boundaries = _get_grouped_param_order(param_names)
    ordered_params = [param_names[i] for i in ordered_indices]
    ordered_S1 = S1[ordered_indices]
    ordered_ST = ST[ordered_indices]
    ordered_S2 = S2[np.ix_(ordered_indices, ordered_indices)]

    # === Calculate Circular Positions ===
    radius = 1.0
    x, y, angles = _calculate_radial_positions(
        n_params, radius, start_angle=90.0,
        group_boundaries=group_boundaries, gap_angle=12.0
    )

    # === Calculate Scaling ===
    s1_min, s1_max = ordered_S1.min(), max(ordered_S1.max(), 0.01)
    st_min, st_max = ordered_ST.min(), max(ordered_ST.max(), 0.01)

    # Use absolute value for S2, ignoring diagonal
    S2_abs = np.abs(ordered_S2)
    np.fill_diagonal(S2_abs, 0)
    s2_significant = S2_abs[S2_abs > s2_threshold]
    s2_min = s2_significant.min() if len(s2_significant) > 0 else 0.01
    s2_max = max(S2_abs.max(), 0.01)

    # Scale S1 to marker sizes
    s1_normalized = (ordered_S1 - s1_min) / (s1_max - s1_min) if s1_max > s1_min else np.ones(n_params) * 0.5
    marker_sizes = S1_MIN_SIZE + s1_normalized * (S1_MAX_SIZE - S1_MIN_SIZE)

    # Scale ST to ring sizes
    st_normalized = (ordered_ST - st_min) / (st_max - st_min) if st_max > st_min else np.ones(n_params) * 0.5
    ring_sizes = RING_MIN_SIZE + st_normalized * (RING_MAX_SIZE - RING_MIN_SIZE)

    # === Create Figure ===
    fig = plt.figure(figsize=figsize)
    # Main plot takes upper portion, leave room for legend
    ax = fig.add_axes([0.05, 0.2, 0.9, 0.75])
    ax.set_aspect('equal')

    # === Draw S2 Interaction Lines (Bottom Layer) ===
    for i in range(n_params):
        for j in range(i + 1, n_params):  # Upper triangle only
            s2_val = S2_abs[i, j]
            if s2_val > s2_threshold:
                # Scale line width
                s2_normalized = (s2_val - s2_min) / (s2_max - s2_min) if s2_max > s2_min else 0.5
                line_width = S2_MIN_WIDTH + s2_normalized * (S2_MAX_WIDTH - S2_MIN_WIDTH)
                # Alpha based on strength
                alpha = 0.3 + 0.5 * s2_normalized
                ax.plot([x[i], x[j]], [y[i], y[j]],
                       color=RADIAL_LINE_COLOR, linewidth=line_width,
                       alpha=alpha, zorder=1)

    # === Draw ST Rings (Middle Layer) ===
    for i in range(n_params):
        ax.scatter(x[i], y[i], s=ring_sizes[i],
                  facecolors='none', edgecolors=RADIAL_RING_COLOR,
                  linewidths=2.5, zorder=2)

    # === Draw S1 Markers (Top Layer) ===
    for i in range(n_params):
        ax.scatter(x[i], y[i], s=marker_sizes[i],
                  c=RADIAL_MARKER_COLOR, alpha=0.85, zorder=3)

    # === Add Parameter Labels ===
    label_radius = radius * 1.25
    for i, param in enumerate(ordered_params):
        angle_rad = np.deg2rad(angles[i])
        lx = label_radius * np.cos(angle_rad)
        ly = label_radius * np.sin(angle_rad)

        # Get shortened name
        short_name = get_short_param_name(param)

        # Determine text alignment based on position
        angle = angles[i]
        if -90 <= angle <= 90:
            ha = 'left'
            rotation = angle
        else:
            ha = 'right'
            rotation = angle + 180

        ax.annotate(short_name, (lx, ly),
                   ha=ha, va='center', fontsize=9,
                   rotation=rotation, rotation_mode='anchor')

    # === Clean Up Axes ===
    ax.set_xlim(-1.9, 1.9)
    ax.set_ylim(-1.9, 1.9)
    ax.axis('off')

    # === Add Title ===
    plot_title = title if title else metric
    ax.set_title(plot_title, fontsize=14, fontweight='bold', y=1.02)

    # === Add Legend ===
    _add_radial_legend(fig, ax, s1_min, s1_max, st_min, st_max, s2_min, s2_max)

    # === Save Figure ===
    if save:
        if filename is None:
            filename = f"radial_{metric}"
        filepath = FIGURES_DIR / f"{filename}.png"
        plt.savefig(filepath, dpi=DPI_HIGH, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"Saved: {filepath}")

    return fig, ax


# =============================================================================
# PARALLEL COORDINATES PLOTTING
# =============================================================================

def _reorganize_objs_for_parallel(objs, columns_axes, ideal_direction, minmaxs):
    """
    Reorganize objective values for parallel coordinates plotting.

    Normalizes values to [0, 1] and handles min/max direction so that
    the ideal direction is consistent across all axes.

    Parameters
    ----------
    objs : pd.DataFrame
        DataFrame with objective values
    columns_axes : list
        Column names to use as axes
    ideal_direction : str
        Either "top" or "bottom" - where ideal values should appear
    minmaxs : list
        List of "max" or "min" for each axis

    Returns
    -------
    tuple : (normalized_df, tops, bottoms)
        - normalized_df: DataFrame with values in [0, 1]
        - tops: actual values to annotate at top of each axis
        - bottoms: actual values to annotate at bottom of each axis
    """
    if minmaxs is None:
        minmaxs = ["max"] * len(columns_axes)

    objs_reorg = objs[columns_axes].copy()
    for c in objs_reorg.columns:
        objs_reorg[c] = pd.to_numeric(objs_reorg[c], errors="coerce")

    if ideal_direction == "bottom":
        tops = objs_reorg.min(axis=0)
        bottoms = objs_reorg.max(axis=0)
        for i, minmax in enumerate(minmaxs):
            col = objs_reorg.iloc[:, i].astype(float)
            mn, mx = col.min(), col.max()
            span = mx - mn
            if not np.isfinite(span) or span == 0:
                objs_reorg.iloc[:, i] = 0.5
                continue
            if minmax == "max":
                objs_reorg.iloc[:, i] = (mx - col) / (mx - mn)
            else:
                bottoms.iloc[i], tops.iloc[i] = tops.iloc[i], bottoms.iloc[i]
                objs_reorg.iloc[:, i] = (col - mn) / (mx - mn)
    else:  # ideal_direction == "top"
        tops = objs_reorg.max(axis=0)
        bottoms = objs_reorg.min(axis=0)
        for i, minmax in enumerate(minmaxs):
            col = objs_reorg.iloc[:, i].astype(float)
            mn, mx = col.min(), col.max()
            span = mx - mn
            if not np.isfinite(span) or span == 0:
                objs_reorg.iloc[:, i] = 0.5
                continue
            if minmax == "max":
                objs_reorg.iloc[:, i] = (col - mn) / (mx - mn)
            else:
                bottoms.iloc[i], tops.iloc[i] = tops.iloc[i], bottoms.iloc[i]
                objs_reorg.iloc[:, i] = (mx - col) / (mx - mn)

    objs_reorg = objs_reorg.where(np.isfinite(objs_reorg), 0.5)
    return objs_reorg, tops, bottoms


def custom_parallel_coordinates(
    objs, columns_axes=None, axis_labels=None,
    ideal_direction="top", minmaxs=None,
    color_by_continuous=None, color_palette_continuous="RdYlGn",
    color_by_categorical=None, color_dict_categorical=None,
    zorder_by=None, zorder_num_classes=10, zorder_direction="ascending",
    alpha_base=0.8, alpha_other=0.05,
    lw_base=1.5, fontsize=12,
    figsize=(12, 6), fname=None,
    title=None
):
    """
    Create a custom parallel coordinates plot with actual value annotations.

    Parameters
    ----------
    objs : pd.DataFrame
        DataFrame with objective/metric values
    columns_axes : list, optional
        Column names to use as axes. If None, uses all columns.
    axis_labels : list, optional
        Labels for axes. If None, uses column names.
    ideal_direction : str
        "top" or "bottom" - where ideal values appear on axes
    minmaxs : list, optional
        List of "max" or "min" for each axis indicating optimization direction
    color_by_continuous : int, optional
        Index of column to use for continuous coloring
    color_palette_continuous : str
        Matplotlib colormap name for continuous coloring
    color_by_categorical : str, optional
        Column name for categorical coloring
    color_dict_categorical : dict, optional
        Dictionary mapping category values to colors
    zorder_by : int, optional
        Index of column to use for z-ordering
    zorder_num_classes : int
        Number of z-order classes
    zorder_direction : str
        "ascending" or "descending" for z-ordering
    alpha_base : float
        Base alpha for lines
    alpha_other : float
        Alpha for "Other"/dominated lines
    lw_base : float
        Base line width
    fontsize : int
        Font size for labels
    figsize : tuple
        Figure size
    fname : str, optional
        Filename to save figure
    title : str, optional
        Plot title

    Returns
    -------
    matplotlib.figure.Figure
    """
    assert ideal_direction in ["top", "bottom"]
    assert zorder_direction in ["ascending", "descending"]
    if minmaxs is not None:
        for minmax in minmaxs:
            assert minmax in ["max", "min"]
    assert (color_by_continuous is None) or (color_by_categorical is None)

    columns_axes = columns_axes if (columns_axes is not None) else list(objs.columns)
    axis_labels = axis_labels if (axis_labels is not None) else columns_axes

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    fig.subplots_adjust(bottom=0.20)

    objs_reorg, tops, bottoms = _reorganize_objs_for_parallel(
        objs, columns_axes, ideal_direction, minmaxs
    )

    # Plot each line
    for i in range(objs_reorg.shape[0]):
        # Determine color
        if color_by_continuous is not None:
            norm_val = objs_reorg[columns_axes[color_by_continuous]].iloc[i]
            color = colormaps.get_cmap(color_palette_continuous)(norm_val)
        elif color_by_categorical is not None and color_dict_categorical is not None:
            cat_val = objs[color_by_categorical].iloc[i]
            color = color_dict_categorical.get(cat_val, "gray")
        else:
            color = "steelblue"

        # Determine z-order
        if zorder_by is not None:
            norm_val = objs_reorg[columns_axes[zorder_by]].iloc[i]
            xgrid = np.arange(0, 1.001, 1 / zorder_num_classes)
            if zorder_direction == "ascending":
                zorder = 4 + np.sum(norm_val > xgrid)
            else:
                zorder = 4 + np.sum(norm_val < xgrid)
        else:
            zorder = 4

        alpha = alpha_base
        lw = lw_base

        # Check for special labels
        if color_by_categorical is not None:
            cat_val = str(objs[color_by_categorical].iloc[i])

            if cat_val == "Baseline":
                alpha = alpha_base
                lw = max(lw_base, 3.5)
                zorder = 100
            elif cat_val == "Other":
                alpha = alpha_other
                lw = 1.0
                zorder = 2

        # Plot line segments
        for j in range(objs_reorg.shape[1] - 1):
            y = [objs_reorg.iloc[i, j], objs_reorg.iloc[i, j + 1]]
            x = [j, j + 1]
            ax.plot(x, y, c=color, alpha=alpha, zorder=zorder, lw=lw)

    # Add axis annotations (actual values)
    for j in range(len(columns_axes)):
        top_val = tops.iloc[j] if hasattr(tops, 'iloc') else tops[j]
        bottom_val = bottoms.iloc[j] if hasattr(bottoms, 'iloc') else bottoms[j]

        # Format values appropriately
        if abs(top_val) >= 1000:
            top_str = f"{top_val:.0f}"
        elif abs(top_val) >= 10:
            top_str = f"{top_val:.1f}"
        else:
            top_str = f"{top_val:.2f}"

        if abs(bottom_val) >= 1000:
            bottom_str = f"{bottom_val:.0f}"
        elif abs(bottom_val) >= 10:
            bottom_str = f"{bottom_val:.1f}"
        else:
            bottom_str = f"{bottom_val:.2f}"

        ax.annotate(top_str, [j, 1.02], ha="center", va="bottom",
                    zorder=5, fontsize=fontsize-1)
        ax.annotate(bottom_str, [j, -0.02], ha="center", va="top",
                    zorder=5, fontsize=fontsize-1)
        ax.plot([j, j], [0, 1], c="k", zorder=1, lw=1.5)

    # Clean up axes
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ["top", "bottom", "left", "right"]:
        ax.spines[spine].set_visible(False)

    # Add direction arrow
    if ideal_direction == "top":
        ax.arrow(-0.18, 0.1, 0, 0.7, head_width=0.08, head_length=0.05,
                 color="k", lw=1.5)
    else:
        ax.arrow(-0.18, 0.9, 0, -0.7, head_width=0.08, head_length=0.05,
                 color="k", lw=1.5)
    ax.annotate("Direction of\npreference", xy=(-0.38, 0.5), ha="center",
                va="center", rotation=90, fontsize=fontsize-1)

    # Set limits and add axis labels
    n_axes = len(columns_axes)
    ax.set_xlim(-0.55, n_axes - 0.5)
    ax.set_ylim(-0.25, 1.15)

    for i, label in enumerate(axis_labels):
        ax.annotate(label, xy=(i, -0.12), ha="center", va="top", fontsize=fontsize)

    ax.patch.set_alpha(0)

    # Add title if provided
    if title:
        ax.set_title(title, fontsize=fontsize + 2, pad=20)

    # Add colorbar for continuous coloring
    if color_by_continuous is not None:
        mappable = cm.ScalarMappable(cmap=color_palette_continuous)
        vmin = objs[columns_axes[color_by_continuous]].min()
        vmax = objs[columns_axes[color_by_continuous]].max()
        mappable.set_clim(vmin=vmin, vmax=vmax)
        cb = plt.colorbar(mappable, ax=ax, orientation="horizontal",
                          location="bottom", shrink=0.4,
                          label=axis_labels[color_by_continuous],
                          pad=0.18)
        cb.ax.set_xlabel(cb.ax.get_xlabel(), fontsize=fontsize)

    # Add legend for categorical coloring
    elif color_by_categorical is not None and color_dict_categorical is not None:
        present = pd.unique(objs[color_by_categorical].astype(str)).tolist()
        labels_for_legend = [lab for lab in color_dict_categorical.keys() if lab in present]
        handles = []
        for lab in labels_for_legend:
            col = color_dict_categorical[lab]
            if lab == "Baseline":
                lw_leg = max(lw_base, 3.5)
                a = alpha_base
            elif lab == "Other":
                lw_leg = 1.0
                a = alpha_other
            else:
                lw_leg = lw_base
                a = alpha_base
            handles.append(Line2D([0], [0], color=col, lw=lw_leg, alpha=a, label=lab))
        if handles:
            ncols = min(4, len(handles))
            fig.legend(handles=handles, loc="lower center",
                      bbox_to_anchor=(0.5, 0.02),
                      ncol=ncols, frameon=False, fontsize=fontsize)

    plt.tight_layout()

    if fname is not None:
        plt.savefig(fname, bbox_inches="tight", dpi=DPI_HIGH)

    return fig


if __name__ == "__main__":
    # Test plotting with existing results
    from methods.analysis import load_sobol_results

    try:
        formatted, raw = load_sobol_results()
        print("Generating figures from existing results...")
        generate_all_figures(raw)
    except FileNotFoundError:
        print("No Sobol results found. Run 04_analyze_sensitivity.py first.")
