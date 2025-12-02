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

    print(f"\nAll figures saved to {FIGURES_DIR}")


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
