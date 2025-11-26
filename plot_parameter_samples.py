"""
Visualize Sobol parameter samples compared to default values.

This script creates figures showing:
- Parameter value distributions vs baselines
- Storage zone profiles with shifts applied
- Other operational parameters with sample ranges

Usage:
    python plot_parameter_samples.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from pywrdrb.parameters.nyc_operations_config import NYCOperationsConfig

from config import (
    PARAMETER_GROUPS,
    FIGURES_DIR,
    get_active_parameters,
    get_parameter_group,
)
from methods.sampling import load_samples, sample_to_nyc_config


# Plot settings
DPI = 150
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['axes.labelsize'] = 10


# Color scheme for parameter groups
GROUP_COLORS = {
    "delivery": "#1f77b4",
    "individual_mrf": "#2ca02c",
    "flow_target_mrf": "#98df8a",  # lighter green
    "mrf_factor_profiles": "#9467bd",  # purple
    "flood": "#ff7f0e",
    "storage_zones": "#d62728",
}


def plot_parameter_distributions(samples: np.ndarray, problem: dict):
    """
    Create histogram plots showing distribution of sampled values for each parameter.
    One figure per parameter group.
    """
    active_params = get_active_parameters()

    # Group parameters
    groups = {}
    for i, name in enumerate(problem["names"]):
        group = get_parameter_group(name)
        if group not in groups:
            groups[group] = []
        groups[group].append((i, name))

    for group_name, param_list in groups.items():
        n_params = len(param_list)
        fig, axes = plt.subplots(1, n_params, figsize=(4 * n_params, 3.5))

        if n_params == 1:
            axes = [axes]

        color = GROUP_COLORS.get(group_name, "#7f7f7f")

        for ax, (idx, param_name) in zip(axes, param_list):
            param_def = active_params[param_name]
            values = samples[:, idx]
            baseline = param_def["baseline"]
            bounds = param_def["bounds"]

            # Histogram
            ax.hist(values, bins=30, color=color, alpha=0.7, edgecolor='white')

            # Baseline line
            ax.axvline(baseline, color='black', linestyle='--', linewidth=2,
                      label=f'Baseline: {baseline}')

            # Bounds
            ax.axvline(bounds[0], color='gray', linestyle=':', linewidth=1.5)
            ax.axvline(bounds[1], color='gray', linestyle=':', linewidth=1.5)

            ax.set_xlabel(f'{param_def["units"]}')
            ax.set_ylabel('Count')
            ax.set_title(param_name.replace('_', ' ').title())
            ax.legend(fontsize=8)

        fig.suptitle(f'{group_name.replace("_", " ").title()} Parameters',
                    fontsize=12, fontweight='bold', y=1.02)
        plt.tight_layout()

        outfile = FIGURES_DIR / f'param_dist_{group_name}.png'
        plt.savefig(outfile, dpi=DPI, bbox_inches='tight')
        print(f"Saved: {outfile}")
        plt.close()


def plot_parameter_ranges(samples: np.ndarray, problem: dict):
    """
    Create a single figure showing all parameters with baseline, bounds, and sample range.
    """
    active_params = get_active_parameters()
    n_params = len(problem["names"])

    fig, ax = plt.subplots(figsize=(10, max(4, n_params * 0.5)))

    y_positions = np.arange(n_params)

    # Separate parameters with zero baseline (need different normalization)
    zero_baseline_params = []
    normal_params = []

    for i, name in enumerate(problem["names"]):
        param_def = active_params[name]
        if param_def["baseline"] == 0:
            zero_baseline_params.append((i, name))
        else:
            normal_params.append((i, name))

    for i, name in enumerate(problem["names"]):
        param_def = active_params[name]
        baseline = param_def["baseline"]
        bounds = param_def["bounds"]
        values = samples[:, i]

        group = get_parameter_group(name)
        color = GROUP_COLORS.get(group, "#7f7f7f")

        # Handle zero baseline parameters differently
        if baseline == 0:
            # For zero baseline (like zone_vertical_shift), show actual values
            # Scale to fit with other parameters by using bounds range
            bound_range = bounds[1] - bounds[0]
            norm_baseline = 100  # Center at 100
            norm_bounds = [100 + 100 * bounds[0] / (bound_range / 2),
                          100 + 100 * bounds[1] / (bound_range / 2)]
            norm_values = 100 + 100 * values / (bound_range / 2)
        else:
            # Normalize to percentage of baseline for visualization
            norm_baseline = 100
            norm_bounds = [100 * b / baseline for b in bounds]
            norm_values = 100 * values / baseline

        # Sample range (min to max)
        ax.barh(i, norm_values.max() - norm_values.min(),
               left=norm_values.min(), height=0.6,
               color=color, alpha=0.5, label=group if i == 0 else "")

        # Bounds
        ax.plot([norm_bounds[0], norm_bounds[0]], [i - 0.35, i + 0.35],
               color='gray', linewidth=2)
        ax.plot([norm_bounds[1], norm_bounds[1]], [i - 0.35, i + 0.35],
               color='gray', linewidth=2)

        # Baseline
        ax.plot(norm_baseline, i, 'k|', markersize=15, markeredgewidth=2)

    ax.set_yticks(y_positions)
    ax.set_yticklabels([n.replace('_', '\n') for n in problem["names"]], fontsize=9)
    ax.set_xlabel('Percent of Baseline Value')
    ax.axvline(100, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_title('Parameter Sample Ranges vs Baseline', fontweight='bold')

    # Legend for groups
    handles = [Patch(facecolor=c, alpha=0.5, label=g.replace('_', ' ').title())
               for g, c in GROUP_COLORS.items() if g in [get_parameter_group(n) for n in problem["names"]]]
    ax.legend(handles=handles, loc='upper right', fontsize=9)

    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()

    outfile = FIGURES_DIR / 'param_ranges_overview.png'
    plt.savefig(outfile, dpi=DPI, bbox_inches='tight')
    print(f"Saved: {outfile}")
    plt.close()


def plot_storage_zones_default():
    """
    Plot the default storage zone thresholds over the year.
    """
    config = NYCOperationsConfig.from_defaults()

    levels = ['level1b', 'level1c', 'level2', 'level3', 'level4', 'level5']
    level_labels = {
        'level1b': 'L1b (Flood)',
        'level1c': 'L1c (Flood)',
        'level2': 'L2 (Normal)',
        'level3': 'L3 (Watch)',
        'level4': 'L4 (Warning)',
        'level5': 'L5 (Emergency)',
    }

    # Colors from low (blue/flood) to high (red/drought)
    colors = plt.cm.RdYlBu_r(np.linspace(0.1, 0.9, len(levels)))

    fig, ax = plt.subplots(figsize=(10, 5))

    doy = np.arange(1, 367)

    for level, color in zip(levels, colors):
        profile = config.get_storage_zone_profile(level)
        ax.plot(doy, profile * 100, label=level_labels[level], color=color, linewidth=2)

    ax.set_xlabel('Day of Year')
    ax.set_ylabel('Storage Zone Threshold (% of Capacity)')
    ax.set_title('Default NYC Storage Zone Thresholds', fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xlim(1, 366)
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.3)

    # Add month labels
    month_starts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    ax.set_xticks(month_starts)
    ax.set_xticklabels(month_names, fontsize=9)

    plt.tight_layout()

    outfile = FIGURES_DIR / 'storage_zones_default.png'
    plt.savefig(outfile, dpi=DPI, bbox_inches='tight')
    print(f"Saved: {outfile}")
    plt.close()


def plot_storage_zones_with_shifts(samples: np.ndarray, problem: dict, n_samples_to_show: int = 50):
    """
    Plot storage zone profiles showing effect of per-level vertical and time shifts.
    Shows default profile with envelope of sampled profiles.
    """
    param_names = problem["names"]

    # Check if we have the new per-level parameters
    levels_to_plot = ['level1b', 'level1c', 'level2', 'level3', 'level4', 'level5']

    # Check for per-level parameters
    has_per_level = any(f"zone_{level}_vertical_shift" in param_names for level in levels_to_plot)

    if not has_per_level:
        print("No storage zone shift parameters found in samples")
        return

    config_default = NYCOperationsConfig.from_defaults()

    doy = np.arange(1, 367)
    month_starts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    for level in levels_to_plot:
        v_shift_key = f"zone_{level}_vertical_shift"
        t_shift_key = f"zone_{level}_time_shift"

        v_shift_idx = param_names.index(v_shift_key) if v_shift_key in param_names else None
        t_shift_idx = param_names.index(t_shift_key) if t_shift_key in param_names else None

        if v_shift_idx is None and t_shift_idx is None:
            continue

        fig, ax = plt.subplots(figsize=(10, 4.5))

        # Get default profile
        default_profile = config_default.get_storage_zone_profile(level) * 100

        # Generate shifted profiles for a subset of samples
        sample_indices = np.linspace(0, len(samples) - 1, min(n_samples_to_show, len(samples)), dtype=int)

        all_profiles = []
        for idx in sample_indices:
            v_shift = samples[idx, v_shift_idx] if v_shift_idx is not None else 0.0
            t_shift = int(round(samples[idx, t_shift_idx])) if t_shift_idx is not None else 0

            # Apply shifts
            profile = config_default.get_storage_zone_profile(level).copy()
            if t_shift != 0:
                profile = np.roll(profile, t_shift)
            profile = np.clip(profile + v_shift, 0, 1) * 100
            all_profiles.append(profile)

        all_profiles = np.array(all_profiles)

        # Plot envelope
        ax.fill_between(doy, all_profiles.min(axis=0), all_profiles.max(axis=0),
                       color='#d62728', alpha=0.3, label='Sample Range')

        # Plot a few individual sample profiles
        for i in range(min(5, len(all_profiles))):
            ax.plot(doy, all_profiles[i], color='#d62728', alpha=0.2, linewidth=0.5)

        # Plot default
        ax.plot(doy, default_profile, 'k-', linewidth=2.5, label='Default')

        ax.set_xlabel('Day of Year')
        ax.set_ylabel('Storage Threshold (% of Capacity)')
        ax.set_title(f'Storage Zone {level.replace("level", "Level ")} - Effect of Parameter Shifts',
                    fontweight='bold')
        ax.legend(loc='lower right', fontsize=9)
        ax.set_xlim(1, 366)
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.3)
        ax.set_xticks(month_starts)
        ax.set_xticklabels(month_names, fontsize=9)

        plt.tight_layout()

        outfile = FIGURES_DIR / f'storage_zones_{level}_shifts.png'
        plt.savefig(outfile, dpi=DPI, bbox_inches='tight')
        print(f"Saved: {outfile}")
        plt.close()


def plot_storage_zones_shift_components(samples: np.ndarray, problem: dict):
    """
    Show vertical and time shift effects separately on an example level.
    Uses level3 as an example but checks for per-level parameters.
    """
    param_names = problem["names"]
    level = 'level3'  # Use level 3 as example

    # Check for per-level parameters (new style)
    v_shift_key = f"zone_{level}_vertical_shift"
    t_shift_key = f"zone_{level}_time_shift"

    has_v_shift = v_shift_key in param_names
    has_t_shift = t_shift_key in param_names

    if not has_v_shift and not has_t_shift:
        print("No storage zone shift parameters found in samples")
        return

    config_default = NYCOperationsConfig.from_defaults()
    default_profile = config_default.get_storage_zone_profile(level) * 100

    doy = np.arange(1, 367)
    month_starts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    # Figure 1: Vertical shift effect
    if has_v_shift:
        fig, ax = plt.subplots(figsize=(10, 4.5))

        v_shifts = [-0.05, -0.025, 0, 0.025, 0.05]
        colors = plt.cm.RdBu_r(np.linspace(0.1, 0.9, len(v_shifts)))

        for v_shift, color in zip(v_shifts, colors):
            profile = config_default.get_storage_zone_profile(level).copy()
            profile = np.clip(profile + v_shift, 0, 1) * 100
            label = f'{v_shift:+.1%}' if v_shift != 0 else 'Default (0%)'
            lw = 2.5 if v_shift == 0 else 1.5
            ax.plot(doy, profile, color=color, linewidth=lw, label=label)

        ax.set_xlabel('Day of Year')
        ax.set_ylabel('Storage Threshold (% of Capacity)')
        ax.set_title('Effect of Vertical Shift on Storage Zone Level 3', fontweight='bold')
        ax.legend(title='Vertical Shift', loc='lower right', fontsize=9)
        ax.set_xlim(1, 366)
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.3)
        ax.set_xticks(month_starts)
        ax.set_xticklabels(month_names, fontsize=9)

        plt.tight_layout()
        outfile = FIGURES_DIR / 'storage_zones_vertical_shift_effect.png'
        plt.savefig(outfile, dpi=DPI, bbox_inches='tight')
        print(f"Saved: {outfile}")
        plt.close()

    # Figure 2: Time shift effect
    if has_t_shift:
        fig, ax = plt.subplots(figsize=(10, 4.5))

        t_shifts = [-30, -15, 0, 15, 30]
        colors = plt.cm.PuOr(np.linspace(0.1, 0.9, len(t_shifts)))

        for t_shift, color in zip(t_shifts, colors):
            profile = config_default.get_storage_zone_profile(level).copy()
            if t_shift != 0:
                profile = np.roll(profile, t_shift)
            profile = profile * 100
            label = f'{t_shift:+d} days' if t_shift != 0 else 'Default (0 days)'
            lw = 2.5 if t_shift == 0 else 1.5
            ax.plot(doy, profile, color=color, linewidth=lw, label=label)

        ax.set_xlabel('Day of Year')
        ax.set_ylabel('Storage Threshold (% of Capacity)')
        ax.set_title('Effect of Time Shift on Storage Zone Level 3', fontweight='bold')
        ax.legend(title='Time Shift', loc='lower right', fontsize=9)
        ax.set_xlim(1, 366)
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.3)
        ax.set_xticks(month_starts)
        ax.set_xticklabels(month_names, fontsize=9)

        plt.tight_layout()
        outfile = FIGURES_DIR / 'storage_zones_time_shift_effect.png'
        plt.savefig(outfile, dpi=DPI, bbox_inches='tight')
        print(f"Saved: {outfile}")
        plt.close()


def plot_delivery_parameters(samples: np.ndarray, problem: dict):
    """
    Plot delivery constraint parameters - baseline vs sampled values.
    """
    param_names = problem["names"]
    active_params = get_active_parameters()

    # Filter to delivery parameters
    delivery_params = [(i, n) for i, n in enumerate(param_names)
                       if get_parameter_group(n) == "delivery"]

    if not delivery_params:
        print("No delivery parameters found")
        return

    n_params = len(delivery_params)
    fig, axes = plt.subplots(1, n_params, figsize=(4 * n_params, 4))

    if n_params == 1:
        axes = [axes]

    for ax, (idx, param_name) in zip(axes, delivery_params):
        param_def = active_params[param_name]
        values = samples[:, idx]
        baseline = param_def["baseline"]
        bounds = param_def["bounds"]

        # Box plot style with individual points
        parts = ax.violinplot([values], positions=[0], showmeans=False, showmedians=False)
        for pc in parts['bodies']:
            pc.set_facecolor(GROUP_COLORS["delivery"])
            pc.set_alpha(0.5)

        # Baseline
        ax.axhline(baseline, color='black', linestyle='--', linewidth=2, label='Baseline')

        # Bounds
        ax.axhline(bounds[0], color='gray', linestyle=':', linewidth=1.5, label='Bounds')
        ax.axhline(bounds[1], color='gray', linestyle=':', linewidth=1.5)

        # Mean of samples
        ax.axhline(values.mean(), color=GROUP_COLORS["delivery"], linestyle='-',
                  linewidth=1.5, label='Sample Mean')

        ax.set_ylabel(param_def["units"])
        ax.set_title(param_name.replace('_', ' ').title(), fontsize=10)
        ax.set_xticks([])
        ax.legend(fontsize=8, loc='best')
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle('Delivery Constraint Parameters', fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()

    outfile = FIGURES_DIR / 'delivery_parameters.png'
    plt.savefig(outfile, dpi=DPI, bbox_inches='tight')
    print(f"Saved: {outfile}")
    plt.close()


def plot_mrf_parameters(samples: np.ndarray, problem: dict):
    """
    Plot MRF baseline parameters - showing all on one figure.
    Handles both individual_mrf and flow_target_mrf groups.
    """
    param_names = problem["names"]
    active_params = get_active_parameters()

    # Filter to MRF baseline parameters (individual_mrf and flow_target_mrf)
    mrf_groups = ["individual_mrf", "flow_target_mrf"]
    mrf_params = [(i, n) for i, n in enumerate(param_names)
                  if get_parameter_group(n) in mrf_groups]

    if not mrf_params:
        print("No MRF baseline parameters found")
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    x_positions = np.arange(len(mrf_params))
    width = 0.6

    baselines = []
    sample_mins = []
    sample_maxs = []
    sample_means = []
    labels = []
    colors = []

    for idx, param_name in mrf_params:
        param_def = active_params[param_name]
        values = samples[:, idx]
        group = get_parameter_group(param_name)

        baselines.append(param_def["baseline"])
        sample_mins.append(values.min())
        sample_maxs.append(values.max())
        sample_means.append(values.mean())
        labels.append(param_name.replace('mrf_', '').replace('_', '\n').title())
        colors.append(GROUP_COLORS.get(group, "#7f7f7f"))

    # Sample range bars
    for i, (smin, smax, color) in enumerate(zip(sample_mins, sample_maxs, colors)):
        ax.bar(i, smax - smin, bottom=smin, width=width,
              color=color, alpha=0.5, edgecolor=color)

    # Baselines
    ax.scatter(x_positions, baselines, color='black', s=100, marker='_',
              linewidths=3, zorder=5, label='Baseline')

    # Sample means
    for i, (mean, color) in enumerate(zip(sample_means, colors)):
        ax.scatter(i, mean, color=color, s=50, marker='o', zorder=5)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Flow (MGD)')
    ax.set_title('Minimum Required Flow Baseline Parameters', fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    outfile = FIGURES_DIR / 'mrf_parameters.png'
    plt.savefig(outfile, dpi=DPI, bbox_inches='tight')
    print(f"Saved: {outfile}")
    plt.close()


def plot_mrf_factor_profiles(samples: np.ndarray, problem: dict):
    """
    Plot MRF factor profile parameters (seasonal shifts and scales).
    """
    param_names = problem["names"]
    active_params = get_active_parameters()

    # Filter to mrf_factor_profiles parameters
    mrf_profile_params = [(i, n) for i, n in enumerate(param_names)
                          if get_parameter_group(n) == "mrf_factor_profiles"]

    if not mrf_profile_params:
        print("No MRF factor profile parameters found")
        return

    # Separate shift and scale parameters
    shift_params = [(i, n) for i, n in mrf_profile_params if "shift" in n]
    scale_params = [(i, n) for i, n in mrf_profile_params if "scale" in n]

    color = GROUP_COLORS.get("mrf_factor_profiles", "#9467bd")

    # Figure 1: Season boundary shifts
    if shift_params:
        fig, ax = plt.subplots(figsize=(8, 5))

        x_positions = np.arange(len(shift_params))
        width = 0.6

        for i, (idx, param_name) in enumerate(shift_params):
            param_def = active_params[param_name]
            values = samples[:, idx]
            baseline = param_def["baseline"]

            # Box showing sample range
            ax.bar(i, values.max() - values.min(), bottom=values.min(),
                  width=width, color=color, alpha=0.5, edgecolor=color)

            # Baseline line
            ax.scatter(i, baseline, color='black', s=100, marker='_', linewidths=3, zorder=5)

        labels = [n.replace('mrf_', '').replace('_start_shift', '').title()
                  for _, n in shift_params]

        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel('Days')
        ax.set_title('MRF Season Boundary Shift Parameters', fontweight='bold')
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        outfile = FIGURES_DIR / 'mrf_season_shifts.png'
        plt.savefig(outfile, dpi=DPI, bbox_inches='tight')
        print(f"Saved: {outfile}")
        plt.close()

    # Figure 2: Season scale factors
    if scale_params:
        fig, ax = plt.subplots(figsize=(8, 5))

        x_positions = np.arange(len(scale_params))
        width = 0.6

        for i, (idx, param_name) in enumerate(scale_params):
            param_def = active_params[param_name]
            values = samples[:, idx]
            baseline = param_def["baseline"]

            ax.bar(i, values.max() - values.min(), bottom=values.min(),
                  width=width, color=color, alpha=0.5, edgecolor=color)

            ax.scatter(i, baseline, color='black', s=100, marker='_', linewidths=3, zorder=5)

        labels = [n.replace('mrf_', '').replace('_scale', '').title()
                  for _, n in scale_params]

        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel('Scale Factor')
        ax.set_title('MRF Season Scale Factor Parameters', fontweight='bold')
        ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        outfile = FIGURES_DIR / 'mrf_season_scales.png'
        plt.savefig(outfile, dpi=DPI, bbox_inches='tight')
        print(f"Saved: {outfile}")
        plt.close()


def plot_flood_parameters(samples: np.ndarray, problem: dict):
    """
    Plot flood release limit parameters.
    """
    param_names = problem["names"]
    active_params = get_active_parameters()

    # Filter to flood parameters
    flood_params = [(i, n) for i, n in enumerate(param_names)
                    if get_parameter_group(n) == "flood"]

    if not flood_params:
        print("No flood parameters found")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    x_positions = np.arange(len(flood_params))
    width = 0.6

    baselines = []
    sample_mins = []
    sample_maxs = []
    labels = []

    for idx, param_name in flood_params:
        param_def = active_params[param_name]
        values = samples[:, idx]

        baselines.append(param_def["baseline"])
        sample_mins.append(values.min())
        sample_maxs.append(values.max())
        labels.append(param_name.replace('flood_max_', '').title())

    # Sample range bars
    for i, (smin, smax) in enumerate(zip(sample_mins, sample_maxs)):
        ax.bar(i, smax - smin, bottom=smin, width=width,
              color=GROUP_COLORS["flood"], alpha=0.5, edgecolor=GROUP_COLORS["flood"])

    # Baselines
    ax.scatter(x_positions, baselines, color='black', s=100, marker='_',
              linewidths=3, zorder=5, label='Baseline')

    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('Max Release (CFS)')
    ax.set_title('Flood Control Release Limit Parameters', fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    outfile = FIGURES_DIR / 'flood_parameters.png'
    plt.savefig(outfile, dpi=DPI, bbox_inches='tight')
    print(f"Saved: {outfile}")
    plt.close()


def main():
    """Generate all parameter visualization figures."""
    print("=" * 70)
    print("PARAMETER SAMPLE VISUALIZATION")
    print("=" * 70)

    # Load samples
    print("\nLoading samples...")
    try:
        samples, problem = load_samples("sobol")
        print(f"  Loaded {len(samples)} samples with {problem['num_vars']} parameters")
    except FileNotFoundError:
        print("ERROR: No samples found. Run 01_generate_samples.py first.")
        return

    print(f"\nGenerating figures to {FIGURES_DIR}...")

    # 1. Parameter range overview
    print("\n1. Parameter ranges overview...")
    plot_parameter_ranges(samples, problem)

    # 2. Parameter distributions by group
    print("\n2. Parameter distributions by group...")
    plot_parameter_distributions(samples, problem)

    # 3. Default storage zones
    print("\n3. Default storage zone profiles...")
    plot_storage_zones_default()

    # 4. Storage zone shift effects
    print("\n4. Storage zone shift effects...")
    plot_storage_zones_shift_components(samples, problem)

    # 5. Storage zones with sample envelope
    print("\n5. Storage zones with sample envelope...")
    plot_storage_zones_with_shifts(samples, problem)

    # 6. Delivery parameters
    print("\n6. Delivery parameters...")
    plot_delivery_parameters(samples, problem)

    # 7. MRF baseline parameters
    print("\n7. MRF baseline parameters...")
    plot_mrf_parameters(samples, problem)

    # 8. MRF factor profile parameters
    print("\n8. MRF factor profile parameters...")
    plot_mrf_factor_profiles(samples, problem)

    # 9. Flood parameters
    print("\n9. Flood parameters...")
    plot_flood_parameters(samples, problem)

    print("\n" + "=" * 70)
    print("VISUALIZATION COMPLETE")
    print("=" * 70)
    print(f"\nFigures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
