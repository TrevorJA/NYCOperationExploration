"""
Visualization of MRF factor profiles and their sensitivity ranges.

Creates plots showing:
- Baseline MRF factor profiles for each reservoir
- Range of profiles from parameter sampling (seasonal shifts and scales)
- Separate figure for each reservoir with subplots for each drought level

The MRF factor profiles determine the fraction of baseline minimum release
requirements for each day of year, varying by drought level.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from pywrdrb.parameters.nyc_operations_config import NYCOperationsConfig
from methods.mrf_profiles import (
    MRFProfileBuilder,
    RESERVOIRS,
    DROUGHT_LEVELS,
    DEFAULT_SEASONS
)
from config import (
    FIGURES_DIR,
    PARAMETER_GROUPS,
    get_active_parameters
)

# Plot settings
DPI = 150
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.size'] = 9
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['axes.labelsize'] = 9

# Color scheme for drought levels
LEVEL_COLORS = {
    "level1a": "#1a9850",  # Green (normal)
    "level1b": "#66bd63",
    "level1c": "#a6d96a",
    "level2": "#d9ef8b",   # Yellow-green
    "level3": "#fee08b",   # Yellow (watch)
    "level4": "#fdae61",   # Orange (warning)
    "level5": "#d73027",   # Red (emergency)
}

# Drought level display names
LEVEL_NAMES = {
    "level1a": "Level 1a (Normal+)",
    "level1b": "Level 1b (Flood Warning)",
    "level1c": "Level 1c (Flood Watch)",
    "level2": "Level 2 (Normal)",
    "level3": "Level 3 (Drought Watch)",
    "level4": "Level 4 (Drought Warning)",
    "level5": "Level 5 (Drought Emergency)",
}

# Reservoir display names
RESERVOIR_NAMES = {
    "cannonsville": "Cannonsville",
    "pepacton": "Pepacton",
    "neversink": "Neversink",
}


def get_month_ticks():
    """Get day of year positions and labels for month boundaries."""
    month_starts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    return month_starts, month_names


def generate_mrf_profile_samples(config: NYCOperationsConfig,
                                  n_samples: int = 100,
                                  seed: int = 42) -> dict:
    """
    Generate a set of modified MRF factor profiles by sampling the parameter space.

    Parameters
    ----------
    config : NYCOperationsConfig
        Configuration with baseline profiles
    n_samples : int
        Number of samples to generate
    seed : int
        Random seed for reproducibility

    Returns
    -------
    dict
        Dictionary mapping profile names to arrays of shape (n_samples, 366)
    """
    np.random.seed(seed)

    # Get parameter bounds from config
    active_params = get_active_parameters()

    # Get bounds for MRF factor profile parameters
    shift_params = {}
    scale_params = {}

    for param_name, param_def in active_params.items():
        if "mrf_" in param_name and "_start_shift" in param_name:
            season = param_name.replace("mrf_", "").replace("_start_shift", "")
            shift_params[season] = param_def["bounds"]
        elif "mrf_" in param_name and "_scale" in param_name:
            season = param_name.replace("mrf_", "").replace("_scale", "")
            scale_params[season] = param_def["bounds"]

    # Get base profiles
    base_profiles = {}
    for reservoir in RESERVOIRS:
        for level in DROUGHT_LEVELS:
            profile_name = f"{level}_factor_mrf_{reservoir}"
            base_profiles[profile_name] = config.get_mrf_factor_profile(profile_name)

    builder = MRFProfileBuilder(base_profiles)

    # Generate samples
    sampled_profiles = {name: [] for name in base_profiles}

    for _ in range(n_samples):
        # Sample seasonal parameters
        season_shifts = {}
        season_scales = {}

        for season in ["summer", "fall", "winter", "spring"]:
            if season in shift_params:
                bounds = shift_params[season]
                season_shifts[season] = int(round(np.random.uniform(bounds[0], bounds[1])))
            if season in scale_params:
                bounds = scale_params[season]
                season_scales[season] = np.random.uniform(bounds[0], bounds[1])

        # Build modified profiles
        modified = builder.build_all_modified_profiles(season_shifts, season_scales)

        for name, profile in modified.items():
            sampled_profiles[name].append(profile)

    # Convert to arrays
    for name in sampled_profiles:
        sampled_profiles[name] = np.array(sampled_profiles[name])

    return sampled_profiles


def plot_reservoir_mrf_profiles(reservoir: str,
                                 config: NYCOperationsConfig,
                                 sampled_profiles: dict,
                                 figsize: tuple = (14, 12)):
    """
    Create a figure with subplots for each drought level showing MRF factor profiles.

    Parameters
    ----------
    reservoir : str
        Reservoir name (cannonsville, pepacton, neversink)
    config : NYCOperationsConfig
        Configuration with baseline values
    sampled_profiles : dict
        Dictionary of sampled profiles from generate_mrf_profile_samples()
    figsize : tuple
        Figure size
    """
    # Get MRF baseline for this reservoir
    mrf_baseline = config.get_constant(f'mrf_baseline_{reservoir}')

    fig, axes = plt.subplots(len(DROUGHT_LEVELS), 1, figsize=figsize, sharex=True)

    doy = np.arange(1, 367)
    month_starts, month_names = get_month_ticks()

    for idx, level in enumerate(DROUGHT_LEVELS):
        ax = axes[idx]
        profile_name = f"{level}_factor_mrf_{reservoir}"

        # Get baseline profile
        baseline_profile = config.get_mrf_factor_profile(profile_name)
        baseline_release = baseline_profile * mrf_baseline

        # Get sampled profiles
        samples = sampled_profiles[profile_name]
        sampled_releases = samples * mrf_baseline

        # Calculate percentiles for envelope
        p5 = np.percentile(sampled_releases, 5, axis=0)
        p25 = np.percentile(sampled_releases, 25, axis=0)
        p75 = np.percentile(sampled_releases, 75, axis=0)
        p95 = np.percentile(sampled_releases, 95, axis=0)

        color = LEVEL_COLORS[level]

        # Plot envelope (5th-95th percentile)
        ax.fill_between(doy, p5, p95, alpha=0.2, color=color, label='5th-95th percentile')

        # Plot interquartile range (25th-75th)
        ax.fill_between(doy, p25, p75, alpha=0.3, color=color, label='25th-75th percentile')

        # Plot baseline
        ax.plot(doy, baseline_release, color=color, linewidth=2, label='Baseline')

        # Formatting
        ax.set_ylabel('MRF (MGD)', fontsize=9)
        ax.set_title(LEVEL_NAMES[level], fontsize=10, fontweight='bold', loc='left')

        ax.set_xlim(1, 366)
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

        # Add season shading
        add_season_shading(ax)

        if idx == 0:
            ax.legend(loc='upper right', fontsize=8)

        # Y-axis formatting
        ax.set_ylim(bottom=0)

    # X-axis formatting on bottom plot
    axes[-1].set_xlabel('Day of Year', fontsize=10)
    axes[-1].set_xticks(month_starts)
    axes[-1].set_xticklabels(month_names, fontsize=8)

    # Add title
    fig.suptitle(f'{RESERVOIR_NAMES[reservoir]} Reservoir\nMRF Factor Profiles by Drought Level',
                 fontsize=12, fontweight='bold', y=0.995)

    plt.tight_layout()
    plt.subplots_adjust(top=0.94)

    return fig


def add_season_shading(ax):
    """Add light background shading to indicate seasons."""
    season_colors = {
        "spring": "#98df8a",  # Light green
        "summer": "#ffeda0",  # Light yellow
        "fall": "#fdbe85",    # Light orange
        "winter": "#c6dbef",  # Light blue
    }

    ylim = ax.get_ylim()

    for season, season_def in DEFAULT_SEASONS.items():
        start = season_def["start_doy"]
        end = season_def["end_doy"]
        color = season_colors[season]

        if start <= end:
            ax.axvspan(start, end, alpha=0.1, color=color, zorder=0)
        else:
            # Wrap-around (winter)
            ax.axvspan(start, 366, alpha=0.1, color=color, zorder=0)
            ax.axvspan(1, end, alpha=0.1, color=color, zorder=0)

    ax.set_ylim(ylim)


def plot_all_levels_comparison(reservoir: str,
                                config: NYCOperationsConfig,
                                figsize: tuple = (12, 6)):
    """
    Create a single plot comparing all drought levels for a reservoir.

    Parameters
    ----------
    reservoir : str
        Reservoir name
    config : NYCOperationsConfig
        Configuration with baseline values
    figsize : tuple
        Figure size
    """
    mrf_baseline = config.get_constant(f'mrf_baseline_{reservoir}')

    fig, ax = plt.subplots(figsize=figsize)

    doy = np.arange(1, 367)
    month_starts, month_names = get_month_ticks()

    for level in DROUGHT_LEVELS:
        profile_name = f"{level}_factor_mrf_{reservoir}"
        profile = config.get_mrf_factor_profile(profile_name)
        release = profile * mrf_baseline

        ax.plot(doy, release, color=LEVEL_COLORS[level], linewidth=1.5,
                label=LEVEL_NAMES[level])

    ax.set_xlabel('Day of Year', fontsize=10)
    ax.set_ylabel('Minimum Release Requirement (MGD)', fontsize=10)
    ax.set_title(f'{RESERVOIR_NAMES[reservoir]} - Baseline MRF by Drought Level',
                 fontsize=11, fontweight='bold')

    ax.set_xlim(1, 366)
    ax.set_ylim(bottom=0)
    ax.set_xticks(month_starts)
    ax.set_xticklabels(month_names, fontsize=9)

    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=8, ncol=2)

    add_season_shading(ax)

    plt.tight_layout()

    return fig


def plot_parameter_effect_demo(config: NYCOperationsConfig,
                                reservoir: str = "cannonsville",
                                level: str = "level3"):
    """
    Create demonstration plots showing how each parameter affects the profiles.

    Parameters
    ----------
    config : NYCOperationsConfig
        Configuration with baseline values
    reservoir : str
        Reservoir to use for demo
    level : str
        Drought level to use for demo
    """
    mrf_baseline = config.get_constant(f'mrf_baseline_{reservoir}')
    profile_name = f"{level}_factor_mrf_{reservoir}"

    # Get base profiles
    base_profiles = {}
    for res in RESERVOIRS:
        for lev in DROUGHT_LEVELS:
            pname = f"{lev}_factor_mrf_{res}"
            base_profiles[pname] = config.get_mrf_factor_profile(pname)

    builder = MRFProfileBuilder(base_profiles)
    baseline_profile = base_profiles[profile_name]
    baseline_release = baseline_profile * mrf_baseline

    doy = np.arange(1, 367)
    month_starts, month_names = get_month_ticks()

    # Figure 1: Season shift effects
    fig1, axes1 = plt.subplots(2, 2, figsize=(12, 8))

    seasons = ["spring", "summer", "fall", "winter"]
    shift_values = [-15, -7, 0, 7, 15]

    for ax, season in zip(axes1.flat, seasons):
        colors = plt.cm.RdBu_r(np.linspace(0.1, 0.9, len(shift_values)))

        for shift, color in zip(shift_values, colors):
            season_shifts = {season: shift}
            modified = builder.apply_seasonal_modifications(
                profile_name, season_shifts, {}
            )
            release = modified * mrf_baseline
            ax.plot(doy, release, color=color, linewidth=1.5,
                   label=f'{shift:+d} days')

        ax.set_title(f'{season.title()} Start Shift', fontsize=10, fontweight='bold')
        ax.set_xlim(1, 366)
        ax.set_ylim(bottom=0)
        ax.set_xticks(month_starts)
        ax.set_xticklabels(month_names, fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=7)
        add_season_shading(ax)

    fig1.suptitle(f'{RESERVOIR_NAMES[reservoir]} ({LEVEL_NAMES[level]})\nEffect of Season Boundary Shifts',
                  fontsize=11, fontweight='bold')
    plt.tight_layout()

    # Figure 2: Season scale effects
    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 8))

    scale_values = [0.9, 0.95, 1.0, 1.05, 1.1]

    for ax, season in zip(axes2.flat, seasons):
        colors = plt.cm.RdBu_r(np.linspace(0.1, 0.9, len(scale_values)))

        for scale, color in zip(scale_values, colors):
            season_scales = {season: scale}
            modified = builder.apply_seasonal_modifications(
                profile_name, {}, season_scales
            )
            release = modified * mrf_baseline
            ax.plot(doy, release, color=color, linewidth=1.5,
                   label=f'{scale:.2f}x')

        ax.set_title(f'{season.title()} Scale Factor', fontsize=10, fontweight='bold')
        ax.set_xlim(1, 366)
        ax.set_ylim(bottom=0)
        ax.set_xticks(month_starts)
        ax.set_xticklabels(month_names, fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=7)
        add_season_shading(ax)

    fig2.suptitle(f'{RESERVOIR_NAMES[reservoir]} ({LEVEL_NAMES[level]})\nEffect of Season Scale Factors',
                  fontsize=11, fontweight='bold')
    plt.tight_layout()

    return fig1, fig2


def main():
    """Generate all MRF factor profile visualizations."""
    print("=" * 70)
    print("MRF FACTOR PROFILE VISUALIZATION")
    print("=" * 70)

    # Load default configuration
    config = NYCOperationsConfig.from_defaults()

    # Print MRF baselines
    print("\nMRF Baselines (MGD):")
    for reservoir in RESERVOIRS:
        baseline = config.get_constant(f'mrf_baseline_{reservoir}')
        print(f"  {RESERVOIR_NAMES[reservoir]}: {baseline:.2f}")

    # Generate sampled profiles
    print("\nGenerating sampled profiles...")
    sampled_profiles = generate_mrf_profile_samples(config, n_samples=200)
    print(f"  Generated {200} samples for each profile")

    # 1. Create comparison plots (baseline only, all levels)
    print("\n1. Creating baseline comparison plots...")
    for reservoir in RESERVOIRS:
        fig = plot_all_levels_comparison(reservoir, config)
        outfile = FIGURES_DIR / f'mrf_profiles_{reservoir}_baseline_comparison.png'
        fig.savefig(outfile, dpi=DPI, bbox_inches='tight')
        print(f"  Saved: {outfile}")
        plt.close(fig)

    # 2. Create detailed plots with sensitivity ranges
    print("\n2. Creating detailed profiles with sensitivity ranges...")
    for reservoir in RESERVOIRS:
        fig = plot_reservoir_mrf_profiles(reservoir, config, sampled_profiles)
        outfile = FIGURES_DIR / f'mrf_profiles_{reservoir}_sensitivity.png'
        fig.savefig(outfile, dpi=DPI, bbox_inches='tight')
        print(f"  Saved: {outfile}")
        plt.close(fig)

    # 3. Create parameter effect demonstration plots
    print("\n3. Creating parameter effect demonstration plots...")
    fig_shift, fig_scale = plot_parameter_effect_demo(config)

    outfile_shift = FIGURES_DIR / 'mrf_profiles_shift_effect_demo.png'
    fig_shift.savefig(outfile_shift, dpi=DPI, bbox_inches='tight')
    print(f"  Saved: {outfile_shift}")
    plt.close(fig_shift)

    outfile_scale = FIGURES_DIR / 'mrf_profiles_scale_effect_demo.png'
    fig_scale.savefig(outfile_scale, dpi=DPI, bbox_inches='tight')
    print(f"  Saved: {outfile_scale}")
    plt.close(fig_scale)

    print("\n" + "=" * 70)
    print("VISUALIZATION COMPLETE")
    print("=" * 70)
    print(f"\nFigures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
