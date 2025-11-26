"""
Configuration for NYC Reservoir Operations Sobol Sensitivity Analysis.

This module defines:
- Simulation settings (dates, inflow type)
- Parameter definitions and bounds for sensitivity analysis
- SALib problem definition helpers
- File paths and directories
- Metric definitions
"""

import os
from pathlib import Path

# =============================================================================
# DIRECTORIES
# =============================================================================
ROOT_DIR = Path(__file__).parent
OUTPUT_DIR = ROOT_DIR / "outputs"
SAMPLES_DIR = OUTPUT_DIR / "samples"
SIMULATIONS_DIR = OUTPUT_DIR / "simulations"
METRICS_DIR = OUTPUT_DIR / "metrics"
ANALYSIS_DIR = OUTPUT_DIR / "analysis"
FIGURES_DIR = ROOT_DIR / "figures"

# Pywr-DRB directory (relative to this project)
PYWRDRB_DIR = ROOT_DIR.parent / "Pywr-DRB"

# Create directories if they don't exist
for d in [OUTPUT_DIR, SAMPLES_DIR, SIMULATIONS_DIR, METRICS_DIR, ANALYSIS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# =============================================================================
# SIMULATION SETTINGS
# =============================================================================
START_DATE = "1945-01-01"
END_DATE = "2023-12-31"
INFLOW_TYPE = "pub_nhmv10_BC_withObsScaled"

# =============================================================================
# SOBOL SAMPLING CONFIGURATION
# =============================================================================
N_SOBOL_SAMPLES = 64  # N in SALib terminology; total = N*(2D+2)
RANDOM_SEED = 42  # For reproducibility

# =============================================================================
# MPI / PARALLEL SETTINGS
# =============================================================================
N_SAMPLES_PER_BATCH = 5  # Memory management for batch processing

# =============================================================================
# NYC RESERVOIR CONSTANTS
# =============================================================================
NYC_RESERVOIRS = ['cannonsville', 'pepacton', 'neversink']

# NYC storage capacities (MG)
NYC_STORAGE_CAPACITIES = {
    "cannonsville": 95706,
    "pepacton": 140190,
    "neversink": 34941
}
NYC_TOTAL_CAPACITY = sum(NYC_STORAGE_CAPACITIES.values())

# =============================================================================
# DROUGHT ZONE MAPPING
# =============================================================================
# Zone numbers (higher = more severe drought)
ZONE_NAMES = {
    6: 'Drought Emergency',  # Most severe
    5: 'Drought Watch',
    4: 'Drought Warning',
    3: 'Normal',
    2: 'Flood Watch',
    1: 'Flood Warning',
}

# =============================================================================
# PARAMETER GROUPS WITH ENABLE/DISABLE FLAGS
# =============================================================================
PARAMETER_GROUPS = {
    "delivery": {
        "enabled": True,
        "description": "NYC and NJ delivery constraints",
        "parameters": {
            "max_nyc_delivery": {
                "baseline": 800.0,
                "bounds": [720.0, 880.0],  # +/- 10%
                "units": "MGD",
                "description": "Maximum NYC delivery baseline"
            },
            "max_nj_daily": {
                "baseline": 120.0,
                "bounds": [108.0, 132.0],
                "units": "MGD",
                "description": "Maximum NJ daily delivery"
            },
            "drought_factor_nyc_level3": {
                "baseline": 0.85,
                "bounds": [0.765, 0.935],
                "units": "fraction",
                "description": "NYC delivery factor at drought level 3 (Watch)"
            },
            "drought_factor_nyc_level5": {
                "baseline": 0.65,
                "bounds": [0.585, 0.715],
                "units": "fraction",
                "description": "NYC delivery factor at drought level 5 (Emergency)"
            },
        }
    },
    "individual_mrf": {
        "enabled": True,
        "description": "Individual reservoir MRF baselines",
        "parameters": {
            "mrf_cannonsville": {
                "baseline": 122.8,
                "bounds": [110.52, 135.08],
                "units": "MGD",
                "description": "Cannonsville MRF baseline"
            },
            "mrf_pepacton": {
                "baseline": 64.63,
                "bounds": [58.17, 71.09],
                "units": "MGD",
                "description": "Pepacton MRF baseline"
            },
            "mrf_neversink": {
                "baseline": 48.47,
                "bounds": [43.62, 53.32],
                "units": "MGD",
                "description": "Neversink MRF baseline"
            },
        }
    },
    "flow_target_mrf": {
        "enabled": False,  # Disabled by default
        "description": "Downstream flow target MRF baselines",
        "parameters": {
            "mrf_montague": {
                "baseline": 1131.05,
                "bounds": [1017.95, 1244.16],
                "units": "MGD",
                "description": "Montague flow target baseline"
            },
            "mrf_trenton": {
                "baseline": 1938.95,
                "bounds": [1745.06, 2132.85],
                "units": "MGD",
                "description": "Trenton flow target baseline"
            },
        }
    },
    "mrf_factor_profiles": {
        "enabled": True,
        "description": "MRF factor profile seasonal modifications",
        "parameters": {
            # Season boundary shifts (days)
            "mrf_summer_start_shift": {
                "baseline": 0,
                "bounds": [-15, 15],
                "units": "days",
                "description": "Shift summer season start (Jun 1 default)"
            },
            "mrf_fall_start_shift": {
                "baseline": 0,
                "bounds": [-15, 15],
                "units": "days",
                "description": "Shift fall season start (Sep 1 default)"
            },
            "mrf_winter_start_shift": {
                "baseline": 0,
                "bounds": [-15, 15],
                "units": "days",
                "description": "Shift winter season start (Dec 1 default)"
            },
            "mrf_spring_start_shift": {
                "baseline": 0,
                "bounds": [-15, 15],
                "units": "days",
                "description": "Shift spring season start (May 1 default)"
            },
            # Season scaling factors
            "mrf_summer_scale": {
                "baseline": 1.0,
                "bounds": [0.9, 1.1],
                "units": "fraction",
                "description": "Scale factor for summer MRF releases"
            },
            "mrf_fall_scale": {
                "baseline": 1.0,
                "bounds": [0.9, 1.1],
                "units": "fraction",
                "description": "Scale factor for fall MRF releases"
            },
            "mrf_winter_scale": {
                "baseline": 1.0,
                "bounds": [0.9, 1.1],
                "units": "fraction",
                "description": "Scale factor for winter MRF releases"
            },
            "mrf_spring_scale": {
                "baseline": 1.0,
                "bounds": [0.9, 1.1],
                "units": "fraction",
                "description": "Scale factor for spring MRF releases"
            },
        }
    },
    "flood": {
        "enabled": True,
        "description": "Flood control release limits",
        "parameters": {
            "flood_max_cannonsville": {
                "baseline": 4200.0,
                "bounds": [3780.0, 4620.0],
                "units": "CFS",
                "description": "Max flood release Cannonsville"
            },
            "flood_max_pepacton": {
                "baseline": 2400.0,
                "bounds": [2160.0, 2640.0],
                "units": "CFS",
                "description": "Max flood release Pepacton"
            },
            "flood_max_neversink": {
                "baseline": 3400.0,
                "bounds": [3060.0, 3740.0],
                "units": "CFS",
                "description": "Max flood release Neversink"
            },
        }
    },
    "storage_zones": {
        "enabled": True,
        "description": "Storage zone threshold adjustments (per-level)",
        "parameters": {
            # Vertical shifts for each level (fraction of capacity)
            "zone_level1b_vertical_shift": {
                "baseline": 0.0,
                "bounds": [-0.05, 0.05],
                "units": "fraction",
                "description": "Vertical shift for level1b (Flood Warning) zone threshold"
            },
            "zone_level1c_vertical_shift": {
                "baseline": 0.0,
                "bounds": [-0.05, 0.05],
                "units": "fraction",
                "description": "Vertical shift for level1c (Flood Watch) zone threshold"
            },
            "zone_level2_vertical_shift": {
                "baseline": 0.0,
                "bounds": [-0.05, 0.05],
                "units": "fraction",
                "description": "Vertical shift for level2 (Normal) zone threshold"
            },
            "zone_level3_vertical_shift": {
                "baseline": 0.0,
                "bounds": [-0.05, 0.05],
                "units": "fraction",
                "description": "Vertical shift for level3 (Drought Watch) zone threshold"
            },
            "zone_level4_vertical_shift": {
                "baseline": 0.0,
                "bounds": [-0.05, 0.05],
                "units": "fraction",
                "description": "Vertical shift for level4 (Drought Warning) zone threshold"
            },
            "zone_level5_vertical_shift": {
                "baseline": 0.0,
                "bounds": [-0.05, 0.05],
                "units": "fraction",
                "description": "Vertical shift for level5 (Drought Emergency) zone threshold"
            },
            # Time shifts for each level (days)
            "zone_level1b_time_shift": {
                "baseline": 0,
                "bounds": [-30, 30],
                "units": "days",
                "description": "Time shift for level1b zone seasonal pattern"
            },
            "zone_level1c_time_shift": {
                "baseline": 0,
                "bounds": [-30, 30],
                "units": "days",
                "description": "Time shift for level1c zone seasonal pattern"
            },
            "zone_level2_time_shift": {
                "baseline": 0,
                "bounds": [-30, 30],
                "units": "days",
                "description": "Time shift for level2 zone seasonal pattern"
            },
            "zone_level3_time_shift": {
                "baseline": 0,
                "bounds": [-30, 30],
                "units": "days",
                "description": "Time shift for level3 zone seasonal pattern"
            },
            "zone_level4_time_shift": {
                "baseline": 0,
                "bounds": [-30, 30],
                "units": "days",
                "description": "Time shift for level4 zone seasonal pattern"
            },
            "zone_level5_time_shift": {
                "baseline": 0,
                "bounds": [-30, 30],
                "units": "days",
                "description": "Time shift for level5 zone seasonal pattern"
            },
        }
    },
}

# =============================================================================
# STORAGE ZONE HIERARCHY (for crossing constraints)
# =============================================================================
# Order from least severe to most severe
# At any given day, thresholds must maintain: level1b >= level1c >= level2 >= level3 >= level4 >= level5
ZONE_LEVELS_ORDERED = ["level1b", "level1c", "level2", "level3", "level4", "level5"]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_active_parameters():
    """
    Return dict of all enabled parameters for SALib problem definition.

    Returns
    -------
    dict
        Dictionary mapping parameter names to their definitions
    """
    active = {}
    for group_name, group in PARAMETER_GROUPS.items():
        if group["enabled"]:
            for param_name, param_def in group["parameters"].items():
                active[param_name] = param_def
    return active


def get_salib_problem(parameter_subset=None):
    """
    Generate SALib problem definition.

    Parameters
    ----------
    parameter_subset : list, optional
        List of parameter names to include. If None, use all active parameters.

    Returns
    -------
    dict
        SALib problem definition with keys: num_vars, names, bounds
    """
    if parameter_subset is None:
        active_params = get_active_parameters()
        parameter_subset = list(active_params.keys())
    else:
        active_params = get_active_parameters()

    names = []
    bounds = []

    for param_name in parameter_subset:
        if param_name not in active_params:
            raise ValueError(f"Unknown or disabled parameter: {param_name}")
        param = active_params[param_name]
        names.append(param_name)
        bounds.append(param["bounds"])

    return {
        "num_vars": len(names),
        "names": names,
        "bounds": bounds
    }


def get_parameter_group(param_name):
    """
    Get the group name for a given parameter.

    Parameters
    ----------
    param_name : str
        Parameter name

    Returns
    -------
    str or None
        Group name or None if not found
    """
    for group_name, group in PARAMETER_GROUPS.items():
        if param_name in group["parameters"]:
            return group_name
    return None


def print_config_summary():
    """Print a summary of the current configuration."""
    print("=" * 60)
    print("NYC OPERATIONS SENSITIVITY ANALYSIS CONFIGURATION")
    print("=" * 60)

    print(f"\nSimulation Period: {START_DATE} to {END_DATE}")
    print(f"Inflow Type: {INFLOW_TYPE}")
    print(f"Sobol Samples (N): {N_SOBOL_SAMPLES}")

    active_params = get_active_parameters()
    n_params = len(active_params)
    n_total_samples = N_SOBOL_SAMPLES * (2 * n_params + 2)

    print(f"\nActive Parameters: {n_params}")
    print(f"Total Simulations: {n_total_samples}")

    print("\nParameter Groups:")
    for group_name, group in PARAMETER_GROUPS.items():
        status = "ENABLED" if group["enabled"] else "DISABLED"
        n_group_params = len(group["parameters"])
        print(f"  {group_name}: {status} ({n_group_params} parameters)")

        if group["enabled"]:
            for param_name, param_def in group["parameters"].items():
                bounds = param_def["bounds"]
                print(f"    - {param_name}: [{bounds[0]}, {bounds[1]}] {param_def['units']}")

    print("=" * 60)


# =============================================================================
# METRICS CONFIGURATION
# =============================================================================

# List of metrics to calculate (can be extended)
METRICS_TO_CALCULATE = [
    "montague_min_flow_mgd",
    "pct_time_drought_watch",
    "pct_time_drought_warning",
    "pct_time_drought_emergency",
    "nyc_min_storage_pct",
]


if __name__ == "__main__":
    # Print configuration summary when run directly
    print_config_summary()
