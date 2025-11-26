"""
Sobol sampling and configuration generation for sensitivity analysis.

This module provides:
- Sobol sequence generation using SALib
- Conversion of samples to NYCOperationsConfig objects
- Sample I/O utilities
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from SALib.sample import sobol as sobol_sample

from pywrdrb.parameters.nyc_operations_config import NYCOperationsConfig

from config import (
    PARAMETER_GROUPS,
    get_salib_problem,
    get_active_parameters,
    SAMPLES_DIR,
    RANDOM_SEED,
    ZONE_LEVELS_ORDERED
)


def generate_sobol_samples(n_samples: int, parameter_subset=None, seed=None) -> tuple:
    """
    Generate Sobol sequence samples for sensitivity analysis.

    Parameters
    ----------
    n_samples : int
        Number of base samples (N). Total samples = N*(2D+2) where D is parameter count.
    parameter_subset : list, optional
        List of parameter names to include. If None, use all active parameters.
    seed : int, optional
        Random seed for reproducibility. If None, uses RANDOM_SEED from config.

    Returns
    -------
    tuple : (samples array, problem dict)
        samples: np.ndarray of shape (n_total_samples, n_parameters)
        problem: SALib problem definition dictionary
    """
    if seed is None:
        seed = RANDOM_SEED

    problem = get_salib_problem(parameter_subset)

    samples = sobol_sample.sample(
        problem,
        n_samples,
        calc_second_order=True,
        seed=seed
    )

    return samples, problem


def save_samples(samples: np.ndarray, problem: dict, filename: str = "sobol"):
    """
    Save samples and problem definition to files.

    Parameters
    ----------
    samples : np.ndarray
        Sobol sample array
    problem : dict
        SALib problem definition
    filename : str
        Base filename (without extension)
    """
    # Save samples as CSV
    df = pd.DataFrame(samples, columns=problem["names"])
    df.index.name = "sample_id"
    df.to_csv(SAMPLES_DIR / f"{filename}_samples.csv")

    # Save problem definition as JSON
    problem_json = {
        "num_vars": problem["num_vars"],
        "names": problem["names"],
        "bounds": [list(b) for b in problem["bounds"]]  # Convert to list for JSON
    }
    with open(SAMPLES_DIR / f"{filename}_problem.json", "w") as f:
        json.dump(problem_json, f, indent=2)

    print(f"Saved {len(samples)} samples to {SAMPLES_DIR / f'{filename}_samples.csv'}")
    print(f"Saved problem definition to {SAMPLES_DIR / f'{filename}_problem.json'}")


def load_samples(filename: str = "sobol") -> tuple:
    """
    Load samples and problem definition from files.

    Parameters
    ----------
    filename : str
        Base filename (without extension)

    Returns
    -------
    tuple : (samples array, problem dict)
    """
    # Load samples
    df = pd.read_csv(SAMPLES_DIR / f"{filename}_samples.csv", index_col="sample_id")
    samples = df.values

    # Load problem definition
    with open(SAMPLES_DIR / f"{filename}_problem.json", "r") as f:
        problem = json.load(f)

    return samples, problem


def sample_to_nyc_config(sample_values: np.ndarray, problem: dict) -> NYCOperationsConfig:
    """
    Convert a single Sobol sample to NYCOperationsConfig.

    Parameters
    ----------
    sample_values : np.ndarray
        Parameter values for one sample (length D).
    problem : dict
        SALib problem definition with parameter names.

    Returns
    -------
    NYCOperationsConfig : Configured operations object
    """
    # Start with default configuration
    config = NYCOperationsConfig.from_defaults()

    # Build parameter dictionary from sample
    param_dict = {name: value for name, value in zip(problem["names"], sample_values)}

    # =========================================================================
    # Apply delivery constraint modifications
    # =========================================================================
    delivery_updates = {}

    if "max_nyc_delivery" in param_dict:
        delivery_updates["max_nyc_delivery"] = param_dict["max_nyc_delivery"]

    if "max_nj_daily" in param_dict:
        delivery_updates["max_nj_daily"] = param_dict["max_nj_daily"]

    # Handle drought factors for NYC
    # Default baseline factors: [1e6, 1e6, 1e6, 1e6, 0.85, 0.7, 0.65] for levels 1a, 1b, 1c, 2, 3, 4, 5
    if any(k.startswith("drought_factor_nyc") for k in param_dict):
        # Get current factors from config
        baseline_factors = []
        for level in ['level1a', 'level1b', 'level1c', 'level2', 'level3', 'level4', 'level5']:
            factor = config.get_constant(f'{level}_factor_delivery_nyc', 1.0)
            baseline_factors.append(factor)
        baseline_factors = np.array(baseline_factors)

        # Apply modifications to specific levels
        # Note: level3 = index 4, level4 = index 5, level5 = index 6
        if "drought_factor_nyc_level3" in param_dict:
            baseline_factors[4] = param_dict["drought_factor_nyc_level3"]
        if "drought_factor_nyc_level4" in param_dict:
            baseline_factors[5] = param_dict["drought_factor_nyc_level4"]
        if "drought_factor_nyc_level5" in param_dict:
            baseline_factors[6] = param_dict["drought_factor_nyc_level5"]

        delivery_updates["drought_factors_nyc"] = baseline_factors

    if delivery_updates:
        config.update_delivery_constraints(**delivery_updates)

    # =========================================================================
    # Apply MRF baseline modifications
    # =========================================================================
    mrf_updates = {}

    if "mrf_cannonsville" in param_dict:
        mrf_updates["cannonsville"] = param_dict["mrf_cannonsville"]
    if "mrf_pepacton" in param_dict:
        mrf_updates["pepacton"] = param_dict["mrf_pepacton"]
    if "mrf_neversink" in param_dict:
        mrf_updates["neversink"] = param_dict["mrf_neversink"]
    if "mrf_montague" in param_dict:
        mrf_updates["montague"] = param_dict["mrf_montague"]
    if "mrf_trenton" in param_dict:
        mrf_updates["trenton"] = param_dict["mrf_trenton"]

    if mrf_updates:
        config.update_mrf_baselines(**mrf_updates)

    # =========================================================================
    # Apply flood limit modifications
    # =========================================================================
    flood_updates = {}

    if "flood_max_cannonsville" in param_dict:
        flood_updates["max_release_cannonsville"] = param_dict["flood_max_cannonsville"]
    if "flood_max_pepacton" in param_dict:
        flood_updates["max_release_pepacton"] = param_dict["flood_max_pepacton"]
    if "flood_max_neversink" in param_dict:
        flood_updates["max_release_neversink"] = param_dict["flood_max_neversink"]

    if flood_updates:
        config.update_flood_limits(**flood_updates)

    # =========================================================================
    # Apply MRF factor profile modifications (seasonal shifts and scales)
    # =========================================================================
    from methods.mrf_profiles import (
        MRFProfileBuilder,
        build_season_params_from_sample,
        has_mrf_profile_params,
        RESERVOIRS,
        DROUGHT_LEVELS
    )

    if has_mrf_profile_params(param_dict):
        season_shifts, season_scales = build_season_params_from_sample(param_dict)

        # Get base profiles from config
        base_profiles = {}
        for reservoir in RESERVOIRS:
            for level in DROUGHT_LEVELS:
                profile_name = f"{level}_factor_mrf_{reservoir}"
                base_profiles[profile_name] = config.get_mrf_factor_profile(profile_name)

        # Build modified profiles
        builder = MRFProfileBuilder(base_profiles)
        modified_profiles = builder.build_all_modified_profiles(season_shifts, season_scales)

        # Apply to config
        for profile_name, profile_values in modified_profiles.items():
            parts = profile_name.split("_factor_mrf_")
            level = parts[0]
            reservoir = parts[1]
            config.update_mrf_factors(
                reservoir=reservoir,
                level=level,
                daily_factors=profile_values
            )

    # =========================================================================
    # Apply storage zone modifications (per-level vertical + time shifts)
    # with zone crossing constraints
    # =========================================================================
    zone_params_present = any(
        f"zone_{level}_vertical_shift" in param_dict or
        f"zone_{level}_time_shift" in param_dict
        for level in ZONE_LEVELS_ORDERED
    )

    if zone_params_present:
        zone_profiles = {}

        # Apply independent shifts to each level
        for level in ZONE_LEVELS_ORDERED:
            v_shift_key = f"zone_{level}_vertical_shift"
            t_shift_key = f"zone_{level}_time_shift"

            profile = config.get_storage_zone_profile(level)

            # Apply time shift first (positive = later in year)
            t_shift = int(round(param_dict.get(t_shift_key, 0)))
            if t_shift != 0:
                profile = np.roll(profile, t_shift)

            # Apply vertical shift and clip to valid range [0, 1]
            v_shift = param_dict.get(v_shift_key, 0.0)
            profile = np.clip(profile + v_shift, 0, 1)

            zone_profiles[level] = profile

        # Enforce zone crossing constraints
        # Hierarchy: level1b >= level1c >= level2 >= level3 >= level4 >= level5
        # When zones intersect, use the lower (less severe) zone
        for i in range(1, len(ZONE_LEVELS_ORDERED)):
            less_severe = ZONE_LEVELS_ORDERED[i-1]
            more_severe = ZONE_LEVELS_ORDERED[i]
            zone_profiles[more_severe] = np.minimum(
                zone_profiles[more_severe],
                zone_profiles[less_severe]
            )

        # Update config with constrained profiles
        for level in ZONE_LEVELS_ORDERED:
            config.update_storage_zones(level=level, daily_values=zone_profiles[level])

    return config


def get_sample_config_batch(samples: np.ndarray, problem: dict,
                            start_idx: int, end_idx: int) -> list:
    """
    Generate batch of NYCOperationsConfig objects for sample range.

    Parameters
    ----------
    samples : np.ndarray
        Full sample array
    problem : dict
        SALib problem definition
    start_idx : int
        Starting sample index (inclusive)
    end_idx : int
        Ending sample index (exclusive)

    Returns
    -------
    list : List of (sample_id, NYCOperationsConfig) tuples
    """
    configs = []
    for i in range(start_idx, min(end_idx, len(samples))):
        config = sample_to_nyc_config(samples[i], problem)
        configs.append((i, config))
    return configs


def print_sample_summary(samples: np.ndarray, problem: dict):
    """
    Print summary statistics for generated samples.

    Parameters
    ----------
    samples : np.ndarray
        Sobol sample array
    problem : dict
        SALib problem definition
    """
    n_samples, n_params = samples.shape

    print("\n" + "=" * 60)
    print("SOBOL SAMPLE SUMMARY")
    print("=" * 60)
    print(f"Total samples: {n_samples}")
    print(f"Parameters: {n_params}")
    print(f"Base N: {n_samples // (2 * n_params + 2)}")
    print()

    print("Parameter ranges in samples:")
    print("-" * 60)
    for i, name in enumerate(problem["names"]):
        bounds = problem["bounds"][i]
        actual_min = samples[:, i].min()
        actual_max = samples[:, i].max()
        print(f"  {name}:")
        print(f"    Bounds: [{bounds[0]:.4f}, {bounds[1]:.4f}]")
        print(f"    Actual: [{actual_min:.4f}, {actual_max:.4f}]")
    print("=" * 60)


if __name__ == "__main__":
    # Test sample generation
    from config import N_SOBOL_SAMPLES

    print("Testing Sobol sample generation...")

    samples, problem = generate_sobol_samples(N_SOBOL_SAMPLES)
    print_sample_summary(samples, problem)

    # Test conversion to config (first sample)
    print("\nTesting sample-to-config conversion...")
    config = sample_to_nyc_config(samples[0], problem)
    print(f"Successfully created NYCOperationsConfig from sample 0")
    print(f"  NYC delivery limit: {config.get_constant('max_flow_baseline_delivery_nyc')}")
