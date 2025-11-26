"""
Sobol sensitivity analysis using SALib.

This module provides:
- First and total order Sobol indices calculation
- Second order interaction indices
- Bootstrap confidence intervals
- Results storage and formatting
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from SALib.analyze import sobol

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import ANALYSIS_DIR, METRICS_TO_CALCULATE


def calculate_sobol_indices(samples: np.ndarray,
                            metrics_df: pd.DataFrame,
                            problem: dict,
                            metric_columns: list = None,
                            calc_second_order: bool = True,
                            n_bootstrap: int = 1000,
                            confidence_level: float = 0.95) -> dict:
    """
    Calculate Sobol sensitivity indices for all metrics.

    Parameters
    ----------
    samples : np.ndarray
        Original Sobol samples from SALib (not used directly, but needed for validation)
    metrics_df : pd.DataFrame
        DataFrame with calculated metrics (must have sample_id column)
    problem : dict
        SALib problem definition
    metric_columns : list, optional
        List of metric column names to analyze. Default: METRICS_TO_CALCULATE
    calc_second_order : bool
        If True, calculate second-order interaction indices
    n_bootstrap : int
        Number of bootstrap resamples for confidence intervals
    confidence_level : float
        Confidence level for intervals (0-1)

    Returns
    -------
    dict : Results organized by metric name
        {
            "metric_name": {
                "S1": first order indices (array),
                "S1_conf": confidence intervals (array),
                "ST": total order indices (array),
                "ST_conf": confidence intervals (array),
                "S2": second order indices (matrix, if calc_second_order),
                "S2_conf": confidence intervals (matrix, if calc_second_order),
                "parameter_names": list of parameter names
            }
        }
    """
    if metric_columns is None:
        metric_columns = [m for m in METRICS_TO_CALCULATE if m in metrics_df.columns]

    # Ensure metrics are aligned with samples by sample_id
    if "sample_id" in metrics_df.columns:
        metrics_df = metrics_df.set_index("sample_id").sort_index()

    n_samples = len(samples)
    n_metrics = len(metrics_df)

    if n_samples != n_metrics:
        raise ValueError(
            f"Sample count ({n_samples}) does not match metrics count ({n_metrics}). "
            "Ensure all simulations completed successfully."
        )

    results = {}

    print(f"Calculating Sobol indices for {len(metric_columns)} metrics...")
    print(f"  Samples: {n_samples}")
    print(f"  Parameters: {problem['num_vars']}")
    print(f"  Second order: {calc_second_order}")
    print(f"  Bootstrap samples: {n_bootstrap}")
    print()

    for metric in metric_columns:
        if metric not in metrics_df.columns:
            print(f"  Skipping {metric}: not found in metrics")
            continue

        Y = metrics_df[metric].values

        # Check for valid values
        valid_mask = ~np.isnan(Y) & ~np.isinf(Y)
        n_invalid = (~valid_mask).sum()

        if n_invalid > 0:
            print(f"  Warning: {metric} has {n_invalid} invalid values")
            # Replace with median for stability
            Y = Y.copy()
            Y[~valid_mask] = np.nanmedian(Y)

        # Check for constant values (would cause division by zero)
        if np.std(Y) < 1e-10:
            print(f"  Skipping {metric}: constant values (no variance)")
            results[metric] = {"error": "constant values"}
            continue

        try:
            Si = sobol.analyze(
                problem,
                Y,
                calc_second_order=calc_second_order,
                num_resamples=n_bootstrap,
                conf_level=confidence_level,
                print_to_console=False
            )

            results[metric] = {
                "S1": Si["S1"],
                "S1_conf": Si["S1_conf"],
                "ST": Si["ST"],
                "ST_conf": Si["ST_conf"],
                "parameter_names": problem["names"]
            }

            if calc_second_order:
                results[metric]["S2"] = Si["S2"]
                results[metric]["S2_conf"] = Si["S2_conf"]

            print(f"  {metric}: S1 sum = {Si['S1'].sum():.3f}, ST sum = {Si['ST'].sum():.3f}")

        except Exception as e:
            print(f"  Error analyzing {metric}: {e}")
            results[metric] = {"error": str(e)}

    return results


def format_sobol_results(sobol_results: dict) -> pd.DataFrame:
    """
    Format Sobol results into a tidy DataFrame for easy analysis.

    Parameters
    ----------
    sobol_results : dict
        Output from calculate_sobol_indices

    Returns
    -------
    pd.DataFrame : Columns [metric, parameter, S1, S1_conf, ST, ST_conf]
    """
    rows = []

    for metric, indices in sobol_results.items():
        if "error" in indices:
            continue

        param_names = indices["parameter_names"]

        for i, param in enumerate(param_names):
            row = {
                "metric": metric,
                "parameter": param,
                "S1": indices["S1"][i],
                "S1_conf": indices["S1_conf"][i],
                "ST": indices["ST"][i],
                "ST_conf": indices["ST_conf"][i],
            }
            rows.append(row)

    return pd.DataFrame(rows)


def get_interaction_matrix(sobol_results: dict, metric: str) -> pd.DataFrame:
    """
    Extract second-order interaction matrix for a specific metric.

    Parameters
    ----------
    sobol_results : dict
        Sobol analysis results
    metric : str
        Metric name

    Returns
    -------
    pd.DataFrame : Symmetric matrix of S2 indices
    """
    if metric not in sobol_results:
        raise ValueError(f"Metric {metric} not found in results")

    indices = sobol_results[metric]

    if "error" in indices:
        raise ValueError(f"Results contain error for {metric}: {indices['error']}")

    if "S2" not in indices:
        raise ValueError(f"Second-order indices not calculated for {metric}")

    param_names = indices["parameter_names"]

    S2_matrix = pd.DataFrame(
        indices["S2"],
        index=param_names,
        columns=param_names
    )

    return S2_matrix


def rank_parameters(sobol_results: dict,
                    metric: str,
                    index_type: str = "ST") -> pd.DataFrame:
    """
    Rank parameters by sensitivity index for a specific metric.

    Parameters
    ----------
    sobol_results : dict
        Sobol analysis results
    metric : str
        Metric name
    index_type : str
        "S1" for first-order or "ST" for total-order (default)

    Returns
    -------
    pd.DataFrame : Ranked parameters with indices and confidence bounds
    """
    if metric not in sobol_results or "error" in sobol_results[metric]:
        raise ValueError(f"Valid results not found for {metric}")

    indices = sobol_results[metric]

    df = pd.DataFrame({
        "parameter": indices["parameter_names"],
        "index": indices[index_type],
        "conf": indices[f"{index_type}_conf"]
    })

    df["lower"] = df["index"] - df["conf"]
    df["upper"] = df["index"] + df["conf"]

    df = df.sort_values("index", ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)

    return df


def save_sobol_results(sobol_results: dict, filename: str = "sobol"):
    """
    Save Sobol results to files.

    Parameters
    ----------
    sobol_results : dict
        Output from calculate_sobol_indices
    filename : str
        Base filename (without extension)
    """
    # Save formatted results as CSV
    formatted = format_sobol_results(sobol_results)
    formatted.to_csv(ANALYSIS_DIR / f"{filename}_indices.csv", index=False)

    # Save raw results as JSON (for interaction matrices, etc.)
    json_results = {}
    for metric, indices in sobol_results.items():
        if "error" in indices:
            json_results[metric] = indices
        else:
            json_results[metric] = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in indices.items()
            }

    with open(ANALYSIS_DIR / f"{filename}_raw.json", "w") as f:
        json.dump(json_results, f, indent=2)

    print(f"Saved Sobol results to {ANALYSIS_DIR}")
    print(f"  - {filename}_indices.csv (formatted)")
    print(f"  - {filename}_raw.json (raw)")


def load_sobol_results(filename: str = "sobol") -> tuple:
    """
    Load Sobol results from files.

    Parameters
    ----------
    filename : str
        Base filename (without extension)

    Returns
    -------
    tuple : (formatted DataFrame, raw dict)
    """
    # Load formatted results
    formatted = pd.read_csv(ANALYSIS_DIR / f"{filename}_indices.csv")

    # Load raw results
    with open(ANALYSIS_DIR / f"{filename}_raw.json", "r") as f:
        raw = json.load(f)

    # Convert lists back to numpy arrays
    for metric in raw:
        if "error" not in raw[metric]:
            for key in ["S1", "S1_conf", "ST", "ST_conf"]:
                if key in raw[metric]:
                    raw[metric][key] = np.array(raw[metric][key])
            if "S2" in raw[metric]:
                raw[metric]["S2"] = np.array(raw[metric]["S2"])
                raw[metric]["S2_conf"] = np.array(raw[metric]["S2_conf"])

    return formatted, raw


def print_sobol_summary(sobol_results: dict, top_n: int = 5):
    """
    Print summary of Sobol analysis results.

    Parameters
    ----------
    sobol_results : dict
        Output from calculate_sobol_indices
    top_n : int
        Number of top parameters to show per metric
    """
    print("\n" + "=" * 70)
    print("SOBOL SENSITIVITY ANALYSIS SUMMARY")
    print("=" * 70)

    for metric, indices in sobol_results.items():
        print(f"\n{metric}:")
        print("-" * 50)

        if "error" in indices:
            print(f"  ERROR: {indices['error']}")
            continue

        # Get ranked parameters
        try:
            ranked = rank_parameters(sobol_results, metric, "ST")
            print(f"  Top {top_n} influential parameters (Total Order):")
            for _, row in ranked.head(top_n).iterrows():
                print(f"    {row['rank']}. {row['parameter']}: "
                      f"ST = {row['index']:.4f} "
                      f"[{row['lower']:.4f}, {row['upper']:.4f}]")
        except Exception as e:
            print(f"  Could not rank parameters: {e}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    # Test loading and summarizing results (if available)
    try:
        formatted, raw = load_sobol_results()
        print_sobol_summary(raw)
    except FileNotFoundError:
        print("No Sobol results found. Run 04_analyze_sensitivity.py first.")
