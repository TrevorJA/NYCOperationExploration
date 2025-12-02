"""
Calculate performance metrics for sensitivity analysis.

This module provides:
- Modular metric extraction from simulation outputs
- Aggregation across samples
- Metric storage utilities

Metrics are designed to be easily extensible - just add new functions
and register them in METRIC_FUNCTIONS.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add parent directory to path for imports when running as script
sys.path.insert(0, str(Path(__file__).parent.parent))

import pywrdrb

from config import (
    SIMULATIONS_DIR,
    METRICS_DIR,
    NYC_RESERVOIRS,
    NYC_TOTAL_CAPACITY,
    ZONE_NAMES,
    METRICS_TO_CALCULATE
)


# =============================================================================
# METRIC CALCULATION FUNCTIONS
# =============================================================================

def calculate_montague_flow_3day_min(data: dict) -> float:
    """
    Calculate minimum 3-day rolling average flow at Montague.

    Parameters
    ----------
    data : dict
        Dictionary containing loaded simulation data

    Returns
    -------
    float : Minimum 3-day rolling average Montague flow in MGD
    """
    montague_flow = data["major_flow"]["delMontague"]
    rolling_3day = montague_flow.rolling(window=3, min_periods=3).mean()
    return float(rolling_3day.min())


def calculate_max_nyc_monthly_shortage_pct(data: dict) -> float:
    """
    Calculate maximum monthly NYC shortage as percentage of demand.

    Shortage = max(0, demand - diversion). Returns maximum monthly shortage
    as a percentage of total monthly demand.

    Parameters
    ----------
    data : dict
        Dictionary containing loaded simulation data

    Returns
    -------
    float : Maximum monthly shortage as percentage of demand
    """
    nyc_demand = data["ibt_demands"]["demand_nyc"]
    nyc_diversion = data["ibt_diversions"]["delivery_nyc"]

    # Calculate daily shortage (demand - diversion, floored at 0)
    daily_shortage = (nyc_demand - nyc_diversion).clip(lower=0)

    # Resample to monthly totals
    monthly_shortage = daily_shortage.resample('M').sum()
    monthly_demand = nyc_demand.resample('M').sum()

    # Calculate shortage as percentage of demand
    # Avoid division by zero
    monthly_shortage_pct = 100.0 * monthly_shortage / monthly_demand.replace(0, np.nan)

    return float(monthly_shortage_pct.max())


def calculate_pct_time_drought_emergency(data: dict) -> float:
    """
    Calculate percentage of time in Drought Emergency (zone 6).

    Parameters
    ----------
    data : dict
        Dictionary containing loaded simulation data

    Returns
    -------
    float : Percentage of days in Drought Emergency
    """
    nyc_zone = data["res_level"]["nyc"]
    n_days = len(nyc_zone)
    n_emergency = (nyc_zone == 6).sum()
    return 100.0 * float(n_emergency) / n_days


def calculate_nyc_min_storage_pct(data: dict) -> float:
    """
    Calculate minimum NYC combined storage as percentage of capacity.

    Parameters
    ----------
    data : dict
        Dictionary containing loaded simulation data

    Returns
    -------
    float : Minimum storage percentage
    """
    nyc_storage = data["res_storage"][NYC_RESERVOIRS].sum(axis=1)
    min_storage = nyc_storage.min()
    return 100.0 * float(min_storage) / NYC_TOTAL_CAPACITY


# =============================================================================
# METRIC REGISTRY
# =============================================================================

# Register all metric functions here
# Key = metric name, Value = function that takes data dict and returns float
METRIC_FUNCTIONS = {
    "montague_flow_3day_min_mgd": calculate_montague_flow_3day_min,
    "nyc_min_storage_pct": calculate_nyc_min_storage_pct,
    "max_nyc_monthly_shortage_pct": calculate_max_nyc_monthly_shortage_pct,
    "pct_time_drought_emergency": calculate_pct_time_drought_emergency,
}


# =============================================================================
# DATA LOADING
# =============================================================================

def load_simulation_data(output_file: str, start_date: str = None, end_date: str = None,
                         warmup_days: int = 7) -> dict:
    """
    Load simulation output data needed for metric calculations.

    Parameters
    ----------
    output_file : str
        Path to HDF5 output file
    start_date : str, optional
        Start date to filter data (format: YYYY-MM-DD). If None, use all data.
    end_date : str, optional
        End date to filter data (format: YYYY-MM-DD). If None, use all data.
    warmup_days : int, optional
        Number of days to skip at the start for model warmup. Default is 7.

    Returns
    -------
    dict : Dictionary containing loaded data series
    """
    import pandas as pd
    from datetime import timedelta

    # Load data using pywrdrb
    data_obj = pywrdrb.Data()

    # Load required results sets
    results_sets = ['major_flow', 'res_storage', 'res_level', 'ibt_diversions', 'ibt_demands']

    data_obj.load_output(
        output_filenames=[output_file],
        results_sets=results_sets
    )

    # Get the dataset key (should be only one)
    dataset_key = list(data_obj.major_flow.keys())[0]
    realization = 0  # Single realization for sensitivity analysis

    # Extract relevant data series
    data = {
        "major_flow": data_obj.major_flow[dataset_key][realization],
        "res_storage": data_obj.res_storage[dataset_key][realization],
        "res_level": data_obj.res_level[dataset_key][realization],
        "ibt_diversions": data_obj.ibt_diversions[dataset_key][realization],
        "ibt_demands": data_obj.ibt_demands[dataset_key][realization],
    }

    # Determine start date with warmup offset
    if start_date is not None:
        start_dt = pd.to_datetime(start_date) + timedelta(days=warmup_days)
    elif warmup_days > 0:
        # Apply warmup to actual data start
        first_date = data["major_flow"].index.min()
        start_dt = first_date + timedelta(days=warmup_days)
    else:
        start_dt = None

    end_dt = pd.to_datetime(end_date) if end_date else None

    # Filter to date range
    if start_dt is not None or end_dt is not None:
        for key in data:
            df = data[key]
            if start_dt is not None:
                df = df[df.index >= start_dt]
            if end_dt is not None:
                df = df[df.index <= end_dt]
            data[key] = df

    return data


# =============================================================================
# METRIC CALCULATION
# =============================================================================

def calculate_sample_metrics(sample_id: int, output_file: str,
                             metrics: list = None,
                             start_date: str = None,
                             end_date: str = None) -> dict:
    """
    Calculate all performance metrics for a single simulation sample.

    Parameters
    ----------
    sample_id : int
        Sample identifier
    output_file : str
        Path to HDF5 output file
    metrics : list, optional
        List of metric names to calculate. If None, uses METRICS_TO_CALCULATE from config.
    start_date : str, optional
        Start date to filter data (format: YYYY-MM-DD). If None, use all data.
    end_date : str, optional
        End date to filter data (format: YYYY-MM-DD). If None, use all data.

    Returns
    -------
    dict : Dictionary of metric values (includes sample_id)
    """
    if metrics is None:
        metrics = METRICS_TO_CALCULATE

    result = {"sample_id": sample_id}

    try:
        # Load simulation data
        data = load_simulation_data(output_file, start_date=start_date, end_date=end_date)

        # Calculate each metric
        for metric_name in metrics:
            if metric_name not in METRIC_FUNCTIONS:
                print(f"  Warning: Unknown metric '{metric_name}', skipping")
                result[metric_name] = np.nan
                continue

            try:
                value = METRIC_FUNCTIONS[metric_name](data)
                result[metric_name] = value
            except Exception as e:
                print(f"  Error calculating {metric_name} for sample {sample_id}: {e}")
                result[metric_name] = np.nan

    except Exception as e:
        print(f"  Error loading data for sample {sample_id}: {e}")
        for metric_name in metrics:
            result[metric_name] = np.nan

    return result


def calculate_all_metrics(simulation_results: pd.DataFrame = None,
                          metrics: list = None) -> pd.DataFrame:
    """
    Calculate metrics for all completed simulations.

    Parameters
    ----------
    simulation_results : pd.DataFrame, optional
        DataFrame with sample_id and output_file columns.
        If None, scans the simulations directory.
    metrics : list, optional
        List of metric names to calculate.

    Returns
    -------
    pd.DataFrame : Metrics for all samples
    """
    if simulation_results is None:
        # Scan output directory for simulation files
        output_files = sorted(SIMULATIONS_DIR.glob("sample_*.hdf5"))
        simulation_results = pd.DataFrame([
            {
                "sample_id": int(f.stem.split("_")[1]),
                "output_file": str(f),
                "status": "success"
            }
            for f in output_files
        ])

    # Filter to successful simulations
    if "status" in simulation_results.columns:
        successful = simulation_results[simulation_results["status"] == "success"]
    else:
        successful = simulation_results

    n_total = len(successful)
    print(f"Calculating metrics for {n_total} samples...")

    all_metrics = []

    for i, (_, row) in enumerate(successful.iterrows()):
        sample_id = row["sample_id"]
        output_file = row["output_file"]

        if (i + 1) % 50 == 0 or i == 0:
            print(f"  Processing sample {i + 1}/{n_total} (ID {sample_id})")

        metrics_dict = calculate_sample_metrics(sample_id, output_file, metrics)
        all_metrics.append(metrics_dict)

    df = pd.DataFrame(all_metrics)

    # Report summary
    print(f"\nMetrics Summary:")
    for col in df.columns:
        if col != "sample_id":
            valid = df[col].notna().sum()
            print(f"  {col}: {valid}/{n_total} valid values")

    return df


def save_metrics(metrics_df: pd.DataFrame, filename: str = "metrics"):
    """
    Save metrics DataFrame to CSV.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Metrics DataFrame
    filename : str
        Output filename (without extension)
    """
    output_path = METRICS_DIR / f"{filename}.csv"
    metrics_df.to_csv(output_path, index=False)
    print(f"Saved metrics to {output_path}")


def load_metrics(filename: str = "metrics") -> pd.DataFrame:
    """
    Load metrics DataFrame from CSV.

    Parameters
    ----------
    filename : str
        Input filename (without extension)

    Returns
    -------
    pd.DataFrame
    """
    return pd.read_csv(METRICS_DIR / f"{filename}.csv")


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def list_available_metrics():
    """Print list of available metrics."""
    print("Available Metrics:")
    print("-" * 60)
    for name, func in METRIC_FUNCTIONS.items():
        doc = func.__doc__.split("\n")[1].strip() if func.__doc__ else "No description"
        print(f"  {name}")
        print(f"    {doc}")
    print("-" * 60)


def add_custom_metric(name: str, func: callable):
    """
    Add a custom metric function to the registry.

    Parameters
    ----------
    name : str
        Metric name
    func : callable
        Function that takes data dict and returns float
    """
    METRIC_FUNCTIONS[name] = func
    print(f"Added custom metric: {name}")


if __name__ == "__main__":
    # List available metrics
    list_available_metrics()

    # Test metric calculation on a sample file (if exists)
    test_file = SIMULATIONS_DIR / "sample_000000.hdf5"
    if test_file.exists():
        print(f"\nTesting metrics on {test_file}...")
        metrics = calculate_sample_metrics(0, str(test_file))
        print("Results:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
    else:
        print(f"\nNo test file found at {test_file}")
        print("Run simulations first to test metric calculation.")
