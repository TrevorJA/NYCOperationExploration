"""
Run Pywr-DRB simulations for Sobol sensitivity analysis.

This module provides:
- Single simulation execution with custom NYCOperationsConfig
- MPI-based parallel execution across samples
- Results storage utilities
"""

import os
import glob
import math
import numpy as np
from pathlib import Path

import pywrdrb

from config import (
    START_DATE,
    END_DATE,
    INFLOW_TYPE,
    SIMULATIONS_DIR,
    N_SAMPLES_PER_BATCH
)

# Conditional MPI import
try:
    from mpi4py import MPI
    MPI_AVAILABLE = True
except ImportError:
    MPI_AVAILABLE = False


def get_output_filename(sample_id: int) -> str:
    """Get output filename for a sample."""
    return str(SIMULATIONS_DIR / f"sample_{sample_id:06d}.hdf5")


def get_model_filename(sample_id: int) -> str:
    """Get model JSON filename for a sample."""
    models_dir = SIMULATIONS_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    return str(models_dir / f"sample_{sample_id:06d}.json")


def run_single_simulation(sample_id: int, nyc_config,
                          start_date: str = START_DATE,
                          end_date: str = END_DATE,
                          inflow_type: str = INFLOW_TYPE,
                          cleanup_model: bool = True) -> dict:
    """
    Run a single Pywr-DRB simulation with custom NYC operations config.

    Parameters
    ----------
    sample_id : int
        Unique identifier for this sample
    nyc_config : NYCOperationsConfig
        Custom operations configuration
    start_date : str
        Simulation start date
    end_date : str
        Simulation end date
    inflow_type : str
        Type of inflow data to use
    cleanup_model : bool
        If True, delete model JSON file after simulation

    Returns
    -------
    dict : Simulation metadata including output file path and status
    """
    output_file = get_output_filename(sample_id)
    model_file = get_model_filename(sample_id)

    try:
        # Model options
        model_options = {
            "nyc_nj_demand_source": "historical",
        }

        # Build model with custom NYC operations config
        mb = pywrdrb.ModelBuilder(
            inflow_type=inflow_type,
            start_date=start_date,
            end_date=end_date,
            options=model_options,
            nyc_operations_config=nyc_config
        )

        # Make and save model
        mb.make_model()
        mb.write_model(model_file)

        # Load model
        model = pywrdrb.Model.load(model_file)

        # Get parameters to export (subset for efficiency)
        all_parameter_names = [p.name for p in model.parameters if p.name]
        export_param_names = _get_sensitivity_export_parameters(all_parameter_names)
        export_parameters = [p for p in model.parameters if p.name in export_param_names]

        # Setup output recorder
        recorder = pywrdrb.OutputRecorder(
            model=model,
            output_filename=output_file,
            parameters=export_parameters
        )

        # Run simulation
        model.run()

        # Cleanup
        del model
        if cleanup_model and os.path.exists(model_file):
            os.remove(model_file)

        return {
            "sample_id": sample_id,
            "output_file": output_file,
            "status": "success"
        }

    except Exception as e:
        return {
            "sample_id": sample_id,
            "output_file": None,
            "status": f"error: {str(e)}"
        }


def _get_sensitivity_export_parameters(all_parameter_names: list) -> list:
    """
    Get subset of parameters needed for sensitivity metrics.

    Focus on:
    - Major flows (for Montague/Trenton flow metrics)
    - Reservoir storage (for NYC storage metrics)
    - Drought zone levels (for drought metrics)
    - MRF targets and releases
    """
    prefixes = [
        "major_flow_",
        "res_storage_",
        "res_level_",
        "mrf_target_",
        "delivery_",
        "demand_",
    ]

    export_params = []
    for name in all_parameter_names:
        if any(name.startswith(p) for p in prefixes):
            export_params.append(name)

    return export_params


def run_parallel_simulations_mpi(samples: np.ndarray, problem: dict,
                                  start_date: str = START_DATE,
                                  end_date: str = END_DATE,
                                  inflow_type: str = INFLOW_TYPE):
    """
    Run all simulations in parallel using MPI.

    This function should be called from within an MPI context (mpiexec).

    Parameters
    ----------
    samples : np.ndarray
        Full sample array from SALib
    problem : dict
        SALib problem definition
    start_date : str
        Simulation start date
    end_date : str
        Simulation end date
    inflow_type : str
        Type of inflow data

    Returns
    -------
    list or None
        List of result dicts on rank 0, None on other ranks
    """
    from methods.sampling import sample_to_nyc_config

    if not MPI_AVAILABLE:
        raise RuntimeError("MPI not available. Install mpi4py to use parallel execution.")

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    n_samples = len(samples)

    if rank == 0:
        print(f"Starting MPI parallel simulations")
        print(f"  Total samples: {n_samples}")
        print(f"  MPI ranks: {size}")
        print(f"  Samples per rank: ~{n_samples // size}")

    # Distribute samples across ranks
    all_sample_ids = list(range(n_samples))
    my_sample_ids = list(np.array_split(all_sample_ids, size)[rank])

    print(f"Rank {rank}: Processing {len(my_sample_ids)} samples")

    # Run simulations for assigned samples
    local_results = []

    for i, sample_id in enumerate(my_sample_ids):
        # Progress reporting
        if (i + 1) % 10 == 0 or i == 0:
            print(f"Rank {rank}: Sample {i + 1}/{len(my_sample_ids)} (global ID {sample_id})")

        # Generate config from sample
        config = sample_to_nyc_config(samples[sample_id], problem)

        # Run simulation
        result = run_single_simulation(
            sample_id=sample_id,
            nyc_config=config,
            start_date=start_date,
            end_date=end_date,
            inflow_type=inflow_type
        )

        local_results.append(result)

        # Report status
        if result["status"] != "success":
            print(f"Rank {rank}: Sample {sample_id} FAILED - {result['status']}")

    # Synchronize all ranks
    comm.Barrier()

    # Gather results at rank 0
    all_results = comm.gather(local_results, root=0)

    if rank == 0:
        # Flatten list of lists
        combined_results = []
        for rank_results in all_results:
            combined_results.extend(rank_results)

        # Sort by sample_id
        combined_results.sort(key=lambda x: x["sample_id"])

        # Summary
        n_success = sum(1 for r in combined_results if r["status"] == "success")
        n_failed = len(combined_results) - n_success

        print(f"\nSimulation Summary:")
        print(f"  Successful: {n_success}")
        print(f"  Failed: {n_failed}")

        return combined_results
    else:
        return None


def run_simulations_serial(samples: np.ndarray, problem: dict,
                           start_date: str = START_DATE,
                           end_date: str = END_DATE,
                           inflow_type: str = INFLOW_TYPE,
                           sample_ids: list = None) -> list:
    """
    Run simulations serially (for testing or small runs).

    Parameters
    ----------
    samples : np.ndarray
        Full sample array from SALib
    problem : dict
        SALib problem definition
    start_date : str
        Simulation start date
    end_date : str
        Simulation end date
    inflow_type : str
        Type of inflow data
    sample_ids : list, optional
        Specific sample IDs to run. If None, run all samples.

    Returns
    -------
    list : List of result dicts
    """
    from methods.sampling import sample_to_nyc_config

    if sample_ids is None:
        sample_ids = list(range(len(samples)))

    n_samples = len(sample_ids)
    print(f"Running {n_samples} simulations serially...")

    results = []

    for i, sample_id in enumerate(sample_ids):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Sample {i + 1}/{n_samples} (ID {sample_id})")

        # Generate config from sample
        config = sample_to_nyc_config(samples[sample_id], problem)

        # Run simulation
        result = run_single_simulation(
            sample_id=sample_id,
            nyc_config=config,
            start_date=start_date,
            end_date=end_date,
            inflow_type=inflow_type
        )

        results.append(result)

        if result["status"] != "success":
            print(f"  Sample {sample_id} FAILED - {result['status']}")

    # Summary
    n_success = sum(1 for r in results if r["status"] == "success")
    n_failed = len(results) - n_success

    print(f"\nSimulation Summary:")
    print(f"  Successful: {n_success}")
    print(f"  Failed: {n_failed}")

    return results


def save_simulation_results(results: list, filename: str = "simulation_results"):
    """
    Save simulation results metadata to CSV.

    Parameters
    ----------
    results : list
        List of result dicts from simulation runs
    filename : str
        Output filename (without extension)
    """
    import pandas as pd

    df = pd.DataFrame(results)
    output_path = SIMULATIONS_DIR / f"{filename}.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved simulation results to {output_path}")


def load_simulation_results(filename: str = "simulation_results"):
    """
    Load simulation results metadata from CSV.

    Parameters
    ----------
    filename : str
        Input filename (without extension)

    Returns
    -------
    pd.DataFrame
    """
    import pandas as pd
    return pd.read_csv(SIMULATIONS_DIR / f"{filename}.csv")


def verify_simulation_outputs(results: list = None) -> dict:
    """
    Verify that simulation output files exist and are valid.

    Parameters
    ----------
    results : list, optional
        List of result dicts. If None, scans output directory.

    Returns
    -------
    dict : Verification summary
    """
    if results is None:
        # Scan output directory
        output_files = list(SIMULATIONS_DIR.glob("sample_*.hdf5"))
        sample_ids = []
        for f in output_files:
            try:
                sid = int(f.stem.split("_")[1])
                sample_ids.append(sid)
            except (IndexError, ValueError):
                pass
        results = [{"sample_id": sid, "output_file": str(SIMULATIONS_DIR / f"sample_{sid:06d}.hdf5"), "status": "unknown"} for sid in sample_ids]

    n_total = len(results)
    n_exists = 0
    n_valid = 0
    missing = []

    for r in results:
        if r["output_file"] and os.path.exists(r["output_file"]):
            n_exists += 1
            # Check file size (basic validity check)
            if os.path.getsize(r["output_file"]) > 1000:
                n_valid += 1
        else:
            missing.append(r["sample_id"])

    summary = {
        "total": n_total,
        "exists": n_exists,
        "valid": n_valid,
        "missing": len(missing),
        "missing_ids": missing[:20]  # First 20 missing IDs
    }

    print(f"Verification Summary:")
    print(f"  Total samples: {n_total}")
    print(f"  Files exist: {n_exists}")
    print(f"  Files valid: {n_valid}")
    print(f"  Missing: {len(missing)}")

    if missing:
        print(f"  First missing IDs: {missing[:20]}")

    return summary


if __name__ == "__main__":
    # Test single simulation (no MPI)
    print("Testing single simulation...")

    from methods.sampling import generate_sobol_samples, sample_to_nyc_config
    from config import N_SOBOL_SAMPLES

    # Generate a small sample set
    samples, problem = generate_sobol_samples(4)  # Small N for testing
    print(f"Generated {len(samples)} samples")

    # Test single simulation with sample 0
    print("\nRunning test simulation with sample 0...")
    config = sample_to_nyc_config(samples[0], problem)

    result = run_single_simulation(
        sample_id=0,
        nyc_config=config,
        cleanup_model=True
    )

    print(f"Result: {result}")
