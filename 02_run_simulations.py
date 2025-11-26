"""
02: Run Pywr-DRB simulations for all Sobol samples.

This script runs Pywr-DRB simulations for each parameter sample.
Supports both MPI parallel execution (for HPC) and serial execution.

Usage:
    # Serial execution (for testing)
    python 02_run_simulations.py

    # MPI parallel execution (for HPC)
    mpiexec -n 16 python 02_run_simulations.py

    # Run specific sample range (for debugging)
    python 02_run_simulations.py --start 0 --end 10

Example:
    mpiexec -n 64 python 02_run_simulations.py
"""

import argparse
import sys
from pathlib import Path

# Add methods to path
sys.path.insert(0, str(Path(__file__).parent))

from config import START_DATE, END_DATE, INFLOW_TYPE, SIMULATIONS_DIR
from methods.sampling import load_samples
from methods.simulation import (
    run_parallel_simulations_mpi,
    run_simulations_serial,
    save_simulation_results,
    verify_simulation_outputs,
    MPI_AVAILABLE
)


def main(serial: bool = False, start_idx: int = None, end_idx: int = None):
    """Run simulations for all Sobol samples."""

    print("=" * 70)
    print("PYWR-DRB SOBOL SENSITIVITY SIMULATIONS")
    print("=" * 70)

    # Load samples
    print("\nLoading samples...")
    samples, problem = load_samples("sobol")

    n_total = len(samples)
    n_params = problem["num_vars"]

    print(f"  Total samples: {n_total}")
    print(f"  Parameters: {n_params}")
    print(f"  Simulation period: {START_DATE} to {END_DATE}")
    print(f"  Inflow type: {INFLOW_TYPE}")

    # Determine execution mode
    if serial or not MPI_AVAILABLE:
        if not MPI_AVAILABLE and not serial:
            print("\n  WARNING: MPI not available, falling back to serial execution")

        print("\n  Execution mode: SERIAL")

        # Handle sample range
        if start_idx is not None or end_idx is not None:
            start = start_idx if start_idx is not None else 0
            end = end_idx if end_idx is not None else n_total
            sample_ids = list(range(start, min(end, n_total)))
            print(f"  Sample range: {start} to {end}")
        else:
            sample_ids = None

        # Run serial simulations
        print("\nStarting simulations...")
        results = run_simulations_serial(
            samples, problem,
            start_date=START_DATE,
            end_date=END_DATE,
            inflow_type=INFLOW_TYPE,
            sample_ids=sample_ids
        )

        # Save results
        save_simulation_results(results)

    else:
        # MPI execution
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        if rank == 0:
            print(f"\n  Execution mode: MPI PARALLEL")
            print(f"  MPI ranks: {comm.Get_size()}")

        # Run MPI parallel simulations
        results = run_parallel_simulations_mpi(
            samples, problem,
            start_date=START_DATE,
            end_date=END_DATE,
            inflow_type=INFLOW_TYPE
        )

        # Save results (rank 0 only)
        if rank == 0 and results is not None:
            save_simulation_results(results)

    # Verification (rank 0 or serial)
    if not MPI_AVAILABLE or (MPI_AVAILABLE and MPI.COMM_WORLD.Get_rank() == 0):
        print("\nVerifying outputs...")
        verify_simulation_outputs()

        print("\n" + "=" * 70)
        print("SIMULATIONS COMPLETE")
        print("=" * 70)
        print(f"\nOutputs saved to: {SIMULATIONS_DIR}")
        print(f"Next step: Calculate metrics with 03_calculate_metrics.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Pywr-DRB simulations for Sobol samples")
    parser.add_argument("--serial", action="store_true",
                        help="Force serial execution (no MPI)")
    parser.add_argument("--start", type=int, default=None,
                        help="Start sample index (for partial runs)")
    parser.add_argument("--end", type=int, default=None,
                        help="End sample index (for partial runs)")

    args = parser.parse_args()
    main(serial=args.serial, start_idx=args.start, end_idx=args.end)
