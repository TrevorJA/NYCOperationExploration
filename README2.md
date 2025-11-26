# NYC Reservoir Operations Sobol Sensitivity Analysis

This project implements a Sobol variance decomposition sensitivity analysis for NYC reservoir operational rules using SALib. The analysis determines which operational parameters have the greatest influence on system performance metrics.

## Key Questions

1. How sensitive are outcome metrics to +/- 10% changes in NYC operational rules?
2. Which specific parameters or parameter groups have the greatest influence on outcomes?

## Quick Start

```bash
# 1. Generate Sobol samples
python 01_generate_samples.py

# 2. Run simulations (MPI for HPC)
mpiexec -n 64 python 02_run_simulations.py

# 3. Calculate metrics
python 03_calculate_metrics.py

# 4. Analyze sensitivity
python 04_analyze_sensitivity.py

# 5. Generate figures
python 05_visualize_results.py
```

## Project Structure

```
NYCOperationExploration/
├── methods/
│   ├── __init__.py
│   ├── sampling.py           # Sobol sampling and config generation
│   ├── simulation.py         # Run Pywr-DRB simulations (MPI-parallel)
│   ├── metrics.py            # Extract performance metrics (modular)
│   ├── analysis.py           # SALib Sobol analysis
│   └── plotting.py           # Visualization functions
├── config.py                 # Centralized configuration
├── 01_generate_samples.py    # Generate Sobol samples
├── 02_run_simulations.py     # Run simulations (MPI-enabled)
├── 03_calculate_metrics.py   # Extract metrics from outputs
├── 04_analyze_sensitivity.py # Compute Sobol indices
├── 05_visualize_results.py   # Generate figures
├── outputs/                  # Generated outputs
│   ├── samples/              # Sobol samples
│   ├── simulations/          # HDF5 simulation outputs
│   ├── metrics/              # Calculated metrics
│   └── analysis/             # Sobol indices
├── figures/                  # Generated figures
└── README.md
```

## Configuration

All settings are centralized in `config.py`:

### Simulation Settings

```python
START_DATE = "1945-01-01"
END_DATE = "2023-12-31"
INFLOW_TYPE = "pub_nhmv10_BC_withObsScaled"
N_SOBOL_SAMPLES = 64  # Total samples = N * (2D + 2)
```

### Parameter Groups

Enable/disable parameter groups by setting `enabled: True/False`:

```python
PARAMETER_GROUPS = {
    "delivery": {"enabled": True, ...},     # NYC/NJ delivery constraints
    "mrf": {"enabled": True, ...},          # Minimum required flows
    "flood": {"enabled": True, ...},        # Flood control limits
    "storage_zones": {"enabled": True, ...} # Storage zone adjustments
}
```

### Parameters

| Group | Parameter | Baseline | Bounds | Units |
|-------|-----------|----------|--------|-------|
| delivery | max_nyc_delivery | 800 | [720, 880] | MGD |
| delivery | max_nj_daily | 120 | [108, 132] | MGD |
| delivery | drought_factor_nyc_level3 | 0.85 | [0.765, 0.935] | fraction |
| delivery | drought_factor_nyc_level5 | 0.65 | [0.585, 0.715] | fraction |
| mrf | mrf_cannonsville | 122.8 | [110.52, 135.08] | MGD |
| mrf | mrf_pepacton | 64.63 | [58.17, 71.09] | MGD |
| mrf | mrf_neversink | 48.47 | [43.62, 53.32] | MGD |
| mrf | mrf_montague | 1131.05 | [1017.95, 1244.16] | MGD |
| mrf | mrf_trenton | 1938.95 | [1745.06, 2132.85] | MGD |
| flood | flood_max_cannonsville | 4200 | [3780, 4620] | CFS |
| flood | flood_max_pepacton | 2400 | [2160, 2640] | CFS |
| flood | flood_max_neversink | 3400 | [3060, 3740] | CFS |
| storage_zones | zone_vertical_shift | 0.0 | [-0.05, 0.05] | fraction |
| storage_zones | zone_time_shift_days | 0 | [-30, 30] | days |

### Metrics

| Metric | Description |
|--------|-------------|
| `montague_min_flow_mgd` | Minimum daily flow at Montague |
| `pct_time_drought_watch` | % days in Drought Watch or worse (zones 5-6) |
| `pct_time_drought_warning` | % days in Drought Warning or worse (zones 4-6) |
| `pct_time_drought_emergency` | % days in Drought Emergency (zone 6) |
| `nyc_min_storage_pct` | Minimum NYC combined storage (% of capacity) |

## Sample Size

Sobol method requires `N * (2D + 2)` samples, where:
- N = base sample count (set in config)
- D = number of parameters

With 14 parameters:
- N=64: 1,920 simulations
- N=128: 3,840 simulations
- N=256: 7,680 simulations

## Running on HPC

Example SLURM script:

```bash
#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=32
#SBATCH --time=24:00:00

module load python/3.9
module load mpi

cd /path/to/NYCOperationExploration

# Generate samples (single process)
python 01_generate_samples.py --n-samples 128

# Run simulations (parallel)
mpirun -np 128 python 02_run_simulations.py

# Calculate metrics (single process)
python 03_calculate_metrics.py

# Analyze sensitivity
python 04_analyze_sensitivity.py

# Generate figures
python 05_visualize_results.py
```

## Adding Custom Metrics

Metrics are modular and easy to extend. Add new metrics in `methods/metrics.py`:

```python
def calculate_my_metric(data: dict) -> float:
    """Calculate my custom metric."""
    # data contains: major_flow, res_storage, res_level
    return float(some_calculation)

# Register the metric
METRIC_FUNCTIONS["my_metric"] = calculate_my_metric
```

Then add to `METRICS_TO_CALCULATE` in `config.py`.

## Output Files

### Samples
- `outputs/samples/sobol_samples.csv` - Parameter values for each sample
- `outputs/samples/sobol_problem.json` - SALib problem definition

### Simulations
- `outputs/simulations/sample_XXXXXX.hdf5` - Individual simulation outputs
- `outputs/simulations/simulation_results.csv` - Metadata for all runs

### Metrics
- `outputs/metrics/metrics.csv` - Calculated metrics for all samples

### Analysis
- `outputs/analysis/sobol_indices.csv` - Formatted S1, ST indices
- `outputs/analysis/sobol_raw.json` - Full results including S2

### Figures
- `figures/sobol_bars_*.png` - Bar charts per metric
- `figures/tornado_*.png` - Parameter rankings
- `figures/heatmap_*.png` - Multi-metric comparison
- `figures/interactions_*.png` - Second-order interactions

## Drought Zone Reference

| Zone | Name | Description |
|------|------|-------------|
| 6 | Drought Emergency | Most severe |
| 5 | Drought Watch | |
| 4 | Drought Warning | |
| 3 | Normal | |
| 2 | Flood Watch | |
| 1 | Flood Warning | |

## Dependencies

- Python 3.8+
- SALib
- numpy
- pandas
- matplotlib
- mpi4py (for parallel execution)
- pywrdrb (Pywr-DRB model)

## References

- Sobol, I.M. (2001). Global sensitivity indices for nonlinear mathematical models. Mathematics and Computers in Simulation.
- SALib: https://salib.readthedocs.io/
