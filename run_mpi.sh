#!/bin/bash
#SBATCH --job-name=NYCOps
#SBATCH --output=./logs/NYCOps.out
#SBATCH --error=./logs/NYCOps.err
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=40
#SBATCH --time=24:00:00
#SBATCH --mem=0

# Setup
module load python/3.11.5
source venv/bin/activate
np=$(($SLURM_NTASKS_PER_NODE * $SLURM_NNODES))

# Workflow flags (all true by default)
PRESIM=${PRESIM:-false}
GENERATE=${GENERATE:-false}
SIMULATE=${SIMULATE:-true}
METRICS=${METRICS:-true}
ANALYZE=${ANALYZE:-true}
VISUALIZE=${VISUALIZE:-true}

# Create directories
mkdir -p logs outputs/{samples,simulations,metrics,analysis,presim} figures

echo "Running NYC Operations SA with $np ranks on $SLURM_NNODES nodes"

# Execute workflow
[ "$PRESIM" = true ] && python3 00_generate_presimulated_releases.py --force
[ "$GENERATE" = true ] && python3 01_generate_samples.py
[ "$SIMULATE" = true ] && mpirun -np $np python3 02_run_simulations.py
[ "$METRICS" = true ] && mpirun -np $np python3 03_calculate_metrics.py
[ "$ANALYZE" = true ] && python3 04_analyze_sensitivity.py
[ "$VISUALIZE" = true ] && python3 05_visualize_results.py

echo "Workflow complete"
