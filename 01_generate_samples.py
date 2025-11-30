"""
01: Generate Sobol samples for sensitivity analysis.

This script generates Sobol sequence samples using SALib and saves them
for use in subsequent simulation runs.

Usage:
    python 01_generate_samples.py
"""

import sys
from pathlib import Path

# Add methods to path
sys.path.insert(0, str(Path(__file__).parent))

from config import N_SOBOL_SAMPLES, print_config_summary
from methods.sampling import (
    generate_sobol_samples,
    save_samples,
    print_sample_summary
)


def main():
    """Generate and save Sobol samples."""

    print("=" * 70)
    print("SOBOL SAMPLE GENERATION")
    print("=" * 70)

    # Print configuration summary
    print_config_summary()

    # Generate samples
    print("\nGenerating Sobol sequence...")
    samples, problem = generate_sobol_samples(N_SOBOL_SAMPLES)

    # Print summary
    print_sample_summary(samples, problem)

    # Save samples
    print("\nSaving samples...")
    save_samples(samples, problem, filename="sobol")

    print("\n" + "=" * 70)
    print("SAMPLE GENERATION COMPLETE")
    print("=" * 70)
    print(f"\nNext step: Run simulations with 02_run_simulations.py")


if __name__ == "__main__":
    main()
