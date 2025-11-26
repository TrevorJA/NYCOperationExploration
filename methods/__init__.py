"""
Methods module for NYC Reservoir Operations Sensitivity Analysis.

This module provides:
- sampling: Sobol sampling and configuration generation
- simulation: Pywr-DRB simulation execution (MPI-parallel)
- metrics: Performance metric extraction
- analysis: SALib Sobol sensitivity analysis
- plotting: Visualization functions
"""

from . import sampling
from . import simulation
from . import metrics
from . import analysis
from . import plotting

__all__ = [
    "sampling",
    "simulation",
    "metrics",
    "analysis",
    "plotting",
]
