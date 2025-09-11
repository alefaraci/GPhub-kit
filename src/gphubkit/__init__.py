"""===============================================================================

    GPhub-kit: A toolkit for benchmarking Gaussian Process Regression libraries.

    Copyright (c) 2024-2025 Alessio Faraci
    Author: Alessio Faraci < alessio.faraci@sigma-clermont.fr >
    License: MIT License.
==================================================================================
"""

from . import data, utils, metrics, plotter, launcher, benchmark
from .launcher.runner import run
from .launcher.processer import postprocess

__all__ = ["benchmark", "data", "launcher", "metrics", "plotter", "postprocess", "run", "utils"]

__app_name__ = "GPhub-kit"
__version__ = "0.1.0"
