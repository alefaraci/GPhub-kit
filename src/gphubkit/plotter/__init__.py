"""Module for plotting."""

import matplotlib.pyplot as plt

from . import radar, prediction, crossvalidation
from .utils import save, adjust, fig_size, latex_formatter, set_square_ratio, get_appropriate_exponent

__all__ = [
    "adjust",
    "crossvalidation",
    "fig_size",
    "get_appropriate_exponent",
    "latex_formatter",
    "prediction",
    "radar",
    "save",
    "set_square_ratio",
]

rc_params = {
    # DPI
    "figure.dpi": 140,
    # LaTeX
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "text.latex.preamble": r"\usepackage{amssymb} \usepackage{amsmath} \usepackage{bm} \usepackage{physics}",
    # Axes
    "axes.linewidth": 0.3,
    # Grid
    "grid.color": "black",
    "grid.linestyle": ":",
    "grid.alpha": 0.3,
    "grid.linewidth": 0.25,
    "xtick.major.width": 0.25,
    "xtick.minor.width": 0.20,
    "ytick.major.width": 0.25,
    "ytick.minor.width": 0.20,
    # Legend
    "legend.edgecolor": "black",
    "patch.linewidth": 0.25,
    "legend.fontsize": 7.5,
    "legend.fancybox": False,
    # Linewidth
    "lines.linewidth": 0.5,
    # Label
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
}


plt.rcParams.update(rc_params)
