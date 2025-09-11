"""Module for plotting cross-validation results."""

from typing import Any

from numpy import ndarray
from matplotlib.axes import Axes
from sklearn.metrics import PredictionErrorDisplay
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, ScalarFormatter, FormatStrFormatter
import numpy as np
import matplotlib.pyplot as plt

from .utils import latex_formatter, get_appropriate_exponent


def plot(pred_y: ndarray, test_y: ndarray, library: str) -> None:
    """Cross-validation plot comparing Kriging predictions with true model response."""
    fig, axs = plt.subplots(ncols=2, figsize=(2 * 6.93 / 2.54, 6.93 / 2.54))
    PredictionErrorDisplay.from_predictions(
        y_true=test_y,
        y_pred=pred_y,
        kind="actual_vs_predicted",
        subsample=None,  # type: ignore
        ax=axs[0],
        scatter_kwargs={
            "color": "#6AD9E5",
            "marker": ".",
            "s": 50,
            "alpha": 0.3,
            "zorder": 2,
            "edgecolors": "k",
        },
        line_kwargs={"color": "k", "lw": 1.5, "zorder": 1, "linestyle": "-"},
    )
    min_value = min(test_y.min(), pred_y.min())
    max_value = max(test_y.max(), pred_y.max())
    max_shift = max(0.05 * abs(min_value), 0.05 * abs(max_value))
    lims = [min_value - max_shift, max_value + max_shift]
    axs[0].plot(lims, lims, "k", lw=1.5, zorder=1)
    axs[0].plot(lims, lims, "k", lw=1.5, zorder=3, alpha=0.35)
    axs[0].set_aspect(1.0 / plt.gca().get_data_ratio(), adjustable="box")
    axs[0].set_xlabel(r"$\hat{y}_{\text{pred}}$")
    axs[0].set_ylabel(r"$y_{\text{test}}$")
    axs[0].grid(visible=True, which="major", linestyle="-", linewidth=0.25, alpha=0.2)
    axs[0].grid(visible=True, which="minor", linewidth=0.20)
    axs[0].minorticks_on()
    axs[0].yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    axs[0].xaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    PredictionErrorDisplay.from_predictions(
        y_true=test_y,
        y_pred=pred_y,
        kind="residual_vs_predicted",
        ax=axs[1],
        subsample=None,  # type: ignore
        scatter_kwargs={
            "color": "#3D7B7B",
            "marker": ".",
            "s": 50,
            "alpha": 0.3,
            "zorder": 2,
            "edgecolors": "k",
        },
        line_kwargs={"color": "k", "lw": 1.5, "zorder": 1, "linestyle": "-"},
    )
    min_value = min(pred_y)
    max_value = max(pred_y)
    max_shift = max(0.05 * abs(min_value), 0.05 * abs(max_value))
    lims = [min_value - max_shift, max_value + max_shift]
    axs[1].plot(lims, [0, 0], "k", lw=1.5, zorder=1)
    axs[1].plot(lims, [0, 0], "k", lw=1.5, zorder=3, alpha=0.35)
    axs[1].set_aspect(1.0 / plt.gca().get_data_ratio(), adjustable="box")
    axs[1].set_xlabel(r"$\hat{y}_{\text{pred}}$")
    axs[1].set_ylabel(r"$y_{\text{test}}-\hat{y}_{\text{pred}}$")
    axs[1].grid(visible=True, which="major", linestyle="-", linewidth=0.25, alpha=0.2)
    axs[1].grid(visible=True, which="minor", linewidth=0.20)
    axs[1].minorticks_on()
    axs[1].xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    exponent = get_appropriate_exponent(test_y - pred_y)
    formatter = lambda x, p: latex_formatter(x, p, exponent)  # noqa: E731
    axs[1].yaxis.set_major_formatter(FuncFormatter(formatter))
    axs[1].text(0, 1.03, rf"$\times 10^{{{exponent}}}$", transform=axs[1].transAxes)
    fig.suptitle(f"Cross-validated predictions for {library}")
    plt.tight_layout()


class CustomScalarFormatter(ScalarFormatter):
    """Custom ScalarFormatter class for scientific notation."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize CustomScalarFormatter."""
        super().__init__(*args, **kwargs)
        self.set_scientific(True)
        self.set_powerlimits((-3, 3))


def plot_density(
    pred_y: np.ndarray,
    test_y: np.ndarray,
    library: str,
    bins: int = 50,
) -> tuple[Figure, Axes]:
    """Create a histogram plot comparing Kriging predictions with true model response.

    Parameters:
    -----------
    pred_y : array-like
        Predicted values from Kriging model
    test_y : array-like
        True model response values
    figsize : tuple, optional
        Figure size (width, height) in inches
    bins : int, optional
        Number of histogram bins

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure object
    ax : matplotlib.axes.Axes
        The created axes object
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(6.93 / 2.54, 6.93 / 2.54))

    # Calculate range for histogram
    min_val = min(np.min(pred_y), np.min(test_y))
    max_val = max(np.max(pred_y), np.max(test_y))
    bin_edges = np.linspace(min_val, max_val, bins + 1)

    # Plot histograms
    ax.hist(
        test_y,
        bins=bin_edges,
        alpha=1,
        color="#244949",
        label=r"$y_{\text{test}}$",
    )
    ax.hist(pred_y, bins=bin_edges, alpha=0.5, color="#B266FF", label=r"$\hat{y}_{\text{pred}}$")

    # Customize plot
    ax.set_xlabel(r"$y$")
    ax.set_ylabel("Density")
    ax.set_title(f"{library}")
    ax.grid(visible=True, which="major", linestyle="-", linewidth=0.25, alpha=0.2)
    ax.grid(visible=True, which="minor", linewidth=0.20)
    ax.minorticks_on()
    ax.legend()
    ax.set_aspect(1.0 / plt.gca().get_data_ratio(), adjustable="box")
    plt.tight_layout()
    return fig, ax


def plot_residuals(residuals: np.ndarray, library: str, nbins: int = 30) -> tuple[Figure, Axes]:
    """Create a histogram plot of residuals."""
    fig, ax = plt.subplots(figsize=(6.93 / 2.54, 6.93 / 2.54))
    counts, bins, _ = ax.hist(
        residuals,
        bins=nbins,
        density=True,
        alpha=0.6,
        color="skyblue",
        edgecolor="black",
    )
    ax.set_xlabel(r"$y-\hat{y}_{\text{pred}}$")
    ax.set_ylabel("Density")
    ax.grid(visible=True, which="major", linestyle="-", linewidth=0.25, alpha=0.2)
    ax.grid(visible=True, which="minor", linewidth=0.20)
    ax.minorticks_on()
    ax.set_aspect(1.0 / plt.gca().get_data_ratio(), adjustable="box")
    ax.set_title(f"{library}")
    plt.tight_layout()
    return fig, ax
