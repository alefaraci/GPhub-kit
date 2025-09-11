"""Utility functions for plotter module."""

from pathlib import Path

import numpy as np


def save(plt, path: Path, filename: str = "autosave", format="png") -> None:  # noqa: ANN001
    """Save plot."""
    path.mkdir(parents=True, exist_ok=True)
    plt.savefig(f"{path}/{filename}.{format}", dpi=400, format=f"{format}")


def fig_size(figsize: str = "squareSmall", width: float = 6.93, factor: float = 2.54) -> tuple[float, float]:
    """Set figure size.

    unit_base: centimeters = 6.93 - factor: inches = 2.54

    squareColorbar | squareColorbar2 | squareSmall | squareMedium | squareBig |
    rectangleSmall | rectangleMedium | rectangleBig
    """
    unit_base = width / factor
    register_fig_size = {
        "squareColorbar": (1.3 * unit_base, unit_base),
        "squareColorbar2": (1.43 * unit_base, unit_base),
        "squareSmall": (unit_base, unit_base),
        "squareMedium": (2 * unit_base, 2 * unit_base),
        "squareBig": (width, width),
        "rectangleSmall": (2 * unit_base, unit_base),
        "rectangleMedium": (4 * unit_base, 2 * unit_base),
        "rectangleBig": (2 * width, width),
    }
    return register_fig_size[figsize]


def adjust(plt, left: float = 0.2, right: float = 0.98, bottom: float = 0.15, top: float = 0.96) -> None:  # noqa: ANN001
    """Adjust spacing for inner margins of a plot."""
    plt.subplots_adjust(left=left, right=right, bottom=bottom, top=top)


def set_square_ratio(ax) -> None:  # noqa: ANN001
    """Set square ratio of a plot."""
    ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable="box")


def latex_formatter(x: np.ndarray, p, exponent: float | None = None) -> str:
    """Format numbers with the given exponent."""
    if exponent is None:
        return f"${x:.1f}$"
    scaled_x = x / (10**exponent)
    return f"${scaled_x:.1f}$"


def get_appropriate_exponent(data: np.ndarray) -> int:
    """Determine the appropriate exponent for the data scale."""
    max_abs = np.max(np.abs(data))
    if max_abs == 0:
        return 0
    return int(np.floor(np.log10(max_abs)))
