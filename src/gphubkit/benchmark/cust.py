"""Custom benchmarks."""

from attrs import define
import numpy as np

from .base import GPhubkitBenchmark


@define
class Custom(GPhubkitBenchmark):
    """Class for custom benchmarks."""

    id: str
    scale_x: bool
    standardize_y: bool
    train_x: np.ndarray
    test_x: np.ndarray
    train_y: np.ndarray
    test_y: np.ndarray

    def __attrs_post_init__(self) -> None:
        """Initialize the Custom class."""
        self._preprocess()
