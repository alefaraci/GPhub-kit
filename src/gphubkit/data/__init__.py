"""Package for databases handling."""

from .loader import load
from .splitter import split_dataset
from .synthetic import generate_synthetic

__all__ = [
    "load",
    "generate_synthetic",
    "split_dataset",
]
