"""Module for loading existing data from CSV files."""

from pathlib import Path

import numpy as np
import polars as pl


def _loader_csv(file_path: Path, data_type: str) -> tuple[np.ndarray, ...]:
    data_x = pl.read_csv(source=file_path.resolve() / f"{data_type}_x.csv", has_header=False).to_numpy()
    data_y = pl.read_csv(source=file_path.resolve() / f"{data_type}_y.csv", has_header=False).to_numpy()
    return data_x, data_y


def load(file_path: Path) -> tuple[np.ndarray, ...]:
    """Load train, and test sets from CSV files."""
    train_x, train_y = _loader_csv(file_path, "train")
    test_x, test_y = _loader_csv(file_path, "test")
    return train_x, test_x, train_y, test_y
