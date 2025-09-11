"""Synthetic data generation module."""

from typing import Literal
from pathlib import Path
from collections.abc import Callable

from pyDOE import lhs
from sklearn.model_selection import train_test_split
import numpy as np

SamplingMethod = Literal["uniform", "lhs"]


def __check_dim(data: np.ndarray) -> np.ndarray:
    return data.reshape(-1, 1) if data.ndim == 1 else data


def __generate_samples(
    bounds: np.ndarray,
    data_size: int,
    method: SamplingMethod = "uniform",
    seed: int | None = None,
) -> np.ndarray:
    """Generate samples using the specified sampling method."""
    if seed is not None:
        np.random.seed(seed)

    dim = bounds.shape[1]

    match method:
        case "uniform":
            return np.random.uniform(
                low=bounds[0],
                high=bounds[1],
                size=(data_size, dim),
            )
        case "lhs":
            lhs_samples: np.ndarray = lhs(dim, samples=data_size)  # type: ignore
            return (lhs_samples * (bounds[1] - bounds[0])) + bounds[0]


def generate_synthetic(
    bounds: np.ndarray,
    fn: Callable,
    data_size: int,
    train_size: int,
    file_path: Path,
    *,
    method: SamplingMethod = "uniform",
    seed: int | None = None,
) -> tuple[np.ndarray, ...]:
    """Generate synthetic dataset with optional preprocessing.

    Args:
        file_path: Directory path to save the generated data
        bounds: Array of shape (2, dim) with lower and upper bounds
        fn: Function to evaluate on generated samples
        data_size: Total number of samples to generate
        train_size: Number of training samples
        method: Sampling method ("uniform" or "lhs")
        seed: Random seed for reproducibility

    Returns:
        tuple: (dataset_x, train_x, test_x, dataset_y, train_y, test_y)
    """
    test_size = data_size - train_size

    dataset_x = __generate_samples(bounds, data_size, method, seed)
    dataset_y = fn(dataset_x)

    file_path.mkdir(parents=True, exist_ok=True)

    train_x, test_x, train_y, test_y = train_test_split(dataset_x, dataset_y, test_size=test_size, random_state=seed)
    np.savetxt(file_path.resolve() / "dataset_x.csv", dataset_x, delimiter=",")
    np.savetxt(file_path.resolve() / "dataset_y.csv", dataset_y, delimiter=",")
    np.savetxt(file_path.resolve() / "train_x.csv", train_x, delimiter=",")
    np.savetxt(file_path.resolve() / "train_y.csv", train_y, delimiter=",")
    np.savetxt(file_path.resolve() / "test_x.csv", test_x, delimiter=",")
    np.savetxt(file_path.resolve() / "test_y.csv", test_y, delimiter=",")

    return __check_dim(train_x), __check_dim(test_x), __check_dim(train_y), __check_dim(test_y)  # type: ignore
