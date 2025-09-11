"""Module to split a dataset to train and test set."""

from pathlib import Path

from sklearn.model_selection import train_test_split
import numpy as np

from .loader import _loader_csv


def __check_dim(data: np.ndarray) -> np.ndarray:
    return data.reshape(-1, 1) if data.ndim == 1 else data


def split_dataset(file_path: Path, test_size: float) -> tuple[np.ndarray, ...]:
    """Split dataset in train and test data."""
    dataset_x, dataset_y = _loader_csv(file_path, "dataset")
    train_x, test_x, train_y, test_y = train_test_split(dataset_x, dataset_y, test_size=test_size)
    np.savetxt(file_path / "train_x.csv", train_x, delimiter=",")
    np.savetxt(file_path / "train_y.csv", train_y, delimiter=",")
    np.savetxt(file_path / "test_x.csv", test_x, delimiter=",")
    np.savetxt(file_path / "test_y.csv", test_y, delimiter=",")
    return __check_dim(train_x), __check_dim(test_x), __check_dim(train_y), __check_dim(test_y)
