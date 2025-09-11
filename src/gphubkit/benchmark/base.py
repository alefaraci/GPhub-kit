"""Benchmark base class."""

from abc import ABC, abstractmethod
from pathlib import Path
import shutil

from attrs import field
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np

from ..utils import get_main_script_path


class GPhubkitBenchmark(ABC):
    """GPhubkitBenchmark class benchmark."""

    id: str
    scale_x: bool = True
    standardize_y: bool = True
    train_x: np.ndarray = field(default=None)
    test_x: np.ndarray = field(default=None)
    train_y: np.ndarray = field(default=None)
    test_y: np.ndarray = field(default=None)
    train_x_scaled: np.ndarray = field(default=None)
    train_y_std: np.ndarray = field(default=None)
    test_x_scaled: np.ndarray = field(default=None)
    test_y_std: np.ndarray = field(default=None)
    scaler_x: MinMaxScaler | None = field(default=None)
    scaler_y: StandardScaler | None = field(default=None)

    @abstractmethod
    def __attrs_post_init__(self) -> None:
        """Initialize the GPhubkitBenchmark class."""

    def _preprocess(self) -> None:
        """Preprocess train set."""
        if self.scale_x:
            self.scaler_x = MinMaxScaler(feature_range=(0, 1))
            self.train_x_scaled = self.scaler_x.fit_transform(self.train_x)
            self.test_x_scaled = self.scaler_x.transform(self.test_x)

        else:
            self.train_x_scaled = self.train_x
            self.test_x_scaled = self.test_x
        if self.standardize_y:
            self.scaler_y = StandardScaler()
            self.train_y_std = self.scaler_y.fit_transform(self.train_y)
            self.test_y_std = self.scaler_y.transform(self.test_y)  # type: ignore
        else:
            self.train_y_std = self.train_y
            self.test_y_std = self.test_y

    def _copy_data_locally(self, source_data_path: Path) -> Path:
        """Copy data directory to caller's local data directory."""
        caller_dir = get_main_script_path()
        local_data_dir = caller_dir / "data"
        if local_data_dir.exists():
            shutil.rmtree(local_data_dir)
        shutil.copytree(source_data_path, local_data_dir)
        return local_data_dir

    def run(self) -> None:
        """Run all scripts in the caller's local script directory."""
        from ..launcher import runner

        runner.run(self)

    @staticmethod
    def postprocess() -> None:
        """Postprocess results in the caller's local results directory."""
        from ..launcher import processer

        processer.postprocess()
