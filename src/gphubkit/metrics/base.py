"""Module for storing and computing metrics for Gaussian Process Regression libraries."""

from pathlib import Path

from attrs import field, define
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, median_absolute_error
import numpy as np
import polars as pl

from ..utils import table, console


@define
class GPlibrary:
    """GPlibrary class for storing and computing metrics for Gaussian Process Regression libraries."""

    library: str
    train_x: np.ndarray = field(default=None)
    train_y: np.ndarray = field(default=None)
    test_x: np.ndarray = field(default=None)
    test_y: np.ndarray = field(default=None)
    pred_y: np.ndarray = field(default=None)
    pred_var: np.ndarray = field(default=None)
    train_time: float = field(default=None)
    pred_time: float = field(default=None)
    train_memory: float = field(default=None)
    pred_memory: float = field(default=None)

    def __attrs_post_init__(self) -> None:
        """Initialize the GPlibrary class."""
        self.train_x = pl.read_csv(source=Path("data").resolve() / "train_x.csv", has_header=False).to_numpy()
        self.train_y = pl.read_csv(source=Path("data").resolve() / "train_y.csv", has_header=False).to_numpy()
        self.test_x = pl.read_csv(source=Path("data").resolve() / "test_x.csv", has_header=False).to_numpy()
        self.test_y = pl.read_csv(source=Path("data").resolve() / "test_y.csv", has_header=False).to_numpy()
        results = pl.read_parquet(source=Path("results/storage").resolve() / f"lib_{self.library}.parquet")
        self.pred_y = results["pred_y"].list.explode().to_numpy().reshape(-1, 1)
        self.pred_var = results["pred_var"].list.explode().to_numpy().reshape(-1, 1)
        self.train_time = results["train_time"].item()
        self.pred_time = results["pred_time"].item()
        self.train_memory = results["train_memory"].item()
        self.pred_memory = results["pred_memory"].item()

    @property
    def mae(self) -> np.ndarray:
        """Mean Absolute Error."""
        return mean_absolute_error(self.test_y, self.pred_y)  # type: ignore

    @property
    def mse(self) -> np.ndarray:
        """Mean Squared Error."""
        return mean_squared_error(self.test_y, self.pred_y)  # type: ignore

    @property
    def rmse(self) -> np.ndarray:
        """Root Mean Squared Error."""
        return np.sqrt(self.mse)

    @property
    def medae(self) -> np.ndarray:
        """Median Absolute Error."""
        return median_absolute_error(self.test_y, self.pred_y)  # type: ignore

    @property
    def r2(self) -> np.ndarray:
        """R² score."""
        return r2_score(self.test_y, self.pred_y)  # type: ignore

    @property
    def nlpd(self) -> np.ndarray:
        """Negative Log Predictive Density."""
        pred_var = np.maximum(self.pred_var, 1e-6)
        log_var_term = 0.5 * np.log(2 * np.pi * pred_var)
        squared_error_term = 0.5 * ((self.test_y - self.pred_y) ** 2) / pred_var
        # Compute NLPD for each prediction and return average NLPD
        return np.mean(log_var_term + squared_error_term)  # type: ignore

    @property
    def msll(self) -> float:
        """Mean Standardized Log Loss."""
        # Clip variances to prevent numerical instability
        pred_var = np.maximum(self.pred_var, 1e-6)

        # Compute log loss manually for better numerical stability
        log_var_term = 0.5 * np.log(2 * np.pi * pred_var)
        squared_error_term = 0.5 * ((self.test_y - self.pred_y) ** 2) / pred_var
        nll = log_var_term + squared_error_term

        # Compute baseline with stabilized variance
        baseline_mean = np.mean(self.test_y)
        baseline_var = np.maximum(np.var(self.test_y), 1e-6)

        # Compute baseline log loss manually
        baseline_log_var = 0.5 * np.log(2 * np.pi * baseline_var)
        baseline_squared_error = 0.5 * ((self.test_y - baseline_mean) ** 2) / baseline_var
        baseline_nll = baseline_log_var + baseline_squared_error

        # Compute MSLL
        return np.mean(nll - baseline_nll)  # type: ignore

    @property
    def residuals(self) -> np.ndarray:
        """Residuals."""
        return self.test_y - self.pred_y

    @property
    def _metrics_row(self) -> list[str]:
        """Metrics row."""
        return [
            f"{self.library}",
            f"{self.mae:.4e}",
            f"{self.rmse:.4e}",
            f"{self.mse:.4e}",
            f"{self.medae:.4e}",
            f"{self.r2:.4f}",
            f"{self.nlpd:.4f}",
            f"{self.msll:.4f}",
            f"{self.train_time:.4f} s",
            f"{self.pred_time:.4f} s",
            f"{self.train_memory:.2f} MB",
            f"{self.pred_memory:.2f} MB",
        ]

    @property
    def _metrics_header(self) -> list[str]:
        """Metrics header."""
        return [
            "Library",
            "MAE",
            "RMSE",
            "MSE",
            "MedAE",
            "R²",
            "NLPD",
            "MSLL",
            "t_train",
            "t_pred",
            "train_memory",
            "pred_memory",
        ]

    def print_metrics(self) -> None:
        """Print metrics."""
        tab = table(headers=self._metrics_header, title="Metrics")
        tab.add_row(*self._metrics_row)
        console.log(tab)

    def plot_results(self, path: Path, *, show: bool = False) -> None:
        """Plot results."""
        from .. import plotter

        plotter.crossvalidation.plot(self.pred_y, self.test_y, self.library)
        plotter.save(plotter.plt, path=path / "crossvalidation", filename=f"lib_{self.library}")
        plotter.crossvalidation.plot_density(self.pred_y, self.test_y, self.library)
        plotter.save(plotter.plt, path=path / "density", filename=f"lib_{self.library}")
        plotter.crossvalidation.plot_residuals(self.residuals, library=f"{self.library}")
        plotter.save(plotter.plt, path=path / "residuals", filename=f"residuals_{self.library}")
        plotter.prediction.plot(gplib=self)
        plotter.save(plotter.plt, path=path / "prediction", filename=f"lib_{self.library}")
        plotter.plt.show() if show else plotter.plt.close("all")
