"""Internal benchmarks."""

from pathlib import Path

from attrs import define

from .base import GPhubkitBenchmark
from ..data import load


@define
class CompositeShell(GPhubkitBenchmark):
    """Generic BM class for internal benchmarks."""

    dim: int
    train_size: int
    id: str = "COMPOSITE"
    scale_x: bool = True
    standardize_y: bool = True

    def __load_bm(self) -> None:
        """Load toy benchmark datasets and copy to local data directory."""
        source_database = (
            Path(__file__).parent / "database" / "COMPOSITE" / f"Dim{self.dim:02d}" / f"{self.train_size}" / "data"
        )
        local_data_dir = self._copy_data_locally(source_database)
        self.train_x, self.test_x, self.train_y, self.test_y = load(local_data_dir)

    def __attrs_post_init__(self) -> None:
        """Initialize the BM class."""
        self.__load_bm()
        self._preprocess()
