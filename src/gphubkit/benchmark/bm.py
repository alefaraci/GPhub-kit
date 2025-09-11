"""Internal benchmarks."""

from pathlib import Path

from attrs import define

from .base import GPhubkitBenchmark
from ..data import load


@define
class BM(GPhubkitBenchmark):
    """Generic BM class for internal benchmarks."""

    id: str
    scale_x: bool = True
    standardize_y: bool = True

    def __load_bm(self) -> None:
        """Load toy benchmark datasets and copy to local data directory."""
        source_database = Path(__file__).parent / "database" / self.id / "data"
        local_data_dir = self._copy_data_locally(source_database)
        self.train_x, self.test_x, self.train_y, self.test_y = load(local_data_dir)

    def __attrs_post_init__(self) -> None:
        """Initialize the BM class."""
        self.__load_bm()
        self._preprocess()


@define
class BM01(BM):
    """BM_01 class benchmark for toy function fn_BM_01."""

    id: str = "BM_01"


@define
class BM02(BM):
    """BM_02 class benchmark for toy function fn_BM_02."""

    id: str = "BM_02"


@define
class BM03(BM):
    """BM_03 class benchmark for toy function fn_BM_03."""

    id: str = "BM_03"


@define
class BM04(BM):
    """BM_04 class benchmark for toy function fn_BM_04."""

    id: str = "BM_04"


@define
class BM05(BM):
    """BM_05 class benchmark for toy function fn_BM_05."""

    id: str = "BM_05"


@define
class BM06(BM):
    """BM_06 class benchmark for toy function fn_BM_06."""

    id: str = "BM_06"


@define
class BM07(BM):
    """BM_07 class benchmark for toy function fn_BM_07."""

    id: str = "BM_07"


@define
class BM08(BM):
    """BM_08 class benchmark for toy function fn_BM_08."""

    id: str = "BM_08"
