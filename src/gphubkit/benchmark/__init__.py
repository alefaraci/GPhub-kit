"""Benchmarking module for GPhub-kitPro."""

from .bm import BM01, BM02, BM03, BM04, BM05, BM06, BM07
from .base import GPhubkitBenchmark
from .comp import CompositeShell
from .cust import Custom
from .toys import fn_BM_01, fn_BM_02, fn_BM_03, fn_BM_04, fn_BM_05, fn_BM_06, fn_BM_07

__all__ = [
    "BM01",
    "BM02",
    "BM03",
    "BM04",
    "BM05",
    "BM06",
    "BM07",
    "CompositeShell",
    "Custom",
    "GPhubkitBenchmark",
    "fn_BM_01",
    "fn_BM_02",
    "fn_BM_03",
    "fn_BM_04",
    "fn_BM_05",
    "fn_BM_06",
    "fn_BM_07",
]
