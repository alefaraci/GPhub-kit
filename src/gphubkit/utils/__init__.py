"""Utility functions for the gphubkit package."""

from typing import Any
from pathlib import Path
from functools import wraps
from collections.abc import Callable
import time
import logging
import threading

from rich import pretty
from rich.table import Table
from rich.console import Console
from rich.logging import RichHandler
from rich_gradient import Gradient
from rich.traceback import install
import psutil

from .logo_ascii import colors, gphubkit_logo


def disp_logo(console: Console) -> None:
    """Display the GPhubkit logo in the console."""
    console.width = 90
    console.log(Gradient(gphubkit_logo, colors=colors))  # type: ignore
    console.width = 175


def get_main_script_path() -> Path:
    """Get the main script path."""
    return Path.cwd()


def stats(func: Callable) -> Callable:
    """Decorator to measure the time and memory usage of a function."""

    @wraps(func)
    def wrapper_stats(*args: Any, **kwargs: Any) -> tuple:
        process = psutil.Process()

        def get_rss() -> int:
            """Get the RSS memory of the current process and its children in bytes."""
            mem = process.memory_info().rss
            for child in process.children(recursive=True):
                try:
                    mem += child.memory_info().rss
                except psutil.NoSuchProcess:
                    continue
            return mem

        baseline_mem = get_rss()
        peak_mem = baseline_mem

        stop_monitoring = threading.Event()

        def monitor_memory() -> None:
            """Monitor memory usage of the current process."""
            nonlocal peak_mem
            while not stop_monitoring.is_set():
                current_memory = get_rss()
                peak_mem = max(current_memory, peak_mem)
                time.sleep(0.01)

        monitor_thread = threading.Thread(target=monitor_memory)
        monitor_thread.start()

        result = None
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
        finally:
            end_time = time.time()
            stop_monitoring.set()
            monitor_thread.join()

        elapsed_time = end_time - start_time

        final_mem = get_rss()
        peak_mem = max(peak_mem, final_mem)

        memory_used = max(0, peak_mem - baseline_mem) / 1024**2

        return result, elapsed_time, memory_used

    return wrapper_stats


def table(headers: list[str], title: str | None = None) -> Table:
    """Create a rich Table object with custom styles."""
    style_1 = {"style": "blue", "no_wrap": True, "justify": "left"}
    style_2 = {"style": "cyan", "no_wrap": True, "justify": "right"}
    fields = {}
    for i, header in enumerate(headers):
        fields[header] = style_1 if i == 0 else style_2
    table = Table(
        title=f"--- {title} ---",
        title_style="dark_orange",
        title_justify="left",
        header_style="orchid",
        show_edge=True,
        pad_edge=False,
        expand=False,
    )
    for field, style in fields.items():
        table.add_column(field, **style)
    return table


def get_logger() -> logging.Logger:
    """Configure logging for GPhub-kitPro."""
    file_log = Path.cwd() / "results" / "logger.log"
    file_log.parent.mkdir(parents=True, exist_ok=True)
    file_console = Console(file=file_log.open("w"), width=175)
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(console=file_console, show_time=True, show_path=True)],
    )

    logging.captureWarnings(True)
    return logging.getLogger(__name__)


install()
pretty.install()
console = Console(record=True, width=175, log_time=False, log_path=False)
disp_logo(console)
