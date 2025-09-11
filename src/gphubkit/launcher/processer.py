"""Post-process module for GPhub-kit."""

from typing import cast
from pathlib import Path
import os
import time

from rich.live import Live
from rich.spinner import Spinner
from rich.progress import Progress, BarColumn, TextColumn, SpinnerColumn, TimeElapsedColumn, TaskProgressColumn

from .. import plotter
from ..utils import Console, table, console, gphubkit_logo, get_main_script_path
from ..metrics import GPlibrary


def __get_script_files(scripts_dir: Path) -> list[str]:
    """Get all library files and organize by language."""
    return [f for f in os.listdir(scripts_dir) if (scripts_dir / f).is_file()]


def __get_libs(all_files: list[str]) -> list[str]:
    """Get all library files and organize by language."""
    return [f for f in all_files if f.endswith(".parquet")]


def __postprocess_library(scripts_dir: Path, library: str, *, display: bool = False) -> GPlibrary:
    """Post-process a library."""
    lib = GPlibrary(library=library)
    lib.print_metrics() if display else None
    lib.plot_results(path=scripts_dir.resolve())
    return lib


def postprocess() -> None:
    """Post-process all libraries."""
    caller_dir = get_main_script_path()
    all_scripts = __get_script_files(caller_dir / "results" / "storage")
    all_libs = __get_libs(all_scripts)

    tab, gp_libs = None, {}

    console.print("[bold yellow]🛠️  Post-processing results...")

    with Progress(
        SpinnerColumn(),
        TextColumn(" "),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        "[progress.description]{task.description}",
        console=console,
    ) as progress:
        task_id = progress.add_task("[bold green]🔄  Running libraries...", total=len(all_libs))
        for script_file in all_libs:
            extension = script_file.split(".")[-1]
            filename = script_file.removesuffix(f".{extension}").removeprefix("lib_")
            scripts_dir = caller_dir / "results" / "img"
            progress.update(task_id, description=f"[cyan] |  [light_coral]Library: [blue]{filename}", cas="cwqd")
            gp_libs[filename] = __postprocess_library(scripts_dir, filename, display=False)
            tab = table(headers=gp_libs[filename]._metrics_header, title="Report Metrics") if tab is None else tab
            tab.add_row(*gp_libs[filename]._metrics_row)
            progress.advance(task_id)
        progress.update(task_id, description="[cyan] |  [bold green]✅  Done!\n", cas="cwqd")

    spinner = Spinner(
        "dots",
        text="[bold yellow]  Creating comparative plots and report...[/] [cyan] |  [bold cyan]⏳  Working...[/]",
        speed=2.5,
    )
    with Live(spinner, console=console, refresh_per_second=10):
        # COMPARISON PLOTS
        out_path = caller_dir / "results" / "img"
        plotter.radar._adimensional_metrics_by_library(gp_libs, path=out_path)
        plotter.radar._metrics(gp_libs, path=out_path)

        with Path(os.devnull).open("w") as devnull:
            writing_console = Console(record=True, width=175, log_time=False, log_path=False, file=devnull)
            writing_console.print(gphubkit_logo)
            writing_console.print(tab)
            writing_console.save_text((caller_dir / "results" / "report.log").resolve().__str__())
        time.sleep(0.25)
        spinner.update(
            text="[bold yellow]  Creating comparative plots and report...[/] [cyan] |  [bold green]✅  Done!\n"
        )
