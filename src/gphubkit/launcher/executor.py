"""Main executor engines for running GP benchmarks across languages."""

from io import StringIO
from abc import ABC, abstractmethod
from pathlib import Path
import sys
import time
import logging
import threading
import contextlib
import importlib.util

from rpy2 import robjects
from rpy2.robjects import numpy2ri
import numpy as np
import matlab
import psutil
import juliacall as jl
import matlab.engine

from ..utils import stats


class PythonGPLibrary(ABC):
    """Python GP library."""

    @abstractmethod
    def __init__(self, train_x: np.ndarray, train_y: np.ndarray) -> None:
        """Initialize the Python library for SMT."""
        msg = "Subclasses must implement init method"
        raise NotImplementedError(msg)

    @abstractmethod
    def train(self) -> None:
        """Train the model."""
        msg = "Subclasses must implement train method"
        raise NotImplementedError(msg)

    @abstractmethod
    def predict(self, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict the values and variances of the model."""
        msg = "Subclasses must implement predict method"
        raise NotImplementedError(msg)


def python_executor(
    script_path: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    logger: logging.Logger,
) -> tuple[np.ndarray, np.ndarray, float, float, float, float]:
    """Run a Python script with the given data."""
    # Dynamically import the module from script_path
    path = Path(script_path)
    module_name = path.stem
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        msg = f"Could not load spec for module at {script_path}"
        raise ImportError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    def init_model() -> PythonGPLibrary:
        """Initialize the model."""
        # Find the PythonGPLibrary subclass in the module
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, type) and issubclass(attr, PythonGPLibrary) and attr is not PythonGPLibrary:
                return attr(train_x, train_y)
        msg = f"No PythonGPLibrary subclass found in {script_path}"
        raise ValueError(msg)

    @stats
    def train_model(model: PythonGPLibrary) -> None:
        """Train the model."""
        model.train()

    @stats
    def predict_model(model: PythonGPLibrary) -> tuple[np.ndarray, np.ndarray]:
        """Predict with the model."""
        return model.predict(test_x)

    # Initialize
    model = init_model()
    # Train
    _, train_time, train_memory = train_model(model)
    # Predict
    (pred_y, pred_var), pred_time, pred_memory = predict_model(model)

    return pred_y, pred_var, train_time, train_memory, pred_time, pred_memory


def matlab_executor(
    script_path: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    logger: logging.Logger,
) -> tuple[np.ndarray, np.ndarray, float, float, float, float]:
    """Run the MATLAB script with the given data."""

    def get_matlab_pids() -> set[int]:
        """Get the PIDs of all running MATLAB processes."""
        pids = set()
        for proc in psutil.process_iter(["pid", "name"]):
            try:
                if "matlab" in proc.info["name"].lower() or "MATLAB" in proc.info["name"]:
                    pids.add(proc.info["pid"])
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return pids

    # Find existing MATLAB PIDs
    existing_matlab_pids = get_matlab_pids()

    # Start MATLAB engine with output capture
    eng = matlab.engine.start_matlab("-nodisplay")

    # Find the new MATLAB PID
    current_matlab_pids = get_matlab_pids()
    new_pids = current_matlab_pids - existing_matlab_pids

    matlab_pids_to_monitor: list[int] = []
    if new_pids:
        # We found the new process
        matlab_pids_to_monitor = list(new_pids)
        info_msg = f"Monitoring specific MATLAB process(es): {matlab_pids_to_monitor}"
        # logging.info(info_msg)
        logger.info(info_msg)

    else:
        msg = "Could not identify a specific new MATLAB process. Aborting measurement."
        logger.error(msg)
        raise RuntimeError(msg)

    # Convert data to MATLAB arrays
    train_x_matlab = matlab.double(train_x.tolist())
    train_y_matlab = matlab.double(train_y.reshape(-1, 1).tolist())
    test_x_matlab = matlab.double(test_x.tolist())

    # Add data to MATLAB workspace
    eng.workspace["train_x"] = train_x_matlab  # type: ignore
    eng.workspace["train_y"] = train_y_matlab  # type: ignore
    eng.workspace["test_x"] = test_x_matlab  # type: ignore

    def evaluator(action: str) -> tuple[float, float]:
        """Evaluate the model and measure MATLAB memory consumption using external monitoring."""
        eng.workspace["ACTION"] = action  # type: ignore

        def get_rss() -> float:
            """Get the total RSS memory of the monitored MATLAB processes."""
            total_rss = 0
            for pid in matlab_pids_to_monitor:
                try:
                    p = psutil.Process(pid)
                    total_rss += p.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            return total_rss

        baseline_memory = get_rss()
        peak_memory = baseline_memory

        stop_monitoring = threading.Event()
        monitor_thread = None

        def monitor_memory() -> None:
            """Monitor memory usage of MATLAB processes."""
            nonlocal peak_memory
            while not stop_monitoring.is_set():
                current_memory = get_rss()
                peak_memory = max(current_memory, peak_memory)
                time.sleep(0.01)  # Poll every 10ms

        if matlab_pids_to_monitor:
            monitor_thread = threading.Thread(target=monitor_memory)
            monitor_thread.start()

        start_time = time.time()

        # Capture MATLAB output
        output_buffer = StringIO()
        try:
            eng.run(script_path, nargout=0, stdout=output_buffer, stderr=output_buffer)  # type: ignore
            # Log any output from MATLAB
            output = output_buffer.getvalue()
            if output.strip():
                info_msg = f"MATLAB output: {output.strip()}"
                logger.info(info_msg)
        except Exception as e:
            err_msg = f"MATLAB execution error: {e}"
            logger.exception(err_msg)
        finally:
            output_buffer.close()

        end_time = time.time()

        if monitor_thread:
            stop_monitoring.set()
            monitor_thread.join()

        memory_used = max(0, peak_memory - baseline_memory) / 1024**2  # Convert to MB
        elapsed_time = end_time - start_time

        return elapsed_time, memory_used

    # Initialize MATLAB package (no memory measurement needed)
    eng.workspace["ACTION"] = "init"  # type: ignore
    output_buffer = StringIO()
    try:
        eng.run(script_path, nargout=0, stdout=output_buffer, stderr=output_buffer)  # type: ignore
        output = output_buffer.getvalue()
        if output.strip():
            info_msg = f"MATLAB init output: {output.strip()}"
            logger.info(info_msg)
    except Exception as e:
        err_msg = f"MATLAB init error: {e}"
        logger.exception(err_msg)
    finally:
        output_buffer.close()

    # Train the model with memory measurement
    train_time, train_memory = evaluator(action="train")
    # Predict using the trained model with memory measurement
    pred_time, pred_memory = evaluator(action="test")

    pred_y = np.array(eng.workspace["pred_y"])  # type: ignore
    pred_var = np.array(eng.workspace["pred_var"])  # type: ignore

    # Quit MATLAB engine
    eng.quit()  # type: ignore
    return pred_y, pred_var, train_time, train_memory, pred_time, pred_memory


def julia_executor(
    script_path: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    logger: logging.Logger,
) -> tuple[np.ndarray, np.ndarray, float, float, float, float]:
    """Run the Julia script with the given data."""
    # Initialize Julia
    se = jl.Main  # type: ignore

    # Convert data to Julia arrays
    se.train_x = train_x
    se.train_y = train_y.reshape(-1, 1)
    se.test_x = test_x

    @stats
    def evaluator(action: str) -> None:
        """Evaluate an action in Julia."""
        se.ACTION = action

        try:
            # Capture Julia's output to buffers
            se.seval("""
                using Base: IOBuffer
                original_stdout = stdout
                original_stderr = stderr
                stdout_buffer = IOBuffer()
                stderr_buffer = IOBuffer()
                redirect_stdout(stdout_buffer)
                redirect_stderr(stderr_buffer)
            """)

            se.include(script_path)

            # Get captured output
            se.seval("""
                redirect_stdout(original_stdout)
                redirect_stderr(original_stderr)
                captured_stdout = String(take!(stdout_buffer))
                captured_stderr = String(take!(stderr_buffer))
            """)

            # Log any captured output
            stdout_output = str(se.captured_stdout)
            stderr_output = str(se.captured_stderr)

            if stdout_output.strip():
                info_msg = f"Julia output: {stdout_output.strip()}"
                logger.info(info_msg)
            if stderr_output.strip():
                error_msg = f"Julia error output: {stderr_output.strip()}"
                logger.warning(error_msg)

        except Exception as e:
            err_msg = f"Julia execution error: {e}"
            logger.exception(err_msg)
        finally:
            # Restore original stdout/stderr
            try:
                se.seval("""
                    redirect_stdout(original_stdout)
                    redirect_stderr(original_stderr)
                """)
            except:
                pass

        return se

    # Initialize Julia package
    se, _, _ = evaluator(action="init")
    # Train the model
    se, train_time, train_memory = evaluator(action="train")
    # Predict using the trained model
    se, pred_time, pred_memory = evaluator(action="test")

    pred_y = np.array(se.pred_y)
    pred_var = np.array(se.pred_var)

    return pred_y, pred_var, train_time, train_memory, pred_time, pred_memory


def r_executor(
    script_path: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    logger: logging.Logger,
) -> tuple[np.ndarray, np.ndarray, float, float, float, float]:
    """Run the R script with the given data."""
    # Use context manager for numpy to R conversion
    with numpy2ri.converter.context():
        # Convert data to R objects
        robjects.globalenv["train_x"] = train_x
        robjects.globalenv["train_y"] = train_y.reshape(-1, 1)
        robjects.globalenv["test_x"] = test_x

        r_source = robjects.r["source"]

        @stats
        def evaluator(action: str) -> None:
            """Evaluate an action in R."""
            robjects.globalenv["ACTION"] = action
            # Temporarily disable numpy conversion to avoid S4 object issues
            with robjects.conversion.localconverter(robjects.default_converter):
                r_source(script_path)  # type: ignore

        # Initialize R package
        _, _, _ = evaluator(action="init")

        # Train the model
        _, train_time, train_memory = evaluator(action="train")
        # Predict using the trained model
        _, pred_time, pred_memory = evaluator(action="test")

        # Extract values directly from global environment
        pred_y = np.array(robjects.globalenv["pred_y"])
        pred_var = np.array(robjects.globalenv["pred_var"])

        return pred_y, pred_var, train_time, train_memory, pred_time, pred_memory
