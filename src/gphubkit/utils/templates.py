"""Templates for the libraries."""


def tmpl_py(lib_name: str) -> str:
    """Return the template for a Python library."""
    return f"""\"\"\"{lib_name} Python library.\"\"\"

# This script is designed to be called from Python with different ACTIONs.
# The variables train_x, train_y, test_x are passed from Python.

# import your_python_library_here
import numpy as np

from gphubkit.launcher.executor import PythonGPLibrary


class {lib_name}Library(PythonGPLibrary):
    \"\"\"{lib_name} Library.\"\"\"

    def __init__(self, train_x: np.ndarray, train_y: np.ndarray) -> None:
        \"\"\"INITIALIZE {lib_name}.\"\"\"
        # ...
        # self.gp_model = ...

    def train(self) -> None:
        \"\"\"TRAINING.\"\"\"
        # ...

    def predict(self, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        \"\"\"PREDICTION.\"\"\"
        # pred_y = ....
        # pred_var = ....
        return pred_y, pred_var
"""


def tmpl_m(lib_name: str) -> str:
    """Return the template for a Matlab library."""
    return f"""% {lib_name} Matlab library.

# This script is designed to be called from Python with diff\rent ACTIONs.
# The variables train_x, train_y, test_x are passed from Python.

switch ACTION
    case 'init'
        % INITIALIZE {lib_name}.
        % addpath('path/to/the/matlab/library/')
        % ...
    case 'train'
        % TRAINING.
        % ...
    case 'test'
        % PREDICTION.
        % [pred_y, pred_var] = ...
end
"""


def tmpl_jl(lib_name: str) -> str:
    """Return the template for a Julia library."""
    return f"""# {lib_name} Julia library.

# using ...

# This script is designed to be called from Python with different ACTIONs.
# The variables train_x, train_y, test_x are passed from Python.

if ACTION == "init"
    # INITIALIZE {lib_name}.
    # global ...

elseif ACTION == "train"
    # TRAINING.
    ...

elseif ACTION == "test"
    # PREDICTION.
    # ...
    # global pred_y, pred_var = ...
end

"""


def tmpl_r(lib_name: str) -> str:
    """Return the template for a R library."""
    return f"""# {lib_name} R library.

# This script is designed to be called from Python with different ACTIONs.
# The variables train_x, train_y, test_x are passed from Python.

switch(ACTION,
  "init" = {{
    # INITIALIZE {lib_name}.
    # library(import_your_R_library_here)
    # ...
    # Convert data to appropriate formats
    train_x <<- as.matrix(train_x)
    train_y <<- as.numeric(train_y)
    test_x <<- as.matrix(test_x)
  }},
  "train" = {{
    # TRAINING.
    # ...
  }},
  "test" = {{
    # PREDICTION.
    # ...
    # pred_y <<- ...
    # pred_var <<- ...
  }}
)
"""
