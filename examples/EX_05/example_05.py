"""Benchmark with synthetic dataset."""

import numpy as np

import gphubkit as gpk


# TOY MODEL
def toy_fn(X: np.ndarray) -> np.ndarray:
    """Toy function."""
    a, r, s = 1, 6, 10
    b = 5.1 / (4 * np.pi**2)
    c = 5 / np.pi
    t = 1 / (8 * np.pi)
    term1 = a * (X[:, 1] - b * X[:, 0] ** 2 + c * X[:, 0] - r) ** 2
    term2 = s * (1 - t) * np.cos(X[:, 0])
    return term1 + term2 + s


if __name__ == "__main__":
    # GENERATE SYNTHETIC DATA
    train_x, test_x, train_y, test_y = gpk.data.generate_synthetic(
        file_path=gpk.utils.Path(__file__).parent.resolve() / "data",
        fn=toy_fn,
        bounds=np.array([[0, 0], [15, 12]]),
        data_size=100,
        train_size=20,
        method="lhs",
        seed=32,
    )
    # INITIALIZE BENCHMARK
    benchmark = gpk.benchmark.Custom(
        id="custom",
        train_x=train_x,
        train_y=train_y,
        test_x=test_x,
        test_y=test_y,
        scale_x=True,
        standardize_y=True,
    )
    # RUN BENCHMARK
    gpk.run(benchmark)
    # POSTPROCESS LIBRARIES
    gpk.postprocess()
