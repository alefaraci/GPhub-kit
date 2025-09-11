"""Custom benchmark from local data."""

import gphubkit as gpk

if __name__ == "__main__":
    # LOAD LOCAL DATA
    train_x, test_x, train_y, test_y = gpk.data.load(file_path=gpk.utils.Path(__file__).parent.resolve() / "data")
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
