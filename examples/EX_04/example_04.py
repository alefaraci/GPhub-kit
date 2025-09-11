"""Custom benchmark from local dataset."""

import gphubkit as gpk

if __name__ == "__main__":
    # LOAD LOCAL DATASET WITH SPLIT INTO TRAINING AND TEST SETS
    train_x, test_x, train_y, test_y = gpk.data.split_dataset(
        file_path=gpk.utils.Path(__file__).parent.resolve() / "data",
        test_size=0.3,
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
