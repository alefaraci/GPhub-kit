"""Composite Shell benchmark."""

import gphubkit as gpk

if __name__ == "__main__":
    # LOAD BENCHMARK
    benchmark = gpk.benchmark.CompositeShell(dim=2, train_size=100)
    # RUN BENCHMARK
    gpk.run(benchmark)
    # POSTPROCESS LIBRARIES
    gpk.postprocess()
