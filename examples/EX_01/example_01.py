"""BM_02 benchmark."""

import gphubkit as gpk

if __name__ == "__main__":
    # LOAD BENCHMARK
    benchmark = gpk.benchmark.BM02()
    # RUN BENCHMARK
    gpk.run(benchmark)
    # POSTPROCESS LIBRARIES
    gpk.postprocess()
