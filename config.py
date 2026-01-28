"""Configuration constants for contextual bandit algorithms."""

# Bootstrap samples for variance estimation
BOOTSTRAP_SAMPLES_SMALL = 5  # Used in GeneralTopTwo (concept)
BOOTSTRAP_SAMPLES_LARGE = 20  # Used in GeneralThompson and GeneralTopTwoLinear

# Probability threshold for early stopping
DELTA = 0.0001

# Maximum loop iterations (for finding second-best arm)
MAX_LOOP_ITERATIONS = 10000

# Thread settings (for limiting numpy parallelism)
OMP_NUM_THREADS = 1
OPENBLAS_NUM_THREADS = 1
MKL_NUM_THREADS = 1
