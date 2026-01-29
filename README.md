# Contextual Bandits for Best-Arm Identification

A research implementation of contextual bandit algorithms for the multi-armed bandit problem. This repo contains implementations several algorithms that identify the optimal arm in linear and nonlinear reward settings.

## Algorithms

### Linear Bandit Algorithms

These algorithms assume rewards are linear in arm features: $\mathbb{E}[r|x] = x^\top θ^*$

- **Thompson Sampling** (`ThompsonSampling`): Samples from the posterior over θ and pulls the arm with highest expected reward under the sample.

- **Top-Two Thompson Sampling** (`TopTwoAlgorithm`): Identifies the two most likely optimal arms under posterior samples, then pulls the arm that best distinguishes between them.

- **XY-Static** (`XYStatic`): Computes an optimal static allocation using Frank-Wolfe optimization for G-optimal experimental design, then samples arms according to this fixed design.

- **XY-Adaptive** (`XYAdaptive`): Adaptively updates the sampling allocation at each step based on the current posterior over likely optimal arms.

- **GeneralTopTwoLinear** (`GeneralTopTwoLinear`): Top-Two algorithm using a Gaussian posterior, compatible with the general concept bandit interface.

### Concept Bandit Algorithms

These algorithms work with general function classes, not just linear:

- **GeneralThompson** (`GeneralThompson`): Thompson Sampling for general function classes using posterior sampling (e.g., CatBoost bootstrap).

- **GeneralTopTwo** (`GeneralTopTwo`): Top-Two algorithm for concept bandits using bootstrap-based posteriors.

## Installation

### Using uv (recommended)

```bash
# Initialize project and install dependencies
uv sync

# Optional: Add ray for distributed computing
uv add ray
```

### Using pip

```bash
pip install -r requirements.txt
```

## Usage

### Running Experiments

Create a JSON configuration file (e.g., `config.json`) specifying the experiment parameters:

```json
{
  "global": {
    "T": 1000,
    "sigma": 1.0,
    "reps": 10,
    "cpu": 4,
    "parallelize": "mp",
    "instance": {
      "name": "sphere",
      "params": {"K": 10, "d": 5}
    }
  },
  "algs": [
    {
      "name": "ts",
      "alg_class": "ThompsonSampling",
      "params": {},
      "active": true
    },
    {
      "name": "toptwo",
      "alg_class": "TopTwoAlgorithm",
      "params": {},
      "active": true
    }
  ]
}
```

Run the experiment:

```bash
uv run python run.py --config config.json --path ./results
```

This will save a plot of identification rates to `results/results.png`.

### Problem Instances

The following problem instances are available:

- **sphere**: Arms and optimal parameter uniformly sampled from the unit sphere
  - Parameters: `K` (number of arms), `d` (dimension)

- **soare**: The Soare instance, a standard hard instance for best-arm identification
  - Parameters: `d` (dimension), `alpha` (angle parameter controlling difficulty)

- **entropy**: Binary entropy maximization (nonlinear rewards)
  - Parameters: `K` (number of arms)

## File Structure

```
contextualbandits/
├── bandit_type.py      # Base classes (Linear, Concept) and TopTwoMixin
├── config.py           # Configuration constants
├── distribution.py     # Posterior distributions (Gaussian, Kitten/CatBoost)
├── instance.py         # Problem instance generators
├── library_concept.py  # Concept bandit algorithms
├── library_linear.py   # Linear bandit algorithms
├── run.py              # Main experiment runner
├── utils.py            # Utility functions (Frank-Wolfe, etc.)
├── pyproject.toml      # Project configuration for uv/pip
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## Configuration

Key constants in `config.py`:

- `BOOTSTRAP_SAMPLES_SMALL/LARGE`: Number of bootstrap samples for variance estimation
- `DELTA`: Probability threshold for early stopping (default: 0.0001)
- `MAX_LOOP_ITERATIONS`: Maximum iterations when searching for second-best arm
- Thread settings for numpy parallelism control
