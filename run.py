"""Main script for running contextual bandit experiments."""

import os
from config import OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, MKL_NUM_THREADS

os.environ['OMP_NUM_THREADS'] = str(OMP_NUM_THREADS)
os.environ['OPENBLAS_NUM_THREADS'] = str(OPENBLAS_NUM_THREADS)
os.environ['MKL_NUM_THREADS'] = str(MKL_NUM_THREADS)

import argparse
import json
import multiprocessing as mp
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from distribution import GenericFunction
from instance import get_instance
from utils import compute_Y, get_alg


def worker(
    alg: dict,
    X: NDArray,
    Y: NDArray,
    f_star: Union[NDArray, GenericFunction],
    T: int,
    sigma: float,
    ix: int
) -> list[int]:
    """Run a single algorithm instance.

    Args:
        alg: Algorithm configuration dict.
        X: Arm feature matrix.
        Y: Pairwise difference matrix.
        f_star: True reward function.
        T: Time horizon.
        sigma: Observation noise.
        ix: Instance index.

    Returns:
        List of recommended arm indices at each time step.
    """
    print('algorithm', alg, 'name', alg['name'])
    np.random.seed()
    algorithm_instance = get_alg(alg, X, Y, f_star, T, sigma, ix)
    algorithm_instance.run(logging_period=100)
    return algorithm_instance.arms_recommended


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run contextual bandit experiments'
    )
    parser.add_argument('--config', type=str, required=True,
                        help='Path to JSON configuration file')
    parser.add_argument('--path', default=os.getcwd(), type=str,
                        help='Output directory for results')
    args = parser.parse_args()

    with open(args.config, "r") as f:
        params = json.load(f)
        print(params)

    T: int = params['global']['T']
    sigma: float = params['global']['sigma']
    reps: int = params['global']['reps']
    cpu: int = params['global']['cpu']
    path: str = args.path
    print('OUR PATH', path)

    X, f_star = get_instance(**params['global']['instance'])
    Y = compute_Y(X)

    algorithms = [alg for alg in params['algs'] if alg['active']]
    runs = []

    for alg in algorithms:
        runs += [(alg, X, Y, f_star, T, sigma, i) for i in range(reps)]

    if params['global']['parallelize'] == 'mp':
        pool = mp.Pool(cpu, maxtasksperchild=1000)
        all_results = pool.starmap(worker, runs)
    else:
        import ray
        ray.init()
        all_results = []
        for a in runs:
            all_results.append(worker(*a))

    # Handle both linear (ndarray) and concept (GenericFunction) cases
    if hasattr(f_star, 'evaluate'):
        idx_star = np.argmax(f_star.evaluate(X))
    else:
        idx_star = np.argmax(X @ f_star)
    K = X.shape[0]
    d = X.shape[1]
    xaxis = np.arange(T)

    for i, alg in enumerate(algorithms):
        results = all_results[reps * i: reps * (i + 1)]
        print('results', [len(results[j]) for j in range(reps)])
        results_arr = np.array(results)
        m = (results_arr == idx_star).mean(axis=0)
        s = (results_arr == idx_star).std(axis=0) / np.sqrt(reps)
        plt.plot(xaxis, m)
        plt.fill_between(xaxis, m - s, m + s, alpha=0.2, label=alg['alg_class'])

    plt.xlabel('time')
    plt.ylabel('identification rate')
    plt.legend()
    plt.savefig(path + '/results.png')
