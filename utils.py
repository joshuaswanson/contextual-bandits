"""Utility functions for contextual bandit algorithms."""

from typing import TYPE_CHECKING, Optional, Union

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt

import distribution

if TYPE_CHECKING:
    import library_concept
    import library_linear


def A(X: NDArray, lambda_: NDArray) -> NDArray:
    """Compute weighted design matrix.

    Args:
        X: Feature matrix of shape (n, d).
        lambda_: Weight vector of shape (n,).

    Returns:
        Weighted covariance matrix X^T diag(lambda) X of shape (d, d).
    """
    return X.T @ np.diag(lambda_) @ X


def calc_max_mat_norm(Y: NDArray, A_inv: NDArray) -> tuple[float, int]:
    """Find the maximum matrix norm over rows of Y.

    Args:
        Y: Matrix of shape (n, d).
        A_inv: Inverse matrix of shape (d, d).

    Returns:
        Tuple of (max_norm, index) where max_norm is the maximum y^T A_inv y
        and index is the row achieving it.
    """
    n = Y.shape[0]
    res = np.zeros(n)
    for i in range(n):
        y = Y[i]
        res[i] = y.T @ A_inv @ y
    ind = np.argmax(res)
    return res[ind], ind


def compute_Y(X: NDArray) -> NDArray:
    """Compute pairwise difference matrix for experimental design.

    Args:
        X: Feature matrix of shape (n, d).

    Returns:
        Matrix of shape (n*(n-1)/2, d) containing all pairwise differences X[i] - X[j].
    """
    n = X.shape[0]
    Y = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            Y.append(X[i] - X[j])
    return np.array(Y)


def FW(
    X: NDArray,
    Y: NDArray,
    reg_l2: float = 0,
    iters: int = 500,
    step_size: float = 1,
    logging_step: int = 10000,
    verbose: bool = False,
    initial: Optional[NDArray] = None
) -> tuple[NDArray, NDArray, list[float]]:
    """Frank-Wolfe algorithm for G-optimal experimental design.

    Computes the optimal allocation over arms to minimize the maximum
    variance of pairwise comparisons.

    Args:
        X: Arm feature matrix of shape (n, d).
        Y: Pairwise difference matrix.
        reg_l2: L2 regularization parameter.
        iters: Number of iterations.
        step_size: Initial step size (decays as 1/(t+2)).
        logging_step: How often to log when verbose=True.
        verbose: Whether to plot progress.
        initial: Initial design (uniform if None).

    Returns:
        Tuple of (design, rho, history) where:
        - design: Optimal allocation vector of shape (n,).
        - rho: Final variance values.
        - history: List of max rho values at logging steps.
    """
    n, d = X.shape
    I = np.eye(n)
    if initial is not None:
        design = initial
    else:
        design = np.ones(n)
        design /= design.sum()
    eta = step_size
    grad_norms = []
    history = []

    for count in range(1, iters):
        A_inv = np.linalg.pinv(X.T @ np.diag(design) @ X + reg_l2 * np.eye(d))
        rho = np.matmul(
            np.matmul(Y.reshape(-1, 1, d), A_inv),
            Y.reshape(-1, d, 1)
        ).reshape(Y.shape[0],)
        y_opt = Y[np.argmax(rho), :]
        g = y_opt @ A_inv @ X.T
        g = -g * g

        eta = step_size / (count + 2)
        imin = np.argmin(g)
        design = (1 - eta) * design + eta * I[imin]
        grad_norms.append(np.linalg.norm(g - np.sum(g) / n * np.ones(n)))

        if verbose and count % logging_step == 0:
            history.append(np.max(rho))
            fig, ax = plt.subplots(1, 2)
            ax[0].plot(grad_norms)
            ax[1].plot(design)
            plt.show()

    return design, rho, history


def pi(self, theta: NDArray, V: NDArray, idx: int, repeat: int = 10000) -> float:
    """Estimate probability that arm idx is optimal via Monte Carlo.

    Args:
        self: Object with self.X attribute (arm features).
        theta: Current parameter estimate.
        V: Covariance matrix for sampling.
        idx: Arm index to evaluate.
        repeat: Number of Monte Carlo samples.

    Returns:
        Estimated probability that arm idx is optimal.
    """
    x_star = self.X[idx]
    count = 0
    for _ in range(repeat):
        random_theta = np.random.multivariate_normal(theta, V)
        count += (idx == np.argmax(self.X @ theta))
    return count / repeat


def fast_rank_one(B: NDArray, v: NDArray) -> NDArray:
    """Sherman-Morrison formula for rank-one update of inverse.

    Computes (B^{-1} + vv^T)^{-1} given B^{-1}.

    Args:
        B: Current inverse matrix.
        v: Vector for rank-one update.

    Returns:
        Updated inverse matrix.
    """
    x = B @ v
    return B - np.outer(x, x) / (1 + v.T @ B @ v)


def get_alg(
    alg: dict,
    X: NDArray,
    Y: NDArray,
    f_star: Union[NDArray, distribution.GenericFunction],
    T: int,
    sigma: float,
    ix: int
):
    """Factory function to create algorithm instances from config.

    Args:
        alg: Algorithm configuration dict with 'name', 'alg_class', and 'params'.
        X: Arm feature matrix.
        Y: Pairwise difference matrix.
        f_star: True reward function (array for linear, GenericFunction for nonlinear).
        T: Time horizon.
        sigma: Observation noise.
        ix: Instance index (appended to name for logging).

    Returns:
        Instantiated algorithm object.

    Raises:
        Exception: If f_star type is invalid for concept algorithms.
    """
    import library_linear
    import library_concept

    name = alg['name'] + f'_{ix}'
    cls = alg['alg_class']
    params = alg['params']
    if cls == 'ThompsonSampling':
        return library_linear.ThompsonSampling(X, Y, f_star, T, sigma, name)
    elif cls == 'TopTwoAlgorithm':
        return library_linear.TopTwoAlgorithm(X, Y, f_star, T, sigma, name)
    elif cls == 'XYStatic':
        return library_linear.XYStatic(X, Y, f_star, T, sigma, name)
    elif cls == 'XYAdaptive':
        return library_linear.XYAdaptive(X, Y, f_star, T, sigma, name)
    elif cls == 'GeneralTopTwoLinear':
        return library_linear.GeneralTopTwoLinear(X, Y, f_star, T, sigma, name)
    elif cls == 'GeneralThompson':
        pi_dist = distribution.get_distribution(params['distribution'])
        if type(f_star) is np.ndarray:
            gen_star = distribution.GenericFunction(lambda x: x @ f_star, sigma)
        elif type(f_star) is not distribution.GenericFunction:
            raise Exception('f_star must be a GenericFunction object or a np.ndarray')
        else:
            gen_star = f_star
        return library_concept.GeneralThompson(X, gen_star, pi_dist, T, sigma, name)
    elif cls == 'GeneralTopTwo':
        pi_dist = distribution.get_distribution(params['distribution'])
        if type(f_star) is np.ndarray:
            gen_star = distribution.GenericFunction(lambda x: x @ f_star, sigma)
        elif type(f_star) is not distribution.GenericFunction:
            raise Exception('f_star must be a GenericFunction object or a np.ndarray')
        else:
            gen_star = f_star
        return library_concept.GeneralTopTwo(X, gen_star, pi_dist, T, sigma, name)
