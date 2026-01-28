"""Problem instance generators for contextual bandit experiments."""

from typing import Union

import numpy as np
from numpy.typing import NDArray

import distribution


def sphere(K: int, d: int) -> tuple[NDArray, NDArray]:
    """Generate arms and optimal parameter for a unit sphere instance.

    Arms are sampled uniformly from the unit sphere, and the optimal
    parameter vector is also on the unit sphere.

    Args:
        K: Number of arms.
        d: Dimension of each arm.

    Returns:
        Tuple of (X, theta_star) where:
        - X: Arm feature matrix of shape (K, d), rows are unit vectors.
        - theta_star: Optimal parameter vector of shape (d,), unit norm.
    """
    X = np.random.randn(K, d) / np.sqrt(d)
    norms = np.linalg.norm(X, axis=1).reshape(K, 1)
    X /= norms
    theta_star = np.random.randn(d)
    theta_star = theta_star / np.linalg.norm(theta_star)
    return X, theta_star


def soare(d: int, alpha: float) -> tuple[NDArray, NDArray]:
    """Generate the Soare instance for best-arm identification.

    This is a standard hard instance where one arm is close to optimal
    but requires many samples to distinguish.

    Args:
        d: Dimension of the arm space.
        alpha: Angle parameter controlling problem difficulty.

    Returns:
        Tuple of (X, theta_star) where:
        - X: Arm feature matrix of shape (d+1, d).
        - theta_star: Optimal parameter vector (2 * e_1).
    """
    X = np.eye(d)
    e_1, e_2 = X[:2]
    x_prime = np.cos(alpha) * e_1 + np.sin(alpha) * e_2
    X = np.concatenate([X, np.array([x_prime])])
    return X, 2 * e_1


def entropy(K: int) -> tuple[NDArray, distribution.GenericFunction]:
    """Generate an entropy maximization instance.

    The reward function is the binary entropy function.

    Args:
        K: Number of arms.

    Returns:
        Tuple of (X, f_star) where:
        - X: Arm feature matrix of shape (K, 1) with values in (0, 1).
        - f_star: GenericFunction representing binary entropy.
    """
    f_star = distribution.GenericFunction(
        lambda x: -x * np.log(x) - (1 - x) * np.log(1 - x),
        0.1
    )
    return np.random.rand(K).reshape(-1, 1), f_star


def get_instance(
    name: str,
    params: dict
) -> Union[tuple[NDArray, NDArray], tuple[NDArray, distribution.GenericFunction]]:
    """Factory function to create problem instances from config.

    Args:
        name: Instance type ('soare', 'sphere', or 'entropy').
        params: Dictionary of parameters to pass to the instance generator.

    Returns:
        Tuple of (X, f_star) where X is the arm matrix and f_star is
        either a parameter vector (linear) or GenericFunction (nonlinear).
    """
    if name == 'soare':
        f = soare
    elif name == 'sphere':
        f = sphere
    elif name == 'entropy':
        f = entropy
    return f(**params)
