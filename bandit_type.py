"""Base classes for contextual bandit algorithms."""

from __future__ import annotations
from typing import Any, Union

import numpy as np
from numpy.typing import NDArray

import distribution
from config import DELTA


class TopTwoMixin:
    """Mixin class containing shared Top-Two algorithm logic.

    Classes using this mixin must have:
    - self.X: arm feature matrix
    - self.n: number of arms
    - self.T: time horizon
    - self.pi: posterior distribution
    - self.gen_star: true function for pulling arms
    - self.B: number of bootstrap samples
    - self.delta: probability threshold
    - self.pulled: list to track pulled arms
    - self.arms_recommended: list to track recommendations
    - self.name: algorithm name
    """

    X: NDArray
    n: int
    T: int
    pi: Any
    gen_star: distribution.GenericFunction
    B: int
    delta: float
    pulled: list[int]
    arms_recommended: list[int]
    name: str

    def _find_second_best(self, idx1: int, k: int) -> tuple[int, bool]:
        """Find second-best arm different from idx1.

        Args:
            idx1: Index of the current best arm.
            k: Batch size for sampling candidates.

        Returns:
            Tuple of (idx2, converged) where converged is False if loop limit hit.
        """
        a = 0
        idx2 = idx1
        while idx1 == idx2:
            f2s = self.pi.sample(k)
            for f2 in f2s:
                idx2 = np.argmax(f2.evaluate(self.X))
                if idx1 != idx2:
                    break
            a += k
            if a > 1 / self.delta:
                return idx2, False
        return idx2, True

    def _compute_exploration_variance(
        self,
        x1: NDArray,
        x2: NDArray,
        t: int
    ) -> list[float]:
        """Compute variance of expected difference for each arm.

        Args:
            x1: First top arm features.
            x2: Second top arm features.
            t: Current time step (used for weighting).

        Returns:
            List of variances for each arm.
        """
        v = []
        for idx in range(self.n):
            x = self.X[idx]
            expected_diff = 0.0
            expected_diff_squared = 0.0

            for _ in range(self.B):
                gen_b1 = self.pi.sample()
                y_b1 = gen_b1.pull(x)
                weight = t
                pi_plus = self.pi.update_posterior(
                    x * weight,
                    self._wrap_observation(y_b1),
                    copy=True
                )
                gen_b2 = pi_plus.sample()
                diff = gen_b2.evaluate(x1) - gen_b2.evaluate(x2)
                expected_diff += diff
                expected_diff_squared += diff ** 2
            v.append(expected_diff_squared / self.B - (expected_diff / self.B) ** 2)
        return v

    def _wrap_observation(self, y: Any) -> Any:
        """Wrap observation for update_posterior. Override in subclasses if needed.

        Args:
            y: Raw observation value.

        Returns:
            Wrapped observation suitable for the posterior's update_posterior method.
        """
        return y

    def _pad_recommendations(self) -> None:
        """Pad arms_recommended to length T if algorithm terminated early."""
        quit_len = len(self.arms_recommended)
        if quit_len < self.T:
            rec = self.arms_recommended[-1]
            for _ in range(quit_len, self.T):
                self.arms_recommended.append(rec)

    def run_top_two(self, logging_period: int = 1, k: int = 10) -> None:
        """Run the Top-Two algorithm.

        Args:
            logging_period: How often to print progress.
            k: Batch size for sampling second-best candidates.
        """
        for t in range(self.T):
            f1 = self.pi.sample()
            idx1 = np.argmax(f1.evaluate(self.X))
            x1 = self.X[idx1]

            idx2, converged = self._find_second_best(idx1, k)
            if not converged:
                break

            x2 = self.X[idx2]
            v = self._compute_exploration_variance(x1, x2, t)

            min_idx = np.argmin(v)
            self.pulled.append(min_idx)
            x_n = self.X[min_idx]
            y_n = self.gen_star.pull(x_n)
            self.pi.update_posterior(x_n, self._wrap_observation(y_n))

            fhat = self.pi.map()
            idx = np.argmax(fhat.evaluate(self.X))
            self.arms_recommended.append(idx)

            if t % logging_period == 0:
                print('general run', self.name, 'iter', t, "/", self.T, end="\r")

        self._pad_recommendations()


class Linear:
    """Base class for linear bandit algorithms.

    Linear bandits assume rewards are linear in the arm features:
    E[r | x] = x^T theta_star
    """

    def __init__(
        self,
        X: NDArray,
        Y: NDArray,
        theta_star: NDArray,
        T: int,
        sigma: float = 1,
        name: str = ""
    ) -> None:
        """Initialize linear bandit.

        Args:
            X: Arm feature matrix of shape (n_arms, d).
            Y: Pairwise difference matrix for experimental design.
            theta_star: True parameter vector.
            T: Time horizon (number of rounds).
            sigma: Observation noise standard deviation.
            name: Algorithm instance name for logging.
        """
        self.X = X
        self.Y = Y
        self.n, self.d = X.shape
        self.theta_star = theta_star

        self.V = np.eye(self.d)

        self.T = T
        self.sigma = sigma
        self.name = name
        self.arms_recommended: list[int] = []


class Concept:
    """Base class for concept bandit algorithms.

    Concept bandits work with general function classes, not just linear.
    """

    def __init__(
        self,
        X: NDArray,
        gen_star: distribution.GenericFunction,
        pi: distribution.Distribution,
        T: int,
        sigma: float = 1,
        name: str = ""
    ) -> None:
        """Initialize concept bandit.

        Args:
            X: Arm feature matrix of shape (n_arms, d).
            gen_star: True reward function (GenericFunction object).
            pi: Prior/posterior distribution over functions.
            T: Time horizon (number of rounds).
            sigma: Observation noise standard deviation.
            name: Algorithm instance name for logging.

        Raises:
            Exception: If gen_star is not a GenericFunction.
        """
        self.X = X
        self.n = X.shape[0]
        if type(gen_star) is not distribution.GenericFunction:
            raise Exception('gen_star must be a GenericFunction object')
        self.gen_star = gen_star
        self.pi = pi

        self.T = T
        self.sigma = sigma
        self.name = name
        self.arms_recommended: list[int] = []
