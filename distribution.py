"""Probability distributions and function representations for bandit algorithms."""

from __future__ import annotations
from typing import Callable, Optional, Union

import numpy as np
from numpy.typing import NDArray
from catboost import CatBoostRegressor


def get_distribution(dist: dict) -> Distribution:
    """Factory function to create distribution instances from config dict.

    Args:
        dist: Dictionary with 'name' and 'd' (dimension) keys.

    Returns:
        Distribution instance (Kitten or Gaussian).
    """
    if dist['name'] == 'Kitten':
        return Kitten(np.empty((0, dist['d'])), np.empty((0)), None)
    elif dist['name'] == 'Gaussian':
        d = dist['d']
        return Gaussian(np.zeros(d), np.eye(d))


class Distribution:
    """Base class for probability distributions over functions."""
    pass


class GenericFunction:
    """Wrapper for callable functions with noise for bandit observations."""

    def __init__(self, f: Callable[[NDArray], NDArray], sigma: float = 1) -> None:
        """Initialize a generic function.

        Args:
            f: Callable that takes feature array and returns values.
            sigma: Standard deviation of observation noise.
        """
        self.f = f
        self.sigma = sigma

    def pull(self, x: NDArray) -> NDArray:
        """Evaluate the function at x and add random noise.

        Args:
            x: Input features.

        Returns:
            Function value plus Gaussian noise.
        """
        noise = np.random.randn(*x.shape[:-1], 1) * self.sigma
        return self.evaluate(x) + noise.squeeze(axis=-1)

    def evaluate(self, x: NDArray) -> Union[float, NDArray]:
        """Evaluate the function at x without noise.

        Args:
            x: Input features.

        Returns:
            Function value(s).
        """
        y = self.f(x)
        if type(y) is list or type(y) is np.ndarray:
            if len(y) == 1:
                y = y[0]
        return y


class Gaussian(Distribution):
    """Gaussian posterior distribution for linear bandits."""

    def __init__(
        self,
        theta: NDArray,
        V: NDArray,
        Vinv: Optional[NDArray] = None,
        S: Union[int, NDArray] = 0,
        sigma: float = 1
    ) -> None:
        """Initialize Gaussian posterior.

        Args:
            theta: Mean parameter vector.
            V: Covariance matrix (precision).
            Vinv: Inverse covariance (computed if None).
            S: Sufficient statistic for updates.
            sigma: Observation noise standard deviation.
        """
        super().__init__()
        self.theta = theta
        self.V = V
        self.S = S
        self.sigma = sigma
        if Vinv is None:
            self.Vinv = np.linalg.inv(V)
        else:
            self.Vinv = Vinv

    def update_posterior(
        self,
        x: NDArray,
        y: Union[float, NDArray],
        copy: bool = False
    ) -> Optional[Gaussian]:
        """Update posterior with new observation.

        Args:
            x: Feature vector of pulled arm.
            y: Observed reward.
            copy: If True, return new Gaussian; otherwise update in-place.

        Returns:
            New Gaussian if copy=True, else None.
        """
        if copy:
            V = self.V + np.outer(x, x)
            S = self.S + np.dot(x, y)
            theta = np.linalg.inv(V) @ S
            return Gaussian(theta, V, Vinv=np.linalg.inv(V), S=S)
        else:
            self.V += np.outer(x, x)
            self.S += x * y
            self.Vinv = np.linalg.inv(self.V)
            self.theta = self.Vinv @ self.S

    def sample(self, k: int = 1) -> Union[GenericFunction, list[GenericFunction]]:
        """Sample function(s) from the posterior.

        Args:
            k: Number of samples to draw.

        Returns:
            Single GenericFunction if k=1, else list of GenericFunctions.
        """
        if k == 1:
            theta_tilde = np.random.multivariate_normal(self.theta, self.Vinv)
            return GenericFunction(lambda x: x @ theta_tilde, self.sigma)

        theta_tilde = np.random.multivariate_normal(self.theta, self.Vinv, size=k)
        return [GenericFunction(lambda x: x @ theta.T, self.sigma) for theta in theta_tilde]

    def map(self) -> GenericFunction:
        """Return the MAP (maximum a posteriori) estimate.

        Returns:
            GenericFunction using the posterior mean.
        """
        return GenericFunction(lambda x: x @ self.theta, self.sigma)


class Kitten(Distribution):
    """CatBoost-based posterior for non-linear function estimation."""

    def __init__(
        self,
        Xtrain: NDArray,
        Ytrain: NDArray,
        f: Optional[GenericFunction],
        sigma: float = 0.5
    ) -> None:
        """Initialize Kitten posterior.

        Args:
            Xtrain: Training feature matrix.
            Ytrain: Training target values.
            f: Current function estimate (random if None).
            sigma: Observation noise standard deviation.
        """
        super().__init__()
        self.sigma = sigma
        if f is not None:
            self.f = f
        else:
            self.f = GenericFunction(lambda x: np.random.rand(x.shape[0]), self.sigma)

        self.Xtrain = Xtrain
        self.Ytrain = Ytrain

    def update_posterior(
        self,
        x: NDArray,
        y: list,
        copy: bool = False
    ) -> Optional[Kitten]:
        """Update posterior with new observation.

        Args:
            x: Feature vector of pulled arm.
            y: List containing observed reward.
            copy: If True, return new Kitten; otherwise update in-place.

        Returns:
            New Kitten if copy=True, else None.
        """
        if x.ndim == 1:
            x = x[:, np.newaxis]

        f = CatBoostRegressor(
            iterations=20,
            random_seed=np.random.randint(100000),
            verbose=False
        )
        if copy:
            Xtrain = np.concatenate((self.Xtrain, x))
            Ytrain = np.concatenate((self.Ytrain, y))
            if self.Xtrain.shape[0] < 10:
                f = GenericFunction(
                    lambda x: self.sigma * np.random.rand(x.shape[0]),
                    self.sigma
                )
                return Kitten(Xtrain, Ytrain, f)
            else:
                f.fit(Xtrain, Ytrain)
                f = GenericFunction(lambda x: f.predict(x), self.sigma)
            return Kitten(Xtrain, Ytrain, f)
        else:
            self.Xtrain = np.concatenate((self.Xtrain, x))
            self.Ytrain = np.concatenate((self.Ytrain, y))
            if self.Xtrain.shape[0] < 10:
                return
            f.fit(self.Xtrain, self.Ytrain)
            self.f = GenericFunction(lambda x: f.predict(x), self.sigma)

    def sample(self, k: int = 1) -> Union[GenericFunction, list[GenericFunction]]:
        """Sample function(s) from the posterior via bootstrap.

        Args:
            k: Number of samples to draw.

        Returns:
            Single GenericFunction if k=1, else list of GenericFunctions.
        """
        if self.Xtrain.shape[0] < 10:
            effs = [
                GenericFunction(
                    lambda x: self.sigma * np.random.rand(x.shape[0]),
                    self.sigma
                )
                for _ in range(k)
            ]
        else:
            effs = []
            for _ in range(k):
                model = CatBoostRegressor(
                    iterations=20,
                    random_seed=np.random.randint(100000),
                    verbose=False
                )
                n = len(self.Xtrain)
                idx = np.random.choice(n, n)
                f_tilde = model.fit(
                    self.Xtrain[idx],
                    self.Ytrain[idx] + np.random.randn(n) * self.sigma
                )
                effs.append(GenericFunction(lambda x: f_tilde.predict(x), self.sigma))

        if k == 1:
            return effs[0]
        return effs

    def map(self) -> GenericFunction:
        """Return the current function estimate.

        Returns:
            Current GenericFunction estimate.
        """
        return self.f
