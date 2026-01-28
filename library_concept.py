"""Concept bandit algorithms for best-arm identification with general function classes."""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from bandit_type import Concept, TopTwoMixin
from config import BOOTSTRAP_SAMPLES_SMALL, BOOTSTRAP_SAMPLES_LARGE, DELTA
from distribution import Distribution, GenericFunction


class GeneralThompson(Concept):
    """Thompson Sampling for concept bandits.

    Works with general function classes (not just linear) using
    posterior sampling over the function space.
    """

    def __init__(
        self,
        X: NDArray,
        gen_star: GenericFunction,
        pi: Distribution,
        T: int,
        sigma: float,
        name: str
    ) -> None:
        """Initialize GeneralThompson algorithm.

        Args:
            X: Arm feature matrix of shape (n_arms, d).
            gen_star: True reward function.
            pi: Prior/posterior distribution over functions.
            T: Time horizon.
            sigma: Observation noise standard deviation.
            name: Algorithm instance name.
        """
        super().__init__(X, gen_star, pi, T, sigma, name)
        self.B = BOOTSTRAP_SAMPLES_LARGE
        self.pulled: list[int] = []
        self.name = name
        self.delta = DELTA

    def run(self, logging_period: int = 1, k: int = 10) -> None:
        """Run the GeneralThompson algorithm.

        Args:
            logging_period: How often to print progress.
            k: Unused (for API consistency with TopTwo variants).
        """
        for t in range(self.T):
            f1 = self.pi.sample()
            best_idx = np.argmax(f1.evaluate(self.X))

            x_n = self.X[best_idx]
            y_n = self.gen_star.pull(x_n)

            self.pi.update_posterior(x_n, [y_n])

            self.fhat = self.pi.map()
            idx = np.argmax(self.fhat.evaluate(self.X))

            self.pulled.append(best_idx)
            self.arms_recommended.append(idx)

            if t % logging_period == 0:
                print(
                    'general run', self.name, 'iter', t, "/", self.T,
                    'idx', best_idx, end="\n"
                )

        quit_len = len(self.arms_recommended)
        rec = self.arms_recommended[-1]
        if quit_len < self.T:
            for _ in range(quit_len, self.T):
                self.arms_recommended.append(rec)


class GeneralTopTwo(TopTwoMixin, Concept):
    """Top-Two algorithm for concept bandits with generic posteriors.

    Uses the shared TopTwoMixin logic with concept bandit posteriors
    (e.g., Kitten/CatBoost).
    """

    def __init__(
        self,
        X: NDArray,
        gen_star: GenericFunction,
        pi: Distribution,
        T: int,
        sigma: float,
        name: str
    ) -> None:
        """Initialize GeneralTopTwo algorithm.

        Args:
            X: Arm feature matrix of shape (n_arms, d).
            gen_star: True reward function.
            pi: Prior/posterior distribution over functions.
            T: Time horizon.
            sigma: Observation noise standard deviation.
            name: Algorithm instance name.
        """
        super().__init__(X, gen_star, pi, T, sigma, name)
        self.B = BOOTSTRAP_SAMPLES_SMALL
        self.pulled: list[int] = []
        self.name = name
        self.delta = DELTA

    def _wrap_observation(self, y: Any) -> list:
        """Wrap observation in a list for concept bandit posteriors.

        Args:
            y: Raw observation value.

        Returns:
            Observation wrapped in a list.
        """
        return [y]

    def run(self, logging_period: int = 1, k: int = 10) -> None:
        """Run the algorithm.

        Args:
            logging_period: How often to print progress.
            k: Batch size for sampling second-best candidates.
        """
        self.run_top_two(logging_period=logging_period, k=k)
