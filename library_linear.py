"""Linear bandit algorithms for best-arm identification."""

from typing import Union

import numpy as np
from numpy.typing import NDArray

from bandit_type import Linear, TopTwoMixin
from config import BOOTSTRAP_SAMPLES_LARGE, DELTA, MAX_LOOP_ITERATIONS
from utils import FW, compute_Y, fast_rank_one
from distribution import Gaussian, GenericFunction


class ThompsonSampling(Linear):
    """Thompson Sampling for linear bandits.

    Samples from the posterior over theta and pulls the arm with
    highest expected reward under the sample.
    """

    def __init__(
        self,
        X: NDArray,
        Y: NDArray,
        theta_star: NDArray,
        T: int,
        sigma: float,
        name: str
    ) -> None:
        """Initialize Thompson Sampling algorithm.

        Args:
            X: Arm feature matrix of shape (n_arms, d).
            Y: Pairwise difference matrix.
            theta_star: True parameter vector.
            T: Time horizon.
            sigma: Observation noise standard deviation.
            name: Algorithm instance name.
        """
        super().__init__(X, Y, theta_star, T, sigma, name)
        self.B = np.matmul(X.reshape(-1, self.d, 1), X.reshape(-1, 1, self.d))
        self.Vinv = np.linalg.inv(self.V)

    def run(self, logging_period: int = 1) -> None:
        """Run the Thompson Sampling algorithm.

        Args:
            logging_period: How often to print progress.
        """
        theta = np.zeros(self.d)
        S = 0
        for t in range(self.T):
            theta_hat = np.random.multivariate_normal(theta, self.Vinv)
            best_idx = np.argmax(self.X @ theta_hat)
            x_n = self.X[best_idx]
            y_n = x_n @ self.theta_star + self.sigma * np.random.randn()
            self.V += np.outer(x_n, x_n)
            self.Vinv = fast_rank_one(self.Vinv, x_n)
            S += x_n * y_n
            theta = np.linalg.inv(self.V) @ S
            self.arms_recommended.append(np.argmax(self.X @ theta))
            if t % logging_period == 0:
                print('ts run', self.name, 'iter', t, "/", self.T, end="\r")


class TopTwoAlgorithm(Linear):
    """Top-Two Thompson Sampling for linear bandits.

    Identifies the top two arms under posterior samples and pulls
    the arm that best distinguishes between them.
    """

    def __init__(
        self,
        X: NDArray,
        Y: NDArray,
        theta_star: NDArray,
        T: int,
        sigma: float,
        name: str
    ) -> None:
        """Initialize Top-Two algorithm.

        Args:
            X: Arm feature matrix of shape (n_arms, d).
            Y: Pairwise difference matrix.
            theta_star: True parameter vector.
            T: Time horizon.
            sigma: Observation noise standard deviation.
            name: Algorithm instance name.
        """
        super().__init__(X, Y, theta_star, T, sigma, name)
        self.B = np.matmul(X.reshape(-1, self.d, 1), X.reshape(-1, 1, self.d))
        self.Vinv = np.linalg.inv(self.V)
        self.toptwo: list[list[int]] = []
        self.pulled: list[int] = []
        self.k = 10

    def run(self, logging_period: int = 1) -> None:
        """Run the Top-Two algorithm.

        Args:
            logging_period: How often to print progress.
        """
        theta = np.zeros(self.d)
        S = 0

        for t in range(self.T):
            theta_1 = np.random.multivariate_normal(theta, self.Vinv)
            best_idx = np.argmax(self.X @ theta_1)
            x_1 = self.X[best_idx]

            theta_2 = np.random.multivariate_normal(theta, self.Vinv)
            best_idx_2 = np.argmax(self.X @ theta_2)

            a = 0
            while best_idx == best_idx_2:
                theta_2_mat = np.random.multivariate_normal(
                    mean=theta,
                    cov=self.Vinv,
                    size=self.k
                )
                max_x2_vec = np.argmax(self.X @ theta_2_mat.transpose(), axis=0)
                if any(max_x2_vec != best_idx):
                    best_idx_2 = max_x2_vec[np.where(max_x2_vec != best_idx)[0][0]]
                a += 1
                if a > MAX_LOOP_ITERATIONS:
                    break
            if a > MAX_LOOP_ITERATIONS:
                break

            x_2 = self.X[best_idx_2]
            self.toptwo.append([best_idx, best_idx_2])

            min_idx = np.argmin(
                (x_1 - x_2) @ np.linalg.inv(self.V + self.B) @ (x_1 - x_2)
            )
            self.pulled.append(min_idx)
            x_n = self.X[min_idx]
            y_n = self.theta_star @ x_n + self.sigma * np.random.randn()

            self.V += np.outer(x_n, x_n)
            self.Vinv = fast_rank_one(self.Vinv, x_n)
            S += x_n * y_n
            theta = self.Vinv @ S
            self.theta = theta
            self.arms_recommended.append(np.argmax(self.X @ theta))

            if t % logging_period == 0:
                print('toptwo run', self.name, 'iter', t, "/", self.T, end="\r")

        quit_len = len(self.arms_recommended)
        rec = self.arms_recommended[-1]
        if quit_len < self.T:
            for _ in range(quit_len, self.T):
                self.arms_recommended.append(rec)


class XYStatic(Linear):
    """XY-Static algorithm using G-optimal experimental design.

    Computes a static allocation using Frank-Wolfe and samples
    arms according to this fixed design.
    """

    def __init__(
        self,
        X: NDArray,
        Y: NDArray,
        theta_star: NDArray,
        T: int,
        sigma: float,
        name: str
    ) -> None:
        """Initialize XY-Static algorithm.

        Args:
            X: Arm feature matrix of shape (n_arms, d).
            Y: Pairwise difference matrix.
            theta_star: True parameter vector.
            T: Time horizon.
            sigma: Observation noise standard deviation.
            name: Algorithm instance name.
        """
        super().__init__(X, Y, theta_star, T, sigma, name)

    def run(self, logging_period: int = 1) -> None:
        """Run the XY-Static algorithm.

        Args:
            logging_period: How often to print progress.
        """
        lam_f, _, _ = FW(self.X, self.Y, iters=1000)
        del self.Y
        S = 0
        for t in range(self.T):
            idx = np.random.choice(self.n, p=lam_f)
            x_n = self.X[idx]
            y_n = x_n @ self.theta_star + self.sigma * np.random.randn()
            self.V += np.outer(x_n, x_n)
            S += x_n * y_n
            theta = np.linalg.inv(self.V) @ S
            self.arms_recommended.append(np.argmax(self.X @ theta))
            if t % logging_period == 0:
                print('xy static run', self.name, 'iter', t, "/", self.T, end="\r")


class XYAdaptive(Linear):
    """XY-Adaptive algorithm with adaptive experimental design.

    Updates the sampling allocation at each step based on the
    current posterior over likely optimal arms.
    """

    def __init__(
        self,
        X: NDArray,
        Y: NDArray,
        theta_star: NDArray,
        T: int,
        sigma: float,
        name: str
    ) -> None:
        """Initialize XY-Adaptive algorithm.

        Args:
            X: Arm feature matrix of shape (n_arms, d).
            Y: Pairwise difference matrix.
            theta_star: True parameter vector.
            T: Time horizon.
            sigma: Observation noise standard deviation.
            name: Algorithm instance name.
        """
        super().__init__(X, Y, theta_star, T, sigma, name)
        self.k = 5
        self.Vinv = np.linalg.inv(self.V)

    def run(
        self,
        logging_period: int = 1,
        FW_verbose: bool = False,
        FW_logging_period: int = 100
    ) -> None:
        """Run the XY-Adaptive algorithm.

        Args:
            logging_period: How often to print progress.
            FW_verbose: Whether to show Frank-Wolfe progress.
            FW_logging_period: Logging frequency for Frank-Wolfe.
        """
        S = 0
        lam_f = np.ones(self.n) / self.n
        theta = np.zeros(self.d)
        for t in range(self.T):
            theta_mat = np.random.multivariate_normal(
                mean=theta,
                cov=self.Vinv,
                size=self.k
            )
            max_x_vec = np.argmax(self.X @ theta_mat.transpose(), axis=0)

            X_t = self.X[max_x_vec]
            Y_t = compute_Y(X_t)
            lam_f, _, _ = FW(
                self.X, Y_t,
                initial=lam_f,
                iters=20,
                step_size=1,
                logging_step=FW_logging_period,
                verbose=FW_verbose
            )

            ind_n = np.random.choice(self.X.shape[0], p=lam_f)

            x_n = self.X[ind_n]
            y_n = x_n @ self.theta_star + self.sigma * np.random.randn()

            self.V += np.outer(x_n, x_n)
            self.Vinv = np.linalg.inv(self.V)
            S += x_n * y_n
            theta = self.Vinv @ S
            self.arms_recommended.append(np.argmax(self.X @ theta))

            if t % logging_period == 0:
                print('xy adaptive run', self.name, 'iter', t, "/", self.T, end="\r")


class GeneralTopTwoLinear(TopTwoMixin, Linear):
    """Top-Two algorithm for linear bandits with Gaussian posterior.

    Uses the shared TopTwoMixin logic with a Gaussian posterior.
    """

    def __init__(
        self,
        X: NDArray,
        Y: NDArray,
        gen_star: Union[NDArray, GenericFunction],
        T: int,
        sigma: float,
        name: str
    ) -> None:
        """Initialize GeneralTopTwoLinear algorithm.

        Args:
            X: Arm feature matrix of shape (n_arms, d).
            Y: Pairwise difference matrix.
            gen_star: True reward function (array or GenericFunction).
            T: Time horizon.
            sigma: Observation noise standard deviation.
            name: Algorithm instance name.
        """
        super().__init__(X, Y, gen_star, T, sigma, name)
        self.pi = Gaussian(np.zeros(self.d), self.V)
        if type(gen_star) is np.ndarray:
            self.gen_star = GenericFunction(lambda x: x @ gen_star, sigma)
        else:
            self.gen_star = gen_star
        self.B = BOOTSTRAP_SAMPLES_LARGE
        self.pulled: list[int] = []
        self.name = name
        self.delta = DELTA

    def run(self, logging_period: int = 1, k: int = 10) -> None:
        """Run the algorithm.

        Args:
            logging_period: How often to print progress.
            k: Batch size for sampling second-best candidates.
        """
        self.run_top_two(logging_period=logging_period, k=k)
