import numpy as np


class GaussianDistribution:
    def __init__(self, mu: float | np.ndarray, Sigma: float | np.ndarray) -> None:
        """Gaussian distribution

        Args:
            mu (float | np.ndarray): mean or mean vector
            Sigma (float | np.ndarray): variance (not std!) or covariance matrix
        """
        if isinstance(mu, np.ndarray) ^ isinstance(Sigma, np.ndarray):
            raise TypeError("mu and Sigma should be of the same type")
        if isinstance(mu, np.ndarray):
            if len(mu) != len(Sigma):
                raise ValueError("mu and Sigma should have the same length")
            np.linalg.cholesky(Sigma)  # check if Sigma is positive definite
        self._d = len(mu) if isinstance(mu, np.ndarray) else 1  # dimension
        self._mu = mu
        self._Sigma = Sigma

    def pdf(self, x: float | np.ndarray) -> float:
        """probability density function

        Args:
            x (float | np.ndarray): independent variable

        Returns:
            float: the probability density of x under N(mu, Sigma)
        """
        if isinstance(x, np.ndarray) ^ (self._d > 1):
            raise TypeError("x should be of the same type with mu")
        if isinstance(x, np.ndarray) and len(x) != self._d:
            raise ValueError("x and mu should have the same length")
        centralized_x = x - self._mu
        if self._d > 1:
            # multivariate
            det = np.linalg.det(self._Sigma)
            inv = np.linalg.inv(self._Sigma)
            exponent = -0.5 * centralized_x.dot(inv).dot(centralized_x)
        else:
            # univariate
            det = self._Sigma
            inv = 1 / (self._Sigma)
            exponent = -0.5 * inv * centralized_x**2
        coef = 1 / np.sqrt(((2 * np.pi) ** self._d) * det)
        return coef * np.exp(exponent)


class GaussianMixtureModel:
    def __init__(
        self, k: int, max_iterations: int = 50, stopping_criteria: float = 0.001
    ) -> None:
        """Estimate the unknown parameters of Gaussian mixture distribution with EM
            (Expectation Maximum) algorithm

        Args:
            max_iterations (int): maximum iteration times. Defualts to 50.
            stopping_criteria (float): stop iterating when new_ll - old_ll < criteria.
                Defaults to 0.001.
        """
        self._k = k  # number of clusters
        self._max_iterations = max_iterations
        self._stopping_criteria = stopping_criteria
        self._X = None  # training set
        self._n = None  # sample size
        self._d = None  # dimension of the Gaussian distribution
        self.alpha_arr: np.ndarray | None = None
        self.mu_arr: np.ndarray | None = None
        self.Sigma_arr: np.ndarray | None = None
        self.ll_lst = []  # log likelihood value list

    @staticmethod
    def _report(iteration: int, new_ll: float, ll_improvement: float) -> None:
        """Report the result of each iteration"""
        if iteration == 1:
            line = "{:^13}|{:^24}|{:^15}"
            print(line.format("iteration", "log likelihood value", "improvement"))
            print("-" * 13 + " " + "-" * 24 + " " + "-" * 15)
        line = "{:>13d} {:>24.3f} {:>15.3f}"
        print(line.format(iteration, new_ll, ll_improvement))

    def _gamma(self, z: int, x: float | np.ndarray) -> float:
        """Conditional probability P(z|x) denoted as `gamma`

        Args:
            z (int): value of latent variable in EM algorithm
            x (float | np.ndarray): value of observed variable in EM algorithm

        Returns:
            float: the probability of z given x
        """
        gauss = GaussianDistribution(self.mu_arr[z], self.Sigma_arr[z])
        p_z_x = self.alpha_arr[z] * gauss.pdf(x)
        p_x = 0
        for j in range(self._k):
            gauss = GaussianDistribution(self.mu_arr[j], self.Sigma_arr[j])
            p_x += self.alpha_arr[j] * gauss.pdf(x)
        return p_z_x / p_x

    def _log_likelihood(self) -> float:
        """log likelihood value of the samples X"""
        ll = 0
        for i in range(self._n):
            prob = 0
            for j in range(self._k):
                gauss = GaussianDistribution(self.mu_arr[j], self.Sigma_arr[j])
                prob += self.alpha_arr[j] * gauss.pdf(self._X[i])
            ll += np.log(prob)
        return ll

    def _init_params(self) -> None:
        """Initialize parameters"""
        # alpha_arr is initially k equal values
        self.alpha_arr = np.ones((self._k,)) / self._k
        # mu_arr is initially k samples drawn from X
        self.mu_arr = self._X[np.random.randint(0, self._n, self._k)]
        # Sigma_arr depends on the dimension of the sample
        if self._d > 1:
            # Sigma_arr is initially k identity matrices
            self.Sigma_arr = np.array([np.eye(self._d) for _ in range(self._k)])
        else:
            # Sigma_arr is initially k 1s
            self.sigma_arr = np.ones((self._k,))

    def _update_params(self) -> None:
        """Update parameters"""
        new_alpha_arr = np.zeros(self.alpha_arr.shape)
        new_mu_arr = np.zeros(self.mu_arr.shape)
        new_Sigma_arr = np.zeros(self.Sigma_arr.shape)
        for j in range(self._k):
            weights = np.array([self._gamma(j, self._X[i]) for i in range(self._n)])
            # update alpha_arr
            sum_weights = np.sum(weights)
            new_alpha_arr[j] = sum_weights / self._n
            # update mu_arr
            mu_frac1 = (weights * self._X.T).sum(axis=1)
            new_mu_arr[j] = mu_frac1 / sum_weights if sum_weights > 1e-9 else 0
            # update Sigma_arr
            centralized_x = self._X - new_mu_arr[j]
            raw = [
                weights[i] * np.outer(centralized_x[i], centralized_x[i])
                for i in range(self._n)
            ]
            Sigma_frac1 = np.array(raw).sum(axis=0)
            new_Sigma_arr[j] = Sigma_frac1 / sum_weights if sum_weights > 1e-9 else 0
        self.alpha_arr = new_alpha_arr
        self.mu_arr = new_mu_arr
        self.Sigma_arr = new_Sigma_arr

    def fit(self, X: np.ndarray, mute: bool = False) -> None:
        """Train GMM by EM algorithm

        Args:
            X (np.ndarray): training set of observed variable
            k (int): number of the Gaussian distribution mixed (hyperparameter)
            mute (bool, optional): report or not in the process. Defaults to False.
        """
        if X.ndim > 2:
            raise ValueError("dimension of X should not be larger than 2")
        self._X = X
        self._n = X.shape[0]
        self._d = X.shape[1] if X.ndim == 2 else 1
        # initialize parameters
        self._init_params()
        # update parameters
        for i in range(self._max_iterations):
            self.ll_lst.append(self._log_likelihood())
            self._update_params()
            new_ll = self._log_likelihood()
            if new_ll < self.ll_lst[-1]:
                break
            else:
                ll_improvement = new_ll - self.ll_lst[-1]
                self.ll_lst.append(new_ll)
                if not mute:
                    self._report(i + 1, new_ll, ll_improvement)
                # stop iterating
                if ll_improvement < self._stopping_criteria:
                    break

    def predict(self, x: float | np.ndarray) -> tuple[int, np.ndarray]:
        """Predict the most likely cluster and the probability densities of a sample
            based on this trained GMM

        Args:
            x (float | np.ndarray): a sample to predict

        Returns:
            tuple[int, np.ndarray]:
                int: index of the cluster which x is most likely drawn from
                np.ndarray: probability densities of x coming from each cluster
        """
        if isinstance(x, np.ndarray):
            if self._d == 1:
                raise TypeError("x should be a scalar")
            elif self._d != x.shape[0]:
                raise ValueError("sample should have the same dimension with model")
        elif self._d > 1:
            raise TypeError("x should be a vector")
        probs = []
        for j in range(self._k):
            gauss = GaussianDistribution(self.mu_arr[j], self.Sigma_arr[j])
            probs.append(gauss.pdf(x))
        probs_arr = np.array(probs)
        idx_cluster = np.argmax(probs)
        return idx_cluster, probs_arr