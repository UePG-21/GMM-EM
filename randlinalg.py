import numpy as np
import scipy.linalg


def is_symmetric(A: np.ndarray) -> bool:
    if np.allclose(A, A.T):
        return True
    return False


def is_toeplitz(A: np.ndarray, onlyOffDiag: bool = False) -> bool:
    if A.ndim != 2:
        return False
    row, col = A[0, :], A[:, 0]
    for i in range(int(onlyOffDiag), len(row) - 1):
        for j in range(len(row) - 1 - i):
            if A[i + j, j] != row[i]:
                return False
    for i in range(1, len(col) - 1):
        for j in range(len(row) - 1 - i):
            if A[j, i + j] != col[i]:
                return False
    return True


def is_positve_definite(A: np.ndarray) -> tuple[bool, np.ndarray | None]:
    if not is_symmetric(A):
        return False, None
    try:
        L = np.linalg.cholesky(A)
        return True, L
    except np.linalg.LinAlgError:
        return False, None


def thpd(
    size: int, n: np.ndarray | None = None, diags: np.ndarray | None = None
) -> np.ndarray:
    """Generate a Toeplitz Hermitian positive definite (thpd) matrix by Vandermonde
        decomposition. The default are often used to generate random correlation matrix

    Args:
        size: the size of the thpd matrix
        n: n determines the x, where x is the vector to generate Vandermonde matrix,
            x(n) = exp(i2πn)
            v_{1} = (1, x_{1}, ..., x_{1}^{n-1}).T
            V = (v_{1}, v_{2}, ... v_{n})
        diags: the diagonal elements

    Returns:
        T: a Toeplitz Hermitian positive definite matrix
    """
    if n is None:
        # default is a random thpd
        n = np.random.randn(size)
    elif len(n) != size:
        raise ValueError("length of n should be equal to size")
    if diags is None:
        # default is all 1s on the diagonal
        diags = np.array([1 for _ in range(size)]) / size
    elif len(diags) != size:
        raise ValueError("length of diags should be equal to size")
    x = np.exp(2j * np.pi * n)  # x(n) = exp(i2πn)
    D = np.diag([1 for _ in range(size)]) / size  # diagonal matrix
    V = np.vander(x, increasing=True).T  # Vandermonde matrix
    T = np.real(V.dot(D).dot(np.conj(V.T)))  # T = VDV^H is then a thpd
    return T


def cov2corr(cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Transform covariance matrix to correlation matrix"""
    if not is_positve_definite(cov):
        raise ValueError("covariance matrix must be positive definite")
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    # fix possible numerical error
    corr[corr < -1], corr[corr > 1] = -1, 1
    return corr, std


def corr2cov(corr: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Transform correlationmatrix to covariance matrix"""
    if not is_positve_definite(corr):
        raise ValueError("correlation matrix must be positive definite")
    if std.ndim != 1:
        raise ValueError("std must be a 1-D ndarray")
    if len(corr) != len(std):
        raise ValueError("correlation matrix and std must be the same length")
    return corr * np.outer(std, std)


def first_order_auto_cov(
    size: int, rho: float, std: np.ndarray | None = None
) -> np.ndarray:
    """Generate a covariance matrix of the values in an autocorrelative time series, we
        can use it to generate random first order autocorrelative time series
        the correlation matrix is like:
            (   1        rho      rho^2     rho^3     ...      rho^n   )
            (  rho        1        rho      rho^2     ...    rho^{n-1} )
            (  ...       ...       ...       ...      ...       ...    )
            ( rho^n   rho^{n-1} rho^{n-2} rho^{n-3}   ...        1     )
        then cov = corr * np.outer(std, std)

    Args:
        size (int): length of the covariance matrix
        rho (float): first order correlation coefficient
        std (np.ndarray | None, optional): standard deviation vector

    Returns:
        np.ndarray: covariance matrix, so is a THPD matrix
    """
    if std is None:
        std = np.ones(size)
    corr = scipy.linalg.toeplitz([rho**i for i in range(size)])
    return corr2cov(corr, std)


class RandomVector:
    def __init__(
        self, mu: np.ndarray, Sigma: np.ndarray, autocorrelate: bool = False
    ) -> None:
        """Random vector generator, considering every elements in the vector is a
            random variable, using `mu` and `Sigma` to describe them

        Args:
            mu: mean vector of r.v.s
            Sigma: covariance matrix of r.v.s
        """
        if not (isinstance(mu, np.ndarray) and isinstance(Sigma, np.ndarray)):
            raise TypeError("mu and Sigma must be np.ndarray")
        if len(mu) != len(Sigma):
            raise ValueError("mu and Sigma should have the same length")
        if autocorrelate:
            std = np.sqrt(np.diag(Sigma))
            corr = Sigma / np.outer(std, std)
            if not is_toeplitz(corr):
                raise ValueError("Sigma must be a Toeplitz matrix if autocorrelated")
        isPosDef, self.L = is_positve_definite(Sigma)
        if not isPosDef:
            raise ValueError("Sigma must be positive definite")
        self.size = len(mu)
        self.mu = mu
        self.Sigma = Sigma

    def __call__(self, rand_seed: int | None = None) -> np.ndarray:
        """generate a random vector"""
        if rand_seed is not None:
            np.random.seed(rand_seed)
        z = np.random.randn(self.size)
        x = np.dot(self.L, z)
        return x + self.mu


class RandomTimeSeries:
    def __init__(self, autocorr: bool) -> None:
        self.autocorr = autocorr

    def __call__(self, size: int, snr: float) -> np.ndarray:
        """Generate a random time series

        Args:
            size (int): length of time series
            snr (float): signal noise ratio

        Returns:
            np.ndarray: random time seires
        """
        if snr < 0:
            raise ValueError("signal noise ratio should not less than 0")
        white_noise = np.random.randn(size)
        auto_ts = np.sin(np.linspace(0, 2 * np.pi, size))
        return (snr * auto_ts + white_noise) / snr if self.autocorr else white_noise


ts_auto = RandomTimeSeries(True)
ts_nona = RandomTimeSeries(False)


class RandomCovMatrix:
    def __init__(self, homoscedastic: bool = True, autocorrelate: bool = False) -> None:
        """Initialize a random covariance matrix generator

        Args:
            homoscedastic (bool): diagonal elements of cov matrix are equal or not
            autocorrelate (bool): off-diagonal elements of corr are equal or not
        """
        self.homoscedastic = homoscedastic
        self.autocorrelate = autocorrelate

    def __call__(
        self, size: int, corr_snr: int | bool = False, rand_seed: int | None = None
    ) -> np.ndarray:
        """Random covariance matrix generator

        Args:
            size (int): length of the covariance matrix
            corr_snr (bool): if the off-diagonal elements in covariance matrix are as
                large as possible, if True, then it represents `snr`. Defaults to False
            rand_seed (int | None, optional): random seed. Defaults to None

        Returns:
            np.ndarray: covariance matrix, positive definite
        """
        if rand_seed is not None:
            np.random.seed(rand_seed)
        # initialize a random covariance matrix
        if corr_snr:
            w = np.array([ts_auto(size, corr_snr) for _ in range(size)])
        else:
            w = np.random.normal(size=(size, size))
        cov = np.dot(w.T, w) / size  # full rank a.s.
        # transform covariance matrix to correlation matrix
        corr, std = cov2corr(cov)
        # fix possible numerical error
        corr[corr < -1], corr[corr > 1] = -1, 1
        # generate a random correlation matrix (thpd) for autocorrelated time-series
        if self.autocorrelate:
            corr = thpd(size)
        if self.homoscedastic:
            std = np.ones((size,))
        # transform correlation matrix back to covariance matrix
        cov = corr2cov(corr, std)
        return cov


cov_homo_auto = RandomCovMatrix(True, True)
cov_homo_nona = RandomCovMatrix(True, False)
cov_nonh_auto = RandomCovMatrix(False, True)
cov_nonh_nona = RandomCovMatrix(False, False)


if __name__ == "__main__":
    n = 2000
    cov1 = cov_homo_auto(n, 1)
    cov2 = cov_homo_nona(n, 1)
    cov3 = cov_nonh_auto(n, 1)
    cov4 = cov_nonh_nona(n, 1)
    print(cov1)
    print(cov2)
    print(cov3)
    print(cov4)
    np.linalg.cholesky(cov1)
    np.linalg.cholesky(cov2)
    np.linalg.cholesky(cov3)
    np.linalg.cholesky(cov4)
