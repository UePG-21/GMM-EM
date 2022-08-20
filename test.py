import numpy as np
from gmm import GaussianMixtureModel
from randlinalg import RandomVector
from matplotlib import pyplot as plt

if __name__ == "__main__":
    mu1 = np.array([10, 10, 10])
    Sigma1 = np.eye(3)
    mu2 = np.array([-2, -2, -4])
    Sigma2 = np.eye(3)
    mu3 = np.array([3, 4, 5])
    Sigma3 = np.array([[1, 0.5, 0.25], [0.5, 1, 0.5], [0.25, 0.5, 1]])
    rv1 = RandomVector(mu1, Sigma1)
    rv2 = RandomVector(mu2, Sigma2)
    rv3 = RandomVector(mu3, Sigma3)

    n1 = 200
    n2 = 500
    n3 = 300

    X = np.array(
        [rv1() for _ in range(n1)]
        + [rv2() for _ in range(n2)]
        + [rv3() for _ in range(n3)]
    )
    Y = X[:200]

    nan_pos = [
        (0, 1), (1, 2), (13, 0), (45, 2), (78, 1),
        (230, 0), (259, 1), (290, 2), (340, 1), (536, 0),
        (777, 1), (812, 0), (943, 2)
    ]
    for i in nan_pos:
        X[i] = np.nan

    gmm = GaussianMixtureModel(3)
    gmm.fit(X)
    print(gmm.alpha_arr)
    print(gmm.mu_arr)
    print(gmm.Sigma_arr)

    plt.plot(gmm.ll_lst)
    plt.show()
