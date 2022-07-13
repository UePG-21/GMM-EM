import numpy as np
from matplotlib import pyplot as plt
from randutils import RandomVector
from gmm import GaussianMixtureModel


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

    gmm = GaussianMixtureModel(3)
    gmm.fit(X)
    print(gmm.alpha_arr)
    print(gmm.mu_arr)
    print(gmm.Sigma_arr)
    print(gmm.predict(X[0]))

    plt.plot(gmm.ll_lst)
    plt.show()
