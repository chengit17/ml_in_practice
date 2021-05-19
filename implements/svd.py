import numpy as np


def svd(X):
    u, s, vh = np.linalg.svd(X)
    return u, s, vh


if __name__ == '__main__':
    mean = np.random.rand(5)
    covar = np.eye(5, 5)

    covar[0, 0] = 2
    covar[1, 1] = 1
    covar[2, 2] = 10
    covar[3, 3] = 3
    covar[4, 4] = 0.1
    
    samples = np.random.multivariate_normal(mean, covar, 1000)
    u, s, vh = svd(samples)

    print('u: ', u)
    print('s: ', s)
    print('vh: ', vh)