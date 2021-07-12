import numpy as np


def gaussian_prob(X, mean, var):
    """
    Compute single-variate gaussian probability for several points.

    Argments:
        X: np.ndarray of shape (N,)
        mean: float 
        var: float

    Returns:
        probs: np.ndarray of shape (N,) 
            the probability at each sample
    """
    return (1.0 / np.sqrt(2 * np.pi * var)) * np.exp(-(X - mean) ** 2 / (2 * var))


def multivariate_gaussian_prob(X, mean, cov):
    """
    Compute multi-variate gaussian probability for several points.

    Argments:
        X: np.ndarray of shape (N, M)
        mean: np.ndarray of shape (M,) 
        cov: np.ndarray of shape (M, M)

    Returns:
        probs: np.ndarray of shape (N,) 
            the probability at each sample
    """
    dim = X.shape[1]
    frac_term = 1.0 / (np.sqrt(2. * np.pi) ** dim * np.sqrt(np.linalg.det(cov)))
    exp_term = np.exp(0 - np.sum(np.dot(X - mean, np.linalg.inv(cov)) * (X - mean), axis=1) / 2.)
    probs = frac_term * exp_term
    return probs


def log_gaussian_prob(x_i, mean, cov):
    """
    Compute the log gaussian probability at a single point.

    Argments:
        x_i : np.ndarray of shape (M,)
        mean: np.ndarray of shape (M,)
        cov: np.ndarray of shape (M, M)

    Returns:
        the probability at each `x_i`
    """
    dim = len(mean)
    a = dim * np.log(2 * np.pi)
    _, b = np.linalg.slogdet(cov)
    y = np.linalg.solve(cov, x_i - cov)
    c = np.dot(x_i - mean, y)
    return -0.5 * (a + b + c)