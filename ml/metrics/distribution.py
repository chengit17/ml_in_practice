import numpy as np
import scipy.linalg


def _squared_mahalanobis_distance_directly(x, mean, covariance):
    d = x - mean
    return d.dot(np.linalg.inv(covariance)).dot(d.T)


def _squared_mahalanobis_distance_using_cholesky_decomposition(x, mean, covariance):
    d = x - mean
    cholesky_factor = np.linalg.cholesky(covariance)
    z = scipy.linalg.solve_triangular(cholesky_factor, d.T, lower=True,
        check_finite=False, overwrite_b=True)
    squared_maha = np.sum(z * z, axis=0)
    return squared_maha


def squared_mahalanobis_distance(x, mean, covariance):
    return _squared_mahalanobis_distance_using_cholesky_decomposition(x, mean, covariance)


def mahalanobis_distance(x, mean, covariance):
    """计算马氏距离
    Notes:
        Mahalonobis distance is the distance between a point and a distribution.
    """
    return np.sqrt(squared_mahalanobis_distance(x, mean, covariance))