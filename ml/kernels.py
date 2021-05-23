import numpy as np
from . import tensor_utils


class Kernel:
    def __call__(self, X, Y):
        return self.transform(X, Y)

    def transform(self, X, Y):
        raise NotImplementedError


class LinearKernel(Kernel):
    def transform(self, X, Y):
        return np.dot(X, Y.T)


class PolynomialKernel(Kernel):
    def __init__(self, d):
        assert(d >= 1)
        self.d = d

    def transform(self, X, Y):
        return np.dot(X, Y.T) ** self.d


class ExpKernel(Kernel):
    def transform(self, X):
        return np.exp(-X)


class SigmoidKernel(Kernel):
    def __init__(self, beta, theta):
        assert(beta > 0)
        assert(theta < 0)
        self.beta = beta
        self.theta = theta

    def transform(self, X):
        return np.tanh(self.beta * np.dot(X, X.T) + self.theta)


class GaussianKernel(Kernel):
    def __init__(self, sigma):
        assert(sigma > 0)
        self.sigma = sigma

    def transform(self, X, Y):
        """
        Arguments
        ---------
        X : np.ndarray, shape of (n_samples, n_features)
        """
        if Y is not None:
            X = X[:, np.newaxis, :]
            Z = X - Y
        else:
            Z = X
        return np.exp(-np.sum(Z ** 2, axis=-1) / (2 * self.sigma ** 2))


class MultiVariateGaussianKernel(Kernel):
    def __init__(self, Sigma):
        self.Sigma = tensor_utils.check_square_matrix(Sigma)
        self.feature_dim = self.Sigma.shape[0]

    def transform(self, X, Y):
        X = tensor_utils.check_2darray(X)
        return self._compute_multivariate_kernel(X, Y, self.Sigma)

    def _compute_multivariate_kernel(self, X, covariance):
        n_samples, n_features = X.shape
        covar_chol = linalg.cholesky(covariance, lower=True)
        precision_chol = linalg.solve_triangular(covar_chol, np.eye(n_features), lower=True).T
                
        matrix_chol_diag = matrix_chol.ravel()[:, ::(n_features + 1)]
        log_det = np.sum(np.log(matrix_chol_diag), axis=1)

        z = (X - Y) @ precision_chol
        log_prob = -.5 * (n_features * np.log(2 * np.pi) + np.sum(np.square(z), axis=1)) + log_det
        return np.exp(log_prob)


class LaplaceKernel(Kernel):
    def __init__(self, sigma):
        assert(sigma > 0)
        self.sigma = sigma

    def transform(self, X, Y):
        """
        Arguments
        ---------
        X : np.ndarray, shape of (n_samples, n_features)
        """
        if Y is not None:
            X = X[:, np.newaxis, :]
            Z = X - Y
        else:
            Z = X
        return np.exp(-np.sqrt(np.sum(Z ** 2, axis=-1)) / (2 * self.sigma))


class LinearCombinedKernel(Kernel):
    def __init__(self, kernels, weights):
        self.kernels = kernels
        self.weights = weights

    def transform(self, X, Y):
        return np.sum([w * k(X, Y) for (w, k) in zip(self.weights, self.kernels)])


class ProdCombinedKernel(Kernel):
    def __init__(self, kernels):
        self.kernels = kernels

    def transform(self, X, Y):
        return np.multiply([k(X, Y) for k in self.kernels])


class CombinedKernel(Kernel):
    def __init__(self, kernel, func):
        self.kernel = kernel
        self.func = func

    def transform(self, X, Y):
        return func(X) * self.kernel(X, Y) * func(Y)