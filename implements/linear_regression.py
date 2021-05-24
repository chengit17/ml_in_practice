import itertools
import numpy as np


def check_X(X):
    X = np.asarray(X)
    if len(X.shape) == 1:
        X = X[:, np.newaxis]
    return X

def check_y(y):
    y = np.asarray(y)
    if len(y.shape) == 1:
        y = y[:, np.newaxis]
    return y


def check_consistent_length(*arrays):
    lengths = [len(array) for array in arrays]
    uniques = np.unique(lengths)
    if len(uniques) > 1:
        raise ValueError('X and y should has the same length!')


def check_X_and_y(X, y):
    X, y = check_X(X), check_y(y)
    check_consistent_length(X, y)
    return X, y


class LinearRegression:
    def __init__(self):
        self.coefficients = None
        self.iterpret = None

    def fit(self, X, y):
        X, y = check_X_and_y(X, y)
        n_samples, n_features = X.shape
        I = np.ones((n_samples, 1))
        X = np.hstack([I, X])
        W = np.linalg.inv(X.T @ X) @ X.T @ y  # ((d, n) @ (n, d))^-1 @ (d, n) @ (n, 1) -> (d, 1)
        self.coefficients = W[1:]
        self.intercept = W[0]

    def predict(self, X):
        X = check_X(X)
        n_samples, n_features = X.shape
        return np.ravel(X @ self.coefficients + self.intercept) # (n,)


class PolynoimalTransform:
    def __init__(self, degree):
        self.degree = degree
        
    def transform(self, X):
        X = check_X(X)
        n_samples, n_features = X.shape
        features = []
        for i in range(self.degree):
            tuples = itertools.combinations_with_replacement(range(n_features), i + 1)
            for t in tuples:
                features.append(np.prod(X[:, t], axis=1, keepdims=True))
        return np.concatenate(features, axis=1)


def test_linear_regression():
    x = np.linspace(0, 10, 100)
    y = 2 * x + 1 + np.random.normal(0, 0.1)
    lr = LinearRegression()
    lr.fit(x, y)

    print('lr.coefficients: ', lr.coefficients)
    print('lr.intercept: ', lr.intercept)
    
    x_pred = np.linspace(9, 15, 50)
    y_pred = lr.predict(x_pred)
    print('y_pred: ', y_pred)


def test_ploynomial_transform():
    t = PolynoimalTransform(2)

    X = np.random.rand(10, 4)
    X_ploy = t.transform(X)
    print('X.shape: ', X.shape)
    print('X_ploy.shape: ', X_ploy.shape)


if __name__ == '__main__':
    test_linear_regression()
    test_ploynomial_transform()