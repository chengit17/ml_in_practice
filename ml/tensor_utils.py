import numpy as np


def check_consistent_length(*arrays):
    lengths = [len(array) for array in arrays]
    uniques = np.unique(lengths)
    if len(uniques) > 1:
        raise ValueError('X and y should has the same length!')


def is_1darray(x):
    x = np.asarray(x)
    shape = x.shape
    if len(shape) == 1:
        return True
    if len(shape) == 2 and shape[1] == 1:
        return True
    return False


def check_1darray(x):
    x = np.asarray(x)
    shape = x.shape
    if len(shape) == 1:
        return x
    if len(shape) == 2 and shape[1] == 1:
        return np.ravel(x)
    raise ValueError(f'bad data with shape: {shape}')


def is_2darray(X):
    X = np.asarray(X)
    shape = X.shape
    if len(shape) == 2:
        return True
    return False


def check_2darray(X):
    X = np.asarray(X)
    shape = X.shape
    if len(shape) == 2:
        return X
    raise ValueError(f'bad data with shape: {shape}')


def check_square_matrix(X):
    X = np.asarray(X)
    shape = X.shape
    if len(shape) == 2 and shape[0] == shape[1]:
        return X
    raise ValueError(f'bad data with shape: {shape}')


def check_X_and_y(X, y):
    X, y = np.asarray(X), np.asarray(y)

    y = make_1darray(y)
    check_consistent_length(X, y)

    return X, y


def check_samples(X, **kwargs):
    pass


def tensort_to_vector(t):
    t = np.asarray(t)
    shape = X.shape
    return t.ravel()