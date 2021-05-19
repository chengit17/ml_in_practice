import numpy as np
from itertools import product
from functools import wraps


def distances_metric(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        x, y = args[0], args[1]

        x = np.asarray(x)
        y = np.asarray(y)

        # assert x.shape == y.shape, '`x` and `y` should has same shapes!'
        _x_is_vector = x.ndim == 1
        _y_is_vector = y.ndim == 1
        if _x_is_vector:
            x = x[np.newaxis, :]
        if _y_is_vector:
            y = y[np.newaxis, :]
        if _x_is_vector and _y_is_vector:
            return func(x, y)[0]
        return func(x, y)
    return wrapper


def pairwise_distances_metric(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        x, y = args[0], args[1]

        x = np.asarray(x)
        y = np.asarray(y)
        x = x[:, np.newaxis, :]
        return func(x, y)
    return wrapper