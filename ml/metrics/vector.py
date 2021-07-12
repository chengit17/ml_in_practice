import numpy as np
from functools import wraps


def vector_distance(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        x, y = args[0], args[1]

        x = np.asarray(x)
        y = np.asarray(y)

        # assert x.shape == y.shape, '`x` and `y` must has the same shapes!'
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


@vector_distance
def euclidean_distance(x, y):
    return np.sqrt(np.sum((x - y)**2, axis=-1))


@vector_distance
def manhattan_distance(x, y):
    return np.sum(np.abs(x, y), axis=-1)


@vector_distance
def cosine_distance(x, y):
    den = np.sum(x * y, axis=-1)
    dum = np.sqrt(np.sum(x**2, axis=-1)) * np.sqrt(np.sum(y**2, axis=-1))
    return den / dum + 1e-5


@vector_distance
def chebyshev_distance(x, y):
    return np.max(np.abs(x - y), axis=-1)


@vector_distance
def minkowsiki_distance(x, y, p):
    return np.sqrt(np.sum(np.power(np.abs(x - y), p), axis=-1), 1.0 / p)