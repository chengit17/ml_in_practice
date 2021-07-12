import numpy as np
from functools import wraps
from . import vector as vector_metrics


def pairwise_vector_distances(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        x, y = args[0], args[1]
        x = np.asarray(x)
        y = np.asarray(y)
        x = x[:, np.newaxis, :]
        return func(x, y)
    return wrapper


@pairwise_vector_distances
def euclidean_distances(X, Y):
    return vector_metrics.euclidean_distance(X, Y)


@pairwise_vector_distances
def cosine_distances(X, Y):
    return vector_metrics.cosine_distance(X, Y)