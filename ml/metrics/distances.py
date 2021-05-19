import numpy as np
from .util import distances_metric


@distances_metric
def euclidean_distance(x, y):
    return np.sqrt(np.sum((x - y)**2, axis=-1))


@distances_metric
def manhattan_distance(x, y):
    return np.sum(np.abs(x, y), axis=-1)


@distances_metric
def cosine_distance(x, y):
    den = np.sum(x * y, axis=-1)
    dum = np.sqrt(np.sum(x**2, axis=-1)) * np.sqrt(np.sum(y**2, axis=-1))
    return den / dum + 1e-5 


@distances_metric
def chebyshev_distance(x, y):
    return np.max(np.abs(x - y), axis=-1)


@distances_metric
def minkowsiki_distance(x, y, p):
    return np.sqrt(np.sum(np.power(np.abs(x - y), p), axis=-1), 1.0 / p)