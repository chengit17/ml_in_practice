import numpy as np


def l0_norm(X):
    return np.count_nonzero(X)


def l1_norm(X):
    return np.sum(np.abs(X))


def l2_norm(X):
    return np.sqrt(np.sum(X ** 2))