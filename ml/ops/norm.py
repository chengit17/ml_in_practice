import numpy as np


def l0_norm(X):
    return np.count_nonzero(X)


def l1_norm(X):
    return np.sum(np.abs(X))


def l2_norm(X):
    return np.sqrt(np.sum(X ** 2))


def mse(X, Y):
    return np.mean((X - Y) ** 2)


def rmse(X, Y):
    return np.sqrt(mse(X, Y))


def mae(X, Y):
    return np.mean(np.abs(X - Y))


def rmae(X, Y):
    return np.sqrt(mae(X, Y))