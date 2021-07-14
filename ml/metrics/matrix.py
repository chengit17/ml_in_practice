import numpy as np


def mse(X, Y):
    return np.mean((X - Y) ** 2)


def rmse(X, Y):
    return np.sqrt(mse(X, Y))


def mae(X, Y):
    return np.mean(np.abs(X - Y))


def rmae(X, Y):
    return np.sqrt(mae(X, Y))