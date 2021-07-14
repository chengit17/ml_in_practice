import numpy as np
from ml.metrics import mae


class GradientDescentSolver:
    def __init__(self, forward, gradient, learning_rate=1e-2, stop_tol=1e-2):
        self.forward = forward
        self.gradient = gradient
        self.learning_rate = learning_rate
        self.stop_tol = stop_tol

    def solve(self, X, y, init_weights):
        weights = init_weights
        grads = np.zeros_like(init_weights)

        y_true = y
        while True:
            y_pred = self.forward(X, weights)
            new_grads = self.gradient(X, y_pred, y_true)

            diff = mae(new_grads, grads)
            if diff < self.stop_tol:
                break

            grads = new_grads
            weights += grads * self.learning_rate

        return weights