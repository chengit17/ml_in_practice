from ml import solvers
import numpy as np
from ml.utils.data_check import check_X_and_y
from ml.utils.data_check import check_X
from ml.solvers.gradient_descent import GradientDescentSolver


def category(logits):
    return np.argmax(logits, axis=-1)


def one_hot(labels):
    labels = np.asarray(labels, dtype=np.int)
    m = np.max(labels)
    num_classes = m + 1
    num_samples = len(labels)
    logits = np.zeros((num_samples, num_classes))
    for i in range(num_samples):
        logits[i, labels[i]] = 1.0
    return logits


class LogisticRegression:
    def __init__(self, lr=0.01, tol=1e-2):
        self.lr = lr
        self.tol = tol

    def fit(self, X, y):
        X, y = check_X_and_y(X, y)
        self._solve(X, y)

    def predict_scores(self, X):
        X = check_X(X)
        return self._predict_labels(X, self._weights)

    def predict(self, X):
        X = check_X(X)
        return self._predict_labels(X, self._weights)

    def _solve(self, X, y):
        logits_true = one_hot(y)

        self.num_features = X.shape[1]
        self.num_classes = int(np.max(y)) + 1

        init_weights = self._init_weights(self.num_features, self.num_classes)

        forward = lambda X, weights: self._predict_logits(X, weights)
        gradient = lambda X, y_pred, y_true: X.T @ (y_true - y_pred)
        solver = GradientDescentSolver(forward, gradient, learning_rate=self.lr, stop_tol=self.tol)

        self._weights = solver.solve(X, logits_true, init_weights)

    def _init_weights(self, num_features, num_classes):
        return 0.2 * np.random.rand(num_features, num_classes) - 0.1

    def _predict_logits(self, X, weights):
        h = X @ weights # (num_samples, num_classes)
        logits = np.exp(h) / np.exp(h).sum(axis=-1, keepdims=True) # (num_samples, num_classes)
        return logits

    def _predict_labels(self, X, weights):
        return np.argmax(self._predict_logits(X, weights), axis=-1)


def test_logistic_regression():
    from sklearn import datasets
    from sklearn.preprocessing import normalize
    from sklearn.metrics import accuracy_score
    
    X, y = datasets.load_digits(return_X_y=True)
    X = normalize(X)

    total_samples = X.shape[0]
    train_ratio = 0.6
    test_start_idx = int(train_ratio * total_samples)
    X_train = X[0:test_start_idx]
    y_train = y[0:test_start_idx]
    X_test = X[test_start_idx:]
    y_test = y[test_start_idx:]

    lr = LogisticRegression(lr=0.001, tol=0.01)

    lr.fit(X_train, y_train)
    
    y_test_pred = lr.predict(X_test)
    acc = accuracy_score(y_test_pred, y_test)
    print('acc: ', acc)


if __name__ == '__main__':
    test_logistic_regression()