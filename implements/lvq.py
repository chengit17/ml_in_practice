import numpy as np
from ml.metrics import distances


class LVQ:
    def __init__(self, n_vectors, vector_labels, metric='euclid', 
                 lr=1e-2, batch_size=10, max_iters=1000):
        self.n_vectors = n_vectors
        self.vector_labels = vector_labels

        if metric == 'euclid':
            self._metric = distances.euclidean_distance
        self.lr = lr
        self.batch_size = batch_size
        self.max_iters = max_iters

        self._vectors_X = None
        self._vectors_y = None

    def fit(self, X, y):
        self._init_vectors(X, y, self.n_vectors, self.vector_labels)
        
        for _ in range(self.max_iters):
            X_sub, y_sub = self._random_choose_samples(X, y, self.batch_size)

            dists = self._metric(X_sub, self._vectors_X)
            nearest_vec_idxs = np.argmin(dists, axis=-1)

            for i in range(len(X_sub)):
                nearest_vec_idx = nearest_vec_idxs[i]
                advance = self.lr * (X_sub[i] - self._vectors_X[nearest_vec_idx])
                if y_sub[i] != self._vectors_y[nearest_vec_idx]:
                    advance = -advance
                self._vectors_X[nearest_vec_idx] += advance


    def _init_vectors(self, X, y, n_vectors, vector_labels):
        vectors_X = []
        vectors_y = []
        for i in range(n_vectors):
            target = vector_labels[i]
            X_sub, _ = np.where(y == target, X, X)
            idx = np.random.randint(0, len(X_sub))
            vectors_X.append(X_sub[idx])
            vectors_y.append(target)

        self._vectors_X = np.asarray(vectors)
        self._vectors_y = np.asarray(vectors_y)

    def _random_choose_samples(self, X, y, size):
        idxs = np.random.choice(range(len(X)), size=size)
        return np.copy(X[idxs, ...])
        
    def predict(self, X):
        pass