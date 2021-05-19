import numpy as np
from ml.metrics import distances


class KMeans:
    def __init__(self, n_clusters, metric='euclid', epsilon=1e-2):
        self.n_clusters = n_clusters
        self.epsilon = epsilon

        if metric == 'euclid':
            self._metric = distances.euclidean_distance

        self._centroids = []

    def fit(self, X, y=None):
        self._init_centroids(X)
        
        while True:
            cluster_marks = self._e_step(X)
            new_centroids = self._m_step(X, cluster_marks)
            
            try:
                if self._stop_criterion(new_centroids):
                    break
            finally:
                self._centroids = new_centroids

    def predict(self, X):
        return self._e_step(X)

    def _e_step(self, X):
        dists = self._metric(X, self._centroids)
        cluster_marks = np.argmin(dists, axis=-1)
        return cluster_marks

    def _m_step(self, X, cluster_marks)
        new_centroids = np.copy(self._centroids)
        for c in range(self.n_clusters):
            new_centroids[c, ...] = X[cluster_marks == c, ...].mean(axis=-1)
        return new_centroids

    def _init_centroids(self, X):
        idxs = np.random.choice(range(len(X)), size=self.n_clusters)
        self._centroids = np.copy(X[idxs, ...])

    def _stop_criterion(self, new_centroids):
        diff = np.mean(np.abs(new_centroids - self._centroids))
        if diff < self.epsilon:
            return True
        return False