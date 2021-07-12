import numpy as np
from queue import Queue
from ml import metrics


class DBSCAN:
    def __init__(self, neighbor_eps=1e-2, neighbor_min_size=3, metric='euclid'):
        self.neighbor_eps = neighbor_eps
        self.neighbor_min_size = neighbor_min_size

        if metric == 'euclid':
            self.metric = metrics.euclidean_distance

        self._clusters = []

    def fit(self, X, y=None):
        self.neighbors = self._compute_neighbors_of_pts(X, X, self.neighbor_eps)

        centroids = self._init_centroids(X, self.neighbors)

        unaccess = np.ones((len(X),), dtype=np.bool)
        while not len(centroids):
            copied_unaccess = unaccess.copy()
            queue = Queue()
            centroid_idx = np.random.choice(centroids, size=1)
            queue.put(centroid_idx)
            unaccess[centroid_idx] = 0

            while not queue.empty():
                pt_idx = queue.get()
                neighbor = self._query_neighbor_of_pt(pt_idx)
                if np.nonzero(neighbor) >= self.neighbor_min_size:
                    delta = neighbor and unaccess
                    for idx in delta:
                        queue.put(idx)
                    unaccess[delta == 1] = 0
            
            cluster = copied_unaccess - unaccess
            self._clusters.append(cluster)
            centroids.remove(centroid_idx)    

    def predict(self, X):
        return self._e_step(X)

    def _init_centroids(self, X, neighbors):
        centroids = []
        for idx, neighbor in enumerate(neighbors):
            if np.nonzero(neighbor) >= self.neighbor_min_size:
                centroids.append(idx)
        return centroids

    def _compute_neighbors_of_pts(self, pts, D, epsilon):
        pr_dists = self.metric(pts, D)
        masks = (pr_dists < epsilon)
        
        neighbors = []
        for i, mask in enumerate(masks):
            neighbors.append(mask)
        return neighbors

    def _query_neighbor_of_pt(self, idx):
        return self._neighbors[idx]