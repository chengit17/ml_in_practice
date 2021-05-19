import numpy as np

from ml.structures.priority_queue import PriorityQueue
from ml.metrics import distances


class BallTreeNode:
    def __init__(self, centroid=None, X=None, y=None):
        self.centroid = centroid
        self.data = X
        self.targets = y
        
        self.is_leaf = False
        self.radius = None
        self.left = None
        self.right = None

    def __repr__(self):
        repr_fmt = 'BallTreeNode(is_leaf={}, centroid={})'
        return repr_fmt.format(self.is_leaf, self.centroid)


class BallTree:
    def __init__(self, leaf_size=40, metric=distances.euclidean_distance, **kwargs):
        self.leaf_size = leaf_size
        self.metric = metric
        self.root = None

    def fit(self, X, y=None):
        centroid, left_X, left_y, right_X, right_y = self._split_data(X, y)
        root = BallTreeNode(centroid=centroid)
        root.radius = np.max([self.metric(centroid, x) for x in X])
        root.left = self._build_tree(left_X, left_y)
        root.right = self._build_tree(right_X, right_y)
        self.root = root

    def _build_tree(self, X, y=None):
        centroid, left_X, left_y, right_X, right_y = self._split_data(X, y)
        
        if X.shape[0] <= self.leaf_size:
            leaf = BallTreeNode(centroid=centroid, X=X, y=y)
            leaf.radius = np.max([self.metric(centroid, x) for x in X])
            leaf.is_leaf = True
            return leaf

        node = BallTreeNode(centroid=centroid)
        node.radius = np.max([self.metric(centroid, x) for x in X])
        node.left = self._build_tree(left_X, left_y)
        node.right = self._build_tree(right_X, right_y)
        return node
    
    def _split_data(self, X, y=None):
        split_dim = np.argmax(np.var(X, axis=0))
        sort_idxs = np.argsort(X[:, split_dim])
        sort_X, sort_y = X[sort_idxs], y[sort_idxs] if y is not None else None

        median_idx = X.shape[0] // 2
        centroid = sort_X[median_idx]
        left_X, left_y = sort_X[:median_idx], sort_y[:median_idx] if sort_y is not None else None
        right_X, right_y = sort_X[median_idx:], sort_y[median_idx:] if sort_y is not None else None

        return centroid, left_X, left_y, right_X, right_y

    def query(self, x, k=1, return_distance=False):
        pq = PriorityQueue(capacity=k, order='min')
    
        pq = self._knn(x, k, pq, self.root)
        
        k_nearests = []
        nearests = pq
        for n in nearests:
            point, _ = n
            if return_distance:
                k_nearests.append((n, self.metric(point, x)))
            else:
                k_nearests.append(n)
        return k_nearests
            
    def _knn(self, x, k, pq, root):
        dist_to_ball_centroid = self.metric(x, root.centroid)
        dist_to_ball = dist_to_ball_centroid - root.radius
        dist_to_farest_neighbor = self.metric(x, pq.last()[0]) if len(pq) > 0 else np.inf

        if len(pq) == k and dist_to_ball >= dist_to_farest_neighbor:
            return pq

        if root.is_leaf:
            targets = [None] * root.data.shape[0] if root.targets is None else root.targets
            for point, target in zip(root.data, targets):
                dist_to_x = self.metric(x, point)
                if dist_to_x < dist_to_farest_neighbor:
                    pq.push(item=(point, target), priority=dist_to_x)
        else:
            close_to_left = self.metric(x, root.left.centroid) < self.metric(x, root.right.centroid)
            pq = self._knn(x, k, pq, root.left if close_to_left else root.right)
            pq = self._knn(x, k, pq, root.right if close_to_left else root.left)
        return pq