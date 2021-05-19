import numpy as np

from ml.structures.ball_tree import BallTree
from ml.metrics import distances

size = int(10e2)
dim = 3
X = np.random.rand(size, dim)
y = np.random.randint(0, 10, size)

print('X: ', X)
print('y: ', y)

bt = BallTree(leaf_size=40, metric=distances.euclidean_distance)
bt.fit(X, y)
knns = bt.query(np.random.rand(dim), k=20, return_distance=True)
print('knns: ', knns)


# from sklearn.neighbors import BallTree
# bt = BallTree(X)
# print(bt.query(np.random.rand(1, dim), k=5, return_distance=True))