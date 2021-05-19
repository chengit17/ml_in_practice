import numpy as np
from numpy import testing
from ml.metrics.distribution import mahalanobis_distance


def test_squared_mahalanobis_distance():
    fake_data = np.random.normal([0, 0, 0], scale=[1.0, 1.0, 1.0], size=(100, 3))

    mean = fake_data.mean(axis=0)
    cov = np.cov(fake_data, rowvar=False)

    point = np.asarray([1.0, 2.0, 3.0])
    maha_dist = mahalanobis_distance(point, mean, cov)
    print('maha_dist: ', maha_dist)

    points = np.random.normal([0, 0, 0], scale=[1.0, 1.0, 1.0], size=(10, 3))
    maha_dists = mahalanobis_distance(points, mean, cov)
    print('maha_dists: ', maha_dists)