from ml.metrics.base import pairwise_distances_metric
from ml.metrics import distance


@pairwise_distances_metric
def pairwise_euclidean_distances(X, Y):
    return distance.euclidean_distance(X, Y)