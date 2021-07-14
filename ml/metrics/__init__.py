from sys import implementation
from .vector import euclidean_distance
from .vector import manhattan_distance
from .vector import cosine_distance
from .vector import chebyshev_distance
from .vector import minkowsiki_distance

from .vector_pairwised import euclidean_distances
from .vector_pairwised import cosine_distances

from .distribution import mahalanobis_distance

from .matrix import mae
from .matrix import mse
from .matrix import rmse
from .matrix import rmae