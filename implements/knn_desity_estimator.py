import numpy as np
from scipy import linalg
from sklearn.neighbors import BallTree

from ml.utils.data_check import check_X
from ml.utils.data_check import check_square_matrix
from ml.utils.data_check import is_multivariate_data


class Kernel:
    def __call__(self, X):
        return self.transform(X)

    def transform(self, X):
        raise NotImplementedError


class GaussianKernel(Kernel):
    """单元高斯核
    """
    def __init__(self, sigma):
        assert(sigma > 0)
        self.sigma = sigma

    def transform(self, X):
        """
        Arguments
        ---------
        X : np.ndarray, shape of (n_samples,)
        """
        # X = tensor_utils.check_1darray(X)
        return np.exp(-np.sum(X ** 2, axis=-1) / (2 * self.sigma ** 2))


class MultiVariateGaussianKernel(Kernel):
    """多元高斯核
    """
    def __init__(self, Sigma):
        self.Sigma = check_square_matrix(Sigma)
        self.feature_dim = self.Sigma.shape[0]

    def transform(self, X):
        X = check_X(X)
        return self._compute_multivariate_kernel(X, self.Sigma)

    def _compute_multivariate_kernel(self, X, covariance):
        _, n_features = X.shape
        covar_chol = linalg.cholesky(covariance, lower=True)
        precision_chol = linalg.solve_triangular(covar_chol, np.eye(n_features), lower=True).T
                
        covar_chol_diag = covar_chol.ravel()[:, ::(n_features + 1)]
        log_det = np.sum(np.log(covar_chol_diag), axis=1)

        z = X @ precision_chol
        log_prob = -.5 * (n_features * np.log(2 * np.pi) + np.sum(np.square(z), axis=1)) + log_det
        prob = np.exp(log_prob)
        return prob


class KnnDensity:
    """基于Knn的密度估计.

    实现细节：
    1. 因为Knn密度是不连续的，而且它的积分为无穷大，而非１. 因此不能表示为概率密度函数. 
       这里的实现在Knn基础上加入了高斯核，能够得到更光滑的密度估计.

    2. 该算法涉及到两个距离度量: a. 核函数; b. k近邻搜索

        对于K近邻搜索，为了处理高维数据，使用了BallTree进行优化。
        各维度尺度不一致对多维数据的距离度有很大的影响。鉴于欧式距离无法做到在各维度上尺度一致，在数据具有多个维度时，建议指定度量方法为马氏距离（默认为欧式距离）。
       
    记住: 
    1. 在实例化`KnnDensity`时，如果给定的`metric`参数为`mahalanobis`, fit的数据应该具有多维度.
    """
    def __init__(self, k=5, metric='euclidean', leaf_size=40):
        self.k = k
        self.metric = metric
        self.leaf_size = leaf_size
        
    def fit(self, X, y=None):
        bt_kwargs = {}
        bt_kwargs['metric'] = self.metric
        if not is_multivariate_data(X):
            if self.metric == 'mahalanobis':
                raise RuntimeError('X must be at least two-dimensional')
        else:
            covar = np.cov(X, rowvar=False)
            self._train_samples_covar = covar
            if self.metric == 'mahalanobis':
                bt_kwargs['V'] = covar
        bt_kwargs['leaf_size'] = self.leaf_size
        self._ball_tree = BallTree(X, **bt_kwargs)
        self._train_samples = X
        
    def score_samples(self, X):
        # 计算X到其第K个近邻的距离
        dists, idxs = self._ball_tree.query(X, k=self.k, return_distance=True)
        knn_dists = dists[:, -1] # (len(X),)
        
        if not is_multivariate_data(self._train_samples):
            kernel = GaussianKernel(sigma=1.)
        else:
            kernel = MultiVariateGaussianKernel(Sigma=self._train_samples_covar)

        # 估计X的概率值
        N = self._train_samples.shape[0]
        projected_vals = kernel(
            (X[:, np.newaxis, :] - self._train_samples) / knn_dists[:, np.newaxis, np.newaxis]
        ) # (len(X), len(self._train_samples))
        sum_projected_vals = projected_vals.sum(axis=1) # (len(X),)
        return (1.0 / (N * knn_dists) * sum_projected_vals).flatten()


def test_sinlge_variate_data():
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    def make_data(N, f=0.3, rseed=1):
        rand = np.random.RandomState(rseed)
        x = rand.randn(N)
        x[int(N * f):] += 5
        return x
    
    x = make_data(500)
    kd = KnnDensity(k=64)
    kd.fit(x[:, np.newaxis])

    x_d = np.linspace(-20, 20, 1000)
    probs = kd.score_samples(x_d[:, np.newaxis])

    sns.set()
    plt.fill_between(x_d, probs, alpha=0.5)
    plt.show()


if __name__ == '__main__':
    test_sinlge_variate_data()