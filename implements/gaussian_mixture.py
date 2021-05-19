import random
import numpy as np
import scipy as sci
from scipy import linalg

from sklearn.cluster import KMeans


logsumexp = sci.special.logsumexp


def _check_X(X, n_features=None, n_clusters=None):
    X = np.asarray(X)
    if X.dtype not in (np.float64, np.float32):
        raise ValueError(f'dtype of X should be `np.float`')

    if n_features is not None:
        if X.shape[1] != n_features:
            raise ValueError(f'ndim of X should be {n_features}')

    if n_clusters is not None:
        if X.shape[0] < n_clusters:
            raise ValueError(f'nsamples of X should be greater than {n_clusters}')

    return X


def _compute_log_gaussian_prob(X, means, covariances, covariance_type):
    """计算对数高斯概率

    Arguments
    ---------
    X : np.ndarray, shape (n_samples, n_features)
        样本
    means : np.ndarray, shape (n_clusters, n_features)
        均值向量
    covariances : np.ndarray
        协方差矩阵
    covariance_type : string
        协方差类型: {'full', 'diag', 'tied', 'spherical'}

    Returns
    -------
    log_probs : np.ndarray, shape (n_samples, n_clusters)
        对数概率
    """
    n_samples, n_features = X.shape
    n_clusters = means.shape[0]
    log_prob = np.empty((n_samples, n_clusters))

    # 计算精度矩阵K（协方差矩阵Σ的逆）的Cholesky分解Q（下三角矩阵）
    precisions_chol = _compute_precision_cholesky(covariances, covariance_type)
    # 计算精度矩阵K的Cholesky分解的对数行列式|Q|
    log_det = _compute_log_determinant_of_precision_cholesky(precisions_chol, n_features, covariance_type)

    if covariance_type == 'full':
        for c in range(n_clusters):
            z = (X - means[c]) @ precisions_chol[c]
            log_prob[:, c] = np.sum(np.square(z), axis=1)
    elif covariance_type == 'tied':
        for c in range(n_clusters):
            precision_chol = precisions_chol
            z = (X - means[c]) @ precision_chol
            log_prob[:, c] = np.sum(np.square(z), axis=1)
    elif covariance_type == 'diag':
        for c in range(n_clusters):
            z = (X - means[c]) * precisions_chol[c, ...]
            log_prob[:, c] = np.sum(np.square(z), axis=1)
    elif covariance_type == 'spherical':
        for c in range(n_clusters):
            z = (X - means[c]) * precisions_chol[c]
            log_prob[:, c] = np.sum(np.square(z), axis=1)
    
    log_prob = -.5 * (n_features * np.log(2 * np.pi) + log_prob) + log_det
    return log_prob


def _compute_precision_cholesky(covariances, covariance_type):
    """计算精度矩阵K的Cholesky分解Q.

    Arguments
    ---------
    covariances : np.ndarray
        协方差矩阵.
    covariance_type : string
        协方差矩阵类型, 取以下值::
            `full`表示每个component各自具有一般形式协方差矩阵. 形状为(n_components, n_features, n_features)
            `tied`表示所有component都共享一个一般形式的协方差矩阵. 形状为(n_features, n_features)
            `diag`表示每个component各自具有对角形式协方差矩阵（各维度之间是独立的）. 形状为(n_components, n_features)
            `spherical`表示每个component各自具有单数值的方差，也就是各个维度的方差一致. 形状为(n_components,)

    Returns
    -------
    precisions_chol : np.ndarray
        精度矩阵的下三角Cholesky分解.

        `full`:: shape of (n_components, n_features, n_features)
        `tied`:: shape of (n_features, n_features)
        `diag`:: shape of (n_components, n_features)
        `spherical`:: shape of (n_components,)
    """
    
    if covariance_type == 'full':
        n_components, n_features, _ = covariances.shape
        precisions_chol = np.empty((n_components, n_features, n_features))
        for c, covariance in enumerate(covariances):
            # 为什么要需要下三角矩阵(lower=True) ?
            covar_chol = linalg.cholesky(covariance, lower=True)
            # 协方差矩阵和精度矩阵的Cholesky分解矩阵之间的关系是: LQ^T = I ?
            precisions_chol[c] = linalg.solve_triangular(covar_chol, np.eye(n_features), lower=True).T # 该矩阵也为下三角矩阵
    elif covariance_type == 'tied':
        covar_chol = linalg.cholesky(covariances)
        precisions_chol = linalg.solve_triangular(covar_chol, np.eye(n_features), lower=True).T
    elif covariance_type in ('diag', 'spherical'):
        # 对于diag, spherical形式的精度矩阵，其Cholesky分解等于其精度的平方根（QQ^T = K -> Q = sqrt(K))
        precisions_chol = 1. / np.sqrt(covariances)
    return precisions_chol


def _compute_log_determinant_of_precision_cholesky(matrix_chol, n_features, covariance_type):
    """计算矩阵的Cholesky分解的对数行列式.

    Cholesky分解的上三角元素为0，其行列式等于对角线元素之和.

    Arguments
    ---------
    matrix_chol : np.ndarray
        矩阵的Cholesky分解.
    n_features : int
        特征维度.
    covariance_type : str, one of {'full', 'tied', 'diag', 'shperival'}
        协方差类型.

    Returns
    -------
    chol_log_det : float
        对数行列式
    """
    if covariance_type == 'full':
        n_clusters = matrix_chol.shape[0]
        # 将matrix_chol展平后通过slice将取对角线上的元素
        matrix_chol_diag = matrix_chol.reshape(n_clusters, -1)[:, ::(n_features + 1)]
        chol_log_det = np.sum(np.log(matrix_chol_diag), axis=1)
    elif covariance_type == 'tied':
        chol_log_det = np.sum(np.log(np.diag(matrix_chol)))
    elif covariance_type == 'diag':
        chol_log_det = np.sum(np.log(matrix_chol), axis=1)
    elif covariance_type == 'spherival':
        chol_log_det = n_features * (np.log(matrix_chol))
    return chol_log_det


def _estimate_gaussian_means(X, resp, nk):
    """估计高斯混合各成分的均值.

    Arguments
    ---------
    X : np.ndarray, shape of (n_samples, n_features)
        样本.
    resp : np.ndarray, shape of (n_samples, n_components)
        样布由各成分生成的后验概率γ_{ij} = p(c_{j} | x_{i})
    nk : np.darray, shape of (n_components,)
        np.sum(resp, axis=0)

    Returns
    -------
    means : np.adarray
        均值.
    """
    return np.dot(resp.T, X) / nk[:, np.newaxis]


def _estimate_gaussian_covariances(X, resp, nk, means, reg_covar=1e-6, covariance_type='full'):
    """估计高斯混合各成分的协方差矩阵.

    Arguments
    ---------
    X : np.ndarray, shape of (n_samples, n_features)
        样本.
    means : np.ndarray, shape of (n_components, n_features)
        分布均值.
    resp : np.ndarray, shape of (n_samples, n_components)
        样布由各成分生成的后验概率γ_{ij} = p(z_{j} | x_{i})
    nk : np.darray, shape of (n_components,)
        每个成分的计数.
    reg_covar : float
        加在协方差矩阵的对角线上的非负数正则化.
    matrix_chol : np.ndarray
        矩阵的Cholesky分解.
    n_features : int
        特征维度n.
    covariance_type : str, one of {'full', 'tied', 'diag', 'shperival'}
        协方差类型.

    Returns
    -------
    covariances : np.adarray
        协方差矩阵
    """
    n_components, n_features = means.shape
    if covariance_type == 'full':
        covariances = np.empty((n_components, n_features, n_features))
        for c in range(n_components):
            diff = X - means[c]
            covariances[c] = np.dot(resp[:, c] * diff.T, diff) / nk[c]
            # 给对角线元素添加非负数正则化
            covariances[c].flat[::(n_features+1)] += reg_covar
        return covariances
    elif covariance_type == 'tied':
        avg_X2 = np.dot(X.T, X)
        avg_mean2 = np.dot(nk * means.T, means)
        covariance = avg_X2 - avg_mean2
        covariance /= nk.sum()
        covariance.flat[(::n_features+1)] += reg_covar
        return covariance
    elif covariance_type == 'diag':
        avg_X2 = np.dot(resp.T, X ** 2) / nk[:, np.newaxis]
        avg_means2 = means ** 2
        avg_X_means = means * np.dot(resp.T, X) / nk[:, np.newaxis]
        covariance = avg_X2 - 2 * avg_X_means + avg_means2 + reg_covar
        return covariance
    elif covariance_type == 'spherical':
        avg_X2 = np.dot(resp.T, X ** 2) / nk[:, np.newaxis]
        avg_means2 = means ** 2
        avg_X_means = means * np.dot(resp.T, X) / nk[:, np.newaxis]
        covariance = avg_X2 - 2 * avg_X_means + avg_means2 + reg_covar
        covariance = covariance.mean(axis=1)
        return covariance


class GaussianMixture:
    def __init__(self, n_clusters=2, covariance_type='full', max_iter=100, 
                 tol=1e-3, reg_covar=1e-6, random_state=None, init_method='random'):
        self.n_clusters = n_clusters
        self.covariance_type = covariance_type
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.random_state = random_state
        self.init_method = init_method

    def fit(self, X, y=None):
        X = _check_X(X, None, self.n_clusters)
        self._initialize_parameters(X)

        prev_ll = -np.inf
        for _ in range(self.max_iter):
            log_resp = self._e_step(X)
            self._m_step(X, log_resp)
            ll = self._estimate_log_likelihood(X)
            converged = np.abs(ll - prev_ll) < self.tol
            if converged:
                break
            prev_ll = ll
        return self

    def _initialize_parameters(self, X):
        """初始化GMM模型的参数（means, covars, weights）.
        """
        n_samples, n_features = X.shape
        random_state = self.random_state or np.random.RandomState(42)

        if self.init_method == 'random1':
            self._pi = np.ones((self.n_clusters)) / self.n_clusters
            self._means = X[random.sample(range(n_samples), self.n_clusters)]
            self._covars = np.array([np.identity(n_features) for _ in range(self.n_clusters)])
        elif self.init_method == 'random2':
            resp = random_state.rand(n_samples, self.n_clusters)
            resp /= np.sum(resp, axis=1)[:, np.newaxis]
            self._pi, self._means, self._covars = self._estimate_parameters(X, resp)
        elif self.init_method == 'kmeans':
            resp = np.zeros((n_samples, self.n_clusters))
            labels = KMeans(self.n_clusters).fit_predict(X)
            resp[np.arange(n_samples), labels] = 1
            self._pi, self._means, self._covars = self._estimate_parameters(X, resp)

    def _estimate_parameters(self, X, resp):
        """估计GMM模型的参数（means, covars, weights）.
        """
        n_samples, _ = X.shape
        nk = np.sum(resp, axis=0) + 10 * np.finfo(resp.dtype).eps

        pi = nk / n_samples
        means = _estimate_gaussian_means(X, resp, nk)
        covars = _estimate_gaussian_covariances(X, resp, nk, means, self.reg_covar, self.covariance_type)

        return pi, means, covars

    def _check_fitted(self):
        fitted = all([hasattr(self, '_means'), hasattr(self, '_covars'), hasattr(self, '_pi')])
        assert fitted, RuntimeError('the model has not fitted')
        
    def predict(self, X, y=None):
        """预测样本的标签.
        """
        self._check_fitted()
        X = _check_X(X, self._means.shape[1])
        return self._estimate_weighted_log_prob(X).argmax(axis=1)

    def predict_proba(self, X):
        """预测样本由各成分生成的概率.
        """
        self._check_fitted()
        X = _check_X(X, self._means.shape[1])
        log_resp = self._estimate_log_prob_resp(X)
        return np.exp(log_resp)

    def score_samples(self, X):
        """预测样本生成的概率.
        """
        self._check_fitted()
        X = _check_X(X, self._means.shape[1])
        return logsumexp(self._estimate_weighted_log_prob(X), axis=1)

    def sample(self, n_samples=1):
        pass
        
    def _estimate_log_likelihood(self, X):
        """计算X在当前参数下的的对数似然.

        混合模型的对数似然具有logsumexp的形式，可借由logsumexp函数快速计算.

        Returns
        -------
        likehood : float
            对数似然.
        """
        weighted_log_prob = self._estimate_weighted_log_prob(X)
        return logsumexp(weighted_log_prob, axis=1).sum()

    def _estimate_log_prob_resp(self, X):
        """计算样本由各成分生成的对数后验概率.

        对于分母的归一化项，具有logsumexp的形式，可借由logsumexp函数快速计算.

        Returns
        -------
        log_resp : np.ndarray, shape (n_samples, n_components)
            对数后验概率.
        """
        weighted_log_prob = self._estimate_weighted_log_prob(X)
        log_prob_norm = logsumexp(weighted_log_prob, axis=1)
        log_resp = weighted_log_prob - log_prob_norm[:, np.newaxis]
        return log_resp

    def _estimate_weighted_log_prob(self, X):
        """计算样本由各成分生成的对数概率（似然）.

        样本的对数概率 = 样本在各成分下的对数高斯概率log P(Z | X) + 各成分的对数权重

        Returns
        -------
        weighted_log_prob : np.ndarray, shape (n_samples, n_components)
            加权的对数概率.
        """
        log_prob = _compute_log_gaussian_prob(X, self._means, self._covars, self.covariance_type)
        weighted_log_prob = log_prob + np.log(self._pi)
        return weighted_log_prob
        
    def _e_step(self, X):
        """EM迭代算法中的E Step
        用于估计后验概率.
        """
        return self._estimate_log_prob_resp(X)

    def _m_step(self, X, log_resp):
        """EM迭代算法中的M Step.
        用于估计参数.
        """
        print(f'old pi: {self._pi}')
        print(f'old means: {self._means}')
        print(f'old covars: {self._covars}')
        resp = np.exp(log_resp)
        print(f'resp: {resp}')
        self._pi, self._means, self._covars = self._estimate_parameters(X, resp)
        print(f'new pi: {self._pi}')
        print(f'new means: {self._means}')
        print(f'new covars: {self._covars}')