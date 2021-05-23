import numpy as np
from ml.kernels import GaussianKernel


def gaussian_kernel(X):
    return np.exp(-np.sum(X ** 2, axis=-1) / 2)


class KernelDensity:
    """核密度估计.
    """
    def __init__(self, winsize=1, kernel='gaussian'):
        self.winsize = winsize
        self.kernel = gaussian_kernel
        
    def fit(self, X, y=None):
        self._fit_samples = X
        
    def score_samples(self, X):
        # 估计X的概率值
        N = self._fit_samples.shape[0]
        projected_vals = self.kernel(
            (X[:, np.newaxis, :] - self._fit_samples) / self.winsize
        ) # (len(X), len(self._fit_samples))
        sum_projected_vals = projected_vals.sum(axis=1) # (len(X),)
        return (1.0 / (N * self.winsize) * sum_projected_vals).flatten()


if __name__ == '__main__':
    def make_data(N, f=0.3, rseed=1):
        rand = np.random.RandomState(rseed)
        x = rand.randn(N)
        x[int(N * f):] += 5
        return x
    
    x = make_data(500)
    kd = KernelDensity(winsize=1)
    kd.fit(x[:, np.newaxis])

    x_d = np.linspace(-20, 20, 1000)
    probs = kd.score_samples(x_d[:, np.newaxis])

    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set()
    plt.fill_between(x_d, probs, alpha=0.5)
    plt.show()