import numpy as np


def pca(X, keep_dim=None):
    n_smaples = X.shape[0]
    
    mean = np.mean(X, axis=0)
    centered = X - mean
    covar = np.dot(centered.T, centered)
    eigen_values, eigen_vecs = np.linalg.eig(covar)
    sortidxs = np.argsort(eigen_values)
    eigen_vecs = eigen_vecs[sortidxs, :]
    
    if keep_dim is None:
        keep_dim = n_smaples
    keep_dim = min(keep_dim, n_smaples)

    projected = np.dot(X, eigen_vecs[:keep_dim, :].T)
    return projected


if __name__ == '__main__':
    mean = np.random.rand(5)
    covar = np.eye(5, 5)

    covar[0, 0] = 2
    covar[1, 1] = 1
    covar[2, 2] = 10
    covar[3, 3] = 3
    covar[4, 4] = 0.1
    # covar[0, 3]= 0.5
    # covar[3, 0]= 0.5
    # covar[2, 4]= 5
    # covar[4, 2]= 5
    
    samples = np.random.multivariate_normal(mean, covar, 1000)
    print('samples_covar: ', np.cov(samples, rowvar=False))
    projected = pca(samples, 2)
    print('projected_covar: ', np.cov(projected, rowvar=False))