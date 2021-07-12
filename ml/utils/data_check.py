import numpy as np


def check_all_arrays_has_consistent_length(*arrays):
    uni_lens = np.unique(len(array) for array in arrays)
    assert len(uni_lens) == 1, 'X and y should has the same length!'


def is_1darray(X):
    X = np.asarray(X)
    return X.ndim == 1


def is_2darray(X):
    X = np.asarray(X)
    return X.ndim == 2

def could_view_as_1darray(X):
    X = np.asarray(X)
    return (X.ndim == 1) or (X.ndim == 2 and 1 in X.shape)


def view_as_1darray(X):
    X = np.asarray(X)
    if X.ndim == 1:
        return X
    elif X.ndim == 2 and 1 in X.shape:
        return np.ravel(X)
    raise ValueError(f'bad data with shape: {X.shape}')


def is_multivariate_data(X):
    X = np.asarray(X)
    if X.ndim == 2 and X.shape[1] > 1:
        return True
    return False


def check_square_matrix(X):
    X = np.asarray(X)
    shape = X.shape
    if X.ndim == 2 and shape[0] == shape[1]:
        return X
    raise ValueError(f'bad data with shape: {shape}')


def check_X(X, feature_row=True):
    X = np.asarray(X)
    assert X.ndim in (1, 2)
    if feature_row:
        if X.ndim == 1:
            X = X[:, np.newaxis]
    else:
        if X.ndim == 1:
            X = X[np.newaxis, :]        
    return X


def check_X_and_y(X, y, feature_row=True, same_dims=False):
    X = np.asarray(X)
    y = np.asarray(y)

    if feature_row:
        assert 0 < X.ndim < 2
        if X.ndim == 1:
            X = X[:, np.newaxis]

        n_samples, _ = X.shape
        if y.ndim == 1:
            if same_dims:
                y = y.reshape((n_samples, -1))
        elif y.ndim == 2:
            assert y.shape[0] == n_samples
    else:
        assert 0 < X.ndim < 2
        if X.ndim == 1:
            X = X[np.newaxis, :]
            
        _, n_samples = X.shape
        if y.ndim == 1:
            if same_dims:
                y = y.reshape((-1, n_samples))
        elif y.ndim == 2:
            assert y.shape[1] == n_samples

    return X, y