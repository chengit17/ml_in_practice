import numpy as np
from collections import Counter

from ml.utils.data_check import check_X_and_y
from ml.probs.pdf import multivariate_gaussian_prob


class GaussianNaiveBayes:
    def __init__(self, priors=None):
        self.n_classes = None
        self.priors = priors

    def fit(self, X, y):
        X, y = check_X_and_y(X, y)

        unique_y = np.unique(y)
        classes = unique_y
        n_classes = len(unique_y)
        n_samples, n_features = X.shape[0], X.shape[1]
        self.classes = classes
        self.n_classes = n_classes

        if self.priors is None:
            classes_count = Counter(y)
            priors = np.zeros((n_classes,), dtype=np.float64)
            for y_i in unique_y:
                i_in_classes = classes.searchsorted(y_i)
                priors[i_in_classes] = 1.0 * classes_count[y_i] / n_samples
            self.priors = priors

        means = np.zeros((n_classes, n_features), dtype=np.float64)
        covariances = np.zeros((n_classes, n_features), dtype=np.float64)
        for i in classes:
            mean = X[y == i].mean(axis=0)
            var = np.var(X[y == i], axis=0)
            means[i] = mean
            covariances[i] = var
        self.means = means
        self.covariances = covariances

    def predict(self, X):
        log_posterior = self._estimate_log_unnormalized_posterior(X, self.priors, self.means, self.covariances, 
                                                                  covariance_type='diag')
        predicted_idxs_in_classes = log_posterior.argmax(axis=1)
        predicted_labels = self.classes[predicted_idxs_in_classes]
        return predicted_labels

    @staticmethod
    def _estimate_log_unnormalized_posterior(X, prior, means, covariances, covariance_type):
        n_classes, n_features = means.shape
        if covariance_type == 'diag':
            log_posterior = []
            for i in range(n_classes):
                log_prior_i = np.log(prior[i])
                log_likelihood_ij = -.5 * (
                    n_features * np.log(2 * np.pi) + 
                    np.sum(np.log(covariances[i])) + 
                    np.sum((X - means[i]) ** 2 / covariances[i], axis=1)
                )
                log_posterior.append(log_prior_i + log_likelihood_ij)
            log_posterior = np.asarray(log_posterior).T
            return log_posterior


class GaussianFullBayes:
    def __init__(self):
        self.n_classes = None
        self.priors = None
        self.gaussian_parameters = None

    def fit(self, X, y):
        X, y = check_X_and_y(X, y)
        unique_y = np.unique(y)
        classes = unique_y
        n_classes = len(unique_y)
        n_samples, n_features = X.shape[0], X.shape[1]

        classes_count = Counter(y)
        priors = np.zeros((n_classes,), dtype=np.float64)
        for y_i in unique_y:
            i_in_classes = classes.searchsorted(y_i)
            priors[i_in_classes] = 1.0 * classes_count[y_i] / n_samples

        gaussian_parameters = []
        for i in classes:
            mean = X[y == i].mean(axis=0)
            cov = np.cov(X[y == i].T)
            gaussian_parameters.append((mean, cov))

        self.n_classes = n_classes
        self.classes = classes
        self.priors = priors
        self.gaussian_parameters = gaussian_parameters

    def predict(self, X):
        posteriors = []
        for i, (mean, cov) in enumerate(self.gaussian_parameters):
            likelihood_ij = multivariate_gaussian_prob(X, mean, cov)
            posterior_ij = likelihood_ij * self.priors[i]
            posteriors.append(posterior_ij[:, np.newaxis])
        posterior = np.hstack(posteriors)
        
        predicted_idxs_in_classes = posterior.argmax(axis=1)
        predicted_labels = self.classes[predicted_idxs_in_classes]
        return predicted_labels