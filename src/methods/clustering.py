"""Clustering utilities for graph Fourier features."""

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import logsumexp
from sklearn.cluster import KMeans


class GMM_Diag:
    """Gaussian mixture model with diagonal covariance matrices.

    The model is intended for graph Fourier coefficients, where graph
    stationarity motivates a diagonal spectral covariance.

    Parameters
    ----------
    K:
        Number of mixture components.
    max_iter:
        Maximum number of expectation-maximization iterations.
    tol:
        Absolute log-likelihood change used as the convergence threshold.
    reg:
        Positive value added to every variance estimate.
    random_state:
        Seed forwarded to the K-means initializer.
    """

    def __init__(
        self,
        K: int,
        max_iter: int = 100,
        tol: float = 1e-4,
        reg: float = 1e-6,
        random_state: int | None = None,
    ):
        if not isinstance(K, (int, np.integer)) or isinstance(K, bool) or K < 1:
            raise ValueError("K must be a positive integer.")
        if (
            not isinstance(max_iter, (int, np.integer))
            or isinstance(max_iter, bool)
            or max_iter < 1
        ):
            raise ValueError("max_iter must be a positive integer.")
        if not np.isscalar(tol) or not np.isfinite(tol) or tol <= 0:
            raise ValueError("tol must be a positive finite number.")
        if not np.isscalar(reg) or not np.isfinite(reg) or reg <= 0:
            raise ValueError("reg must be a positive finite number.")

        self.K = int(K)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.reg = float(reg)
        self.random_state = random_state

    @staticmethod
    def _validate_features(X) -> np.ndarray:
        """Convert and validate a two-dimensional feature matrix."""
        features = np.asarray(X, dtype=float)
        if features.ndim != 2:
            raise ValueError("X must be a two-dimensional feature matrix.")
        if features.shape[0] == 0 or features.shape[1] == 0:
            raise ValueError("X must contain at least one sample and one feature.")
        if not np.all(np.isfinite(features)):
            raise ValueError("X must contain only finite values.")
        return features

    def _check_is_fitted(self) -> None:
        """Raise a clear error when inference is requested before fitting."""
        required = ("mu", "var", "pi")
        if not all(hasattr(self, attribute) for attribute in required):
            raise RuntimeError("The GMM is not fitted. Call fit or init_params first.")

    @staticmethod
    def _log_gaussian(
        X: np.ndarray,
        mean: np.ndarray,
        variance: np.ndarray,
    ) -> np.ndarray:
        """Evaluate a diagonal Gaussian log-density for every row of ``X``."""
        normalization = np.sum(np.log(2 * np.pi * variance))
        squared_distance = np.sum((X - mean) ** 2 / variance, axis=1)
        return -0.5 * (normalization + squared_distance)

    def init_params(self, X):
        """Initialize mixture parameters with K-means and return ``self``."""
        features = self._validate_features(X)
        sample_count, feature_count = features.shape
        if self.K > sample_count:
            raise ValueError("K cannot exceed the number of samples.")

        kmeans = KMeans(
            n_clusters=self.K,
            n_init=10,
            random_state=self.random_state,
        )
        labels = kmeans.fit_predict(features)

        self.mu = kmeans.cluster_centers_.astype(float, copy=True)
        self.var = np.empty((self.K, feature_count), dtype=float)
        global_variance = np.var(features, axis=0) + self.reg

        for cluster in range(self.K):
            cluster_points = features[labels == cluster]
            self.var[cluster] = (
                np.var(cluster_points, axis=0) + self.reg
                if len(cluster_points) > 0
                else global_variance
            )

        counts = np.bincount(labels, minlength=self.K).astype(float)
        self.pi = counts / sample_count
        self.n_features_in_ = feature_count
        return self

    def e_step(self, X) -> tuple[np.ndarray, float]:
        """Compute responsibilities and total log-likelihood."""
        self._check_is_fitted()
        features = self._validate_features(X)
        if features.shape[1] != self.mu.shape[1]:
            raise ValueError("X has a different number of features than the model.")

        log_responsibilities = np.empty((features.shape[0], self.K), dtype=float)
        for cluster in range(self.K):
            log_responsibilities[:, cluster] = (
                np.log(self.pi[cluster] + 1e-12)
                + self._log_gaussian(
                    features,
                    self.mu[cluster],
                    self.var[cluster],
                )
            )

        log_normalizer = logsumexp(
            log_responsibilities,
            axis=1,
            keepdims=True,
        )
        responsibilities = np.exp(log_responsibilities - log_normalizer)
        return responsibilities, float(np.sum(log_normalizer))

    def m_step(self, X, resp):
        """Update mixture parameters from responsibilities and return ``self``."""
        self._check_is_fitted()
        features = self._validate_features(X)
        responsibilities = np.asarray(resp, dtype=float)
        expected_shape = (features.shape[0], self.K)
        if responsibilities.shape != expected_shape:
            raise ValueError(f"resp must have shape {expected_shape}.")
        if (
            not np.all(np.isfinite(responsibilities))
            or np.any(responsibilities < 0)
        ):
            raise ValueError("resp must contain finite, non-negative values.")
        if not np.allclose(responsibilities.sum(axis=1), 1.0):
            raise ValueError("Each row of resp must sum to one.")

        effective_counts = responsibilities.sum(axis=0)
        safe_counts = np.maximum(effective_counts, 1e-12)
        self.pi = safe_counts / safe_counts.sum()
        self.mu = (responsibilities.T @ features) / safe_counts[:, None]

        for cluster in range(self.K):
            difference = features - self.mu[cluster]
            weighted_variance = (
                responsibilities[:, cluster, None] * difference**2
            ).sum(axis=0)
            self.var[cluster] = (
                weighted_variance / safe_counts[cluster] + self.reg
            )
        return self

    def fit(self, X):
        """Fit the model with expectation-maximization and return ``self``."""
        features = self._validate_features(X)
        self.init_params(features)
        previous_likelihood = -np.inf
        self.converged_ = False

        for iteration in range(1, self.max_iter + 1):
            responsibilities, likelihood = self.e_step(features)
            self.m_step(features, responsibilities)

            if abs(likelihood - previous_likelihood) < self.tol:
                self.converged_ = True
                self.n_iter_ = iteration
                self.lower_bound_ = likelihood
                break
            previous_likelihood = likelihood
        else:
            self.n_iter_ = self.max_iter
            _, self.lower_bound_ = self.e_step(features)

        return self

    def predict(self, X) -> np.ndarray:
        """Return the most likely component for every feature vector."""
        responsibilities, _ = self.e_step(X)
        return np.argmax(responsibilities, axis=1)


class ClusteringEvaluator:
    """Evaluate cluster assignments and compute graph Fourier features."""

    @staticmethod
    def evaluate_accuracy(y_true, y_pred, K=None) -> float:
        """Compute clustering accuracy after optimal label permutation.

        ``K`` is accepted for backward compatibility. If supplied, it must be
        at least the number of unique labels in both partitions.
        """
        truth = np.asarray(y_true)
        prediction = np.asarray(y_pred)
        if truth.ndim != 1 or prediction.ndim != 1 or truth.shape != prediction.shape:
            raise ValueError("y_true and y_pred must be one-dimensional arrays of equal length.")
        if truth.size == 0:
            raise ValueError("Label arrays must not be empty.")

        true_labels, true_inverse = np.unique(truth, return_inverse=True)
        predicted_labels, predicted_inverse = np.unique(
            prediction,
            return_inverse=True,
        )
        if K is not None:
            if (
                not isinstance(K, (int, np.integer))
                or isinstance(K, bool)
                or K < 1
            ):
                raise ValueError("K must be a positive integer.")
            if K < max(len(true_labels), len(predicted_labels)):
                raise ValueError("K is smaller than the number of unique labels.")

        contingency = np.zeros(
            (len(true_labels), len(predicted_labels)),
            dtype=int,
        )
        np.add.at(contingency, (true_inverse, predicted_inverse), 1)
        true_indices, predicted_indices = linear_sum_assignment(-contingency)
        matched = contingency[true_indices, predicted_indices].sum()
        return float(matched / truth.size)

    @staticmethod
    def graph_fourier_features(graph, X) -> np.ndarray:
        """Project column-wise vertex signals into row-wise GFT features."""
        signals = np.asarray(X, dtype=float)
        if signals.ndim != 2 or signals.shape[0] != graph.eigenvectors.shape[0]:
            raise ValueError("X must have shape (N, M), with one row per graph node.")
        if not np.all(np.isfinite(signals)):
            raise ValueError(
                "X must be finite. Reconstruct missing observations before "
                "computing graph Fourier features."
            )
        return (graph.eigenvectors.T @ signals).T
