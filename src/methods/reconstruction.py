"""PSD estimation and graph-signal reconstruction algorithms."""

import numpy as np

from .generation import GSPGraph


class SignalReconstructor:
    """Estimate graph PSDs and reconstruct partially observed graph signals.

    Missing observations are represented by ``NaN`` at the public API. Methods
    create zero-filled working copies internally and never mutate input arrays.
    """

    _EPSILON = 1e-8

    def __init__(self, graph: GSPGraph):
        if not isinstance(graph, GSPGraph):
            raise TypeError("graph must be a GSPGraph instance.")
        self.graph = graph

    @staticmethod
    def normalize_gamma(gamma) -> np.ndarray:
        """Clip a PSD estimate to non-negative values and normalize it to [0, 1].

        An all-zero or entirely negative estimate is returned as an all-zero
        array. The input is not modified.
        """
        values = np.asarray(gamma, dtype=float)
        if values.size == 0:
            raise ValueError("gamma must contain at least one value.")
        if not np.all(np.isfinite(values)):
            raise ValueError("gamma values must be finite.")

        non_negative = np.maximum(values, 0.0)
        maximum = float(np.max(non_negative))
        if maximum == 0.0:
            return np.zeros_like(non_negative)
        return non_negative / maximum

    @staticmethod
    def _zero_fill_missing(values) -> tuple[np.ndarray, np.ndarray]:
        """Return an observation mask and a zero-filled working copy."""
        array = np.asarray(values, dtype=float)
        observed = ~np.isnan(array)
        if not np.all(np.isfinite(array[observed])):
            raise ValueError("Observed signal values must be finite.")
        return observed, np.where(observed, array, 0.0)

    @staticmethod
    def _validate_regularization(alpha=None, beta=None) -> None:
        """Validate PSD contrast and regularization parameters."""
        if alpha is not None and (
            isinstance(alpha, (bool, np.bool_))
            or not np.isscalar(alpha)
            or not np.isfinite(alpha)
            or alpha < 0
        ):
            raise ValueError("alpha must be a finite, non-negative number.")
        if beta is not None and (
            isinstance(beta, (bool, np.bool_))
            or not np.isscalar(beta)
            or not np.isfinite(beta)
            or beta <= 0
        ):
            raise ValueError("beta must be a positive finite number.")

    def _validate_signal_matrix(self, Y, *, name: str = "Y") -> np.ndarray:
        """Convert and validate a matrix containing one signal per column."""
        matrix = np.asarray(Y, dtype=float)
        expected_rows = self.graph.number_of_nodes()
        if matrix.ndim != 2 or matrix.shape[0] != expected_rows:
            raise ValueError(
                f"{name} must have shape (N, M), with one row per graph node."
            )
        if matrix.shape[1] == 0:
            raise ValueError(f"{name} must contain at least one signal.")
        self._zero_fill_missing(matrix)
        return matrix

    def _validate_single_signal(self, y) -> np.ndarray:
        """Convert and validate one vertex-domain signal."""
        signal = np.asarray(y, dtype=float)
        if signal.ndim != 1 or signal.shape[0] != self.graph.number_of_nodes():
            raise ValueError("y must contain one value per graph node.")
        self._zero_fill_missing(signal)
        return signal

    def _psd_penalty(self, gamma, alpha: float, scaling: bool) -> np.ndarray:
        """Construct the vertex-domain PSD-weighted penalty matrix."""
        spectrum = np.asarray(gamma, dtype=float)
        expected_size = self.graph.number_of_nodes()
        if spectrum.ndim != 1 or spectrum.shape[0] != expected_size:
            raise ValueError("gamma must contain one value per graph frequency.")
        if not np.all(np.isfinite(spectrum)) or np.any(spectrum < 0):
            raise ValueError("gamma must contain finite, non-negative values.")
        if not isinstance(scaling, (bool, np.bool_)):
            raise TypeError("scaling must be a boolean.")

        weights = np.exp(-alpha * spectrum)
        if scaling:
            lambda_max = float(self.graph.eigenvalues[-1])
            if lambda_max <= self._EPSILON:
                raise ValueError(
                    "PSD reconstruction requires a graph with a positive "
                    "largest Laplacian eigenvalue."
                )
            weights = weights / np.max(weights) * lambda_max

        U = self.graph.eigenvectors
        return (U * weights) @ U.T

    @staticmethod
    def _solve(system: np.ndarray, right_hand_side: np.ndarray) -> np.ndarray:
        """Solve a reconstruction system and provide a meaningful failure."""
        try:
            return np.linalg.solve(system, right_hand_side)
        except np.linalg.LinAlgError as error:
            raise ValueError(
                "The reconstruction system is singular. Ensure every connected "
                "component has an observation or use a strictly positive-definite "
                "regularizer."
            ) from error

    def estimate_gamma(self, Y, sigma2: float = 0.0) -> np.ndarray:
        """Estimate a normalized graph PSD from partial signal realizations.

        The implementation follows the sampled-covariance correction described
        in the thesis and in Yang et al. Missing entries are converted from
        ``NaN`` to zero only in the estimator's working copy. Observed zeros are
        included when estimating the sampling probability and signal mean.

        Parameters
        ----------
        Y:
            Partially observed signal matrix with shape ``(N, M)``.
        sigma2:
            Known additive-noise variance. Must be non-negative.

        Returns
        -------
        numpy.ndarray
            Non-negative PSD estimate with one value per graph frequency and a
            maximum no greater than one. If nothing is observed, returns zeros.
        """
        matrix = self._validate_signal_matrix(Y)
        if (
            isinstance(sigma2, (bool, np.bool_))
            or not np.isscalar(sigma2)
            or not np.isfinite(sigma2)
            or sigma2 < 0
        ):
            raise ValueError("sigma2 must be a finite, non-negative number.")

        observed, zero_filled = self._zero_fill_missing(matrix)
        N, M = matrix.shape
        observed_count = int(observed.sum())
        if observed_count == 0:
            return np.zeros(N, dtype=float)

        sampling_probability = observed_count / (N * M)
        mean = zero_filled.sum() / observed_count
        empirical_second_moment = (zero_filled @ zero_filled.T) / M

        diagonal = np.diag(empirical_second_moment)
        off_diagonal = empirical_second_moment - np.diag(diagonal)
        corrected_diagonal = (
            diagonal
            - sampling_probability * (1 - sampling_probability) * mean**2
            - sampling_probability * sigma2
        ) / sampling_probability
        corrected_off_diagonal = off_diagonal / sampling_probability**2
        corrected_covariance = (
            np.diag(corrected_diagonal) + corrected_off_diagonal
        )

        U = self.graph.eigenvectors
        raw_gamma = np.sum((U.T @ corrected_covariance) * U.T, axis=1)
        return self.normalize_gamma(raw_gamma)

    def reconstruct_smooth(self, y, beta: float = 1.0) -> np.ndarray:
        """Reconstruct signals using combinatorial-Laplacian regularization.

        Parameters
        ----------
        y:
            One partial signal with shape ``(N,)`` or a signal matrix with shape
            ``(N, M)``. Missing entries must be ``NaN``.
        beta:
            Positive weight of the graph-smoothness penalty.
        """
        self._validate_regularization(beta=beta)
        values = np.asarray(y, dtype=float)

        if values.ndim == 2:
            matrix = self._validate_signal_matrix(values, name="y")
            reconstructed = np.empty_like(matrix, dtype=float)
            for column in range(matrix.shape[1]):
                reconstructed[:, column] = self.reconstruct_smooth(
                    matrix[:, column], beta=beta
                )
            return reconstructed

        signal = self._validate_single_signal(values)
        observed, zero_filled = self._zero_fill_missing(signal)
        system = beta * self.graph.L + np.diag(observed.astype(float))
        return self._solve(system, zero_filled)

    def reconstruct_psd_single(
        self,
        y,
        gamma,
        alpha: float = 10.0,
        beta: float = 1.0,
        scaling: bool = True,
    ) -> np.ndarray:
        """Reconstruct one signal using a PSD-adaptive spectral penalty.

        Parameters
        ----------
        y:
            Partial signal with one value per graph node and ``NaN`` at missing
            vertices.
        gamma:
            Non-negative PSD value for every graph frequency.
        alpha:
            Non-negative contrast of ``exp(-alpha * gamma)``.
        beta:
            Positive weight of the PSD penalty.
        scaling:
            If true, normalize spectral weights and scale them by the largest
            graph eigenvalue, as used in the experimental implementation.
        """
        self._validate_regularization(alpha=alpha, beta=beta)
        signal = self._validate_single_signal(y)
        observed, zero_filled = self._zero_fill_missing(signal)
        penalty = self._psd_penalty(gamma, alpha, scaling)

        system = beta * penalty
        system[observed, observed] += 1.0
        return self._solve(system, zero_filled)

    def reconstruct_psd(
        self,
        Y,
        gamma=None,
        alpha: float = 10.0,
        beta: float = 1.0,
        scaling: bool = True,
    ) -> np.ndarray:
        """Reconstruct a signal batch using one global PSD profile.

        If ``gamma`` is omitted, it is estimated from the same partially
        observed batch.
        """
        matrix = self._validate_signal_matrix(Y)
        spectrum = self.estimate_gamma(matrix) if gamma is None else gamma

        reconstructed = np.empty_like(matrix, dtype=float)
        for column in range(matrix.shape[1]):
            reconstructed[:, column] = self.reconstruct_psd_single(
                matrix[:, column],
                spectrum,
                alpha=alpha,
                beta=beta,
                scaling=scaling,
            )
        return reconstructed

    def reconstruct_mixed(
        self,
        Y,
        labels,
        alpha: float = 10.0,
        beta: float = 1.0,
        scaling: bool = True,
    ) -> np.ndarray:
        """Reconstruct signals with one estimated PSD per assigned cluster.

        Parameters
        ----------
        Y:
            Partial signal matrix with shape ``(N, M)``.
        labels:
            One cluster label per signal. Labels need not be consecutive.
        alpha, beta, scaling:
            Parameters forwarded to :meth:`reconstruct_psd`.
        """
        matrix = self._validate_signal_matrix(Y)
        assignments = np.asarray(labels)
        if assignments.ndim != 1 or assignments.shape[0] != matrix.shape[1]:
            raise ValueError("labels must contain one cluster label per signal.")

        reconstructed = np.empty_like(matrix, dtype=float)
        for cluster in np.unique(assignments):
            selected = assignments == cluster
            cluster_signals = matrix[:, selected]
            gamma = self.estimate_gamma(cluster_signals)
            reconstructed[:, selected] = self.reconstruct_psd(
                cluster_signals,
                gamma=gamma,
                alpha=alpha,
                beta=beta,
                scaling=scaling,
            )
        return reconstructed
