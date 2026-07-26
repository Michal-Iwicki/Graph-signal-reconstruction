"""High-level reconstruction workflows for mixtures of graph signals."""

from collections.abc import Iterable

import numpy as np
from scipy.interpolate import UnivariateSpline

from .clustering import ClusteringEvaluator, GMM_Diag
from .generation import GSPGraph
from .reconstruction import SignalReconstructor


class MixedSignalReconstruction:
    """Coordinate reconstruction, clustering, and PSD transfer.

    The class exposes baseline smooth and global-PSD reconstruction as well as
    cluster-specific sequential, iterative, and EM-PSD workflows. PSD profiles
    learned by the PSD-based methods are cached as continuous functions for
    reconstruction on another graph.
    """

    _METHOD_ALIASES = {
        "simultaneus_method": "simultaneous_method",
        "em_psd": "simultaneous_method",
    }

    def __init__(self, graph: GSPGraph):
        if not isinstance(graph, GSPGraph):
            raise TypeError("graph must be a GSPGraph instance.")
        self.graph = graph
        self.base_reconstructor = SignalReconstructor(graph)
        self.last_labels = None
        self.learned_splines = None
        self.best_K_ = None
        self.best_score_ = None

    @staticmethod
    def _reject_unknown_options(options: dict) -> None:
        """Reject misspelled or unsupported workflow parameters."""
        if options:
            names = ", ".join(sorted(options))
            raise TypeError(f"Unexpected reconstruction parameter(s): {names}.")

    @staticmethod
    def _normalize_rows(values: np.ndarray) -> np.ndarray:
        """Normalize each non-negative matrix row independently."""
        matrix = np.maximum(np.asarray(values, dtype=float), 0.0)
        maxima = np.max(matrix, axis=1, keepdims=True)
        return np.divide(
            matrix,
            maxima,
            out=np.zeros_like(matrix),
            where=maxima > SignalReconstructor._EPSILON,
        )

    @staticmethod
    def _validate_iterations(max_iter: int, tol: float | None = None) -> None:
        """Validate iterative-algorithm stopping parameters."""
        if (
            not isinstance(max_iter, (int, np.integer))
            or isinstance(max_iter, bool)
            or max_iter < 1
        ):
            raise ValueError("max_iter must be a positive integer.")
        if tol is not None and (
            not np.isscalar(tol) or not np.isfinite(tol) or tol <= 0
        ):
            raise ValueError("tol must be a positive finite number.")

    @staticmethod
    def _validate_min_cluster_size(min_cluster_size: int) -> int:
        """Validate and return the minimum accepted cluster size."""
        if (
            not isinstance(min_cluster_size, (int, np.integer))
            or isinstance(min_cluster_size, bool)
            or min_cluster_size < 1
        ):
            raise ValueError("min_cluster_size must be a positive integer.")
        return int(min_cluster_size)

    @staticmethod
    def _validate_k_list(K_list: Iterable[int], sample_count: int) -> list[int]:
        """Validate candidate component counts while preserving their order."""
        try:
            candidates = list(K_list)
        except TypeError as error:
            raise ValueError("K_list must be an iterable of positive integers.") from error
        if not candidates:
            raise ValueError("K_list must contain at least one candidate.")

        validated = []
        for candidate in candidates:
            if (
                not isinstance(candidate, (int, np.integer))
                or isinstance(candidate, bool)
                or not 1 <= candidate <= sample_count
            ):
                raise ValueError(
                    "Every value in K_list must be an integer between 1 and "
                    "the number of signals."
                )
            integer_candidate = int(candidate)
            if integer_candidate not in validated:
                validated.append(integer_candidate)
        return validated

    def _fit_gamma_spline(
        self,
        eigenvalues,
        gamma,
        smoothing_factor: float | None = None,
    ):
        """Fit a continuous PSD profile, safely handling repeated eigenvalues."""
        frequencies = np.asarray(eigenvalues, dtype=float)
        spectrum = np.asarray(gamma, dtype=float)
        if (
            frequencies.ndim != 1
            or spectrum.ndim != 1
            or frequencies.shape != spectrum.shape
            or frequencies.size == 0
        ):
            raise ValueError("eigenvalues and gamma must be non-empty vectors of equal length.")
        if not np.all(np.isfinite(frequencies)) or not np.all(np.isfinite(spectrum)):
            raise ValueError("eigenvalues and gamma must contain finite values.")
        if np.any(spectrum < 0):
            raise ValueError("gamma must contain non-negative values.")
        if smoothing_factor is not None and (
            not np.isscalar(smoothing_factor)
            or not np.isfinite(smoothing_factor)
            or smoothing_factor < 0
        ):
            raise ValueError("smoothing_factor must be non-negative or None.")

        order = np.argsort(frequencies)
        sorted_frequencies = frequencies[order]
        sorted_spectrum = spectrum[order]
        unique_frequencies, inverse, counts = np.unique(
            sorted_frequencies,
            return_inverse=True,
            return_counts=True,
        )
        unique_spectrum = (
            np.bincount(inverse, weights=sorted_spectrum) / counts
        )

        if unique_frequencies.size == 1:
            return np.poly1d([float(unique_spectrum[0])])

        degree = min(3, unique_frequencies.size - 1)
        return UnivariateSpline(
            unique_frequencies,
            unique_spectrum,
            k=degree,
            s=smoothing_factor,
            ext=0,
        )

    @staticmethod
    def _evaluate_spline_psd(spline, target_eigenvalues) -> np.ndarray:
        """Evaluate a learned profile and constrain it to normalized PSD bounds."""
        frequencies = np.asarray(target_eigenvalues, dtype=float)
        if frequencies.ndim != 1 or not np.all(np.isfinite(frequencies)):
            raise ValueError("target_eigenvalues must be a finite vector.")
        return np.clip(np.asarray(spline(frequencies), dtype=float), 0.0, 1.0)

    def _cache_splines(
        self,
        psds: dict,
        smoothing_factor: float | None = None,
    ) -> None:
        """Cache continuous versions of cluster-specific PSD estimates."""
        self.learned_splines = {
            cluster: self._fit_gamma_spline(
                self.graph.eigenvalues,
                gamma,
                smoothing_factor=smoothing_factor,
            )
            for cluster, gamma in psds.items()
        }

    def _estimate_cluster_psds(
        self,
        Y: np.ndarray,
        labels: np.ndarray,
        clusters,
        min_cluster_size: int,
    ) -> dict | None:
        """Estimate PSDs or reject a partition containing a small cluster."""
        psds = {}
        for cluster in clusters:
            selected = labels == cluster
            if int(selected.sum()) < min_cluster_size:
                return None
            psds[cluster] = self.base_reconstructor.estimate_gamma(Y[:, selected])
        return psds

    def _reconstruct_with_psds(
        self,
        Y: np.ndarray,
        labels: np.ndarray,
        psds: dict,
        alpha: float,
        beta: float,
    ) -> np.ndarray:
        """Reconstruct each cluster with its supplied PSD."""
        reconstructed = np.empty_like(Y, dtype=float)
        for cluster, gamma in psds.items():
            selected = labels == cluster
            if not np.any(selected):
                continue
            reconstructed[:, selected] = self.base_reconstructor.reconstruct_psd(
                Y[:, selected],
                gamma=gamma,
                alpha=alpha,
                beta=beta,
            )
        return reconstructed

    def fit_transform(self, Y, method: str = "smooth", **kwargs) -> np.ndarray:
        """Fit the selected workflow and reconstruct a signal matrix.

        Parameters
        ----------
        Y:
            Partially observed matrix with shape ``(N, M)`` and ``NaN`` at
            missing vertices.
        method:
            One of ``"smooth"``, ``"global_psd"``,
            ``"clustered_reconstruction"``,
            ``"iterative_clustered_reconstruction"``, or
            ``"simultaneous_method"``. The legacy misspelling
            ``"simultaneus_method"`` remains supported.
        **kwargs:
            Method-specific parameters. Unknown parameters raise ``TypeError``.

        Returns
        -------
        numpy.ndarray
            Reconstructed signal matrix with the same shape as ``Y``.
        """
        matrix = self.base_reconstructor._validate_signal_matrix(Y)
        selected_method = self._METHOD_ALIASES.get(method, method)
        options = dict(kwargs)
        self.last_labels = None
        self.learned_splines = None
        self.best_K_ = None
        self.best_score_ = None

        if selected_method == "smooth":
            beta = options.pop("beta", 1.0)
            self._reject_unknown_options(options)
            return self.base_reconstructor.reconstruct_smooth(matrix, beta=beta)

        if selected_method == "global_psd":
            alpha = options.pop("alpha", 10.0)
            beta = options.pop("beta", 1.0)
            sigma2 = options.pop("sigma2", 0.0)
            smoothing_factor = options.pop("smoothing_factor", None)
            self._reject_unknown_options(options)

            gamma = self.base_reconstructor.estimate_gamma(matrix, sigma2=sigma2)
            self._cache_splines({0: gamma}, smoothing_factor)
            self.last_labels = np.zeros(matrix.shape[1], dtype=int)
            self.best_K_ = 1
            return self.base_reconstructor.reconstruct_psd(
                matrix,
                gamma=gamma,
                alpha=alpha,
                beta=beta,
            )

        if selected_method == "clustered_reconstruction":
            if "K_list" in options:
                return self._clustered_reconstruction_auto(matrix, **options)

            if "labels" not in options:
                raise ValueError(
                    "clustered_reconstruction requires labels or K_list."
                )
            labels = np.asarray(options.pop("labels"))
            alpha = options.pop("alpha", 10.0)
            beta = options.pop("beta", 1.0)
            smoothing_factor = options.pop("smoothing_factor", None)
            self._reject_unknown_options(options)
            if labels.ndim != 1 or labels.shape[0] != matrix.shape[1]:
                raise ValueError("labels must contain one value per signal.")

            clusters = np.unique(labels)
            psds = self._estimate_cluster_psds(matrix, labels, clusters, 1)
            reconstructed = self._reconstruct_with_psds(
                matrix,
                labels,
                psds,
                alpha,
                beta,
            )
            self.last_labels = labels.copy()
            self.best_K_ = len(clusters)
            self._cache_splines(psds, smoothing_factor)
            return reconstructed

        if selected_method == "iterative_clustered_reconstruction":
            if "K_list" not in options:
                raise ValueError(
                    "iterative_clustered_reconstruction requires K_list."
                )
            return self._iterative_clustered_reconstruction_auto(matrix, **options)

        if selected_method == "simultaneous_method":
            if "K_list" in options:
                return self._reconstruct_em_psd_auto(matrix, **options)

            K = options.pop("K", 2)
            smoothing_factor = options.pop("smoothing_factor", None)
            reconstructed, gmm, responsibilities = self._reconstruct_em_psd(
                matrix,
                K=K,
                **options,
            )
            labels = np.argmax(responsibilities, axis=1)
            gamma_matrix = self._normalize_rows(gmm.var)
            self.last_labels = labels
            self.best_K_ = int(K)
            self._cache_splines(
                {cluster: gamma_matrix[cluster] for cluster in range(int(K))},
                smoothing_factor,
            )
            return reconstructed

        allowed = (
            "smooth, global_psd, clustered_reconstruction, "
            "iterative_clustered_reconstruction, simultaneous_method"
        )
        raise ValueError(f"Unknown reconstruction method {method!r}. Choose from: {allowed}.")

    def _iterative_clustered_reconstruction_auto(
        self,
        Y,
        K_list,
        max_iter: int = 10,
        alpha: float = 10.0,
        beta: float = 1.0,
        init_beta: float = 0.01,
        min_cluster_size: int = 3,
        random_state: int | None = None,
        smoothing_factor: float | None = None,
    ) -> np.ndarray:
        """Select ``K`` while alternating hard clustering and reconstruction."""
        self._validate_iterations(max_iter)
        minimum_size = self._validate_min_cluster_size(min_cluster_size)
        candidates = self._validate_k_list(K_list, Y.shape[1])
        SignalReconstructor._validate_regularization(alpha=alpha, beta=beta)
        SignalReconstructor._validate_regularization(beta=init_beta)

        best = None
        for K in candidates:
            reconstructed = self.base_reconstructor.reconstruct_smooth(
                Y,
                beta=init_beta,
            )
            previous_labels = None

            for _ in range(max_iter):
                features = ClusteringEvaluator.graph_fourier_features(
                    self.graph,
                    reconstructed,
                )
                gmm = GMM_Diag(K=K, random_state=random_state).fit(features)
                labels = gmm.predict(features)
                if previous_labels is not None and np.array_equal(
                    labels,
                    previous_labels,
                ):
                    break
                previous_labels = labels.copy()
                reconstructed = self.base_reconstructor.reconstruct_mixed(
                    Y,
                    labels,
                    alpha=alpha,
                    beta=beta,
                )

            psds = self._estimate_cluster_psds(
                Y,
                labels,
                range(K),
                minimum_size,
            )
            if psds is None:
                continue
            score = self._calculate_yang_cost(
                Y,
                reconstructed,
                psds,
                labels,
                alpha,
                beta,
            )
            if best is None or score < best["score"]:
                best = {
                    "score": score,
                    "signals": reconstructed.copy(),
                    "labels": labels.copy(),
                    "psds": psds,
                    "K": K,
                }

        return self._finish_model_selection(best, candidates, smoothing_factor)

    def transfer_reconstruction(
        self,
        Y_new,
        new_graph: GSPGraph,
        labels=None,
        alpha: float = 10.0,
        beta: float = 1.0,
    ) -> np.ndarray:
        """Reconstruct signals on a new graph using cached PSD profiles.

        ``labels`` are required when more than one PSD profile was learned.
        Every supplied label must have a corresponding cached profile.
        """
        if not self.learned_splines:
            raise ValueError("No learned PSD profiles are available. Call fit_transform first.")
        new_reconstructor = SignalReconstructor(new_graph)
        matrix = new_reconstructor._validate_signal_matrix(Y_new, name="Y_new")

        if labels is None:
            if len(self.learned_splines) != 1:
                raise ValueError(
                    "labels are required when multiple PSD profiles were learned."
                )
            spline = next(iter(self.learned_splines.values()))
            gamma = self._evaluate_spline_psd(
                spline,
                new_graph.eigenvalues,
            )
            return new_reconstructor.reconstruct_psd(
                matrix,
                gamma=gamma,
                alpha=alpha,
                beta=beta,
            )

        assignments = np.asarray(labels)
        if assignments.ndim != 1 or assignments.shape[0] != matrix.shape[1]:
            raise ValueError("labels must contain one value per new signal.")
        unknown = set(np.unique(assignments)) - set(self.learned_splines)
        if unknown:
            raise ValueError(f"No learned PSD profile for labels: {sorted(unknown)!r}.")

        reconstructed = np.empty_like(matrix, dtype=float)
        for cluster in np.unique(assignments):
            selected = assignments == cluster
            gamma = self._evaluate_spline_psd(
                self.learned_splines[cluster],
                new_graph.eigenvalues,
            )
            reconstructed[:, selected] = new_reconstructor.reconstruct_psd(
                matrix[:, selected],
                gamma=gamma,
                alpha=alpha,
                beta=beta,
            )
        return reconstructed

    def _calculate_yang_cost(
        self,
        Y,
        Y_hat,
        psds: dict,
        labels,
        alpha: float,
        beta: float,
    ) -> float:
        """Evaluate normalized fidelity plus PSD-weighted regularization."""
        observed_signals = self.base_reconstructor._validate_signal_matrix(Y)
        reconstructed = np.asarray(Y_hat, dtype=float)
        assignments = np.asarray(labels)
        if reconstructed.shape != observed_signals.shape:
            raise ValueError("Y_hat must have the same shape as Y.")
        if not np.all(np.isfinite(reconstructed)):
            raise ValueError("Y_hat must contain only finite values.")
        if assignments.shape != (observed_signals.shape[1],):
            raise ValueError("labels must contain one value per signal.")

        edge_count = max(1, self.graph.number_of_edges())
        penalties = {
            cluster: self.base_reconstructor._psd_penalty(
                gamma,
                alpha,
                scaling=True,
            )
            for cluster, gamma in psds.items()
        }

        cost = 0.0
        for column, cluster in enumerate(assignments):
            if cluster not in penalties:
                raise ValueError(f"Missing PSD for cluster {cluster!r}.")
            original = observed_signals[:, column]
            estimate = reconstructed[:, column]
            observed = ~np.isnan(original)
            observed_count = max(1, int(observed.sum()))

            fidelity = np.sum((original[observed] - estimate[observed]) ** 2)
            fidelity /= observed_count
            regularization = (
                beta
                * (estimate.T @ penalties[cluster] @ estimate)
                / edge_count
            )
            cost += fidelity + regularization
        return float(cost)

    def _clustered_reconstruction_auto(
        self,
        Y,
        K_list,
        alpha: float = 10.0,
        beta: float = 1.0,
        init_beta: float = 0.01,
        min_cluster_size: int = 3,
        random_state: int | None = None,
        smoothing_factor: float | None = None,
    ) -> np.ndarray:
        """Select ``K`` for one-pass clustering followed by PSD reconstruction."""
        minimum_size = self._validate_min_cluster_size(min_cluster_size)
        candidates = self._validate_k_list(K_list, Y.shape[1])
        SignalReconstructor._validate_regularization(alpha=alpha, beta=beta)
        SignalReconstructor._validate_regularization(beta=init_beta)

        smooth_signals = self.base_reconstructor.reconstruct_smooth(
            Y,
            beta=init_beta,
        )
        features = ClusteringEvaluator.graph_fourier_features(
            self.graph,
            smooth_signals,
        )

        best = None
        for K in candidates:
            gmm = GMM_Diag(K=K, random_state=random_state).fit(features)
            labels = gmm.predict(features)
            psds = self._estimate_cluster_psds(
                Y,
                labels,
                range(K),
                minimum_size,
            )
            if psds is None:
                continue

            reconstructed = self._reconstruct_with_psds(
                Y,
                labels,
                psds,
                alpha,
                beta,
            )
            score = self._calculate_yang_cost(
                Y,
                reconstructed,
                psds,
                labels,
                alpha,
                beta,
            )
            if best is None or score < best["score"]:
                best = {
                    "score": score,
                    "signals": reconstructed,
                    "labels": labels.copy(),
                    "psds": psds,
                    "K": K,
                }

        return self._finish_model_selection(best, candidates, smoothing_factor)

    def _reconstruct_em_psd(
        self,
        Y,
        K,
        max_iter: int = 10,
        tol: float = 1e-4,
        alpha: float = 10.0,
        beta: float = 1.0,
        init_beta: float = 0.01,
        random_state: int | None = None,
    ):
        """Alternate spectral GMM updates and PSD-based reconstruction."""
        self._validate_iterations(max_iter, tol)
        K = self._validate_k_list([K], Y.shape[1])[0]
        SignalReconstructor._validate_regularization(alpha=alpha, beta=beta)
        SignalReconstructor._validate_regularization(beta=init_beta)

        reconstructed = self.base_reconstructor.reconstruct_smooth(
            Y,
            beta=init_beta,
        )
        features = ClusteringEvaluator.graph_fourier_features(
            self.graph,
            reconstructed,
        )
        gmm = GMM_Diag(K=K, random_state=random_state).init_params(features)
        previous_likelihood = None

        for _ in range(max_iter):
            responsibilities, likelihood = gmm.e_step(features)
            gmm.m_step(features, responsibilities)
            gamma_matrix = self._normalize_rows(gmm.var)
            labels = np.argmax(responsibilities, axis=1)

            reconstructed = self._reconstruct_with_psds(
                Y,
                labels,
                {cluster: gamma_matrix[cluster] for cluster in range(K)},
                alpha,
                beta,
            )
            features = ClusteringEvaluator.graph_fourier_features(
                self.graph,
                reconstructed,
            )
            if (
                previous_likelihood is not None
                and abs(likelihood - previous_likelihood) < tol
            ):
                break
            previous_likelihood = likelihood

        final_responsibilities, _ = gmm.e_step(features)
        gmm.m_step(features, final_responsibilities)
        final_responsibilities, _ = gmm.e_step(features)
        gamma_matrix = self._normalize_rows(gmm.var)
        final_labels = np.argmax(final_responsibilities, axis=1)
        reconstructed = self._reconstruct_with_psds(
            Y,
            final_labels,
            {cluster: gamma_matrix[cluster] for cluster in range(K)},
            alpha,
            beta,
        )
        final_features = ClusteringEvaluator.graph_fourier_features(
            self.graph,
            reconstructed,
        )
        final_responsibilities, _ = gmm.e_step(final_features)
        self.last_labels = np.argmax(final_responsibilities, axis=1)
        return reconstructed, gmm, final_responsibilities

    def _reconstruct_em_psd_auto(
        self,
        Y,
        K_list,
        criterion: str = "yang",
        alpha: float = 10.0,
        beta: float = 1.0,
        min_cluster_size: int = 3,
        smoothing_factor: float | None = None,
        **kwargs,
    ) -> np.ndarray:
        """Select the EM-PSD component count with Yang cost or BIC."""
        if criterion not in {"yang", "bic"}:
            raise ValueError("criterion must be either 'yang' or 'bic'.")
        minimum_size = self._validate_min_cluster_size(min_cluster_size)
        candidates = self._validate_k_list(K_list, Y.shape[1])

        best = None
        for K in candidates:
            reconstructed, gmm, responsibilities = self._reconstruct_em_psd(
                Y,
                K,
                alpha=alpha,
                beta=beta,
                **kwargs,
            )
            labels = np.argmax(responsibilities, axis=1)
            if any(np.sum(labels == cluster) < minimum_size for cluster in range(K)):
                continue

            gamma_matrix = self._normalize_rows(gmm.var)
            psds = {
                cluster: gamma_matrix[cluster] for cluster in range(K)
            }
            if criterion == "bic":
                features = ClusteringEvaluator.graph_fourier_features(
                    self.graph,
                    reconstructed,
                )
                _, likelihood = gmm.e_step(features)
                sample_count, dimension = features.shape
                parameter_count = 2 * K * dimension + K - 1
                score = -2 * likelihood + parameter_count * np.log(sample_count)
            else:
                score = self._calculate_yang_cost(
                    Y,
                    reconstructed,
                    psds,
                    labels,
                    alpha,
                    beta,
                )

            if best is None or score < best["score"]:
                best = {
                    "score": float(score),
                    "signals": reconstructed.copy(),
                    "labels": labels.copy(),
                    "psds": psds,
                    "K": K,
                }

        return self._finish_model_selection(best, candidates, smoothing_factor)

    def _finish_model_selection(
        self,
        best: dict | None,
        candidates: list[int],
        smoothing_factor: float | None,
    ) -> np.ndarray:
        """Store and return the best valid candidate."""
        if best is None:
            raise ValueError(
                "No valid clustering was found for K_list="
                f"{candidates}. Reduce min_cluster_size or change the candidates."
            )

        self.last_labels = best["labels"]
        self.best_K_ = best["K"]
        self.best_score_ = best["score"]
        self._cache_splines(best["psds"], smoothing_factor)
        return best["signals"]
