"""Shared helpers for leakage-free synthetic experiment evaluation.

The functions in this module deliberately separate complete signal generation,
observation masking, model fitting, and test-set reconstruction.  This makes it
possible to reuse the same complete signals and nested masks in one-factor
studies without estimating a PSD on the signals used for final evaluation.
"""

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np

from src.methods.clustering import ClusteringEvaluator, GMM_Diag
from src.methods.generation import GSPGraph, SignalGenerator
from src.methods.reconstruction import SignalReconstructor


@dataclass(frozen=True)
class MixtureSplit:
    """Complete train/test signals and their true source assignments."""

    train: np.ndarray
    train_labels: np.ndarray
    test: np.ndarray
    test_labels: np.ndarray


@dataclass(frozen=True)
class ClusteredPSDModel:
    """Objects learned exclusively from the training split."""

    gmm: GMM_Diag
    psds: dict[int, np.ndarray]
    train_cluster_sizes: dict[int, int]
    fallback_clusters: tuple[int, ...]


def draw_nested_mask_uniforms(
    shape: tuple[int, int],
    random_generator: np.random.Generator,
) -> np.ndarray:
    """Draw reusable uniforms for nested observation masks.

    One entry in every signal column is set to zero.  Consequently every
    signal has at least one observation for every strictly positive sampling
    probability, while masks remain nested as the probability increases.
    """
    if len(shape) != 2 or shape[0] < 1 or shape[1] < 1:
        raise ValueError("shape must describe a non-empty signal matrix.")
    uniforms = random_generator.random(shape)
    guaranteed_rows = random_generator.integers(0, shape[0], size=shape[1])
    uniforms[guaranteed_rows, np.arange(shape[1])] = 0.0
    return uniforms


def apply_observation_mask(
    complete: np.ndarray,
    probability: float,
    uniforms: np.ndarray,
) -> np.ndarray:
    """Mask a complete signal matrix with a pre-drawn uniform matrix."""
    signals = np.asarray(complete, dtype=float)
    mask_values = np.asarray(uniforms, dtype=float)
    if signals.ndim != 2 or signals.shape != mask_values.shape:
        raise ValueError("complete and uniforms must be matrices of equal shape.")
    if not np.all(np.isfinite(signals)):
        raise ValueError("complete signals must be finite.")
    if not np.all(np.isfinite(mask_values)):
        raise ValueError("uniforms must be finite.")
    if not np.isfinite(probability) or not 0 < probability <= 1:
        raise ValueError("probability must lie in the interval (0, 1].")
    return np.where(mask_values < probability, signals, np.nan)


def generate_mixture_split(
    generator: SignalGenerator,
    n_train: int,
    n_test: int,
    profiles: Sequence,
    probabilities: Sequence[float],
    *,
    max_attempts: int = 100,
) -> MixtureSplit:
    """Generate independent train/test columns containing every source.

    Complete signals are requested with ``p=1``.  Missing observations are
    applied later, allowing multiple sampling probabilities to share exactly
    the same latent signals.
    """
    component_count = len(profiles)
    for _ in range(max_attempts):
        test, _, test_labels = generator.generate_mixed_signals(
            n_test,
            1.0,
            profiles,
            probabilities,
        )
        train, _, train_labels = generator.generate_mixed_signals(
            n_train,
            1.0,
            profiles,
            probabilities,
        )
        if (
            len(np.unique(train_labels)) == component_count
            and len(np.unique(test_labels)) == component_count
        ):
            return MixtureSplit(train, train_labels, test, test_labels)
    raise ValueError(
        "Could not draw every component in both train and test splits. "
        "Increase split sizes or reduce the component count."
    )


def estimate_group_psds(
    reconstructor: SignalReconstructor,
    observed: np.ndarray,
    labels: np.ndarray,
    n_components: int,
    *,
    min_cluster_size: int = 3,
    fallback: np.ndarray | None = None,
) -> dict[int, np.ndarray]:
    """Estimate training PSDs, optionally regularizing undersized clusters."""
    psds = {}
    for component in range(n_components):
        selected = labels == component
        count = int(np.sum(selected))
        if count < min_cluster_size:
            if fallback is None:
                raise ValueError(
                    f"Cluster {component} has {count} training signals; at "
                    f"least {min_cluster_size} are required."
                )
            psds[component] = fallback
        else:
            psds[component] = reconstructor.estimate_gamma(
                observed[:, selected]
            )
    return psds


def reconstruct_from_group_psds(
    reconstructor: SignalReconstructor,
    observed: np.ndarray,
    labels: np.ndarray,
    psds: dict[int, np.ndarray],
    *,
    alpha: float,
    beta: float,
) -> np.ndarray:
    """Reconstruct test signals using fixed PSDs learned on training data."""
    estimate = np.empty_like(observed, dtype=float)
    for component in np.unique(labels):
        selected = labels == component
        estimate[:, selected] = reconstructor.reconstruct_psd(
            observed[:, selected],
            gamma=psds[int(component)],
            alpha=alpha,
            beta=beta,
        )
    return estimate


def fit_clustered_psd(
    graph: GSPGraph,
    train_observed: np.ndarray,
    n_components: int,
    *,
    smooth_beta: float,
    random_state: int,
    min_cluster_size: int = 3,
) -> ClusteredPSDModel:
    """Fit the proposed clustering and PSD stages on training data only."""
    reconstructor = SignalReconstructor(graph)
    smooth_train = reconstructor.reconstruct_smooth(
        train_observed,
        beta=smooth_beta,
    )
    train_features = ClusteringEvaluator.graph_fourier_features(
        graph,
        smooth_train,
    )
    gmm = GMM_Diag(
        n_components,
        random_state=random_state,
    ).fit(train_features)
    train_labels = gmm.predict(train_features)
    cluster_sizes = {
        component: int(np.sum(train_labels == component))
        for component in range(n_components)
    }
    global_psd = reconstructor.estimate_gamma(train_observed)
    psds = estimate_group_psds(
        reconstructor,
        train_observed,
        train_labels,
        n_components,
        min_cluster_size=min_cluster_size,
        fallback=global_psd,
    )
    fallback_clusters = tuple(
        component
        for component, size in cluster_sizes.items()
        if size < min_cluster_size
    )
    return ClusteredPSDModel(
        gmm=gmm,
        psds=psds,
        train_cluster_sizes=cluster_sizes,
        fallback_clusters=fallback_clusters,
    )


def predict_clustered_psd(
    graph: GSPGraph,
    model: ClusteredPSDModel,
    test_observed: np.ndarray,
    *,
    alpha: float,
    beta: float,
    smooth_beta: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Assign and reconstruct test signals without refitting on the test set."""
    reconstructor = SignalReconstructor(graph)
    smooth_test = reconstructor.reconstruct_smooth(
        test_observed,
        beta=smooth_beta,
    )
    test_features = ClusteringEvaluator.graph_fourier_features(
        graph,
        smooth_test,
    )
    test_labels = model.gmm.predict(test_features)
    estimate = reconstruct_from_group_psds(
        reconstructor,
        test_observed,
        test_labels,
        model.psds,
        alpha=alpha,
        beta=beta,
    )
    return estimate, test_labels


def evaluate_reconstruction_methods(
    graph: GSPGraph,
    train_observed: np.ndarray,
    test_observed: np.ndarray,
    n_components: int,
    *,
    alpha: float,
    beta: float,
    smooth_beta: float,
    random_state: int,
    min_cluster_size: int = 3,
) -> tuple[dict[str, np.ndarray], np.ndarray, ClusteredPSDModel]:
    """Fit on training observations and return test-only reconstructions."""
    reconstructor = SignalReconstructor(graph)
    smooth_test = reconstructor.reconstruct_smooth(
        test_observed,
        beta=smooth_beta,
    )
    global_gamma = reconstructor.estimate_gamma(train_observed)
    global_test = reconstructor.reconstruct_psd(
        test_observed,
        gamma=global_gamma,
        alpha=alpha,
        beta=beta,
    )
    clustered_model = fit_clustered_psd(
        graph,
        train_observed,
        n_components,
        smooth_beta=smooth_beta,
        random_state=random_state,
        min_cluster_size=min_cluster_size,
    )
    proposed_test, predicted_labels = predict_clustered_psd(
        graph,
        clustered_model,
        test_observed,
        alpha=alpha,
        beta=beta,
        smooth_beta=smooth_beta,
    )
    return {
        "proposed": proposed_test,
        "basic_psd": global_test,
        "smoothing": smooth_test,
    }, predicted_labels, clustered_model


def missing_mae(
    truth: np.ndarray,
    estimate: np.ndarray,
    observed: np.ndarray,
) -> float:
    """Return MAE restricted to entries hidden in the test observations."""
    missing = np.isnan(observed)
    if not np.any(missing):
        return float("nan")
    return float(np.mean(np.abs(truth[missing] - estimate[missing])))


def missing_rmse(
    truth: np.ndarray,
    estimate: np.ndarray,
    observed: np.ndarray,
) -> float:
    """Return RMSE restricted to entries hidden in the test observations."""
    missing = np.isnan(observed)
    if not np.any(missing):
        return float("nan")
    errors = truth[missing] - estimate[missing]
    return float(np.sqrt(np.mean(errors**2)))
