"""Shared PSD, data-preparation, and evaluation helpers for experiments.

The functions deliberately separate complete signal generation, observation
masking, model fitting, and test-set reconstruction. This makes it possible to
reuse latent signals and nested masks without leaking test data into training.
"""

from dataclasses import asdict, dataclass, is_dataclass, replace
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.cluster import contingency_matrix
from scipy.optimize import linear_sum_assignment

from src.methods.clustering import ClusteringEvaluator, GMM_Diag
from src.methods.generation import GSPGraph, SignalGenerator
from src.methods.reconstruction import SignalReconstructor


# Resolving output paths from this file makes them independent of the directory
# used to launch an experiment.
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
RECONSTRUCTION_RESULTS_DIR = RESULTS_DIR / "reconstruction"
CLUSTERING_RESULTS_DIR = RESULTS_DIR / "clustering"
CONFIG_RESULTS_DIR = RESULTS_DIR / "configs"


@dataclass(frozen=True)
class ExperimentConfig:
    """Parameters shared by the maintained graph experiments."""

    n_nodes: int = 100
    k_neighbors: int = 10
    n_train: int = 600
    n_test: int = 200
    p_observed: float = 0.5
    n_components: int = 5
    n_runs: int = 10
    seed: int = 42
    alpha: float = 10.0
    psd_beta: float = 0.75
    smooth_beta: float = 0.75


DEFAULT_EXPERIMENT_CONFIG = ExperimentConfig()


def experiment_config(**overrides) -> ExperimentConfig:
    """Create a shared experiment configuration with explicit overrides."""
    return replace(DEFAULT_EXPERIMENT_CONFIG, **overrides)


def config_dict(
    config: ExperimentConfig,
    *,
    exclude: Sequence[str] = (),
    **extra,
) -> dict:
    """Build JSON metadata from shared configuration and suite additions."""
    values = asdict(config)
    for name in exclude:
        values.pop(name, None)
    return {**values, **extra}


def save_csv(
    frame: pd.DataFrame,
    filename: str,
    directory: Path,
    *,
    index: bool = False,
) -> Path:
    """Save an experiment frame in its metric-specific results directory."""
    if Path(filename).name != filename:
        raise ValueError("filename must not contain directory components.")
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / filename
    frame.to_csv(destination, index=index, float_format="%.4f")
    return destination


def clustering_accuracy(true_labels: np.ndarray, predicted_labels: np.ndarray) -> float:
    """Return clustering accuracy after the optimal permutation of labels."""
    table = contingency_matrix(true_labels, predicted_labels)
    rows, columns = linear_sum_assignment(-table)
    return float(table[rows, columns].sum() / table.sum())


def _json_value(value):
    """Convert numpy, dataclass, and callable values to JSON-safe objects."""
    if is_dataclass(value):
        return _json_value(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if callable(value):
        return getattr(value, "__name__", repr(value))
    return value


def save_experiment_config(config: dict, suite: str) -> Path:
    """Store parameters that are constant throughout a suite in JSON."""
    CONFIG_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    destination = CONFIG_RESULTS_DIR / f"{suite}.json"
    destination.write_text(
        json.dumps(_json_value(config), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return destination


def aggregate_experiment_results(
    results: pd.DataFrame,
    varied_columns: Sequence[str],
    *,
    tidy: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reduce repeated runs to the public reconstruction/clustering schema."""
    missing = set(varied_columns) - set(results.columns)
    if missing:
        raise ValueError(f"Missing varied columns: {sorted(missing)!r}.")
    groups = [*varied_columns, "method"]

    if tidy:
        required = {"metric", "value", "method"}
        if missing_required := required - set(results.columns):
            raise ValueError(f"Missing tidy columns: {sorted(missing_required)!r}.")
        selected = results.loc[results["metric"].isin({"mae", "ari", "acc"})]
        summary = (
            selected.groupby([*groups, "metric"], dropna=False)["value"]
            .agg(mean="mean", std="std")
            .reset_index()
        )
        reconstruction = summary.loc[summary["metric"] == "mae", groups].copy()
        mae = summary.loc[summary["metric"] == "mae", [*groups, "mean", "std"]]
        reconstruction = mae.rename(columns={"mean": "mae", "std": "std_mae"})
        clustering = (
            summary.loc[summary["metric"].isin({"ari", "acc"}), [*groups, "metric", "mean"]]
            .pivot(index=groups, columns="metric", values="mean")
            .reset_index()
            .rename(columns={"acc": "clustering_accuracy"})
        )
        clustering.columns.name = None
        clustering = clustering[[*groups, "ari", "clustering_accuracy"]]
        return reconstruction, clustering

    reconstruction = (
        results.groupby(groups, dropna=False)["mae"]
        .agg(mae="mean", std_mae="std")
        .reset_index()
    )
    if "ari" not in results:
        return reconstruction, pd.DataFrame()
    cluster_rows = results.loc[results["ari"].notna()]
    if cluster_rows.empty:
        return reconstruction, pd.DataFrame()
    if "clustering_accuracy" not in cluster_rows:
        raise ValueError(
            "Clustering results must contain 'clustering_accuracy'."
        )
    metrics = {
        "ari": ("ari", "mean"),
        "clustering_accuracy": ("clustering_accuracy", "mean"),
    }
    clustering = cluster_rows.groupby(groups, dropna=False).agg(**metrics).reset_index()
    return reconstruction, clustering


def save_aggregated_results(
    results: pd.DataFrame,
    suite: str,
    varied_columns: Sequence[str],
    config: dict,
    *,
    tidy: bool = False,
) -> tuple[Path, Path | None, Path]:
    """Save compact aggregate CSV files and the constant configuration JSON."""
    reconstruction, clustering = aggregate_experiment_results(
        results, varied_columns, tidy=tidy
    )
    reconstruction_path = save_csv(
        reconstruction,
        f"{suite}_reconstruction_results.csv",
        RECONSTRUCTION_RESULTS_DIR,
    )
    clustering_path = None
    if not clustering.empty:
        clustering_path = save_csv(
            clustering,
            f"{suite}_clustering_results.csv",
            CLUSTERING_RESULTS_DIR,
        )
    config_path = save_experiment_config(config, suite)
    for path in (reconstruction_path, clustering_path, config_path):
        if path is not None:
            print(f"Saved {path}", flush=True)
    return reconstruction_path, clustering_path, config_path


def tracked_range(total: int, label: str):
    """Yield run numbers and report cumulative progress to the console."""
    started_at = perf_counter()
    print(f"[{label}] 0/{total}", flush=True)
    for index in range(total):
        yield index
        elapsed = perf_counter() - started_at
        print(f"[{label}] {index + 1}/{total} ({elapsed:.1f} s)", flush=True)


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


def normalize_psd(values: np.ndarray) -> np.ndarray:
    """Clip a PSD profile to non-negative values and normalize its maximum."""
    normalized = np.maximum(np.asarray(values, dtype=float), 0.0)
    maximum = float(np.max(normalized))
    return normalized / maximum if maximum > 0 else normalized


def gaussian_psd(
    mean: float,
    variance: float,
) -> Callable[[np.ndarray, float], np.ndarray]:
    """Create a Gaussian profile on normalized graph frequencies."""

    def profile(eigenvalues: np.ndarray, lambda_max: float) -> np.ndarray:
        x = np.asarray(eigenvalues, dtype=float) / lambda_max
        values = np.exp(-0.5 * (x - mean) ** 2 / variance)
        return normalize_psd(values)

    return profile


def edge_skewed_psd(
    scale: float,
    side: str,
) -> Callable[[np.ndarray, float], np.ndarray]:
    """Create a smooth half-Gaussian anchored at a spectral boundary."""
    if side not in {"low", "high"}:
        raise ValueError("side must be 'low' or 'high'.")

    def profile(eigenvalues: np.ndarray, lambda_max: float) -> np.ndarray:
        x = np.asarray(eigenvalues, dtype=float) / lambda_max
        distance = x if side == "low" else 1.0 - x
        values = np.exp(-0.5 * (distance / scale) ** 2)
        return normalize_psd(values)

    return profile


def flat_band_psd(
    low: float,
    high: float,
) -> Callable[[np.ndarray, float], np.ndarray]:
    """Create a flat PSD supported on one normalized frequency interval."""

    def profile(eigenvalues: np.ndarray, lambda_max: float) -> np.ndarray:
        x = np.asarray(eigenvalues, dtype=float) / lambda_max
        return ((x >= low) & (x <= high)).astype(float)

    return profile


REFERENCE_PSD_PROFILES = (
    edge_skewed_psd(scale=0.12, side="low"),
    gaussian_psd(mean=0.30, variance=0.010),
    flat_band_psd(low=0.40, high=0.60),
    gaussian_psd(mean=0.70, variance=0.010),
    edge_skewed_psd(scale=0.12, side="high"),
)

REFERENCE_PSD_PROBABILITIES = (0.25, 0.20, 0.20, 0.20, 0.15)


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


def generate_observed_mixture_split(
    generator: SignalGenerator,
    n_train: int,
    n_test: int,
    profiles: Sequence,
    probabilities: Sequence[float],
    observation_probability: float,
    random_generator: np.random.Generator,
) -> tuple[MixtureSplit, np.ndarray, np.ndarray]:
    """Generate a complete mixture split and independently mask train/test."""
    split = generate_mixture_split(
        generator,
        n_train,
        n_test,
        profiles,
        probabilities,
    )
    train_observed = apply_observation_mask(
        split.train,
        observation_probability,
        draw_nested_mask_uniforms(split.train.shape, random_generator),
    )
    test_observed = apply_observation_mask(
        split.test,
        observation_probability,
        draw_nested_mask_uniforms(split.test.shape, random_generator),
    )
    return split, train_observed, test_observed


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


def reconstruction_metric_rows(
    truth: np.ndarray,
    observed: np.ndarray,
    true_labels: np.ndarray,
    estimates: dict[str, np.ndarray],
    predicted_labels: np.ndarray,
    clustered_model: ClusteredPSDModel,
) -> list[dict[str, float | str]]:
    """Build the common per-method metrics used by reconstruction studies."""
    minimum_cluster_size = min(clustered_model.train_cluster_sizes.values())
    fallback_cluster_count = len(clustered_model.fallback_clusters)
    ari = adjusted_rand_score(true_labels, predicted_labels)
    accuracy = clustering_accuracy(true_labels, predicted_labels)

    return [
        {
            "method": method,
            "mae": missing_mae(truth, estimate, observed),
            "min_train_cluster_size": (
                minimum_cluster_size if method == "proposed" else np.nan
            ),
            "fallback_cluster_count": (
                fallback_cluster_count if method == "proposed" else np.nan
            ),
            "ari": ari if method == "proposed" else np.nan,
            "clustering_accuracy": (
                accuracy if method == "proposed" else np.nan
            ),
        }
        for method, estimate in estimates.items()
    ]


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
