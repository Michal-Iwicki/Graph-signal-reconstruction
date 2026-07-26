"""Synthetic experiments defined in Sections 4.2 and 4.3 of the thesis.

The module intentionally uses one-factor-at-a-time studies. A study starts from
one documented baseline configuration and changes only one selected parameter;
it does not perform a Cartesian grid search.
"""

from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from src.methods.clustering import ClusteringEvaluator, GMM_Diag
from src.methods.generation import GSPGraph, GraphFactory, SignalGenerator
from src.methods.reconstruction import SignalReconstructor


PLANNED_SYNTHETIC_VALUES = {
    "topology": (
        "knn",
        "grid",
        "erdos_renyi",
        "barabasi_albert",
        "watts_strogatz",
    ),
    "profile_name": (
        "gaussian",
        "shifted_gaussian",
        "skewed",
        "flat",
    ),
    "n_train": (100, 1_000, 10_000),
    "n_components": (2, 5, 10, 20),
    "observation_probability": (0.1, 0.2, 0.5, 0.8),
}

_PLANNED_EXPERIMENTS = {
    "topology": (
        "single_source",
        "clustering",
        "mixture_reconstruction",
    ),
    "profile_name": ("single_source",),
    "n_train": (
        "single_source",
        "clustering",
        "mixture_reconstruction",
    ),
    "n_components": ("clustering", "mixture_reconstruction"),
    "observation_probability": (
        "single_source",
        "clustering",
        "mixture_reconstruction",
    ),
}


@dataclass(frozen=True)
class SyntheticExperimentConfig:
    """Baseline configuration shared by all synthetic experiments.

    The defaults provide a representative initial experiment without the cost
    of the largest settings listed in the thesis. Planned values can be studied
    independently with :func:`run_planned_parameter_study`.
    """

    topology: str = "knn"
    profile_name: str = "gaussian"
    n_nodes: int = 100
    n_train: int = 1_000
    n_test: int = 100
    n_components: int = 2
    observation_probability: float = 0.5
    n_runs: int = 10
    alpha: float = 10.0
    beta: float = 1.0
    smooth_beta: float = 0.1
    seed: int = 42
    knn_neighbors: int = 10
    erdos_renyi_probability: float = 0.08
    barabasi_albert_m: int = 3
    watts_strogatz_neighbors: int = 6
    watts_strogatz_rewiring: float = 0.1

    def __post_init__(self):
        """Validate configuration values immediately."""
        integer_fields = {
            "n_nodes": (self.n_nodes, 2),
            "n_train": (self.n_train, 1),
            "n_test": (self.n_test, 1),
            "n_components": (self.n_components, 1),
            "n_runs": (self.n_runs, 1),
        }
        for name, (value, minimum) in integer_fields.items():
            if (
                not isinstance(value, (int, np.integer))
                or isinstance(value, bool)
                or value < minimum
            ):
                raise ValueError(
                    f"{name} must be an integer greater than or equal to "
                    f"{minimum}."
                )
        if self.n_components > min(self.n_train, self.n_test):
            raise ValueError(
                "n_components cannot exceed n_train or n_test."
            )
        if (
            not np.isfinite(self.observation_probability)
            or not 0 < self.observation_probability <= 1
        ):
            raise ValueError(
                "observation_probability must lie in the interval (0, 1]."
            )
        if not np.isfinite(self.alpha) or self.alpha < 0:
            raise ValueError("alpha must be finite and non-negative.")
        for name, value in (
            ("beta", self.beta),
            ("smooth_beta", self.smooth_beta),
        ):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite.")


def gaussian_psd(
    center: float = 0.15,
    width: float = 0.12,
) -> Callable[[np.ndarray, float], np.ndarray]:
    """Create a Gaussian PSD profile on normalized graph frequencies."""
    if not 0 <= center <= 1 or not np.isfinite(center):
        raise ValueError("center must lie in the interval [0, 1].")
    if not np.isfinite(width) or width <= 0:
        raise ValueError("width must be positive and finite.")

    def profile(eigenvalues: np.ndarray, lambda_max: float) -> np.ndarray:
        """Evaluate the configured Gaussian profile."""
        normalized = np.asarray(eigenvalues, dtype=float) / lambda_max
        return np.exp(-0.5 * ((normalized - center) / width) ** 2)

    return profile


def skewed_psd(
    shape: float = 2.0,
    decay: float = 8.0,
) -> Callable[[np.ndarray, float], np.ndarray]:
    """Create a right-skewed PSD profile on normalized frequencies."""
    if not np.isfinite(shape) or shape <= 0:
        raise ValueError("shape must be positive and finite.")
    if not np.isfinite(decay) or decay <= 0:
        raise ValueError("decay must be positive and finite.")

    def profile(eigenvalues: np.ndarray, lambda_max: float) -> np.ndarray:
        """Evaluate the configured skewed profile."""
        normalized = np.clip(
            np.asarray(eigenvalues, dtype=float) / lambda_max,
            0.0,
            1.0,
        )
        values = normalized**shape * np.exp(-decay * normalized)
        maximum = float(np.max(values))
        return values / maximum if maximum > 0 else values

    return profile


def flat_psd(
    eigenvalues: np.ndarray,
    lambda_max: float,
) -> np.ndarray:
    """Return a white-noise-like flat PSD profile."""
    del lambda_max
    return np.ones_like(eigenvalues, dtype=float)


def get_psd_profile(
    name: str,
) -> Callable[[np.ndarray, float], np.ndarray]:
    """Return one of the single-source PSD profiles listed in the thesis."""
    profiles = {
        "gaussian": gaussian_psd(center=0.15, width=0.12),
        "shifted_gaussian": gaussian_psd(center=0.55, width=0.12),
        "skewed": skewed_psd(),
        "flat": flat_psd,
    }
    try:
        return profiles[name]
    except KeyError as error:
        choices = ", ".join(profiles)
        raise ValueError(
            f"Unknown profile_name {name!r}. Choose from: {choices}."
        ) from error


def make_mixture_profiles(
    n_components: int,
    width: float = 0.12,
) -> list[Callable[[np.ndarray, float], np.ndarray]]:
    """Create shifted Gaussian PSD sources with increasing overlap for larger K."""
    if (
        not isinstance(n_components, (int, np.integer))
        or isinstance(n_components, bool)
        or n_components < 1
    ):
        raise ValueError("n_components must be a positive integer.")
    centers = np.linspace(0.15, 0.85, int(n_components))
    return [gaussian_psd(float(center), width) for center in centers]


def _grid_shape(n_nodes: int) -> tuple[int, int]:
    """Factor a node count into grid dimensions as close to square as possible."""
    rows = int(np.sqrt(n_nodes))
    while rows > 1 and n_nodes % rows != 0:
        rows -= 1
    return rows, n_nodes // rows


def create_synthetic_graph(
    config: SyntheticExperimentConfig,
    seed: int,
) -> GSPGraph:
    """Create the topology selected in a synthetic experiment configuration."""
    topology = config.topology
    if topology == "knn":
        if not 1 <= config.knn_neighbors < config.n_nodes:
            raise ValueError("knn_neighbors must satisfy 1 <= k < n_nodes.")
        for attempt in range(100):
            np.random.seed(seed + attempt)
            graph = GraphFactory.generate_nn_graph(
                config.n_nodes,
                config.knn_neighbors,
            )
            if nx.is_connected(graph):
                return graph
        raise ValueError(
            "Could not generate a connected k-NN graph. Increase "
            "knn_neighbors."
        )
    if topology == "grid":
        rows, columns = _grid_shape(config.n_nodes)
        return GraphFactory.generate_grid_graph(rows, columns)
    if topology == "erdos_renyi":
        return GraphFactory.generate_erdos_renyi_graph(
            config.n_nodes,
            config.erdos_renyi_probability,
            seed=seed,
        )
    if topology == "barabasi_albert":
        return GraphFactory.generate_barabasi_albert_graph(
            config.n_nodes,
            config.barabasi_albert_m,
            seed=seed,
        )
    if topology == "watts_strogatz":
        return GraphFactory.generate_watts_strogatz_graph(
            config.n_nodes,
            config.watts_strogatz_neighbors,
            config.watts_strogatz_rewiring,
            seed=seed,
        )
    choices = ", ".join(PLANNED_SYNTHETIC_VALUES["topology"])
    raise ValueError(
        f"Unknown topology {topology!r}. Choose from: {choices}."
    )


def _ensure_observed_signal(
    complete: np.ndarray,
    observed: np.ndarray,
    random_generator: np.random.Generator,
) -> np.ndarray:
    """Condition rare all-missing realizations on at least one observation."""
    result = observed.copy()
    all_missing = np.flatnonzero(np.all(np.isnan(result), axis=0))
    for column in all_missing:
        row = int(random_generator.integers(0, result.shape[0]))
        result[row, column] = complete[row, column]
    return result


def _generate_mixture(
    generator: SignalGenerator,
    M: int,
    probability: float,
    profiles,
    random_generator: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a mixture containing at least one realization of every source."""
    component_count = len(profiles)
    proportions = np.full(component_count, 1.0 / component_count)
    for _ in range(100):
        complete, observed, labels = generator.generate_mixed_signals(
            M,
            probability,
            profiles,
            proportions,
        )
        if len(np.unique(labels)) == component_count:
            return (
                complete,
                _ensure_observed_signal(
                    complete,
                    observed,
                    random_generator,
                ),
                labels,
            )
    raise ValueError(
        "Could not sample every mixture component. Increase the number of "
        "signals or reduce n_components."
    )


def _config_metadata(config: SyntheticExperimentConfig) -> dict:
    """Return configuration fields stored with every result row."""
    return asdict(config)


def _add_result(
    rows: list[dict],
    config: SyntheticExperimentConfig,
    run: int,
    experiment: str,
    method: str,
    metric: str,
    value: float,
) -> None:
    """Append one tidy-format metric observation."""
    rows.append(
        {
            **_config_metadata(config),
            "run": run,
            "experiment": experiment,
            "method": method,
            "metric": metric,
            "value": float(value),
        }
    )


def _reconstruction_metrics(
    truth: np.ndarray,
    estimate: np.ndarray,
    observed: np.ndarray,
) -> dict[str, float]:
    """Compute thesis MAE and a diagnostic MAE restricted to missing entries."""
    absolute_error = np.abs(estimate - truth)
    missing = np.isnan(observed)
    return {
        "mae": float(np.mean(absolute_error)),
        "mae_missing": (
            float(np.mean(absolute_error[missing]))
            if np.any(missing)
            else float("nan")
        ),
    }


def _add_reconstruction_results(
    rows: list[dict],
    config: SyntheticExperimentConfig,
    run: int,
    experiment: str,
    method: str,
    truth: np.ndarray,
    estimate: np.ndarray,
    observed: np.ndarray,
) -> None:
    """Append all reconstruction metrics for one method and repetition."""
    for metric, value in _reconstruction_metrics(
        truth,
        estimate,
        observed,
    ).items():
        _add_result(
            rows,
            config,
            run,
            experiment,
            method,
            metric,
            value,
        )


def run_single_source_experiment(
    config: SyntheticExperimentConfig | None = None,
) -> pd.DataFrame:
    """Evaluate PSD estimation and reconstruction for one stationary source.

    The PSD estimator is evaluated with MAE and RMSE. Smoothness and single-PSD
    reconstruction are evaluated with the thesis-wide MAE and missing-only MAE.
    """
    config = config or SyntheticExperimentConfig()
    profile = get_psd_profile(config.profile_name)
    rows = []

    for run in range(config.n_runs):
        run_seed = config.seed + run
        np.random.seed(run_seed)
        random_generator = np.random.default_rng(run_seed)
        graph = create_synthetic_graph(config, run_seed)
        generator = SignalGenerator(graph)
        reconstructor = SignalReconstructor(graph)
        true_gamma = reconstructor.normalize_gamma(
            profile(graph.eigenvalues, float(graph.eigenvalues[-1]))
        )

        training_truth, training_observed = generator.generate_signals(
            config.n_train,
            config.observation_probability,
            profile,
        )
        training_observed = _ensure_observed_signal(
            training_truth,
            training_observed,
            random_generator,
        )
        estimated_gamma = reconstructor.estimate_gamma(training_observed)
        difference = estimated_gamma - true_gamma
        _add_result(
            rows,
            config,
            run,
            "single_source_psd",
            "sampled_covariance",
            "psd_mae",
            np.mean(np.abs(difference)),
        )
        _add_result(
            rows,
            config,
            run,
            "single_source_psd",
            "sampled_covariance",
            "psd_rmse",
            np.sqrt(np.mean(difference**2)),
        )

        test_truth, test_observed = generator.generate_signals(
            config.n_test,
            config.observation_probability,
            profile,
        )
        test_observed = _ensure_observed_signal(
            test_truth,
            test_observed,
            random_generator,
        )
        smooth_estimate = reconstructor.reconstruct_smooth(
            test_observed,
            beta=config.smooth_beta,
        )
        psd_estimate = reconstructor.reconstruct_psd(
            test_observed,
            gamma=estimated_gamma,
            alpha=config.alpha,
            beta=config.beta,
        )
        _add_reconstruction_results(
            rows,
            config,
            run,
            "single_source_reconstruction",
            "smooth",
            test_truth,
            smooth_estimate,
            test_observed,
        )
        _add_reconstruction_results(
            rows,
            config,
            run,
            "single_source_reconstruction",
            "single_psd",
            test_truth,
            psd_estimate,
            test_observed,
        )

    return pd.DataFrame(rows)


def _fit_spectral_gmm(
    graph: GSPGraph,
    signals: np.ndarray,
    n_components: int,
    seed: int,
) -> tuple[GMM_Diag, np.ndarray]:
    """Fit a diagonal GMM to graph Fourier features."""
    features = ClusteringEvaluator.graph_fourier_features(graph, signals)
    model = GMM_Diag(n_components, random_state=seed).fit(features)
    return model, model.predict(features)


def _add_clustering_results(
    rows: list[dict],
    config: SyntheticExperimentConfig,
    run: int,
    experiment: str,
    method: str,
    truth: np.ndarray,
    prediction: np.ndarray,
) -> None:
    """Append clustering accuracy and adjusted Rand index."""
    _add_result(
        rows,
        config,
        run,
        experiment,
        method,
        "acc",
        ClusteringEvaluator.evaluate_accuracy(truth, prediction),
    )
    _add_result(
        rows,
        config,
        run,
        experiment,
        method,
        "ari",
        adjusted_rand_score(truth, prediction),
    )


def run_clustering_experiment(
    config: SyntheticExperimentConfig | None = None,
) -> pd.DataFrame:
    """Evaluate latent-source clustering independently of reconstruction.

    GMM clustering of complete GFT coefficients is an upper reference.
    Clustering after smooth reconstruction is the implementable partial-data
    pipeline described in the thesis.
    """
    config = config or SyntheticExperimentConfig()
    profiles = make_mixture_profiles(config.n_components)
    rows = []

    for run in range(config.n_runs):
        run_seed = config.seed + run
        np.random.seed(run_seed)
        random_generator = np.random.default_rng(run_seed)
        graph = create_synthetic_graph(config, run_seed)
        generator = SignalGenerator(graph)
        complete, observed, labels = _generate_mixture(
            generator,
            config.n_train,
            config.observation_probability,
            profiles,
            random_generator,
        )
        smooth = SignalReconstructor(graph).reconstruct_smooth(
            observed,
            beta=config.smooth_beta,
        )
        _, complete_prediction = _fit_spectral_gmm(
            graph,
            complete,
            config.n_components,
            run_seed,
        )
        _, smooth_prediction = _fit_spectral_gmm(
            graph,
            smooth,
            config.n_components,
            run_seed,
        )
        _add_clustering_results(
            rows,
            config,
            run,
            "source_clustering",
            "complete_gft",
            labels,
            complete_prediction,
        )
        _add_clustering_results(
            rows,
            config,
            run,
            "source_clustering",
            "smooth_gft",
            labels,
            smooth_prediction,
        )

    return pd.DataFrame(rows)


def _estimate_group_psds(
    reconstructor: SignalReconstructor,
    observed: np.ndarray,
    labels: np.ndarray,
    n_components: int,
    fallback: np.ndarray,
) -> dict[int, np.ndarray]:
    """Estimate one PSD per group, using a global fallback for empty groups."""
    result = {}
    for component in range(n_components):
        selected = labels == component
        result[component] = (
            reconstructor.estimate_gamma(observed[:, selected])
            if np.any(selected)
            else fallback
        )
    return result


def _reconstruct_from_group_psds(
    reconstructor: SignalReconstructor,
    observed: np.ndarray,
    labels: np.ndarray,
    psds: dict[int, np.ndarray],
    alpha: float,
    beta: float,
) -> np.ndarray:
    """Apply supplied group PSDs without re-estimating them on test signals."""
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


def run_mixture_reconstruction_experiment(
    config: SyntheticExperimentConfig | None = None,
) -> pd.DataFrame:
    """Compare reconstruction methods for a mixture of PSD sources.

    Compared methods are graph smoothness, one global PSD, PSDs estimated after
    source clustering, and an oracle grouped-PSD reference using true labels.
    The clustered method learns both GMM and PSDs only from training signals.
    """
    config = config or SyntheticExperimentConfig()
    profiles = make_mixture_profiles(config.n_components)
    rows = []

    for run in range(config.n_runs):
        run_seed = config.seed + run
        np.random.seed(run_seed)
        random_generator = np.random.default_rng(run_seed)
        graph = create_synthetic_graph(config, run_seed)
        generator = SignalGenerator(graph)
        reconstructor = SignalReconstructor(graph)
        train_truth, train_observed, train_labels = _generate_mixture(
            generator,
            config.n_train,
            config.observation_probability,
            profiles,
            random_generator,
        )
        del train_truth
        test_truth, test_observed, test_labels = _generate_mixture(
            generator,
            config.n_test,
            config.observation_probability,
            profiles,
            random_generator,
        )

        gamma_global = reconstructor.estimate_gamma(train_observed)
        oracle_psds = _estimate_group_psds(
            reconstructor,
            train_observed,
            train_labels,
            config.n_components,
            gamma_global,
        )

        smooth_train = reconstructor.reconstruct_smooth(
            train_observed,
            beta=config.smooth_beta,
        )
        gmm, predicted_train_labels = _fit_spectral_gmm(
            graph,
            smooth_train,
            config.n_components,
            run_seed,
        )
        clustered_psds = _estimate_group_psds(
            reconstructor,
            train_observed,
            predicted_train_labels,
            config.n_components,
            gamma_global,
        )
        smooth_test = reconstructor.reconstruct_smooth(
            test_observed,
            beta=config.smooth_beta,
        )
        test_features = ClusteringEvaluator.graph_fourier_features(
            graph,
            smooth_test,
        )
        predicted_test_labels = gmm.predict(test_features)

        estimates = {
            "smooth": smooth_test,
            "single_psd": reconstructor.reconstruct_psd(
                test_observed,
                gamma=gamma_global,
                alpha=config.alpha,
                beta=config.beta,
            ),
            "clustered_psd": _reconstruct_from_group_psds(
                reconstructor,
                test_observed,
                predicted_test_labels,
                clustered_psds,
                config.alpha,
                config.beta,
            ),
            "oracle_grouped_psd": _reconstruct_from_group_psds(
                reconstructor,
                test_observed,
                test_labels,
                oracle_psds,
                config.alpha,
                config.beta,
            ),
        }
        for method, estimate in estimates.items():
            _add_reconstruction_results(
                rows,
                config,
                run,
                "mixture_reconstruction",
                method,
                test_truth,
                estimate,
                test_observed,
            )
        _add_clustering_results(
            rows,
            config,
            run,
            "mixture_assignment",
            "smooth_gft",
            test_labels,
            predicted_test_labels,
        )

    return pd.DataFrame(rows)


_EXPERIMENT_RUNNERS = {
    "single_source": run_single_source_experiment,
    "clustering": run_clustering_experiment,
    "mixture_reconstruction": run_mixture_reconstruction_experiment,
}


def run_default_synthetic_experiments(
    config: SyntheticExperimentConfig | None = None,
    experiments: Sequence[str] = (
        "single_source",
        "clustering",
        "mixture_reconstruction",
    ),
) -> pd.DataFrame:
    """Run selected scenarios once at the common baseline configuration."""
    config = config or SyntheticExperimentConfig()
    frames = []
    for experiment in experiments:
        try:
            runner = _EXPERIMENT_RUNNERS[experiment]
        except KeyError as error:
            choices = ", ".join(_EXPERIMENT_RUNNERS)
            raise ValueError(
                f"Unknown experiment {experiment!r}. Choose from: {choices}."
            ) from error
        frames.append(runner(config))
    return pd.concat(frames, ignore_index=True)


def run_one_factor_study(
    parameter: str,
    values: Iterable,
    config: SyntheticExperimentConfig | None = None,
    experiments: Sequence[str] = (
        "single_source",
        "clustering",
        "mixture_reconstruction",
    ),
) -> pd.DataFrame:
    """Vary exactly one configuration field and concatenate tidy results."""
    config = config or SyntheticExperimentConfig()
    if parameter not in asdict(config):
        raise ValueError(f"Unknown configuration parameter {parameter!r}.")
    selected_values = list(values)
    if not selected_values:
        raise ValueError("values must contain at least one setting.")

    frames = []
    for value in selected_values:
        varied_config = replace(config, **{parameter: value})
        frame = run_default_synthetic_experiments(
            varied_config,
            experiments=experiments,
        )
        frame["varied_parameter"] = parameter
        frame["parameter_value"] = value
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def run_planned_parameter_study(
    parameter: str,
    config: SyntheticExperimentConfig | None = None,
    values: Iterable | None = None,
) -> pd.DataFrame:
    """Run a thesis-planned one-factor study with its predefined values."""
    if parameter not in PLANNED_SYNTHETIC_VALUES:
        choices = ", ".join(PLANNED_SYNTHETIC_VALUES)
        raise ValueError(
            f"Unknown planned parameter {parameter!r}. Choose from: {choices}."
        )
    selected_values = (
        PLANNED_SYNTHETIC_VALUES[parameter]
        if values is None
        else values
    )
    return run_one_factor_study(
        parameter,
        selected_values,
        config=config,
        experiments=_PLANNED_EXPERIMENTS[parameter],
    )


def summarize_synthetic_results(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate repetitions into mean, standard deviation, and sample count."""
    required = {"experiment", "method", "metric", "value"}
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"Results are missing columns: {sorted(missing)!r}.")

    group_columns = ["experiment", "method", "metric"]
    if "varied_parameter" in results:
        group_columns.extend(["varied_parameter", "parameter_value"])
    summary = (
        results.groupby(group_columns, dropna=False)["value"]
        .agg(mean="mean", std="std", count="count")
        .reset_index()
    )
    return summary


def plot_synthetic_results(
    results: pd.DataFrame,
) -> dict[tuple[str, str], tuple]:
    """Plot baseline bars or one-factor curves with run-to-run deviation.

    One figure is created for every ``(experiment, metric)`` pair. The function
    returns figures and axes without calling ``plt.show()`` so callers retain
    control over display and file export.
    """
    summary = summarize_synthetic_results(results)
    figures = {}
    is_parameter_study = "varied_parameter" in results

    for (experiment, metric), metric_data in summary.groupby(
        ["experiment", "metric"],
        sort=False,
    ):
        figure, axis = plt.subplots(figsize=(8, 4.5))
        if is_parameter_study:
            parameter = str(metric_data["varied_parameter"].iloc[0])
            values = list(dict.fromkeys(metric_data["parameter_value"]))
            positions = np.arange(len(values), dtype=float)
            for method, method_data in metric_data.groupby(
                "method",
                sort=False,
            ):
                indexed = method_data.set_index("parameter_value")
                means = [indexed.loc[value, "mean"] for value in values]
                deviations = [
                    indexed.loc[value, "std"]
                    if np.isfinite(indexed.loc[value, "std"])
                    else 0.0
                    for value in values
                ]
                axis.errorbar(
                    positions,
                    means,
                    yerr=deviations,
                    marker="o",
                    capsize=3,
                    label=method,
                )
            axis.set_xticks(positions, [str(value) for value in values])
            axis.set_xlabel(parameter)
        else:
            methods = list(metric_data["method"])
            means = metric_data["mean"].to_numpy()
            deviations = metric_data["std"].fillna(0.0).to_numpy()
            axis.bar(
                np.arange(len(methods)),
                means,
                yerr=deviations,
                capsize=3,
            )
            axis.set_xticks(
                np.arange(len(methods)),
                methods,
                rotation=20,
                ha="right",
            )
            axis.set_xlabel("method")

        axis.set(
            title=f"{experiment}: {metric}",
            ylabel=metric,
        )
        if is_parameter_study:
            axis.legend()
        axis.grid(axis="y", alpha=0.25)
        figure.tight_layout()
        figures[(experiment, metric)] = (figure, axis)

    return figures


def save_synthetic_results(
    results: pd.DataFrame,
    path: str | Path,
) -> Path:
    """Save tidy synthetic results to CSV and return the resolved path."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(destination, index=False)
    return destination.resolve()
