"""Transfer PSDs learned on a partial training graph to the full test graph.

Two studies are implemented:

1. ``experiment_single_psd`` -- one stationary graph process;
2. ``experiment_mixed_signals`` -- a mixture of stationary processes.

``TRAIN_VERTEX_VISIBILITY`` controls the fraction of vertices present in the
training graph.  Testing always uses the complete graph.  PSD estimates learned
on the smaller graph are converted into splines on normalized graph frequency
``lambda / lambda_max`` and evaluated on the spectrum of the full graph.

The mixed experiment uses true source labels for both train and test signals.
This deliberately isolates PSD/spline transfer from cross-graph classification,
whose feature dimensions differ when the training graph has fewer vertices.
"""

from collections.abc import Callable, Sequence

import networkx as nx
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline

from src.methods.generation import GSPGraph, GraphFactory, SignalGenerator
from src.methods.reconstruction import SignalReconstructor
from src.experiments._helper import (
    REFERENCE_PSD_PROBABILITIES,
    REFERENCE_PSD_PROFILES,
    apply_observation_mask,
    config_dict,
    draw_nested_mask_uniforms,
    experiment_config,
    missing_mae,
    save_aggregated_results,
    tracked_range,
)


# ============================================================
# 1. SHARED PARAMETERS
# ============================================================

EXPERIMENT_CONFIG = experiment_config(n_nodes=1000)
N_NODES = EXPERIMENT_CONFIG.n_nodes
K_NEIGHBORS = EXPERIMENT_CONFIG.k_neighbors
N_TRAIN = EXPERIMENT_CONFIG.n_train
N_TEST = EXPERIMENT_CONFIG.n_test

# Fraction of full-graph vertices available during training.
TRAIN_VERTEX_VISIBILITY = (0.2, 0.4, 0.6, 0.8, 1.0)

# Testing uses the full graph, but some signal values remain hidden and must be
# reconstructed.
TEST_OBSERVATION_PROBABILITY = EXPERIMENT_CONFIG.p_observed
N_RUNS = EXPERIMENT_CONFIG.n_runs
SEED = EXPERIMENT_CONFIG.seed
ALPHA = EXPERIMENT_CONFIG.alpha
PSD_BETA = EXPERIMENT_CONFIG.psd_beta
SMOOTH_BETA = EXPERIMENT_CONFIG.smooth_beta
SPLINE_SMOOTHING = None

# Profile used by the single-PSD variant.
SINGLE_PSD = REFERENCE_PSD_PROFILES[0]


# ============================================================
# 2. TRAINING GRAPH AS A NESTED SUBGRAPH
# ============================================================


def nested_vertex_order(graph: GSPGraph, rng: np.random.Generator) -> np.ndarray:
    """Draw one vertex order reused by all visibility levels in a run."""
    return rng.permutation(graph.number_of_nodes())


def training_subgraph(
    full_graph: GSPGraph,
    visibility: float,
    vertex_order: np.ndarray,
) -> GSPGraph:
    """Return an induced graph containing a nested fraction of full-graph nodes."""
    if not np.isfinite(visibility) or not 0 < visibility <= 1:
        raise ValueError("visibility must lie in the interval (0, 1].")

    n_full = full_graph.number_of_nodes()
    order = np.asarray(vertex_order)
    if order.ndim != 1 or order.shape[0] != n_full:
        raise ValueError("vertex_order must contain every full-graph row index.")

    n_visible = max(2, int(round(visibility * n_full)))
    visible_row_indices = order[:n_visible]
    visible_nodes = [full_graph.node_order[index] for index in visible_row_indices]
    base_graph = nx.Graph(full_graph)
    return GSPGraph(base_graph.subgraph(visible_nodes).copy())


# ============================================================
# 3. PSD SPLINE ON NORMALIZED FREQUENCIES
# ============================================================


def fit_normalized_psd_spline(
    graph: GSPGraph,
    gamma: np.ndarray,
    smoothing_factor: float | None = SPLINE_SMOOTHING,
) -> Callable[[np.ndarray], np.ndarray]:
    """Fit PSD(lambda/lambda_max), averaging repeated frequencies safely."""
    spectrum = SignalReconstructor.normalize_gamma(gamma)
    lambda_max = float(graph.eigenvalues[-1])
    if lambda_max <= SignalReconstructor._EPSILON:
        raise ValueError("Training graph must have a positive largest eigenvalue.")

    frequencies = graph.eigenvalues / lambda_max
    order = np.argsort(frequencies)
    frequencies = frequencies[order]
    spectrum = spectrum[order]

    unique_frequencies, inverse, counts = np.unique(
        frequencies,
        return_inverse=True,
        return_counts=True,
    )
    unique_spectrum = np.bincount(inverse, weights=spectrum) / counts

    if unique_frequencies.size == 1:
        constant = float(unique_spectrum[0])
        return lambda x: np.full_like(np.asarray(x, dtype=float), constant)

    degree = min(3, unique_frequencies.size - 1)
    spline = UnivariateSpline(
        unique_frequencies,
        unique_spectrum,
        k=degree,
        s=smoothing_factor,
        ext=3,  # Use boundary values outside the fitted range.
    )
    return lambda x: np.clip(np.asarray(spline(x), dtype=float), 0.0, 1.0)


def transfer_gamma(
    spline: Callable[[np.ndarray], np.ndarray],
    target_graph: GSPGraph,
) -> np.ndarray:
    """Evaluate a normalized-frequency PSD spline on a target graph."""
    lambda_max = float(target_graph.eigenvalues[-1])
    if lambda_max <= SignalReconstructor._EPSILON:
        raise ValueError("Target graph must have a positive largest eigenvalue.")
    return SignalReconstructor.normalize_gamma(
        spline(target_graph.eigenvalues / lambda_max)
    )


# ============================================================
# 4. SHARED METRICS
# ============================================================


def metric_row(
    *,
    experiment: str,
    method: str,
    run: int,
    visibility: float,
    train_graph: GSPGraph,
    truth: np.ndarray,
    observed: np.ndarray,
    estimate: np.ndarray,
) -> dict:
    """Build one reconstruction-metric row for a visibility level."""
    return {
        "experiment": experiment,
        "method": method,
        "run": run,
        "train_vertex_visibility": visibility,
        "n_train_vertices": train_graph.number_of_nodes(),
        "n_test_vertices": truth.shape[0],
        "n_train": N_TRAIN,
        "n_test": N_TEST,
        "test_observation_probability": TEST_OBSERVATION_PROBABILITY,
        "mae": missing_mae(truth, estimate, observed),
    }


# ============================================================
# 5. SINGLE-PSD VARIANT
# ============================================================


def experiment_single_psd() -> pd.DataFrame:
    """Learn one PSD on partial graphs and reconstruct on the full graph."""
    rows = []

    for run in tracked_range(N_RUNS, "spline transfer: single PSD"):
        graph_seed = SEED + run
        np.random.seed(graph_seed)
        full_graph = GraphFactory.generate_nn_graph(N_NODES, K_NEIGHBORS)

        rng = np.random.default_rng(SEED + 10_000 * run)
        vertex_order = nested_vertex_order(full_graph, rng)

        # Reuse one full test set and nested test mask for every visibility
        # level within the run.
        np.random.seed(SEED + 20_000 * run)
        test_truth, _ = SignalGenerator(full_graph).generate_signals(
            N_TEST,
            1.0,
            SINGLE_PSD,
        )
        test_uniforms = draw_nested_mask_uniforms(test_truth.shape, rng)
        test_observed = apply_observation_mask(
            test_truth,
            TEST_OBSERVATION_PROBABILITY,
            test_uniforms,
        )
        full_reconstructor = SignalReconstructor(full_graph)
        smooth_estimate = full_reconstructor.reconstruct_smooth(
            test_observed,
            beta=SMOOTH_BETA,
        )

        for visibility in TRAIN_VERTEX_VISIBILITY:
            train_graph = training_subgraph(full_graph, visibility, vertex_order)

            # Use a separate process on the training-graph spectrum while
            # retaining the continuous PSD profile from the full graph.
            np.random.seed(SEED + 30_000 * run + train_graph.number_of_nodes())
            _, train_observed = SignalGenerator(train_graph).generate_signals(
                N_TRAIN,
                1.0,
                SINGLE_PSD,
            )
            train_reconstructor = SignalReconstructor(train_graph)
            gamma_train = train_reconstructor.estimate_gamma(train_observed)
            spline = fit_normalized_psd_spline(train_graph, gamma_train)
            gamma_full = transfer_gamma(spline, full_graph)
            spline_estimate = full_reconstructor.reconstruct_psd(
                test_observed,
                gamma=gamma_full,
                alpha=ALPHA,
                beta=PSD_BETA,
            )

            rows.append(metric_row(
                experiment="single_psd",
                method="transferred_psd_spline",
                run=run,
                visibility=visibility,
                train_graph=train_graph,
                truth=test_truth,
                observed=test_observed,
                estimate=spline_estimate,
            ))
            rows.append(metric_row(
                experiment="single_psd",
                method="smoothing",
                run=run,
                visibility=visibility,
                train_graph=train_graph,
                truth=test_truth,
                observed=test_observed,
                estimate=smooth_estimate,
            ))

    return pd.DataFrame(rows)


# ============================================================
# 6. MIXED-SIGNAL VARIANT
# ============================================================


def estimate_component_splines(
    train_graph: GSPGraph,
    train_observed: np.ndarray,
    train_labels: np.ndarray,
    n_components: int,
) -> dict[int, Callable[[np.ndarray], np.ndarray]]:
    """Estimate one PSD spline per known mixture component."""
    reconstructor = SignalReconstructor(train_graph)
    splines = {}
    for component in range(n_components):
        selected = train_labels == component
        if not np.any(selected):
            raise ValueError(f"Component {component} is absent from training data.")
        gamma = reconstructor.estimate_gamma(train_observed[:, selected])
        splines[component] = fit_normalized_psd_spline(train_graph, gamma)
    return splines


def reconstruct_mixture_from_splines(
    full_graph: GSPGraph,
    test_observed: np.ndarray,
    test_labels: np.ndarray,
    splines: dict[int, Callable[[np.ndarray], np.ndarray]],
) -> np.ndarray:
    """Oracle-label reconstruction used to isolate component PSD transfer."""
    reconstructor = SignalReconstructor(full_graph)
    estimate = np.empty_like(test_observed, dtype=float)
    for component in np.unique(test_labels):
        selected = test_labels == component
        gamma_full = transfer_gamma(splines[int(component)], full_graph)
        estimate[:, selected] = reconstructor.reconstruct_psd(
            test_observed[:, selected],
            gamma=gamma_full,
            alpha=ALPHA,
            beta=PSD_BETA,
        )
    return estimate


def _generate_complete_mixture(
    graph: GSPGraph,
    n_signals: int,
    profiles: Sequence,
    probabilities: Sequence[float],
    *,
    max_attempts: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a complete mixture containing every component."""
    generator = SignalGenerator(graph)
    for _ in range(max_attempts):
        complete, _, labels = generator.generate_mixed_signals(
            n_signals,
            1.0,
            profiles,
            probabilities,
        )
        if len(np.unique(labels)) == len(profiles):
            return complete, labels
    raise ValueError("Could not draw every mixture component.")


def experiment_mixed_signals() -> pd.DataFrame:
    """Transfer component-specific PSD splines from partial to full graphs."""
    rows = []
    n_components = len(REFERENCE_PSD_PROFILES)

    for run in tracked_range(N_RUNS, "spline transfer: mixed PSD"):
        graph_seed = SEED + run
        np.random.seed(graph_seed)
        full_graph = GraphFactory.generate_nn_graph(N_NODES, K_NEIGHBORS)

        rng = np.random.default_rng(SEED + 40_000 * run)
        vertex_order = nested_vertex_order(full_graph, rng)

        np.random.seed(SEED + 50_000 * run)
        test_truth, test_labels = _generate_complete_mixture(
            full_graph,
            N_TEST,
            REFERENCE_PSD_PROFILES,
            REFERENCE_PSD_PROBABILITIES,
        )
        test_uniforms = draw_nested_mask_uniforms(test_truth.shape, rng)
        test_observed = apply_observation_mask(
            test_truth,
            TEST_OBSERVATION_PROBABILITY,
            test_uniforms,
        )
        full_reconstructor = SignalReconstructor(full_graph)
        smooth_estimate = full_reconstructor.reconstruct_smooth(
            test_observed,
            beta=SMOOTH_BETA,
        )

        for visibility in TRAIN_VERTEX_VISIBILITY:
            train_graph = training_subgraph(full_graph, visibility, vertex_order)

            np.random.seed(SEED + 60_000 * run + train_graph.number_of_nodes())
            train_truth, train_labels = _generate_complete_mixture(
                train_graph,
                N_TRAIN,
                REFERENCE_PSD_PROFILES,
                REFERENCE_PSD_PROBABILITIES,
            )

            component_splines = estimate_component_splines(
                train_graph,
                train_truth,
                train_labels,
                n_components,
            )
            component_estimate = reconstruct_mixture_from_splines(
                full_graph,
                test_observed,
                test_labels,
                component_splines,
            )

            # The global spline measures the value of one PSD for the mixture.
            train_reconstructor = SignalReconstructor(train_graph)
            global_gamma = train_reconstructor.estimate_gamma(train_truth)
            global_spline = fit_normalized_psd_spline(train_graph, global_gamma)
            global_estimate = full_reconstructor.reconstruct_psd(
                test_observed,
                gamma=transfer_gamma(global_spline, full_graph),
                alpha=ALPHA,
                beta=PSD_BETA,
            )

            for method, estimate in (
                ("oracle_component_psd_splines", component_estimate),
                ("global_psd_spline", global_estimate),
                ("smoothing", smooth_estimate),
            ):
                rows.append(metric_row(
                    experiment="mixed_signals",
                    method=method,
                    run=run,
                    visibility=visibility,
                    train_graph=train_graph,
                    truth=test_truth,
                    observed=test_observed,
                    estimate=estimate,
                ))

    return pd.DataFrame(rows)


# ============================================================
# 7. EXECUTION AND SUMMARY
# ============================================================


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate reconstruction error over runs."""
    return (
        results.groupby(
            ["experiment", "method", "train_vertex_visibility"],
            as_index=False,
        )
        .agg(
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            n_train_vertices=("n_train_vertices", "first"),
        )
        .sort_values(["experiment", "method", "train_vertex_visibility"])
    )


def run_all() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run both spline-transfer studies and return raw and summary results."""
    single = experiment_single_psd()
    mixed = experiment_mixed_signals()
    results = pd.concat([single, mixed], ignore_index=True)
    return results, summarize(results)


def save_results(results):
    """Save spline-transfer summaries and parameters fixed across visibility."""
    save_aggregated_results(
        results, "spline_transfer",
        ["experiment", "train_vertex_visibility"],
        config_dict(
            EXPERIMENT_CONFIG,
            exclude=("p_observed",),
            test_observation_probability=TEST_OBSERVATION_PROBABILITY,
            spline_smoothing=SPLINE_SMOOTHING,
        ),
    )


def main():
    """Run and save both spline-transfer studies."""
    raw_results, _ = run_all()
    save_results(raw_results)


if __name__ == "__main__":
    main()
