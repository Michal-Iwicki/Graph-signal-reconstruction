"""Testowanie jednego parametru naraz dla stałej mieszaniny 5 PSD."""

import numpy as np
import pandas as pd

from src.methods.generation import GraphFactory, SignalGenerator
from src.experiments._helper import (
    MixtureSplit,
    REFERENCE_PSD_PROBABILITIES,
    REFERENCE_PSD_PROFILES,
    apply_observation_mask,
    draw_nested_mask_uniforms,
    config_dict,
    evaluate_reconstruction_methods,
    experiment_config,
    generate_mixture_split,
    reconstruction_metric_rows,
    save_aggregated_results,
    tracked_range,
)


# ============================================================
# 1. PARAMETRY WSPÓLNE
# ============================================================

EXPERIMENT_CONFIG = experiment_config()
N_TRAIN = EXPERIMENT_CONFIG.n_train
N_TEST = EXPERIMENT_CONFIG.n_test
K_NEIGHBORS = EXPERIMENT_CONFIG.k_neighbors
N_COMPONENTS = EXPERIMENT_CONFIG.n_components
ALPHA = EXPERIMENT_CONFIG.alpha
PSD_BETA = EXPERIMENT_CONFIG.psd_beta
SMOOTH_BETA = EXPERIMENT_CONFIG.smooth_beta
N_RUNS = EXPERIMENT_CONFIG.n_runs
SEED = EXPERIMENT_CONFIG.seed

# Eksperyment 1: zmieniamy tylko liczbę wierzchołków.
NODE_VALUES = [50, 100, 200, 400]
FIXED_VISIBILITY = 0.5

# Eksperyment 2: zmieniamy tylko prawdopodobieństwo obserwacji.
FIXED_NODES = 100
VISIBILITY_VALUES = [0.1, 0.2, 0.5, 0.8]


# ============================================================
# 2. METRYKI I PORÓWNANIE METOD
# ============================================================

def compare_methods(
    graph,
    p_observed,
    seed,
    *,
    split: MixtureSplit | None = None,
    train_uniforms=None,
    test_uniforms=None,
):
    """Uczy modele na treningu i raportuje błąd wyłącznie na teście.

    Opcjonalne ``split`` i macierze uniformów pozwalają wszystkim wartościom
    widoczności używać tych samych sygnałów oraz zagnieżdżonych masek.
    """
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    if split is None:
        split = generate_mixture_split(
            SignalGenerator(graph),
            N_TRAIN,
            N_TEST,
            REFERENCE_PSD_PROFILES,
            REFERENCE_PSD_PROBABILITIES,
        )
    if train_uniforms is None:
        train_uniforms = draw_nested_mask_uniforms(split.train.shape, rng)
    if test_uniforms is None:
        test_uniforms = draw_nested_mask_uniforms(split.test.shape, rng)

    train_observed = apply_observation_mask(
        split.train,
        p_observed,
        train_uniforms,
    )
    test_observed = apply_observation_mask(
        split.test,
        p_observed,
        test_uniforms,
    )
    estimates, predicted_labels, clustered_model = (
        evaluate_reconstruction_methods(
            graph,
            train_observed,
            test_observed,
            N_COMPONENTS,
            alpha=ALPHA,
            beta=PSD_BETA,
            smooth_beta=SMOOTH_BETA,
            random_state=seed,
        )
    )
    return reconstruction_metric_rows(
        split.test,
        test_observed,
        split.test_labels,
        estimates,
        predicted_labels,
        clustered_model,
    )


# ============================================================
# 3. EKSPERYMENT: LICZBA WIERZCHOŁKÓW
# ============================================================

def experiment_number_of_nodes():
    rows = []

    for n_nodes in NODE_VALUES:
        for run in tracked_range(N_RUNS, f"graph size: N={n_nodes}"):
            graph_seed = SEED + 1000 * n_nodes + run
            data_seed = SEED + run
            np.random.seed(graph_seed)

            # k pozostaje stałe, więc średni stopień i E/N są zbliżone.
            graph = GraphFactory.generate_nn_graph(
                N=n_nodes,
                k=K_NEIGHBORS,
            )

            method_results = compare_methods(
                graph=graph,
                p_observed=FIXED_VISIBILITY,
                seed=data_seed,
            )

            for result in method_results:
                rows.append({
                    "experiment": "number_of_nodes",
                    "n_nodes": n_nodes,
                    "p_observed": FIXED_VISIBILITY,
                    "n_train": N_TRAIN,
                    "n_test": N_TEST,
                    "run": run,
                    "graph_seed": graph_seed,
                    "data_seed": data_seed,
                    **result,
                })

    return pd.DataFrame(rows)


# ============================================================
# 4. EKSPERYMENT: PRAWDOPODOBIEŃSTWO OBSERWACJI
# ============================================================

def experiment_visibility():
    rows = []

    for run in tracked_range(N_RUNS, "observation visibility"):
        graph_seed = SEED + run
        np.random.seed(graph_seed)

        # Ten sam graf w obrębie jednego runu dla wszystkich wartości p.
        graph = GraphFactory.generate_nn_graph(
            N=FIXED_NODES,
            k=K_NEIGHBORS,
        )
        data_seed = SEED + 10_000 * run
        np.random.seed(data_seed)
        generator = SignalGenerator(graph)
        split = generate_mixture_split(
            generator,
            N_TRAIN,
            N_TEST,
            REFERENCE_PSD_PROFILES,
            REFERENCE_PSD_PROBABILITIES,
        )
        rng = np.random.default_rng(data_seed)
        train_uniforms = draw_nested_mask_uniforms(split.train.shape, rng)
        test_uniforms = draw_nested_mask_uniforms(split.test.shape, rng)

        for p_observed in VISIBILITY_VALUES:
            method_results = compare_methods(
                graph=graph,
                p_observed=p_observed,
                seed=data_seed,
                split=split,
                train_uniforms=train_uniforms,
                test_uniforms=test_uniforms,
            )

            for result in method_results:
                rows.append({
                    "experiment": "visibility",
                    "n_nodes": FIXED_NODES,
                    "p_observed": p_observed,
                    "n_train": N_TRAIN,
                    "n_test": N_TEST,
                    "run": run,
                    "graph_seed": graph_seed,
                    "data_seed": data_seed,
                    **result,
                })

    return pd.DataFrame(rows)


# ============================================================
# 5. PODSUMOWANIE
# ============================================================

def summarize(results, parameter):
    return (
        results
        .groupby([parameter, "method"])
        .agg(
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            ari_mean=("ari", "mean"),
            ari_std=("ari", "std"),
            min_cluster_size_mean=("min_train_cluster_size", "mean"),
            fallback_clusters_mean=("fallback_cluster_count", "mean"),
            n_runs=("mae", "count"),
        )
        .reset_index()
    )


# ============================================================
# 6. START
# ============================================================

def save_results(results, suite, varied_columns, **config_overrides):
    """Save one graph-size/visibility study with its constant metadata."""
    save_aggregated_results(
        results, suite, varied_columns,
        config_dict(
            EXPERIMENT_CONFIG,
            exclude=varied_columns,
            **config_overrides,
        ),
    )


def main():
    """Run and save both graph-size and observation-visibility studies."""
    node_results = experiment_number_of_nodes()
    visibility_results = experiment_visibility()
    save_results(
        pd.concat([node_results, visibility_results], ignore_index=True),
        "graph_size_visibility", ["experiment", "n_nodes", "p_observed"],
    )


if __name__ == "__main__":
    main()
