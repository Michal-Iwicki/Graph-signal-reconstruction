"""Testowanie jednego parametru naraz dla stałej mieszaniny 5 PSD."""

import numpy as np
import pandas as pd
from scipy.special import ndtr
from sklearn.metrics import adjusted_rand_score

from src.methods.generation import GraphFactory, SignalGenerator
from src.experiments._evaluation import (
    MixtureSplit,
    apply_observation_mask,
    draw_nested_mask_uniforms,
    evaluate_reconstruction_methods,
    generate_mixture_split,
    missing_mae,
)


# ============================================================
# 1. PARAMETRY WSPÓLNE
# ============================================================

N_TRAIN = 600
N_TEST = 200
K_NEIGHBORS = 10          # stałe k => podobny stosunek liczby krawędzi do wierzchołków
N_COMPONENTS = 5

ALPHA = 10.0
PSD_BETA = 1.0
SMOOTH_BETA = 0.1

N_RUNS = 10
SEED = 42

# Eksperyment 1: zmieniamy tylko liczbę wierzchołków.
NODE_VALUES = [50, 100, 200, 400]
FIXED_VISIBILITY = 0.5

# Eksperyment 2: zmieniamy tylko prawdopodobieństwo obserwacji.
FIXED_NODES = 100
VISIBILITY_VALUES = [0.1, 0.2, 0.5, 0.8]


# ============================================================
# 2. STAŁE PROFILE PSD
# ============================================================

def normalize(values):
    values = np.maximum(np.asarray(values, dtype=float), 0.0)
    return values / values.max() if values.max() > 0 else values


def gaussian_psd(mean, variance):
    def profile(eigenvalues, lambda_max):
        x = eigenvalues / lambda_max
        return normalize(np.exp(-0.5 * (x - mean) ** 2 / variance))
    return profile


def flat_band_psd(low, high):
    def profile(eigenvalues, lambda_max):
        x = eigenvalues / lambda_max
        return ((x >= low) & (x <= high)).astype(float)
    return profile


def skewed_psd(location, scale, skew):
    def profile(eigenvalues, lambda_max):
        x = eigenvalues / lambda_max
        z = (x - location) / scale
        values = 2 * np.exp(-0.5 * z**2) * ndtr(skew * z)
        return normalize(values)
    return profile


# Ta sama mieszanina jest używana we wszystkich eksperymentach.
PSD_FUNCTIONS = [
    gaussian_psd(mean=0.10, variance=0.006),       # low
    gaussian_psd(mean=0.30, variance=0.010),       # low-mid
    flat_band_psd(low=0.40, high=0.60),            # mid band
    skewed_psd(location=0.65, scale=0.10, skew=5), # skewed high
    gaussian_psd(mean=0.85, variance=0.008),       # high
]

PSD_PROBABILITIES = [0.25, 0.20, 0.20, 0.20, 0.15]


# ============================================================
# 3. METRYKI I PORÓWNANIE METOD
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
            PSD_FUNCTIONS,
            PSD_PROBABILITIES,
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
    return [
        {
            "method": method,
            "mae": missing_mae(split.test, estimate, test_observed),
            "min_train_cluster_size": (
                min(clustered_model.train_cluster_sizes.values())
                if method == "proposed"
                else np.nan
            ),
            "fallback_cluster_count": (
                len(clustered_model.fallback_clusters)
                if method == "proposed"
                else np.nan
            ),
            "ari": (
                adjusted_rand_score(split.test_labels, predicted_labels)
                if method == "proposed"
                else np.nan
            ),
        }
        for method, estimate in estimates.items()
    ]


# ============================================================
# 4. EKSPERYMENT: LICZBA WIERZCHOŁKÓW
# ============================================================

def experiment_number_of_nodes():
    rows = []

    for n_nodes in NODE_VALUES:
        for run in range(N_RUNS):
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
# 5. EKSPERYMENT: PRAWDOPODOBIEŃSTWO OBSERWACJI
# ============================================================

def experiment_visibility():
    rows = []

    for run in range(N_RUNS):
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
            PSD_FUNCTIONS,
            PSD_PROBABILITIES,
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
# 6. PODSUMOWANIE
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
# 7. START
# ============================================================

if __name__ == "__main__":
    node_results = experiment_number_of_nodes()
    visibility_results = experiment_visibility()

    node_summary = summarize(node_results, "n_nodes")
    visibility_summary = summarize(visibility_results, "p_observed")

    print("\nWpływ liczby wierzchołków:")
    print(node_summary)

    print("\nWpływ prawdopodobieństwa obserwacji:")
    print(visibility_summary)

    node_results.to_csv("nodes_results.csv", index=False)
    node_summary.to_csv("nodes_summary.csv", index=False)

    visibility_results.to_csv("visibility_results.csv", index=False)
    visibility_summary.to_csv("visibility_summary.csv", index=False)
