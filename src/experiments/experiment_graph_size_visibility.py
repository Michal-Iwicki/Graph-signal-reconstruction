"""Testowanie jednego parametru naraz dla stałej mieszaniny 5 PSD."""

import numpy as np
import pandas as pd
from scipy.special import ndtr
from sklearn.metrics import adjusted_rand_score

from src.methods.generation import GraphFactory, SignalGenerator
from src.methods.reconstruction import SignalReconstructor
from src.methods.models import MixedSignalReconstruction


# ============================================================
# 1. PARAMETRY WSPÓLNE
# ============================================================

N_SIGNALS = 600
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

def missing_mae(truth, estimate, observed):
    missing = np.isnan(observed)
    return float(np.mean(np.abs(truth[missing] - estimate[missing])))


def compare_methods(graph, p_observed, seed):
    np.random.seed(seed)

    generator = SignalGenerator(graph)
    reconstructor = SignalReconstructor(graph)

    truth, observed, true_labels = generator.generate_mixed_signals(
        M=N_SIGNALS,
        p=p_observed,
        psd_s=PSD_FUNCTIONS,
        probs=PSD_PROBABILITIES,
    )

    smoothing = reconstructor.reconstruct_smooth(
        observed,
        beta=SMOOTH_BETA,
    )

    gamma_global = reconstructor.estimate_gamma(observed)
    basic_psd = reconstructor.reconstruct_psd(
        observed,
        gamma=gamma_global,
        alpha=ALPHA,
        beta=PSD_BETA,
    )

    proposed_model = MixedSignalReconstruction(graph)
    proposed = proposed_model.fit_transform(
        observed,
        method="clustered_reconstruction",
        K_list=[N_COMPONENTS],
        alpha=ALPHA,
        beta=PSD_BETA,
        init_beta=SMOOTH_BETA,
        min_cluster_size=3,
        random_state=seed,
    )

    return [
        {
            "method": "proposed",
            "mae": missing_mae(truth, proposed, observed),
            "ari": adjusted_rand_score(
                true_labels,
                proposed_model.last_labels,
            ),
        },
        {
            "method": "basic_psd",
            "mae": missing_mae(truth, basic_psd, observed),
            "ari": np.nan,
        },
        {
            "method": "smoothing",
            "mae": missing_mae(truth, smoothing, observed),
            "ari": np.nan,
        },
    ]


# ============================================================
# 4. EKSPERYMENT: LICZBA WIERZCHOŁKÓW
# ============================================================

def experiment_number_of_nodes():
    rows = []

    for n_nodes in NODE_VALUES:
        for run in range(N_RUNS):
            seed = SEED + 1000 * n_nodes + run
            np.random.seed(seed)

            # k pozostaje stałe, więc średni stopień i E/N są zbliżone.
            graph = GraphFactory.generate_nn_graph(
                N=n_nodes,
                k=K_NEIGHBORS,
            )

            method_results = compare_methods(
                graph=graph,
                p_observed=FIXED_VISIBILITY,
                seed=seed,
            )

            for result in method_results:
                rows.append({
                    "experiment": "number_of_nodes",
                    "n_nodes": n_nodes,
                    "p_observed": FIXED_VISIBILITY,
                    "run": run,
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

        for p_observed in VISIBILITY_VALUES:
            seed = SEED + 10_000 * run + int(100 * p_observed)

            method_results = compare_methods(
                graph=graph,
                p_observed=p_observed,
                seed=seed,
            )

            for result in method_results:
                rows.append({
                    "experiment": "visibility",
                    "n_nodes": FIXED_NODES,
                    "p_observed": p_observed,
                    "run": run,
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
