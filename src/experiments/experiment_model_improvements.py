"""Porównanie ulepszeń z models.py na stałym grafie i stałej mieszaninie PSD."""

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from src.methods.generation import GraphFactory, SignalGenerator
from src.methods.models import MixedSignalReconstruction
from src.methods.clustering import ClusteringEvaluator
from src.methods.reconstruction import SignalReconstructor
from experiments._helper import (
    REFERENCE_PSD_PROBABILITIES,
    REFERENCE_PSD_PROFILES,
    generate_observed_mixture_split,
    missing_mae,
    missing_rmse,
)


# ============================================================
# 1. PARAMETRY STAŁE
# ============================================================

N_NODES = 100
K_NEIGHBORS = 10
N_TRAIN = 600
N_TEST = 200
P_OBSERVED = 0.5
N_COMPONENTS = 5

N_RUNS = 20
SEED = 42

ALPHA = 10.0
PSD_BETA = 1.0
INIT_BETA = 0.1

MAX_ITER = 10
TOL = 1e-4


# ============================================================
# 2. TESTOWANE METODY
# ============================================================

METHODS = {
    # # Klasyczna rekonstrukcja oparta wyłącznie na gładkości grafowej.
    # "smoothness": {
    #     "method": "smooth",
    #     "kwargs": {
    #         "beta": INIT_BETA,
    #     },
    # },

    # # Jedno PSD estymowane na całym zbiorze treningowym.
    # "global_psd": {
    #     "method": "global_psd",
    #     "kwargs": {
    #         "alpha": ALPHA,
    #         "beta": PSD_BETA,
    #     },
    # },

    # Metoda bazowa: smoothing -> clustering -> PSD reconstruction.
    "clustered": {
        "method": "clustered_reconstruction",
        "kwargs": {
            "K_list": [N_COMPONENTS],
            "alpha": ALPHA,
            "beta": PSD_BETA,
            "init_beta": INIT_BETA,
            "min_cluster_size": 1,
        },
    },

    # Ulepszenie 1: ponowne klastrowanie po rekonstrukcji.
    "iterative": {
        "method": "iterative_clustered_reconstruction",
        "kwargs": {
            "K_list": [N_COMPONENTS],
            "max_iter": MAX_ITER,
            "alpha": ALPHA,
            "beta": PSD_BETA,
            "init_beta": INIT_BETA,
            "min_cluster_size": 1,
        },
    },

    # Ulepszenie 2: rekonstrukcja wpleciona w aktualizacje EM.
    "simultaneous": {
        "method": "simultaneous_method",
        "kwargs": {
            "K": N_COMPONENTS,
            "max_iter": MAX_ITER,
            "tol": TOL,
            "alpha": ALPHA,
            "beta": PSD_BETA,
            "init_beta": INIT_BETA,
        },
    },
}


# ============================================================
# 3. KLASYFIKACJA NOWYCH SYGNAŁÓW
# ============================================================

def predict_from_training_partition(train_features, labels, test_features):
    """Klasyfikuje test bez używania jego prawdziwych etykiet.

    Dla każdego klastra wyznaczany jest diagonalny model Gaussa w przestrzeni
    GFT. Numery klas pozostają zgodne z PSD zapisanymi przez trenowany model.
    """
    assignments = np.asarray(labels)
    clusters = np.unique(assignments)
    scores = np.empty((test_features.shape[0], len(clusters)), dtype=float)

    for index, cluster in enumerate(clusters):
        selected = assignments == cluster
        cluster_features = train_features[selected]
        mean = np.mean(cluster_features, axis=0)
        variance = np.var(cluster_features, axis=0) + 1e-6
        log_prior = np.log(np.mean(selected))
        scores[:, index] = (
            log_prior
            - 0.5 * np.sum(np.log(2 * np.pi * variance))
            - 0.5 * np.sum(
                (test_features - mean) ** 2 / variance,
                axis=1,
            )
        )
    return clusters[np.argmax(scores, axis=1)]


# ============================================================
# 4. JEDNO POWTÓRZENIE
# ============================================================

def run_one(graph, run, seed):
    """Trenuje każdy wariant na treningu i ocenia na niezależnym teście."""
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    generator = SignalGenerator(graph)
    split, train_observed, test_observed = generate_observed_mixture_split(
        generator,
        N_TRAIN,
        N_TEST,
        REFERENCE_PSD_PROFILES,
        REFERENCE_PSD_PROBABILITIES,
        P_OBSERVED,
        rng,
    )

    reconstructor = SignalReconstructor(graph)
    smooth_train = reconstructor.reconstruct_smooth(
        train_observed,
        beta=INIT_BETA,
    )
    smooth_test = reconstructor.reconstruct_smooth(
        test_observed,
        beta=INIT_BETA,
    )
    train_features = ClusteringEvaluator.graph_fourier_features(
        graph,
        smooth_train,
    )
    test_features = ClusteringEvaluator.graph_fourier_features(
        graph,
        smooth_test,
    )
    rows = []

    for method_name, specification in METHODS.items():
        model = MixedSignalReconstruction(graph)

        kwargs = {
            **specification["kwargs"],
        }
        if method_name not in {"smoothness", "global_psd"}:
            kwargs["random_state"] = seed

        if method_name == "smoothness":
            estimate = smooth_test
            predicted_labels = None
        else:
            model.fit_transform(
                train_observed,
                method=specification["method"],
                **kwargs,
            )
            if method_name == "global_psd":
                estimate = model.transfer_reconstruction(
                    test_observed,
                    graph,
                    alpha=ALPHA,
                    beta=PSD_BETA,
                )
                predicted_labels = None
            else:
                predicted_labels = predict_from_training_partition(
                    train_features,
                    model.last_labels,
                    test_features,
                )
                estimate = model.transfer_reconstruction(
                    test_observed,
                    graph,
                    labels=predicted_labels,
                    alpha=ALPHA,
                    beta=PSD_BETA,
                )

        ari = (
            adjusted_rand_score(split.test_labels, predicted_labels)
            if predicted_labels is not None
            else np.nan
        )

        rows.append({
            "run": run,
            "method": method_name,
            "n_train": N_TRAIN,
            "n_test": N_TEST,
            "mae": missing_mae(split.test, estimate, test_observed),
            "rmse": missing_rmse(split.test, estimate, test_observed),
            "ari": ari,
            "min_train_cluster_size": (
                int(
                    min(
                        np.sum(model.last_labels == cluster)
                        for cluster in np.unique(model.last_labels)
                    )
                )
                if model.last_labels is not None
                and method_name != "global_psd"
                else np.nan
            ),
            "selected_k": model.best_K_,
            "train_cost": model.best_score_,
            "cost": model.best_score_,
        })

    return rows


# ============================================================
# 5. WIELE POWTÓRZEŃ
# ============================================================

def experiment_model_improvements():
    rows = []

    for run in range(N_RUNS):
        seed = SEED + run

        # Nowy graf i dane w każdym runie, wspólne dla wszystkich metod.
        np.random.seed(seed)
        graph = GraphFactory.generate_nn_graph(
            N=N_NODES,
            k=K_NEIGHBORS,
        )

        rows.extend(run_one(graph, run, seed))

    return pd.DataFrame(rows)


# ============================================================
# 6. PODSUMOWANIE
# ============================================================

def summarize(results):
    return (
        results
        .groupby("method")
        .agg(
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            ari_mean=("ari", "mean"),
            ari_std=("ari", "std"),
            n_runs=("mae", "count"),
        )
        .sort_values("mae_mean")
        .reset_index()
    )


# ============================================================
# 7. START
# ============================================================

if __name__ == "__main__":
    results = experiment_model_improvements()
    summary = summarize(results)

    print("\nPorównanie metod:")
    print(summary)

    results.to_csv("model_improvements_results.csv", index=False)
    summary.to_csv("model_improvements_summary.csv", index=False)
