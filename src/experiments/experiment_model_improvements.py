"""Porównanie ulepszeń z models.py na stałym grafie i stałej mieszaninie PSD."""

import numpy as np
import pandas as pd
from scipy.special import ndtr
from sklearn.metrics import adjusted_rand_score

from src.methods.generation import GraphFactory, SignalGenerator
from src.methods.models import MixedSignalReconstruction


# ============================================================
# 1. PARAMETRY STAŁE
# ============================================================

N_NODES = 100
K_NEIGHBORS = 10
N_SIGNALS = 600
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
# 2. STAŁA MIESZANINA PSD
# ============================================================

def normalize(values):
    values = np.maximum(np.asarray(values, dtype=float), 0.0)
    maximum = values.max()
    return values / maximum if maximum > 0 else values


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
        values = 2.0 * np.exp(-0.5 * z**2) * ndtr(skew * z)
        return normalize(values)
    return profile


PSD_FUNCTIONS = [
    gaussian_psd(mean=0.10, variance=0.006),
    gaussian_psd(mean=0.30, variance=0.010),
    flat_band_psd(low=0.40, high=0.60),
    skewed_psd(location=0.65, scale=0.10, skew=5.0),
    gaussian_psd(mean=0.85, variance=0.008),
]

PSD_PROBABILITIES = [0.25, 0.20, 0.20, 0.20, 0.15]


# ============================================================
# 3. METRYKI
# ============================================================

def missing_mae(truth, estimate, observed):
    missing = np.isnan(observed)
    return float(np.mean(np.abs(truth[missing] - estimate[missing])))


def missing_rmse(truth, estimate, observed):
    missing = np.isnan(observed)
    error = truth[missing] - estimate[missing]
    return float(np.sqrt(np.mean(error**2)))


# ============================================================
# 4. TESTOWANE METODY
# ============================================================

METHODS = {
    # Klasyczna rekonstrukcja oparta wyłącznie na gładkości grafowej.
    "smoothness": {
        "method": "smooth",
        "kwargs": {
            "beta": INIT_BETA,
        },
    },

    # Metoda bazowa: smoothing -> clustering -> PSD reconstruction.
    "clustered": {
        "method": "clustered_reconstruction",
        "kwargs": {
            "K_list": [N_COMPONENTS],
            "alpha": ALPHA,
            "beta": PSD_BETA,
            "init_beta": INIT_BETA,
            "min_cluster_size": 3,
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
            "min_cluster_size": 3,
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
# 5. JEDNO POWTÓRZENIE
# ============================================================

def run_one(graph, run, seed):
    np.random.seed(seed)

    generator = SignalGenerator(graph)
    truth, observed, true_labels = generator.generate_mixed_signals(
        M=N_SIGNALS,
        p=P_OBSERVED,
        psd_s=PSD_FUNCTIONS,
        probs=PSD_PROBABILITIES,
    )

    rows = []

    for method_name, specification in METHODS.items():
        model = MixedSignalReconstruction(graph)

        kwargs = {
            **specification["kwargs"],
            "random_state": seed,
        }

        estimate = model.fit_transform(
            observed,
            method=specification["method"],
            **kwargs,
        )

        ari = (
            adjusted_rand_score(true_labels, model.last_labels)
            if model.last_labels is not None
            else np.nan
        )

        rows.append({
            "run": run,
            "method": method_name,
            "mae": missing_mae(truth, estimate, observed),
            "rmse": missing_rmse(truth, estimate, observed),
            "ari": ari,
            "selected_k": model.best_K_,
            "cost": model.best_score_,
        })

    return rows


# ============================================================
# 6. WIELE POWTÓRZEŃ
# ============================================================

def experiment_model_improvements():
    rows = []

    for run in range(N_RUNS):
        seed = SEED + run

        # Nowy graf i nowe dane w każdym runie, ale identyczne dla trzech metod.
        np.random.seed(seed)
        graph = GraphFactory.generate_nn_graph(
            N=N_NODES,
            k=K_NEIGHBORS,
        )

        rows.extend(run_one(graph, run, seed))

    return pd.DataFrame(rows)


# ============================================================
# 7. PODSUMOWANIE
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
        )
        .sort_values("mae_mean")
        .reset_index()
    )


# ============================================================
# 8. START
# ============================================================

if __name__ == "__main__":
    results = experiment_model_improvements()
    summary = summarize(results)

    print("\nPorównanie metod:")
    print(summary)

    results.to_csv("model_improvements_results.csv", index=False)
    summary.to_csv("model_improvements_summary.csv", index=False)
