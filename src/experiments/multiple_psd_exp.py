import numpy as np
import pandas as pd

from src.methods.generation import GraphFactory, SignalGenerator
from experiments._helper import (
    edge_skewed_psd,
    evaluate_reconstruction_methods,
    flat_band_psd,
    gaussian_psd,
    generate_observed_mixture_split,
    reconstruction_metric_rows,
)


# ============================================================
# 1. PARAMETRY STAŁE
# ============================================================

N_NODES = 100
K_NEIGHBORS = 10

N_TRAIN = 600
N_TEST = 200
N_COMPONENTS = 3
P_OBSERVED = 0.5

N_RUNS = 20
SEED = 42

ALPHA = 10.0
PSD_BETA = 1.0
SMOOTH_BETA = 0.1

PSD_TYPES = ["flat", "gaussian", "low_band", "high_band"]


# ============================================================
# 2. FUNKCJA BUDUJĄCA PSD
# ============================================================

def make_psd(psd_type, parameters):
    if psd_type == "flat":
        return flat_band_psd(parameters["low"], parameters["high"])
    if psd_type == "gaussian":
        return gaussian_psd(parameters["mean"], parameters["variance"])
    if psd_type == "low_band":
        return edge_skewed_psd(parameters["scale"], side="low")
    if psd_type == "high_band":
        return edge_skewed_psd(parameters["scale"], side="high")
    raise ValueError(f"Nieznany typ PSD: {psd_type}")


# ============================================================
# 3. LOSOWANIE PARAMETRÓW
# ============================================================

def draw_psd(psd_type, rng):
    if psd_type == "flat":
        low = rng.uniform(0.0, 0.7)
        high = rng.uniform(low + 0.1, min(low + 0.35, 1.0))
        parameters = {"low": low, "high": high}

    elif psd_type == "gaussian":
        parameters = {
            "mean": rng.uniform(0.05, 0.95),
            "variance": rng.uniform(0.0025, 0.025),
        }

    elif psd_type == "low_band":
        parameters = {"scale": rng.uniform(0.08, 0.22)}

    elif psd_type == "high_band":
        parameters = {"scale": rng.uniform(0.08, 0.22)}

    else:
        raise ValueError(f"Nieznany typ PSD: {psd_type}")

    return make_psd(psd_type, parameters), parameters


def draw_mixture(graph, rng):
    # Dla K <= liczby typów profile mają różne rodziny.
    selected_types = rng.choice(
        PSD_TYPES,
        size=N_COMPONENTS,
        replace=N_COMPONENTS > len(PSD_TYPES),
    )

    functions = []
    descriptions = []

    for component, psd_type in enumerate(selected_types):
        # Czasem płaski przedział może nie zawierać żadnej wartości własnej.
        for _ in range(100):
            function, parameters = draw_psd(psd_type, rng)
            gamma = function(graph.eigenvalues, graph.eigenvalues[-1])
            if np.any(gamma > 0):
                break

        functions.append(function)
        descriptions.append({
            "component": component,
            "psd_type": psd_type,
            **parameters,
        })

    return functions, descriptions


# ============================================================
# 4. METRYKA
# ============================================================

# ============================================================
# 5. JEDEN EKSPERYMENT DLA WYLOSOWANEJ MIESZANINY
# ============================================================

def run_one(graph, run_id, seed):
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    generator = SignalGenerator(graph)

    psd_functions, profile_info = draw_mixture(graph, rng)

    split, train_observed, test_observed = generate_observed_mixture_split(
        generator,
        N_TRAIN,
        N_TEST,
        psd_functions,
        np.full(N_COMPONENTS, 1 / N_COMPONENTS),
        P_OBSERVED,
        rng,
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

    results = pd.DataFrame([
        {
            "run": run_id,
            "n_train": N_TRAIN,
            "n_test": N_TEST,
            **row,
        }
        for row in reconstruction_metric_rows(
            split.test,
            test_observed,
            split.test_labels,
            estimates,
            predicted_labels,
            clustered_model,
        )
    ])

    profiles = pd.DataFrame(profile_info)
    profiles.insert(0, "run", run_id)

    return results, profiles


# ============================================================
# 6. WIELE LOSOWAŃ KSZTAŁTÓW I PARAMETRÓW
# ============================================================

def run_many():
    # Graf jest stały. W kolejnych runach zmieniają się PSD, sygnały i maski.
    np.random.seed(SEED)
    graph = GraphFactory.generate_nn_graph(N_NODES, K_NEIGHBORS)

    results = []
    profiles = []

    for run_id in range(N_RUNS):
        run_results, run_profiles = run_one(
            graph,
            run_id,
            SEED + run_id,
        )
        results.append(run_results)
        profiles.append(run_profiles)

    return (
        pd.concat(results, ignore_index=True),
        pd.concat(profiles, ignore_index=True),
    )


# ============================================================
# 7. START
# ============================================================

if __name__ == "__main__":
    results, profiles = run_many()

    summary = (
        results.groupby("method")
        .agg(
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            mae_median=("mae", "median"),
            ari_mean=("ari", "mean"),
            ari_std=("ari", "std"),
            fallback_clusters_mean=("fallback_cluster_count", "mean"),
        )
        .sort_values("mae_mean")
    )

    print(summary)

    results.to_csv("many_psd_results.csv", index=False)
    profiles.to_csv("many_psd_profiles.csv", index=False)
    summary.to_csv("many_psd_summary.csv")
