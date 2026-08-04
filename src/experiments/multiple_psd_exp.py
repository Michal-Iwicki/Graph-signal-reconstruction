import numpy as np
import pandas as pd

from src.methods.generation import GraphFactory, SignalGenerator
from src.experiments._helper import (
    edge_skewed_psd,
    config_dict,
    evaluate_reconstruction_methods,
    experiment_config,
    flat_band_psd,
    gaussian_psd,
    generate_observed_mixture_split,
    reconstruction_metric_rows,
    save_aggregated_results,
    tracked_range,
)


# ============================================================
# 1. PARAMETRY STAŁE
# ============================================================

EXPERIMENT_CONFIG = experiment_config(n_runs=20)
N_NODES = EXPERIMENT_CONFIG.n_nodes
K_NEIGHBORS = EXPERIMENT_CONFIG.k_neighbors
N_TRAIN = EXPERIMENT_CONFIG.n_train
N_TEST = EXPERIMENT_CONFIG.n_test
P_OBSERVED = EXPERIMENT_CONFIG.p_observed
N_RUNS = EXPERIMENT_CONFIG.n_runs
SEED = EXPERIMENT_CONFIG.seed
ALPHA = EXPERIMENT_CONFIG.alpha
PSD_BETA = EXPERIMENT_CONFIG.psd_beta
SMOOTH_BETA = EXPERIMENT_CONFIG.smooth_beta


PSD_TYPES = ["flat", "gaussian", "low_band", "high_band"]
COMPONENT_VALUES = (2, 5, 10, 15)


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


def draw_mixture(graph, rng, n_components):
    """Draw the requested number of random PSD source profiles."""
    selected_types = rng.choice(
        PSD_TYPES,
        size=n_components,
        replace=n_components > len(PSD_TYPES),
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

def run_one(graph, n_components, run_id, seed):
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    generator = SignalGenerator(graph)

    psd_functions, profile_info = draw_mixture(graph, rng, n_components)

    split, train_observed, test_observed = generate_observed_mixture_split(
        generator,
        N_TRAIN,
        N_TEST,
        psd_functions,
        np.full(n_components, 1 / n_components),
        P_OBSERVED,
        rng,
    )
    estimates, predicted_labels, clustered_model = (
        evaluate_reconstruction_methods(
            graph,
            train_observed,
            test_observed,
            n_components,
            alpha=ALPHA,
            beta=PSD_BETA,
            smooth_beta=SMOOTH_BETA,
            random_state=seed,
        )
    )

    results = pd.DataFrame([
        {
            "run": run_id,
            "n_components": n_components,
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
    profiles.insert(0, "n_components", n_components)

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

    for n_components in COMPONENT_VALUES:
        for run_id in tracked_range(
            N_RUNS,
            f"multiple PSD: K={n_components}",
        ):
            run_results, run_profiles = run_one(
                graph,
                n_components,
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

def save_results(results):
    """Save aggregate metrics and the random-PSD experiment parameters."""
    save_aggregated_results(
        results, "multiple_psd", ["n_components"],
        config_dict(
            EXPERIMENT_CONFIG,
            exclude=("n_components",),
            psd_types=PSD_TYPES,
        ),
    )


def main():
    """Run and save the random multiple-PSD experiment."""
    results, _ = run_many()
    save_results(results)


if __name__ == "__main__":
    main()
