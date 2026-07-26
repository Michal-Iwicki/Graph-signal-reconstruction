import numpy as np
import pandas as pd
from scipy.special import ndtr
from sklearn.metrics import adjusted_rand_score

from src.methods.generation import GraphFactory, SignalGenerator
from src.methods.reconstruction import SignalReconstructor
from src.methods.models import MixedSignalReconstruction


# ============================================================
# 1. PARAMETRY STAŁE
# ============================================================

N_NODES = 100
K_NEIGHBORS = 10

N_SIGNALS = 600
N_COMPONENTS = 3
P_OBSERVED = 0.5

N_RUNS = 20
SEED = 42

ALPHA = 10.0
PSD_BETA = 1.0
SMOOTH_BETA = 0.1

PSD_TYPES = ["flat", "gaussian", "skewed", "low_band", "high_band"]


# ============================================================
# 2. FUNKCJA BUDUJĄCA PSD
# ============================================================

def make_psd(psd_type, parameters):
    def psd(eigenvalues, lambda_max):
        x = eigenvalues / lambda_max

        if psd_type == "flat":
            values = (
                (x >= parameters["low"])
                & (x <= parameters["high"])
            ).astype(float)

        elif psd_type == "gaussian":
            mean = parameters["mean"]
            variance = parameters["variance"]
            values = np.exp(-0.5 * (x - mean) ** 2 / variance)

        elif psd_type == "skewed":
            z = (x - parameters["location"]) / parameters["scale"]
            values = 2 * np.exp(-0.5 * z**2) * ndtr(parameters["skew"] * z)

        elif psd_type == "low_band":
            values = (x <= parameters["cutoff"]).astype(float)

        elif psd_type == "high_band":
            values = (x >= parameters["cutoff"]).astype(float)

        else:
            raise ValueError(f"Nieznany typ PSD: {psd_type}")

        maximum = values.max()
        return values / maximum if maximum > 0 else values

    return psd


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

    elif psd_type == "skewed":
        parameters = {
            "location": rng.uniform(0.1, 0.9),
            "scale": rng.uniform(0.05, 0.18),
            "skew": rng.uniform(-10.0, 10.0),
        }

    elif psd_type == "low_band":
        parameters = {"cutoff": rng.uniform(0.12, 0.4)}

    elif psd_type == "high_band":
        parameters = {"cutoff": rng.uniform(0.6, 0.88)}

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

def missing_mae(truth, estimate, observed):
    missing = np.isnan(observed)
    return np.mean(np.abs(truth[missing] - estimate[missing]))


# ============================================================
# 5. JEDEN EKSPERYMENT DLA WYLOSOWANEJ MIESZANINY
# ============================================================

def run_one(graph, run_id, seed):
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    generator = SignalGenerator(graph)
    reconstructor = SignalReconstructor(graph)

    psd_functions, profile_info = draw_mixture(graph, rng)

    truth, observed, true_labels = generator.generate_mixed_signals(
        M=N_SIGNALS,
        p=P_OBSERVED,
        psd_s=psd_functions,
        probs=np.full(N_COMPONENTS, 1 / N_COMPONENTS),
    )

    # Smoothing
    smoothing = reconstructor.reconstruct_smooth(
        observed,
        beta=SMOOTH_BETA,
    )

    # Basic PSD: jedna PSD dla całej mieszaniny
    gamma = reconstructor.estimate_gamma(observed)
    basic_psd = reconstructor.reconstruct_psd(
        observed,
        gamma=gamma,
        alpha=ALPHA,
        beta=PSD_BETA,
    )

    # Proposed: clustering i osobna PSD dla każdego klastra
    proposed_model = MixedSignalReconstruction(graph)
    proposed = proposed_model.fit_transform(
        observed,
        method="clustered_reconstruction",
        K_list=[N_COMPONENTS],
        alpha=ALPHA,
        beta=PSD_BETA,
        init_beta=SMOOTH_BETA,
        random_state=seed,
    )

    results = pd.DataFrame([
        {
            "run": run_id,
            "method": "proposed",
            "mae": missing_mae(truth, proposed, observed),
            "ari": adjusted_rand_score(
                true_labels,
                proposed_model.last_labels,
            ),
        },
        {
            "run": run_id,
            "method": "basic_psd",
            "mae": missing_mae(truth, basic_psd, observed),
            "ari": np.nan,
        },
        {
            "run": run_id,
            "method": "smoothing",
            "mae": missing_mae(truth, smoothing, observed),
            "ari": np.nan,
        },
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
        results.groupby("method")["mae"]
        .agg(["mean", "std", "median"])
        .sort_values("mean")
    )

    print(summary)

    results.to_csv("many_psd_results.csv", index=False)
    profiles.to_csv("many_psd_profiles.csv", index=False)
