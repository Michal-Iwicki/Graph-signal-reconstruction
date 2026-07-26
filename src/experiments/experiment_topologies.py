"""Eksperyment OFAT: zmiana topologii grafu przy stałej liczbie wierzchołków i krawędzi."""

import networkx as nx
import numpy as np
import pandas as pd
from scipy.special import ndtr
from sklearn.metrics import adjusted_rand_score

from src.methods.generation import GSPGraph, GraphFactory, SignalGenerator
from src.methods.reconstruction import SignalReconstructor
from src.methods.models import MixedSignalReconstruction


# ============================================================
# 1. PARAMETRY STAŁE
# ============================================================

N_NODES = 100
TARGET_EDGES = 180

N_SIGNALS = 600
P_OBSERVED = 0.5
N_COMPONENTS = 5

N_RUNS = 10
SEED = 42

ALPHA = 10.0
PSD_BETA = 1.0
SMOOTH_BETA = 0.1

TOPOLOGIES = [
    "knn",
    "grid",
    "erdos_renyi",
    "barabasi_albert",
    "watts_strogatz",
]


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
        values = np.exp(-0.5 * (x - mean) ** 2 / variance)
        return normalize(values)

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
# 3. UJEDNOLICENIE LICZBY KRAWĘDZI
# ============================================================

def match_edge_count(graph, target_edges, rng):
    """Dodaje lub usuwa krawędzie, zachowując spójność grafu."""
    graph = nx.Graph(graph)

    if target_edges < graph.number_of_nodes() - 1:
        raise ValueError("Spójny graf wymaga co najmniej N-1 krawędzi.")

    # Usuwanie losowych krawędzi, ale bez rozspajania grafu.
    while graph.number_of_edges() > target_edges:
        edges = list(graph.edges())
        rng.shuffle(edges)

        removed = False
        for u, v in edges:
            graph.remove_edge(u, v)

            if nx.is_connected(graph):
                removed = True
                break

            graph.add_edge(u, v)

        if not removed:
            raise RuntimeError("Nie można usunąć kolejnej krawędzi bez rozspojenia grafu.")

    # Dodawanie losowych brakujących krawędzi.
    while graph.number_of_edges() < target_edges:
        missing_edges = list(nx.non_edges(graph))
        u, v = missing_edges[rng.integers(len(missing_edges))]
        graph.add_edge(u, v)

    return graph


# ============================================================
# 4. GENEROWANIE TOPOLOGII
# ============================================================

def create_graph(topology, seed):
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    if topology == "knn":
        base = GraphFactory.generate_nn_graph(
            N=N_NODES,
            k=4,
        )

    elif topology == "grid":
        side = int(np.sqrt(N_NODES))
        if side * side != N_NODES:
            raise ValueError("Dla grid N_NODES musi być kwadratem liczby całkowitej.")

        base = GraphFactory.generate_grid_graph(
            rows=side,
            columns=side,
        )

    elif topology == "erdos_renyi":
        probability = 2 * TARGET_EDGES / (N_NODES * (N_NODES - 1))
        base = GraphFactory.generate_erdos_renyi_graph(
            N=N_NODES,
            edge_probability=probability,
            seed=seed,
        )

    elif topology == "barabasi_albert":
        base = GraphFactory.generate_barabasi_albert_graph(
            N=N_NODES,
            m=2,
            seed=seed,
        )

    elif topology == "watts_strogatz":
        base = GraphFactory.generate_watts_strogatz_graph(
            N=N_NODES,
            k=4,
            rewiring_probability=0.1,
            seed=seed,
        )

    else:
        raise ValueError(f"Nieznana topologia: {topology}")

    adjusted = match_edge_count(
        graph=base,
        target_edges=TARGET_EDGES,
        rng=rng,
    )

    return GSPGraph(adjusted)


# ============================================================
# 5. METRYKI I PORÓWNANIE METOD
# ============================================================

def missing_mae(truth, estimate, observed):
    missing = np.isnan(observed)
    return float(np.mean(np.abs(truth[missing] - estimate[missing])))


def compare_methods(graph, seed):
    np.random.seed(seed)

    generator = SignalGenerator(graph)
    reconstructor = SignalReconstructor(graph)

    truth, observed, true_labels = generator.generate_mixed_signals(
        M=N_SIGNALS,
        p=P_OBSERVED,
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
# 6. EKSPERYMENT TOPOLOGII
# ============================================================

def experiment_topology():
    rows = []

    for topology_id, topology in enumerate(TOPOLOGIES):
        for run in range(N_RUNS):
            seed = SEED + 10_000 * topology_id + run
            graph = create_graph(topology, seed)

            method_results = compare_methods(
                graph=graph,
                seed=seed,
            )

            for result in method_results:
                rows.append({
                    "topology": topology,
                    "n_nodes": graph.number_of_nodes(),
                    "n_edges": graph.number_of_edges(),
                    "p_observed": P_OBSERVED,
                    "run": run,
                    **result,
                })

    return pd.DataFrame(rows)


# ============================================================
# 7. PODSUMOWANIE
# ============================================================

def summarize(results):
    return (
        results
        .groupby(["topology", "method"])
        .agg(
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            ari_mean=("ari", "mean"),
            ari_std=("ari", "std"),
        )
        .reset_index()
    )


# ============================================================
# 8. START
# ============================================================

if __name__ == "__main__":
    results = experiment_topology()
    summary = summarize(results)

    print("\nKontrola liczby wierzchołków i krawędzi:")
    print(
        results[["topology", "n_nodes", "n_edges"]]
        .drop_duplicates()
        .sort_values("topology")
    )

    print("\nWyniki:")
    print(summary)

    results.to_csv("topology_results.csv", index=False)
    summary.to_csv("topology_summary.csv", index=False)
