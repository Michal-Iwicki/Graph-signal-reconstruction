"""Eksperyment OFAT dla topologii o zbliżonej naturalnej liczbie krawędzi."""

import networkx as nx
import numpy as np
import pandas as pd

from src.methods.generation import GraphFactory, SignalGenerator
from experiments._helper import (
    REFERENCE_PSD_PROBABILITIES,
    REFERENCE_PSD_PROFILES,
    evaluate_reconstruction_methods,
    generate_observed_mixture_split,
    reconstruction_metric_rows,
)


# ============================================================
# 1. PARAMETRY STAŁE
# ============================================================

N_NODES = 100
TARGET_EDGES = 180

# Parametry wynikają ze standardowych wzorów na liczbę krawędzi.
ERDOS_RENYI_PROBABILITY = 2 * TARGET_EDGES / (N_NODES * (N_NODES - 1))
BARABASI_ALBERT_M = round(
    (N_NODES - np.sqrt(N_NODES**2 - 4 * TARGET_EDGES)) / 2
)
WATTS_STROGATZ_K = 2 * round(TARGET_EDGES / N_NODES)
# Po symetryzacji k-NN ma od N*k/2 do N*k krawędzi. Z wynikającego
# przedziału dla k wybieramy największą liczbę całkowitą.
KNN_NEIGHBORS = int(2 * TARGET_EDGES // N_NODES)

N_TRAIN = 600
N_TEST = 200
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
# 2. NATURALNE GENEROWANIE TOPOLOGII
# ============================================================

def create_connected_knn(max_attempts=100):
    """Losuje naturalny k-NN ponownie, jeśli realizacja jest niespójna."""
    for _ in range(max_attempts):
        graph = GraphFactory.generate_nn_graph(
            N=N_NODES,
            k=KNN_NEIGHBORS,
        )
        if nx.is_connected(graph):
            return graph
    raise RuntimeError("Nie udało się wylosować spójnego grafu k-NN.")


# ============================================================
# 3. GENEROWANIE TOPOLOGII
# ============================================================

def create_graph(topology, seed):
    """Generuje topologię bez późniejszego dodawania lub usuwania krawędzi."""
    np.random.seed(seed)

    if topology == "knn":
        graph = create_connected_knn()

    elif topology == "grid":
        side = int(np.sqrt(N_NODES))
        if side * side != N_NODES:
            raise ValueError("Dla grid N_NODES musi być kwadratem liczby całkowitej.")

        graph = GraphFactory.generate_grid_graph(
            rows=side,
            columns=side,
        )

    elif topology == "erdos_renyi":
        graph = GraphFactory.generate_erdos_renyi_graph(
            N=N_NODES,
            edge_probability=ERDOS_RENYI_PROBABILITY,
            seed=seed,
        )

    elif topology == "barabasi_albert":
        graph = GraphFactory.generate_barabasi_albert_graph(
            N=N_NODES,
            m=BARABASI_ALBERT_M,
            seed=seed,
        )

    elif topology == "watts_strogatz":
        graph = GraphFactory.generate_watts_strogatz_graph(
            N=N_NODES,
            k=WATTS_STROGATZ_K,
            rewiring_probability=0.1,
            seed=seed,
        )

    else:
        raise ValueError(f"Nieznana topologia: {topology}")

    if (
        graph.number_of_nodes() != N_NODES
        or not nx.is_connected(graph)
    ):
        raise RuntimeError("Generator naruszył kontrakt wspólnego N lub spójność.")

    return graph


# ============================================================
# 4. METRYKI I PORÓWNANIE METOD
# ============================================================

def compare_methods(graph, seed):
    """Uczy wszystkie modele na treningu i ocenia je na osobnym teście."""
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
# 5. EKSPERYMENT TOPOLOGII
# ============================================================

def experiment_topology():
    rows = []

    for topology_id, topology in enumerate(TOPOLOGIES):
        for run in range(N_RUNS):
            graph_seed = SEED + 10_000 * topology_id + run
            data_seed = SEED + run
            graph = create_graph(topology, graph_seed)

            method_results = compare_methods(
                graph=graph,
                seed=data_seed,
            )

            for result in method_results:
                degree_values = np.asarray(
                    [degree for _, degree in graph.degree()],
                    dtype=float,
                )
                rows.append({
                    "topology": topology,
                    "n_nodes": graph.number_of_nodes(),
                    "n_edges": graph.number_of_edges(),
                    "mean_degree": float(np.mean(degree_values)),
                    "degree_std": float(np.std(degree_values)),
                    "clustering_coefficient": (
                        nx.average_clustering(graph)
                    ),
                    "average_shortest_path": (
                        nx.average_shortest_path_length(graph)
                    ),
                    "p_observed": P_OBSERVED,
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

def summarize(results):
    return (
        results
        .groupby(["topology", "method"])
        .agg(
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            ari_mean=("ari", "mean"),
            ari_std=("ari", "std"),
            min_cluster_size_mean=("min_train_cluster_size", "mean"),
            fallback_clusters_mean=("fallback_cluster_count", "mean"),
            degree_std_mean=("degree_std", "mean"),
            clustering_mean=("clustering_coefficient", "mean"),
            path_length_mean=("average_shortest_path", "mean"),
            n_runs=("mae", "count"),
        )
        .reset_index()
    )


# ============================================================
# 7. START
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
