"""Eksperyment OFAT: zmiana topologii grafu przy stałej liczbie wierzchołków i krawędzi."""

import networkx as nx
import numpy as np
import pandas as pd
from scipy.special import ndtr
from sklearn.metrics import adjusted_rand_score

from src.methods.generation import GSPGraph, GraphFactory, SignalGenerator
from src.experiments._evaluation import (
    apply_observation_mask,
    draw_nested_mask_uniforms,
    evaluate_reconstruction_methods,
    generate_mixture_split,
    missing_mae,
)
from src.experiments.synthetic import PLANNED_SYNTHETIC_VALUES


# ============================================================
# 1. PARAMETRY STAŁE
# ============================================================

N_NODES = 100
TARGET_EDGES = 180

N_TRAIN = 600
N_TEST = 200
P_OBSERVED = 0.5
N_COMPONENTS = 5

N_RUNS = 10
SEED = 42

ALPHA = 10.0
PSD_BETA = 1.0
SMOOTH_BETA = 0.1

TOPOLOGIES = list(PLANNED_SYNTHETIC_VALUES["topology"])


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
    """Minimalnie koryguje liczbę krawędzi, zachowując spójność grafu.

    Bazowy generator jest zawsze generatorem właściwej rodziny topologicznej.
    Korekta służy wyłącznie ujednoliceniu liczby krawędzi między rodzinami.
    """
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


def create_exact_knn(seed):
    """Tworzy geometryczny k-NN i dopasowuje E bez losowych dalekich połączeń."""
    rng = np.random.default_rng(seed)
    coordinates = rng.random((N_NODES, 2))
    differences = coordinates[:, None, :] - coordinates[None, :, :]
    distances = np.sum(differences**2, axis=2)
    np.fill_diagonal(distances, np.inf)

    graph = nx.Graph()
    graph.add_nodes_from(range(N_NODES))
    nx.set_node_attributes(
        graph,
        {node: tuple(coordinates[node]) for node in graph.nodes},
        "pos",
    )
    for node in graph.nodes:
        neighbors = np.argsort(distances[node])[:3]
        graph.add_edges_from((node, int(neighbor)) for neighbor in neighbors)

    # Łączymy komponenty najkrótszymi możliwymi krawędziami geometrycznymi.
    while not nx.is_connected(graph):
        components = [
            np.asarray(list(group), dtype=int)
            for group in nx.connected_components(graph)
        ]
        best = None
        for first_index, first in enumerate(components):
            for second in components[first_index + 1:]:
                block = distances[np.ix_(first, second)]
                local = np.unravel_index(np.argmin(block), block.shape)
                candidate = (
                    float(block[local]),
                    int(first[local[0]]),
                    int(second[local[1]]),
                )
                if best is None or candidate[0] < best[0]:
                    best = candidate
        _, first_node, second_node = best
        graph.add_edge(first_node, second_node)

    # Usuwamy najdłuższe niekrytyczne połączenia lub dodajemy najkrótsze
    # brakujące połączenia, aby zachować geometryczny charakter grafu.
    while graph.number_of_edges() > TARGET_EDGES:
        removable = sorted(
            graph.edges(),
            key=lambda edge: distances[edge[0], edge[1]],
            reverse=True,
        )
        for first_node, second_node in removable:
            graph.remove_edge(first_node, second_node)
            if nx.is_connected(graph):
                break
            graph.add_edge(first_node, second_node)
        else:
            raise RuntimeError("Nie można dopasować liczby krawędzi k-NN.")

    while graph.number_of_edges() < TARGET_EDGES:
        first_node, second_node = min(
            nx.non_edges(graph),
            key=lambda edge: distances[edge[0], edge[1]],
        )
        graph.add_edge(first_node, second_node)
    return graph


# ============================================================
# 4. GENEROWANIE TOPOLOGII
# ============================================================

def create_graph(topology, seed):
    """Generuje jedną z topologii planowanych w ``synthetic.py``.

    Parametry każdej rodziny są dobierane możliwie blisko ``TARGET_EDGES``.
    Erdős–Rényi jest losowany bezpośrednio jako G(n, m), więc nie wymaga
    późniejszej korekty liczby krawędzi.
    """
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    if topology == "knn":
        base = create_exact_knn(seed)

    elif topology == "grid":
        side = int(np.sqrt(N_NODES))
        if side * side != N_NODES:
            raise ValueError("Dla grid N_NODES musi być kwadratem liczby całkowitej.")

        base = GraphFactory.generate_grid_graph(
            rows=side,
            columns=side,
        )

    elif topology == "erdos_renyi":
        base = None
        for _ in range(100):
            graph_seed = int(rng.integers(0, np.iinfo(np.int32).max))
            candidate = nx.gnm_random_graph(
                N_NODES,
                TARGET_EDGES,
                seed=graph_seed,
            )
            if nx.is_connected(candidate):
                base = candidate
                break
        if base is None:
            raise RuntimeError("Nie udało się wygenerować spójnego G(n, m).")

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

    adjusted = (
        nx.Graph(base)
        if base.number_of_edges() == TARGET_EDGES
        else match_edge_count(
            graph=base,
            target_edges=TARGET_EDGES,
            rng=rng,
        )
    )
    if (
        adjusted.number_of_nodes() != N_NODES
        or adjusted.number_of_edges() != TARGET_EDGES
        or not nx.is_connected(adjusted)
    ):
        raise RuntimeError(
            "Generator topologii naruszył kontrakt wspólnych N/E lub spójność."
        )

    return GSPGraph(adjusted)


# ============================================================
# 5. METRYKI I PORÓWNANIE METOD
# ============================================================

def compare_methods(graph, seed):
    """Uczy wszystkie modele na treningu i ocenia je na osobnym teście."""
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    generator = SignalGenerator(graph)
    split = generate_mixture_split(
        generator,
        N_TRAIN,
        N_TEST,
        PSD_FUNCTIONS,
        PSD_PROBABILITIES,
    )
    train_observed = apply_observation_mask(
        split.train,
        P_OBSERVED,
        draw_nested_mask_uniforms(split.train.shape, rng),
    )
    test_observed = apply_observation_mask(
        split.test,
        P_OBSERVED,
        draw_nested_mask_uniforms(split.test.shape, rng),
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
    rows = []
    for method, estimate in estimates.items():
        rows.append({
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
        })
    return rows


# ============================================================
# 6. EKSPERYMENT TOPOLOGII
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
