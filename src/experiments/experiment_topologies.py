"""Compare graph topologies with similar natural edge counts."""

import networkx as nx
import numpy as np
import pandas as pd

from src.methods.generation import GraphFactory, SignalGenerator
from src.experiments._helper import (
    REFERENCE_PSD_PROBABILITIES,
    REFERENCE_PSD_PROFILES,
    config_dict,
    evaluate_reconstruction_methods,
    experiment_config,
    generate_observed_mixture_split,
    reconstruction_metric_rows,
    save_aggregated_results,
    tracked_range,
)


# ============================================================
# 1. FIXED PARAMETERS
# ============================================================

EXPERIMENT_CONFIG = experiment_config()
N_NODES = EXPERIMENT_CONFIG.n_nodes
TARGET_EDGES = 180

# Derive parameters from the standard edge-count formulas.
ERDOS_RENYI_PROBABILITY = 2 * TARGET_EDGES / (N_NODES * (N_NODES - 1))
BARABASI_ALBERT_M = round(
    (N_NODES - np.sqrt(N_NODES**2 - 4 * TARGET_EDGES)) / 2
)
WATTS_STROGATZ_K = 2 * round(TARGET_EDGES / N_NODES)
WATTS_STROGATZ_REWIRING = 0.1
# A symmetrized k-NN graph has between N*k/2 and N*k edges. Choose the
# largest integer k from the resulting range.
KNN_NEIGHBORS = int(2 * TARGET_EDGES // N_NODES)

N_TRAIN = EXPERIMENT_CONFIG.n_train
N_TEST = EXPERIMENT_CONFIG.n_test
P_OBSERVED = EXPERIMENT_CONFIG.p_observed
N_COMPONENTS = EXPERIMENT_CONFIG.n_components
N_RUNS = EXPERIMENT_CONFIG.n_runs
SEED = EXPERIMENT_CONFIG.seed
ALPHA = EXPERIMENT_CONFIG.alpha
PSD_BETA = EXPERIMENT_CONFIG.psd_beta
SMOOTH_BETA = EXPERIMENT_CONFIG.smooth_beta


TOPOLOGIES = [
    "knn",
    "grid",
    "erdos_renyi",
    "barabasi_albert",
    "watts_strogatz",
]


# ============================================================
# 2. NATURAL TOPOLOGY GENERATION
# ============================================================

def create_connected_knn(max_attempts=100):
    """Resample a natural k-NN graph until it is connected."""
    for _ in range(max_attempts):
        graph = GraphFactory.generate_nn_graph(
            N=N_NODES,
            k=KNN_NEIGHBORS,
        )
        if nx.is_connected(graph):
            return graph
    raise RuntimeError("Could not generate a connected k-NN graph.")


# ============================================================
# 3. TOPOLOGY DISPATCH
# ============================================================

def create_graph(topology, seed):
    """Generate a topology without subsequently adding or removing edges."""
    np.random.seed(seed)

    if topology == "knn":
        graph = create_connected_knn()

    elif topology == "grid":
        side = int(np.sqrt(N_NODES))
        if side * side != N_NODES:
            raise ValueError("N_NODES must be a perfect square for a grid graph.")

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
            rewiring_probability=WATTS_STROGATZ_REWIRING,
            seed=seed,
        )

    else:
        raise ValueError(f"Unknown topology: {topology}")

    if (
        graph.number_of_nodes() != N_NODES
        or not nx.is_connected(graph)
    ):
        raise RuntimeError(
            "Generated graph violates the shared size/connectivity contract."
        )

    return graph


# ============================================================
# 4. METHOD COMPARISON
# ============================================================

def compare_methods(graph, seed):
    """Fit every model on training data and evaluate on a separate test set."""
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
# 5. TOPOLOGY EXPERIMENT
# ============================================================

def experiment_topology():
    """Evaluate reconstruction across all configured graph topologies."""
    rows = []

    for topology_id, topology in enumerate(TOPOLOGIES):
        for run in tracked_range(N_RUNS, f"topology: {topology}"):
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
# 6. ENTRYPOINT
# ============================================================

def save_results(results):
    """Save topology summaries and topology-independent parameters."""
    save_aggregated_results(
        results, "topologies", ["topology"],
        config_dict(
            EXPERIMENT_CONFIG,
            target_edges=TARGET_EDGES,
            erdos_renyi_probability=ERDOS_RENYI_PROBABILITY,
            barabasi_albert_m=BARABASI_ALBERT_M,
            watts_strogatz_neighbors=WATTS_STROGATZ_K,
            watts_strogatz_rewiring=WATTS_STROGATZ_REWIRING,
            knn_neighbors=KNN_NEIGHBORS,
        ),
    )


def main():
    """Run and save the graph-topology experiment."""
    save_results(experiment_topology())


if __name__ == "__main__":
    main()
