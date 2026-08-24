"""Compare model improvements on a fixed graph and PSD mixture."""

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from src.methods.generation import GraphFactory, SignalGenerator
from src.methods.models import MixedSignalReconstruction
from src.methods.clustering import ClusteringEvaluator
from src.methods.reconstruction import SignalReconstructor
from src.experiments._helper import (
    REFERENCE_PSD_PROBABILITIES,
    REFERENCE_PSD_PROFILES,
    config_dict,
    experiment_config,
    generate_observed_mixture_split,
    missing_mae,
    clustering_accuracy,
    save_aggregated_results,
    tracked_range,
)


# ============================================================
# 1. FIXED PARAMETERS
# ============================================================

EXPERIMENT_CONFIG = experiment_config(n_runs=20)
N_NODES = EXPERIMENT_CONFIG.n_nodes
K_NEIGHBORS = EXPERIMENT_CONFIG.k_neighbors
N_TRAIN = EXPERIMENT_CONFIG.n_train
N_TEST = EXPERIMENT_CONFIG.n_test
P_OBSERVED = EXPERIMENT_CONFIG.p_observed
N_COMPONENTS = EXPERIMENT_CONFIG.n_components
N_RUNS = EXPERIMENT_CONFIG.n_runs
SEED = EXPERIMENT_CONFIG.seed
ALPHA = EXPERIMENT_CONFIG.alpha
PSD_BETA = EXPERIMENT_CONFIG.psd_beta
INIT_BETA = EXPERIMENT_CONFIG.smooth_beta

MAX_ITER = 10
TOL = 1e-4


# ============================================================
# 2. METHODS UNDER TEST
# ============================================================

METHODS = {

    "proposed": {
        "method": "clustered_reconstruction",
        "kwargs": {
            "K_list": [N_COMPONENTS],
            "alpha": ALPHA,
            "beta": PSD_BETA,
            "init_beta": INIT_BETA,
            "min_cluster_size": 1,
        },
    },

    # Improvement 1: repeat clustering after reconstruction.
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

    # Improvement 2: incorporate reconstruction into the EM updates.
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
# 3. NEW-SIGNAL CLASSIFICATION
# ============================================================

def predict_from_training_partition(train_features, labels, test_features):
    """Classify test signals without using their ground-truth labels.

    A diagonal Gaussian is fitted in GFT space for each training cluster.
    Cluster identifiers remain aligned with the PSDs stored by the model.
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
# 4. SINGLE RUN
# ============================================================

def run_one(graph, run, seed):
    """Fit each variant on training data and evaluate on independent test data."""
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

        kwargs = {**specification["kwargs"], "random_state": seed}
        model.fit_transform(
            train_observed,
            method=specification["method"],
            **kwargs,
        )
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
        ari = adjusted_rand_score(split.test_labels, predicted_labels)
        accuracy = clustering_accuracy(split.test_labels, predicted_labels)

        rows.append({
            "run": run,
            "method": method_name,
            "n_train": N_TRAIN,
            "n_test": N_TEST,
            "mae": missing_mae(split.test, estimate, test_observed),
            "ari": ari,
            "clustering_accuracy": accuracy,
            "min_train_cluster_size": (
                int(
                    min(
                        np.sum(model.last_labels == cluster)
                        for cluster in np.unique(model.last_labels)
                    )
                )
                if model.last_labels is not None else np.nan
            ),
            "selected_k": model.best_K_,
            "train_cost": model.best_score_,
            "cost": model.best_score_,
        })

    return rows


# ============================================================
# 5. REPEATED RUNS
# ============================================================

def experiment_model_improvements():
    """Compare all configured model variants over repeated random runs."""
    rows = []

    for run in tracked_range(N_RUNS, "model improvements"):
        seed = SEED + run

        # Use a new graph and data in each run, shared by all methods.
        np.random.seed(seed)
        graph = GraphFactory.generate_nn_graph(
            N=N_NODES,
            k=K_NEIGHBORS,
        )

        rows.extend(run_one(graph, run, seed))

    return pd.DataFrame(rows)


# ============================================================
# 6. ENTRYPOINT
# ============================================================

def save_results(results):
    """Save model comparison summaries and method-specific settings."""
    save_aggregated_results(
        results, "model_improvements", [],
        config_dict(
            EXPERIMENT_CONFIG,
            init_beta=INIT_BETA,
            max_iter=MAX_ITER,
            tol=TOL,
            methods=METHODS,
        ),
    )


def main():
    """Run and save the model-improvements experiment."""
    save_results(experiment_model_improvements())


if __name__ == "__main__":
    main()
