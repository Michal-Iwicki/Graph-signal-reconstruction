"""Reproducible experiment workflows used by the project notebooks."""

from collections.abc import Callable, Sequence

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

from src.methods.clustering import ClusteringEvaluator, GMM_Diag
from src.methods.generation import GraphFactory, SignalGenerator
from src.methods.reconstruction import SignalReconstructor
from src.experiments._helper import (
    apply_observation_mask,
    draw_nested_mask_uniforms,
    estimate_group_psds,
    generate_mixture_split,
    reconstruct_from_group_psds,
    tracked_range,
)


def _default_low_pass_psd(eigenvalues: np.ndarray, lambda_max: float) -> np.ndarray:
    """Return the low-pass PSD profile used in baseline experiments."""
    values = np.zeros_like(eigenvalues, dtype=float)
    selected = eigenvalues < lambda_max / 4
    values[selected] = np.sin(4 * np.pi * eigenvalues[selected] / lambda_max)
    return np.maximum(values, 0.0)


def _mae_on_missing(
    truth: np.ndarray,
    estimate: np.ndarray,
    missing: np.ndarray,
) -> float:
    """Compute MAE on missing entries, or NaN when no entry is missing."""
    if not np.any(missing):
        return float("nan")
    return float(np.mean(np.abs(estimate[missing] - truth[missing])))


def run_psd_experiment(
    N: int = 500,
    k: int = 40,
    M: int = 1000,
    p_values: Sequence[float] = (1.0, 0.5, 0.05),
    psd_fn: Callable[[np.ndarray, float], np.ndarray] | None = None,
    seed: int = 42,
) -> dict:
    """Compare theoretical and estimated PSDs at several sampling rates.

    Parameters
    ----------
    N, k:
        Size and neighborhood order of the synthetic k-NN graph.
    M:
        Number of signal realizations used for each PSD estimate.
    p_values:
        Vertex-observation probabilities.
    psd_fn:
        Optional PSD profile. The default is a compact low-pass profile.

    Returns
    -------
    dict
        Graph, theoretical PSD, and a mapping from sampling probability to the
        corresponding estimated PSD.
    """
    np.random.seed(seed)
    random_generator = np.random.default_rng(seed)
    graph = GraphFactory.generate_nn_graph(N, k)
    generator = SignalGenerator(graph)
    reconstructor = SignalReconstructor(graph)
    profile = _default_low_pass_psd if psd_fn is None else psd_fn
    if len(p_values) == 0:
        raise ValueError("p_values must contain at least one probability.")

    eigenvalues = graph.eigenvalues
    lambda_max = float(eigenvalues[-1])
    gamma_true = reconstructor.normalize_gamma(
        profile(eigenvalues, lambda_max)
    )
    estimates = {}
    complete, _ = generator.generate_signals(M, 1.0, profile)
    mask_uniforms = draw_nested_mask_uniforms(
        complete.shape,
        random_generator,
    )

    figure, axes = plt.subplots(
        1,
        len(p_values),
        figsize=(5 * len(p_values), 4),
        squeeze=False,
    )
    for axis, probability in zip(axes[0], p_values):
        observed = apply_observation_mask(
            complete,
            probability,
            mask_uniforms,
        )
        gamma_estimated = reconstructor.estimate_gamma(observed)
        estimates[probability] = gamma_estimated

        axis.plot(eigenvalues, gamma_true, label="True PSD")
        axis.plot(
            eigenvalues,
            gamma_estimated,
            "--",
            label="Estimated PSD",
        )
        axis.set(
            title=f"Sampling p = {probability}",
            xlabel="Eigenvalue (λ)",
            ylabel="Normalized PSD",
        )
        axis.legend()

    figure.tight_layout()
    plt.show()
    return {
        "graph": graph,
        "gamma_true": gamma_true,
        "gamma_estimates": estimates,
    }


def reconstruction_experiment(
    N: int = 500,
    k: int = 40,
    M_train: int = 500,
    p: float = 0.5,
    alpha: float = 10.0,
    beta: float = 1.0,
    gm_beta: float = 0.1,
    head: int | None = None,
    seed: int = 42,
) -> tuple[float, float]:
    """Compare PSD-based and graph-smooth reconstruction on one test signal.

    ``head`` optionally limits the number of sorted samples displayed in the
    line plot. Reconstruction metrics are always computed over all missing
    vertices.
    """
    np.random.seed(seed)
    random_generator = np.random.default_rng(seed)
    graph = GraphFactory.generate_nn_graph(N, k)
    generator = SignalGenerator(graph)
    reconstructor = SignalReconstructor(graph)

    if head is not None and (
        not isinstance(head, (int, np.integer))
        or isinstance(head, bool)
        or head < 1
        or head > N
    ):
        raise ValueError("head must be None or an integer between 1 and N.")

    training_truth, _ = generator.generate_signals(
        M_train,
        1.0,
        _default_low_pass_psd,
    )
    training_observations = apply_observation_mask(
        training_truth,
        p,
        draw_nested_mask_uniforms(
            training_truth.shape,
            random_generator,
        ),
    )
    gamma_estimated = reconstructor.estimate_gamma(training_observations)

    test_signals, _ = generator.generate_signals(
        1,
        1.0,
        _default_low_pass_psd,
    )
    test_observations = apply_observation_mask(
        test_signals,
        p,
        draw_nested_mask_uniforms(
            test_signals.shape,
            random_generator,
        ),
    )
    truth = test_signals[:, 0]
    observed = test_observations[:, 0]
    psd_estimate = reconstructor.reconstruct_psd_single(
        observed,
        gamma_estimated,
        alpha=alpha,
        beta=beta,
    )
    smooth_estimate = reconstructor.reconstruct_smooth(
        observed,
        beta=gm_beta,
    )

    observed_mask = ~np.isnan(observed)
    missing_mask = ~observed_mask
    mae_psd = _mae_on_missing(truth, psd_estimate, missing_mask)
    mae_smooth = _mae_on_missing(truth, smooth_estimate, missing_mask)

    positions = nx.spring_layout(graph, seed=seed)
    order = np.argsort(truth)
    if head is not None:
        order = order[:head]
    figure, axes = plt.subplots(2, 2, figsize=(14, 8))

    axes[0, 0].plot(truth[order], label="True signal", color="C0")
    axes[0, 0].plot(
        psd_estimate[order],
        "--",
        label="PSD reconstruction",
        color="C1",
    )
    axes[0, 0].plot(
        smooth_estimate[order],
        ":",
        label="Smooth reconstruction",
        color="C2",
    )
    axes[0, 0].set_title("Signal values sorted by true intensity")
    axes[0, 0].legend()

    value_min, value_max = float(np.min(truth)), float(np.max(truth))
    titles = [
        "True signal",
        f"PSD reconstruction (MAE={mae_psd:.3f})",
        f"Smooth reconstruction (MAE={mae_smooth:.3f})",
    ]
    plotted_signals = [truth, psd_estimate, smooth_estimate]
    graph_axes = [axes[0, 1], axes[1, 0], axes[1, 1]]
    observed_nodes = np.flatnonzero(observed_mask)

    for axis, title, values in zip(graph_axes, titles, plotted_signals):
        nodes = nx.draw_networkx_nodes(
            graph,
            positions,
            node_color=values,
            cmap="viridis",
            node_size=50,
            ax=axis,
            vmin=value_min,
            vmax=value_max,
        )
        nx.draw_networkx_edges(graph, positions, alpha=0.05, ax=axis)
        nx.draw_networkx_nodes(
            graph,
            positions,
            nodelist=observed_nodes,
            node_color="black",
            node_size=25,
            ax=axis,
        )
        axis.set_title(title)
        figure.colorbar(nodes, ax=axis)
        axis.set_axis_off()

    figure.tight_layout()
    plt.show()
    return mae_psd, mae_smooth


def run_mixed_psd_estimation_exp(
    N: int = 500,
    M: int = 1200,
    p: float = 0.7,
    psd_funcs: Sequence[Callable[[np.ndarray, float], np.ndarray]] | None = None,
    probs: Sequence[float] | None = None,
    psd_names: Sequence[str] | None = None,
    k: int = 25,
    seed: int = 42,
) -> dict:
    """Compare global and source-informed PSD estimates for mixed signals."""
    if psd_funcs is None:
        psd_funcs = [
            lambda x, lm: np.exp(-3.0 * (x / lm)),
            lambda x, lm: np.exp(-20.0 * (x / lm - 0.4) ** 2),
            lambda x, lm: np.exp(-3.0 * (1.0 - x / lm)),
        ]
    if probs is None:
        probs = [0.4, 0.3, 0.3]
    if psd_names is None:
        psd_names = ["Low-pass", "Band-pass", "High-pass"]
    if len(psd_names) != len(psd_funcs):
        raise ValueError("psd_names must contain one name per PSD function.")

    np.random.seed(seed)
    graph = GraphFactory.generate_nn_graph(N, k)
    generator = SignalGenerator(graph)
    reconstructor = SignalReconstructor(graph)
    eigenvalues = graph.eigenvalues
    lambda_max = float(eigenvalues[-1])
    complete, _, labels = generator.generate_mixed_signals(
        M,
        1.0,
        psd_funcs,
        probs,
    )
    observed = apply_observation_mask(
        complete,
        p,
        draw_nested_mask_uniforms(
            complete.shape,
            np.random.default_rng(seed),
        ),
    )

    gamma_global = reconstructor.estimate_gamma(observed)
    gamma_local = {}
    figure, axes = plt.subplots(
        1,
        len(psd_funcs),
        figsize=(6 * len(psd_funcs), 5),
        sharey=True,
        squeeze=False,
    )

    for component, (axis, profile, name) in enumerate(
        zip(axes[0], psd_funcs, psd_names)
    ):
        selected = labels == component
        if np.any(selected):
            local_estimate = reconstructor.estimate_gamma(
                observed[:, selected]
            )
        else:
            local_estimate = np.zeros_like(eigenvalues)
        gamma_local[component] = local_estimate
        gamma_true = reconstructor.normalize_gamma(
            profile(eigenvalues, lambda_max)
        )

        axis.plot(
            eigenvalues,
            gamma_true,
            "g-",
            linewidth=3,
            label="True PSD",
            alpha=0.7,
        )
        axis.plot(
            eigenvalues,
            local_estimate,
            "b--",
            linewidth=2,
            label=f"Estimated (n={int(selected.sum())})",
        )
        axis.plot(
            eigenvalues,
            gamma_global,
            "r:",
            linewidth=2,
            label="Global mixed PSD",
        )
        axis.set(title=name, xlabel="Eigenvalue (λ)")
        axis.legend()
        axis.grid(True, alpha=0.2)

    axes[0, 0].set_ylabel("Normalized PSD")
    figure.suptitle(f"Mixed PSD estimation (M={M}, p={p})", fontsize=15)
    figure.tight_layout()
    plt.show()
    return {
        "graph": graph,
        "labels": labels,
        "gamma_global": gamma_global,
        "gamma_local": gamma_local,
    }


def run_mixed_comparison_experiment(
    N: int = 500,
    k: int = 30,
    M_train: int = 1000,
    M_test: int = 100,
    p: float = 0.4,
    alpha: float = 15.0,
    beta: float = 1.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Compare class-informed and global PSD reconstruction on two sources."""
    np.random.seed(seed)
    random_generator = np.random.default_rng(seed)
    graph = GraphFactory.generate_nn_graph(N, k)
    generator = SignalGenerator(graph)
    reconstructor = SignalReconstructor(graph)
    eigenvalues = graph.eigenvalues
    lambda_max = float(eigenvalues[-1])

    def psd_a(eigenvalues, maximum):
        """Return the lower-frequency mid-band source profile."""
        return np.exp(-100.0 * (eigenvalues / maximum - 0.25) ** 2)

    def psd_b(eigenvalues, maximum):
        """Return the higher-frequency mid-band source profile."""
        return np.exp(-40.0 * (eigenvalues / maximum - 0.6) ** 2)

    profiles = [psd_a, psd_b]
    names = ["Mid-Band A (Low-Shift)", "Mid-Band B (High-Shift)"]
    split = generate_mixture_split(
        generator,
        M_train,
        M_test,
        profiles,
        [0.5, 0.5],
    )
    training_observations = apply_observation_mask(
        split.train,
        p,
        draw_nested_mask_uniforms(
            split.train.shape,
            random_generator,
        ),
    )
    training_labels = split.train_labels

    gamma_global = reconstructor.estimate_gamma(training_observations)
    gamma_informed = []
    for component in range(len(profiles)):
        selected = training_labels == component
        if not np.any(selected):
            raise ValueError(
                "Every source must occur in the training sample. Increase "
                "M_train or choose a different seed."
            )
        gamma_informed.append(
            reconstructor.estimate_gamma(training_observations[:, selected])
        )

    figure_psd, axis_psd = plt.subplots(figsize=(12, 5))
    axis_psd.plot(
        eigenvalues,
        reconstructor.normalize_gamma(psd_a(eigenvalues, lambda_max)),
        "k-",
        alpha=0.15,
        label="True PSD A",
    )
    axis_psd.plot(
        eigenvalues,
        reconstructor.normalize_gamma(psd_b(eigenvalues, lambda_max)),
        "k--",
        alpha=0.15,
        label="True PSD B",
    )
    axis_psd.plot(
        eigenvalues,
        gamma_informed[0],
        color="C2",
        linewidth=2,
        label="Informed PSD A",
    )
    axis_psd.plot(
        eigenvalues,
        gamma_informed[1],
        color="C0",
        linewidth=2,
        label="Informed PSD B",
    )
    axis_psd.plot(
        eigenvalues,
        gamma_global,
        color="C3",
        linewidth=3,
        linestyle=":",
        label="Global mixed PSD",
    )
    axis_psd.set(
        title="PSD shape comparison",
        xlabel="Eigenvalue (λ)",
        ylabel="Normalized amplitude",
    )
    axis_psd.legend()
    axis_psd.grid(True, alpha=0.2)
    figure_psd.tight_layout()
    plt.show()

    test_signals = split.test
    test_labels = split.test_labels
    test_observations = apply_observation_mask(
        test_signals,
        p,
        draw_nested_mask_uniforms(
            test_signals.shape,
            random_generator,
        ),
    )
    source_a_indices = np.flatnonzero(test_labels == 0)
    example_index = (
        int(np.random.choice(source_a_indices))
        if source_a_indices.size
        else 0
    )
    rows = []
    example = None

    for column in range(M_test):
        truth = test_signals[:, column]
        observed = test_observations[:, column]
        label = int(test_labels[column])
        missing = np.isnan(observed)
        informed = reconstructor.reconstruct_psd_single(
            observed,
            gamma_informed[label],
            alpha=alpha,
            beta=beta,
        )
        global_estimate = reconstructor.reconstruct_psd_single(
            observed,
            gamma_global,
            alpha=alpha,
            beta=beta,
        )
        informed_mae = _mae_on_missing(truth, informed, missing)
        global_mae = _mae_on_missing(truth, global_estimate, missing)
        rows.append(
            {
                "Type": names[label],
                "Informed MAE": informed_mae,
                "Global MAE": global_mae,
            }
        )
        if column == example_index:
            example = (
                truth,
                informed,
                global_estimate,
                observed,
                label,
                informed_mae,
                global_mae,
            )

    results = pd.DataFrame(rows)
    truth, informed, global_estimate, observed, label, mae_inf, mae_global = example
    positions = nx.spring_layout(graph, seed=seed)
    observed_nodes = np.flatnonzero(~np.isnan(observed))
    order = np.argsort(truth)
    figure_example, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(truth[order], label="True signal", color="C0", linewidth=2)
    axes[0, 0].plot(
        informed[order],
        "--",
        label="Informed PSD",
        color="C2",
    )
    axes[0, 0].plot(
        global_estimate[order],
        ":",
        label="Global PSD",
        color="C3",
    )
    axes[0, 0].set_title(f"Example: {names[label]} (sorted)")
    axes[0, 0].legend()

    graph_axes = [axes[0, 1], axes[1, 0], axes[1, 1]]
    titles = [
        "True signal",
        f"Informed reconstruction (MAE={mae_inf:.3f})",
        f"Global reconstruction (MAE={mae_global:.3f})",
    ]
    values_to_plot = [truth, informed, global_estimate]
    value_min, value_max = float(truth.min()), float(truth.max())

    for axis, title, values in zip(graph_axes, titles, values_to_plot):
        nodes = nx.draw_networkx_nodes(
            graph,
            positions,
            node_color=values,
            cmap="viridis",
            node_size=50,
            ax=axis,
            vmin=value_min,
            vmax=value_max,
        )
        nx.draw_networkx_edges(graph, positions, alpha=0.05, ax=axis)
        nx.draw_networkx_nodes(
            graph,
            positions,
            nodelist=observed_nodes,
            node_color="black",
            node_size=20,
            ax=axis,
        )
        axis.set_title(title)
        figure_example.colorbar(nodes, ax=axis)
        axis.set_axis_off()

    figure_example.tight_layout()
    plt.show()

    melted = results.melt(
        id_vars=["Type"],
        value_vars=["Informed MAE", "Global MAE"],
        var_name="Method",
        value_name="MAE",
    )
    figure_box, axis_box = plt.subplots(figsize=(10, 6))
    sns.boxplot(
        data=melted,
        x="Type",
        y="MAE",
        hue="Method",
        palette=["#2ecc71", "#e74c3c"],
        order=names,
        ax=axis_box,
    )
    axis_box.set_title(f"MAE across {M_test} mixed signals")
    axis_box.grid(axis="y", alpha=0.3)
    figure_box.tight_layout()
    plt.show()

    return results


def gmm_mixed_signal_experiment(
    graph,
    psd_s,
    probs,
    M: int = 500,
    p: float = 1.0,
    n_runs: int = 10,
    psd_rec: bool = False,
    seed: int = 42,
    refinement_steps: int = 10,
    verbose: bool = False,
    M_test: int | None = None,
) -> dict:
    """Evaluate spectral GMM clustering on an independent test split.

    When ``psd_rec`` is true, reconstruction/GMM refinement is fitted only on
    training signals.  The final GMM and train-estimated PSDs are then frozen
    and used to assign and reconstruct test signals.
    """
    if (
        not isinstance(n_runs, (int, np.integer))
        or isinstance(n_runs, bool)
        or n_runs < 1
    ):
        raise ValueError("n_runs must be a positive integer.")
    if (
        not isinstance(refinement_steps, (int, np.integer))
        or isinstance(refinement_steps, bool)
        or refinement_steps < 1
    ):
        raise ValueError("refinement_steps must be a positive integer.")

    np.random.seed(seed)
    test_count = M if M_test is None else M_test
    accuracies = []
    cluster_count = len(psd_s)
    generator = SignalGenerator(graph)
    reconstructor = SignalReconstructor(graph)

    for run in tracked_range(n_runs, "preliminary GMM"):
        run_seed = seed + run
        random_generator = np.random.default_rng(run_seed)
        split = generate_mixture_split(
            generator,
            M,
            test_count,
            psd_s,
            probs,
        )
        train_observed = apply_observation_mask(
            split.train,
            p,
            draw_nested_mask_uniforms(
                split.train.shape,
                random_generator,
            ),
        )
        test_observed = apply_observation_mask(
            split.test,
            p,
            draw_nested_mask_uniforms(
                split.test.shape,
                random_generator,
            ),
        )
        train_reconstructed = (
            reconstructor.reconstruct_smooth(train_observed, beta=0.0001)
            if p < 1.0
            else train_observed
        )
        train_features = ClusteringEvaluator.graph_fourier_features(
            graph,
            train_reconstructed,
        )
        gmm = GMM_Diag(
            cluster_count,
            random_state=run_seed,
        ).fit(train_features)
        train_prediction = gmm.predict(train_features)

        if psd_rec:
            for _ in range(refinement_steps):
                train_reconstructed = reconstructor.reconstruct_mixed(
                    train_observed,
                    train_prediction,
                )
                train_features = ClusteringEvaluator.graph_fourier_features(
                    graph,
                    train_reconstructed,
                )
                gmm.fit(train_features)
                updated_prediction = gmm.predict(train_features)
                if np.array_equal(updated_prediction, train_prediction):
                    train_prediction = updated_prediction
                    break
                train_prediction = updated_prediction

        test_reconstructed = (
            reconstructor.reconstruct_smooth(test_observed, beta=0.0001)
            if p < 1.0
            else test_observed
        )
        prediction = gmm.predict(
            ClusteringEvaluator.graph_fourier_features(
                graph,
                test_reconstructed,
            )
        )

        if psd_rec:
            train_psds = estimate_group_psds(
                reconstructor,
                train_observed,
                train_prediction,
                cluster_count,
                min_cluster_size=1,
                fallback=reconstructor.estimate_gamma(train_observed),
            )
            for _ in range(refinement_steps):
                test_reconstructed = reconstruct_from_group_psds(
                    reconstructor,
                    test_observed,
                    prediction,
                    train_psds,
                    alpha=10.0,
                    beta=1.0,
                )
                updated_prediction = gmm.predict(
                    ClusteringEvaluator.graph_fourier_features(
                        graph,
                        test_reconstructed,
                    )
                )
                if np.array_equal(updated_prediction, prediction):
                    prediction = updated_prediction
                    break
                prediction = updated_prediction

        accuracies.append(
            ClusteringEvaluator.evaluate_accuracy(
                split.test_labels,
                prediction,
                cluster_count,
            )
        )

    values = np.asarray(accuracies, dtype=float)
    return {
        "mean_acc": float(np.mean(values)),
        "std_acc": float(np.std(values)),
        "all_accs": accuracies,
    }
