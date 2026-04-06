import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import seaborn as sns

from codes.generation import GraphFactory, SignalGenerator, GSPVisualizer
from codes.reconstruction import SignalReconstructor
from codes.clustering import GMM_Diag, ClusteringEvaluator

def run_psd_experiment(N=500, k=40, M=1000, p_values=(1.0, 0.5, 0.05), psd_fn=None):
    graph = GraphFactory.generate_nn_graph(N, k)
    generator = SignalGenerator(graph)
    reconstructor = SignalReconstructor(graph)
    
    eigvals = graph.eigenvalues
    lmax = eigvals.max()

    if psd_fn is None:
        def psd_fn(x, lmax):
            out = np.zeros_like(x)
            mask = x < lmax / 4
            out[mask] = np.sin(4 * np.pi * x[mask] / lmax)
            return np.maximum(out, 0)

    gamma_true = psd_fn(eigvals, lmax)
    gamma_true = np.maximum(gamma_true, 0)
    gamma_true /= (np.max(gamma_true) + 1e-8)

    plt.figure(figsize=(5 * len(p_values), 4))

    for i, p in enumerate(p_values):
        X, Y = generator.generate_signals(M, p, psd_fn)
        Y[Y == 0] = np.nan

        gamma_est = reconstructor.estimate_gamma(Y)
        gamma_est /= (np.max(gamma_est) + 1e-8)

        plt.subplot(1, len(p_values), i + 1)
        plt.plot(eigvals, gamma_true, label="True PSD")
        plt.plot(eigvals, gamma_est, "--", label="Estimated PSD")
        plt.title(f"Sampling p = {p}")
        plt.xlabel("Eigenvalue (λ)")
        plt.ylabel("Normalized PSD")
        plt.legend()

    plt.tight_layout()
    plt.show()

def reconstruction_experiment(N=500, k=40, M_train=500, p=0.5, alpha=10, beta=1.0, gm_beta=0.1, head=None, seed=42):
    np.random.seed(seed)
    
    graph = GraphFactory.generate_nn_graph(N, k)
    generator = SignalGenerator(graph)
    reconstructor = SignalReconstructor(graph)
    
    lmax = graph.eigenvalues.max()

    def psd_fn(x, lmax):
        out = np.zeros_like(x)
        mask = x < lmax / 4
        out[mask] = np.sin(4 * np.pi * x[mask] / lmax)
        return np.maximum(out, 0)

    # Train
    _, Y_train = generator.generate_signals(M_train, p, psd_fn)
    Y_train[Y_train == 0] = np.nan
    gamma_est = reconstructor.estimate_gamma(Y_train)

    # Test
    X_test, Y_test = generator.generate_signals(1, p, psd_fn)
    x_true = X_test[:, 0]
    y_obs = Y_test[:, 0].copy()
    y_obs[y_obs == 0] = np.nan

    # Reconstruct
    x_psd = reconstructor.reconstruct_psd_single(y_obs, gamma_est, alpha=alpha, beta=beta)
    x_smooth = reconstructor.reconstruct_smooth(y_obs, beta=gm_beta)

    # Metrics
    mask_obs = ~np.isnan(y_obs)
    obs_idx = np.where(mask_obs)[0]
    mask_missing = ~mask_obs
    mae_psd = np.mean(np.abs(x_psd[mask_missing] - x_true[mask_missing]))
    mae_smooth = np.mean(np.abs(x_smooth[mask_missing] - x_true[mask_missing]))

    print(f"\n=== Reconstruction results (p={p}) ===")
    print(f"MAE PSD: {mae_psd:.4f}")
    print(f"MAE GM : {mae_smooth:.4f}")

    # Visualization
    pos = nx.spring_layout(graph, seed=seed)
    order = np.argsort(x_true)
    fig = plt.figure(figsize=(14, 8))

    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(x_true[order], label="True Signal", color='C0')
    ax1.plot(x_psd[order], "--", label="PSD Recon", color='C1')
    ax1.plot(x_smooth[order], ":", label="Smooth (GM)", color='C2')
    ax1.set_title("Signal (sorted by true intensity)")
    ax1.legend()

    vmin, vmax = np.min(x_true), np.max(x_true)
    titles = ["True Signal", f"PSD Recon (MAE={mae_psd:.3f})", f"GM Recon (MAE={mae_smooth:.3f})"]
    data_to_plot = [x_true, x_psd, x_smooth]

    for i in range(3):
        ax = plt.subplot(2, 2, i+2)
        nodes = nx.draw_networkx_nodes(graph, pos, node_color=data_to_plot[i], cmap="viridis", node_size=50, ax=ax, vmin=vmin, vmax=vmax)
        nx.draw_networkx_edges(graph, pos, alpha=0.05, ax=ax)
        nx.draw_networkx_nodes(graph, pos, nodelist=obs_idx, node_color="black", node_size=25, ax=ax, label="Observed")
        ax.set_title(titles[i])
        plt.colorbar(nodes, ax=ax)
        ax.axis("off")

    plt.tight_layout()
    plt.show()
    return mae_psd, mae_smooth

def run_mixed_psd_estimation_exp(N=500, M=1200, p=0.7, psd_funcs=None, probs=None, psd_names=None):
    if psd_funcs is None:
        psd_funcs = [
            lambda x, lm: np.exp(-3.0 * (x / lm)),
            lambda x, lm: np.exp(-20.0 * (x / lm - 0.4)**2),
            lambda x, lm: np.exp(-3.0 * (1.0 - x / lm))
        ]
    probs = probs or [0.4, 0.3, 0.3]
    psd_names = psd_names or ["Low-pass", "Band-pass", "High-pass"]

    graph = GraphFactory.generate_nn_graph(N, 25)
    generator = SignalGenerator(graph)
    reconstructor = SignalReconstructor(graph)
    
    eigvals, lmax = graph.eigenvalues, graph.eigenvalues.max()
    _, Y, labels = generator.generate_mixed_signals(M, p, psd_funcs, probs)
    Y_nan = Y.copy()
    Y_nan[Y_nan == 0] = np.nan

    gamma_global = reconstructor.normalize_gamma(reconstructor.estimate_gamma(Y_nan))

    fig, axes = plt.subplots(1, len(psd_funcs), figsize=(18, 5), sharey=True)
    if len(psd_funcs) == 1: axes = [axes]

    for i in range(len(psd_funcs)):
        ax = axes[i]
        mask = (labels == i)
        
        if np.any(mask):
            gamma_local = reconstructor.normalize_gamma(reconstructor.estimate_gamma(Y_nan[:, mask]))
            count = np.sum(mask)
        else:
            gamma_local = np.zeros_like(eigvals)
            count = 0

        gamma_true = reconstructor.normalize_gamma(psd_funcs[i](eigvals, lmax))

        ax.plot(eigvals, gamma_true, 'g-', lw=3, label="True PSD", alpha=0.7)
        ax.plot(eigvals, gamma_local, 'b--', lw=2, label=f"Estimated (n={count})")
        ax.plot(eigvals, gamma_global, 'r:', lw=2, label="Global (Mixed)")

        ax.set_title(psd_names[i])
        ax.set_xlabel("Eigenvalues (λ)")
        if i == 0: ax.set_ylabel("Normalized PSD")
        ax.legend()
        ax.grid(True, alpha=0.2)

    plt.suptitle(f"Mixed PSD Estimation Analysis (M={M}, p={p})", fontsize=15)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def run_mixed_comparison_experiment(
    N=500, 
    k=30, 
    M_train=1000, 
    M_test=100, 
    p=0.4, 
    alpha=15, 
    beta=1.0, 
    seed=42
):
    """
    Comparison of Informed PSD (class-aware) vs. Global PSD (class-blind) reconstruction.
    Uses two distinct Mid-Band-pass signals with different shifts and widths.
    Includes spectral analysis of estimated vs true PSD shapes.
    """
    np.random.seed(seed)

    # 1. Initialize Graph and OOP Helpers
    graph = GraphFactory.generate_nn_graph(N, k)
    generator = SignalGenerator(graph)
    reconstructor = SignalReconstructor(graph)
    
    eigvals = graph.eigenvalues
    lmax = eigvals.max()

    # 2. Define two different Mid-Band-pass PSD functions
    psd_a_fn = lambda x, lm: np.exp(-100.0 * (x / lm - 0.25)**2)
    psd_b_fn = lambda x, lm: np.exp(-40.0 * (x / lm - 0.6)**2)
    
    psd_funcs = [psd_a_fn, psd_b_fn]
    psd_names = ["Mid-Band A (Low-Shift)", "Mid-Band B (High-Shift)"]

    # 3. Train: Estimate PSDs from mixed training data
    X_tr, Y_tr, labels_tr = generator.generate_mixed_signals(M_train, p, psd_funcs, [0.5, 0.5])
    Y_tr_nan = Y_tr.copy()
    Y_tr_nan[Y_tr_nan == 0] = np.nan

    # Estimate Global (Mixed) PSD - Blind to labels
    gamma_global = reconstructor.normalize_gamma(reconstructor.estimate_gamma(Y_tr_nan))
    
    # Estimate Informed PSDs - Specific to each class
    gamma_informed = []
    for i in range(len(psd_funcs)):
        mask = (labels_tr == i)
        gamma_inf = reconstructor.normalize_gamma(reconstructor.estimate_gamma(Y_tr_nan[:, mask]))
        gamma_informed.append(gamma_inf)

    # 4. PSD Shape Comparison (Spectral Analysis)
    plt.figure(figsize=(12, 5))
    
    # True theoretical shapes
    plt.plot(eigvals, reconstructor.normalize_gamma(psd_a_fn(eigvals, lmax)), 'k-', alpha=0.15, label="True PSD A")
    plt.plot(eigvals, reconstructor.normalize_gamma(psd_b_fn(eigvals, lmax)), 'k--', alpha=0.15, label="True PSD B")
    
    # Estimated shapes
    plt.plot(eigvals, gamma_informed[0], color='C2', lw=2, label="Informed PSD A (Estimated)")
    plt.plot(eigvals, gamma_informed[1], color='C0', lw=2, label="Informed PSD B (Estimated)")
    plt.plot(eigvals, gamma_global, color='C3', lw=3, ls=':', label="Global PSD (Mixed)")
    
    plt.title("PSD Shape Comparison: Informed (Class-Aware) vs. Global (Mixed)")
    plt.xlabel("Eigenvalue (λ)")
    plt.ylabel("Normalized Amplitude")
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.show()

    # 5. Test & Aggregate: Evaluate on M_test signals
    X_te, Y_te, labels_te = generator.generate_mixed_signals(M_test, p, psd_funcs, [0.5, 0.5])
    
    results_list = []
    
    # Wybieramy losowy indeks dla sygnału typu "Mid-Band A" (label == 0)
    indices_A = np.where(labels_te == 0)[0]
    # Upewniamy się, że wylosowano przynajmniej jeden taki sygnał
    example_idx = np.random.choice(indices_A) if len(indices_A) > 0 else 0
    
    for i in range(M_test):
        x_true = X_te[:, i]
        y_obs = Y_te[:, i].copy()
        y_obs[y_obs == 0] = np.nan
        label = labels_te[i]
        mask_missing = np.isnan(y_obs)

        # Reconstructions using the OOP reconstructor
        x_inf = reconstructor.reconstruct_psd_single(y_obs, gamma_informed[label], alpha=alpha, beta=beta)
        x_glo = reconstructor.reconstruct_psd_single(y_obs, gamma_global, alpha=alpha, beta=beta)

        # Metrics
        mae_inf = np.mean(np.abs(x_true[mask_missing] - x_inf[mask_missing]))
        mae_glo = np.mean(np.abs(x_true[mask_missing] - x_glo[mask_missing]))

        results_list.append({
            'Type': psd_names[label],
            'Informed MAE': mae_inf,
            'Global MAE': mae_glo
        })
        
        if i == example_idx:
            ex_data = (x_true, x_inf, x_glo, y_obs, label, mae_inf, mae_glo)

    df_results = pd.DataFrame(results_list)

    # 6. Visualization: Example Signal (2x2 Style)
    x_t, x_i, x_g, y_o, lab, m_i, m_g = ex_data
    pos = nx.spring_layout(graph, seed=seed)
    obs_idx = np.where(~np.isnan(y_o))[0]
    order = np.argsort(x_t)

    fig1 = plt.figure(figsize=(14, 10))
    
    # --- Plot 1: Sorted Signal Values ---
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(x_t[order], label="True Signal", color='C0', lw=2)
    ax1.plot(x_i[order], "--", label="Informed PSD Recon", color='C2')
    ax1.plot(x_g[order], ":", label="Global PSD Recon", color='C3')
    ax1.set_title(f"Example: {psd_names[lab]} (Sorted)")
    ax1.legend()

    # --- Plots 2, 3, 4: Graph Heatmaps ---
    titles = ["True Signal", f"Informed Reconstruction (MAE={m_i:.3f})", f"Global Reconstruction (MAE={m_g:.3f})"]
    data_to_plot = [x_t, x_i, x_g]
    vmin, vmax = x_t.min(), x_t.max()

    for i in range(3):
        ax = plt.subplot(2, 2, i+2)
        nodes = nx.draw_networkx_nodes(graph, pos, node_color=data_to_plot[i], 
                                       cmap="viridis", node_size=50, ax=ax, vmin=vmin, vmax=vmax)
        nx.draw_networkx_edges(graph, pos, alpha=0.05, ax=ax)
        nx.draw_networkx_nodes(graph, pos, nodelist=obs_idx, node_color="black", node_size=20, ax=ax)
        ax.set_title(titles[i])
        plt.colorbar(nodes, ax=ax)
        ax.axis("off")

    plt.tight_layout()
    plt.show()

    # 7. Visualization: Aggregated Values (Boxplot)
    plt.figure(figsize=(10, 6))
    df_melt = df_results.melt(id_vars=['Type'], value_vars=['Informed MAE', 'Global MAE'], 
                              var_name='Method', value_name='MAE')
    
    # Wymuszamy kolejność Mid-Band A przed Mid-Band B
    sns.boxplot(data=df_melt, x='Type', y='MAE', hue='Method', 
                palette=["#2ecc71", "#e74c3c"],
                order=["Mid-Band A (Low-Shift)", "Mid-Band B (High-Shift)"])
    
    plt.title(f"MAE Aggregation across {M_test} mixed signals")
    plt.grid(axis='y', alpha=0.3)
    plt.show()

    # Summary Table
    print("\n=== Aggregate Results (Mean MAE) ===")
    print(df_results.groupby('Type')[['Informed MAE', 'Global MAE']].mean())

    return df_results