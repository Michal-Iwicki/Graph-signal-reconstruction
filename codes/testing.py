import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import kneighbors_graph
from codes.generation import (
    generate_nn_graph, 
    generate_signals, 
    GSPGraph
    )
from codes.reconstruction import (
    estimate_gamma,
    reconstruct_psd_single,
    reconstruct_smooth
)

def reconstruction_experiment(
    N=500,
    k=40,
    M_train=500,
    p=0.5,
    alpha=10,
    beta=1.0,
    gm_beta = 0.1,
    head = None,
    seed=42
):

    np.random.seed(seed)

    # =========================================
    # GRAPH
    # =========================================
    graph = generate_nn_graph(N, k)
    eigvals = graph.eigenvalues
    lmax = eigvals.max()

    # =========================================
    # PSD FUNCTION
    # =========================================
    def psd_fn(x, lmax):
        out = np.zeros_like(x)
        mask = x < lmax / 4
        out[mask] = np.sin(4 * np.pi * x[mask] / lmax)
        return np.maximum(out, 0)

    # =========================================
    # 1. TRAIN PSD
    # =========================================
    X_train, Y_train, _ = generate_signals(graph, M_train, p=p, psd_fun=psd_fn)

    Y_train = Y_train.copy()
    Y_train[Y_train == 0] = np.nan

    gamma_est = estimate_gamma(graph, Y_train)

    # =========================================
    # 2. TEST SIGNAL
    # =========================================
    X_test, Y_test, _ = generate_signals(graph, 1, p=p, psd_fun=psd_fn)

    x_true = X_test[:, 0]
    y_obs = Y_test[:, 0]

    y_obs = y_obs.copy()
    y_obs[y_obs == 0] = np.nan

    # =========================================
    # 3. RECONSTRUCTION
    # =========================================
    x_psd = reconstruct_psd_single(graph, y_obs, gamma_est, alpha=alpha, beta=beta)
    x_gm = reconstruct_smooth(graph, y_obs, beta=gm_beta)

    # =========================================
    # 4. METRICS
    # =========================================
    mask_obs = ~np.isnan(y_obs)
    mask_missing = ~mask_obs

    mae_psd = np.mean(np.abs(x_psd[mask_missing] - x_true[mask_missing]))
    mae_gm = np.mean(np.abs(x_gm[mask_missing] - x_true[mask_missing]))

    print(f"\n=== Reconstruction results (p={p}) ===")
    print(f"MAE PSD: {mae_psd:.4f}")
    print(f"MAE GM : {mae_gm:.4f}")

    # =========================================
    # 5. VISUALIZATION
    # =========================================
    pos = nx.spring_layout(graph, seed=seed)
    order = np.argsort(x_true)

    fig = plt.figure(figsize=(14, 8))

    # ---------- SIGNAL ----------
    ax1 = plt.subplot(2, 2, 1)

    ax1.plot(x_true[order], label="true", linewidth=2)
    ax1.plot(x_psd[order], "--", label="PSD")
    ax1.plot(x_gm[order], ":", label="GM")

    # observed
    obs_idx = np.where(mask_obs)[0]
    obs_positions = np.where(np.isin(order, obs_idx))[0]

    ax1.scatter(
        obs_positions,
        x_true[order][np.isin(order, obs_idx)],
        color="black",
        s=12,
        label="observed"
    )

    # missing
    missing_idx = np.where(mask_missing)[0]
    missing_positions = np.where(np.isin(order, missing_idx))[0]

    ax1.scatter(
        missing_positions,
        x_true[order][np.isin(order, missing_idx)],
        color="red",
        s=8,
        label="missing (hidden)"
    )

    ax1.set_title("Signal (sorted)")
    ax1.legend()

    # ---------- TRUE ----------
    ax2 = plt.subplot(2, 2, 2)

    nodes = nx.draw_networkx_nodes(
        graph, pos,
        node_color=x_true,
        cmap="viridis",
        node_size=50,
        ax=ax2
    )
    nx.draw_networkx_edges(graph, pos, alpha=0.2, ax=ax2)
    ax2.set_title("True signal")
    plt.colorbar(nodes, ax=ax2)

    # ---------- PSD ----------
    ax3 = plt.subplot(2, 2, 3)

    nodes = nx.draw_networkx_nodes(
        graph, pos,
        node_color=x_psd,
        cmap="viridis",
        node_size=50,
        ax=ax3
    )
    nx.draw_networkx_edges(graph, pos, alpha=0.2, ax=ax3)

    nx.draw_networkx_nodes(
        graph, pos,
        nodelist=obs_idx,
        node_color="black",
        node_size=20,
        ax=ax3
    )

    ax3.set_title(f"PSD reconstruction\nMAE={mae_psd:.4f}")
    plt.colorbar(nodes, ax=ax3)

    # ---------- GM ----------
    ax4 = plt.subplot(2, 2, 4)

    nodes = nx.draw_networkx_nodes(
        graph, pos,
        node_color=x_gm,
        cmap="viridis",
        node_size=50,
        ax=ax4
    )
    nx.draw_networkx_edges(graph, pos, alpha=0.2, ax=ax4)

    nx.draw_networkx_nodes(
        graph, pos,
        nodelist=obs_idx,
        node_color="black",
        node_size=20,
        ax=ax4
    )

    ax4.set_title(f"GM reconstruction\nMAE={mae_gm:.4f}")
    plt.colorbar(nodes, ax=ax4)

    plt.tight_layout()
    plt.show()

    # =========================================
    # 6. TABLE (missing nodes)
    # =========================================
    df = pd.DataFrame({
        "node": missing_idx,
        "true": x_true[missing_idx],
        "PSD": x_psd[missing_idx],
        "GM": x_gm[missing_idx],
    })

    df["error_PSD"] = np.abs(df["PSD"] - df["true"])
    df["error_GM"]  = np.abs(df["GM"] - df["true"])

    if not head:
        head = len(df)
    print(f"\n=== Missing nodes (first {head}) ===")
    print(df.head(head))

    print("\nMean errors (check):")
    print(f"PSD: {df['error_PSD'].mean():.4f}")
    print(f"GM : {df['error_GM'].mean():.4f}")

    return mae_psd, mae_gm, df

def run_psd_experiment(
    N=500,
    k=40,
    M=1000,
    p_values=(1.0, 0.5, 0.05),
    psd_fn=None
):
    # ===== graph =====
    coords = np.random.rand(N, 2)
    A = kneighbors_graph(coords, k, mode='connectivity', include_self=False)
    G = nx.from_scipy_sparse_array(A)
    graph = GSPGraph(G)

    eigvals = graph.eigenvalues
    lmax = eigvals.max()

    # ===== PSD =====
    if psd_fn is None:
        def psd_fn(x, lmax):
            out = np.zeros_like(x)
            mask = x < lmax / 4
            out[mask] = np.sin(4 * np.pi * x[mask] / lmax)
            return np.maximum(out, 0)

    gamma_true = psd_fn(eigvals,lmax)
    gamma_true = np.maximum(gamma_true, 0)
    gamma_true /= (np.max(gamma_true) + 1e-8)


    plt.figure(figsize=(5 * len(p_values), 4))

    for i, p in enumerate(p_values):

        X, Y, _ = generate_signals(graph, M, p, psd_fn)

        Y = Y.copy()
        Y[Y == 0] = np.nan

        # ===== PSD estimation =====
        gamma_est = estimate_gamma(graph, Y)
        gamma_est /= (np.max(gamma_est) + 1e-8)

        # ===== plot =====
        plt.subplot(1, len(p_values), i + 1)
        plt.plot(eigvals, gamma_true, label="true")
        plt.plot(eigvals, gamma_est, "--", label="estimated")

        plt.title(f"p = {p}")
        plt.xlabel("λ")
        plt.ylabel("PSD")
        plt.legend()

    plt.tight_layout()
    plt.show()