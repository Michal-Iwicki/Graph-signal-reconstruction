import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.neighbors import kneighbors_graph

from codes.generation import GSPGraph, generate_signals
from codes.reconstruction import estimate_gamma
import numpy as np
import matplotlib.pyplot as plt

from codes.generation import generate_nn_graph, generate_signals
from codes.reconstruction import (
    estimate_gamma,
    reconstruct_psd_single,
    reconstruct_smooth
)

def reconstruction_experiment(
    N=500,
    k=40,
    M_train=500,
    p=0.9,
    alpha=10,
    beta=1.0
):

    # ===== graph =====
    graph = generate_nn_graph(N, k)
    eigvals = graph.eigenvalues
    lmax = eigvals.max()

    # ===== PSD =====
    def psd_fn(x, lmax):
        out = np.zeros_like(x)
        mask = x < lmax / 4
        out[mask] = np.sin(4 * np.pi * x[mask] / lmax)
        return np.maximum(out, 0)

    # ===== 1. TRAIN PSD (many signals) =====
    X_train, Y_train, _ = generate_signals(graph, M_train, p=0.9, psd_fun=psd_fn)

    Y_train[Y_train == 0] = np.nan
    gamma_est = estimate_gamma(graph, Y_train)

    # ===== 2. TEST SIGNAL (single realization) =====
    X_test, Y_test, _ = generate_signals(graph, 1, p=p, psd_fun=psd_fn)

    x_true = X_test[:, 0]
    y_obs = Y_test[:, 0]
    y_obs[y_obs == 0] = np.nan

    # ===== 3. RECONSTRUCTION =====
    x_psd = reconstruct_psd_single(graph, y_obs, gamma_est, alpha=alpha, beta=beta)
    x_gm = reconstruct_smooth(graph, y_obs, beta=beta)

    # ===== 4. METRYKA =====
    mask_test = ~np.isnan(y_obs)

    mae_psd = np.mean(np.abs(x_psd[~mask_test] - x_true[~mask_test]))
    mae_gm = np.mean(np.abs(x_gm[~mask_test] - x_true[~mask_test]))

    print(f"MAE PSD: {mae_psd:.4f}")
    print(f"MAE GM : {mae_gm:.4f}")

    # ===== 5. PLOT =====
    plt.figure(figsize=(10, 4))
    plt.plot(x_true, label="true", linewidth=2)
    plt.plot(x_psd, "--", label="PSD reconstruction")
    plt.plot(x_gm, ":", label="GM reconstruction")

    plt.scatter(np.where(mask_test)[0], x_true[mask_test], color="black", s=20, label="observed")

    plt.legend()
    plt.title(f"Reconstruction (p={p})")
    plt.show()

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
        def psd_fn(x):
            out = np.zeros_like(x)
            mask = x < lmax / 4
            out[mask] = np.sin(4 * np.pi * x[mask] / lmax)
            return out

    gamma_true = psd_fn(eigvals)
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