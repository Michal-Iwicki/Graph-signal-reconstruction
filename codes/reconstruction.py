import numpy as np
import networkx as nx
from generation import GSPGraph

# ==========================================================
# PSD ESTIMATION
# ==========================================================

def estimate_gamma(graph: GSPGraph, Y):
    """
    Fast PSD estimation from sampled graph signals.

    Parameters
    ----------
    graph : GSPGraph
    Y : (N, M) array
        sampled signals with NaN on missing vertices

    Returns
    -------
    gamma : (N,)
    """

    U = graph.eigenvectors

    N, M = Y.shape

    mask = ~np.isnan(Y)

    Y0 = np.nan_to_num(Y, nan=0.0)

    p = mask.sum() / (N * M)

    c = Y0.sum() / mask.sum()

    # covariance
    C = np.cov(Y0)

    # bias correction (paper eq.22)
    C -= p * (1 - p) * c**2 * np.eye(N)

    # gamma = diag(U^T C U)
    gamma = np.sum((U.T @ C) * U.T, axis=1) / p

    return np.maximum(gamma, 0)


# ==========================================================
# SMOOTH RECONSTRUCTION (GM)
# ==========================================================

def reconstruct_smooth(graph: GSPGraph, y, beta=1.0):
    """
    Smooth reconstruction (General Model).

    y : signal with NaN on missing vertices
    """
    L = graph.laplacian

    mask = ~np.isnan(y)

    y0 = np.nan_to_num(y, nan=0.0)

    A = beta * L.copy()
    A[mask, mask] += 1

    b = y0.copy()

    return np.linalg.solve(A, b)


# ==========================================================
# PSD RECONSTRUCTION
# ==========================================================

def reconstruct_psd(graph: GSPGraph, y, gamma, alpha=10, beta=1.0):
    """
    PSD reconstruction method.
    """

    U = graph.eigenvectors

    mask = ~np.isnan(y)

    y0 = np.nan_to_num(y, nan=0.0)

    w = np.exp(-alpha * gamma)

    penalty = (U * w) @ U.T

    A = beta * penalty
    A[mask, mask] += 1

    b = y0.copy()

    return np.linalg.solve(A, b)