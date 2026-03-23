import numpy as np
from codes.generation import GSPGraph

# ==========================================================
# PSD ESTIMATION (closer to Eq. 24 from paper)
# ==========================================================

def estimate_gamma(graph: GSPGraph, Y, sigma2=0.0):
    """
    PSD estimation from multiple sampled graph signals.

    Parameters
    ----------
    graph : GSPGraph
    Y : (N, M) array with NaN on missing entries
    sigma2 : noise variance (default 0)

    Returns
    -------
    gamma : (N,)
    """

    U = graph.eigenvectors
    N, M = Y.shape

    mask = (Y != 0)
    Y0 = np.nan_to_num(Y, nan=0.0)

    # p = probability of observation
    p = mask.sum() / (N * M)

    # mean estimate
    c = Y0.sum() / mask.sum()


    C = (Y0 @ Y0.T) / M

    diag = np.diag(C)
    off_diag = C - np.diag(diag)

    diag_corrected = (diag - p*(1-p)*c**2 - p*sigma2) / p
    off_corrected = off_diag / (p**2)

    C_corrected = np.diag(diag_corrected) + off_corrected

    # ===== PSD =====
    gamma = np.sum((U.T @ C_corrected) * U.T, axis=1)

    gamma = normalize_gamma(gamma)

    return np.maximum(gamma, 1e-8)


# ==========================================================
# SMOOTH RECONSTRUCTION (GM baseline)
# ==========================================================

def reconstruct_smooth(graph: GSPGraph, y, beta=1.0):
    """
    Smooth reconstruction (General Model)
    """
    L = graph.L

    mask = ~np.isnan(y)
    y0 = np.nan_to_num(y, nan=0.0)

    A = beta * L.copy()
    A[mask, mask] += 1

    return np.linalg.solve(A, y0)


# ==========================================================
# PSD RECONSTRUCTION (single signal)
# ==========================================================

def reconstruct_psd_single(graph: GSPGraph, y, gamma, alpha=10, beta=1.0, scalling=True):
    """
    PSD reconstruction for a single signal
    """
    U = graph.eigenvectors

    mask = ~np.isnan(y)
    y0 = np.nan_to_num(y, nan=0.0)

    # weight function from paper with normalization
    w = np.exp(-alpha * gamma)
    if scalling:
        w = w / (np.max(w) + 1e-8) * np.max(graph.eigenvalues)

    penalty = (U * w) @ U.T

    A = beta * penalty.copy()
    A[mask, mask] += 1
    

    return np.linalg.solve(A, y0)


# ==========================================================
# PSD RECONSTRUCTION (MULTIPLE SIGNALS)
# ==========================================================

def reconstruct_psd(graph: GSPGraph, Y, gamma=None, alpha=10, beta=1.0):
    """
    PSD reconstruction for multiple signals

    Parameters
    ----------
    Y : (N, M)
    """

    U = graph.eigenvectors
    N, M = Y.shape

    # estimate PSD once (VERY IMPORTANT)
    if gamma is None:
        gamma = estimate_gamma(graph, Y)

    # weight function
    w = np.exp(-alpha * gamma)

    # penalty matrix
    penalty = (U * w) @ U.T

    X_hat = np.zeros_like(Y)

    for m in range(M):
        y = Y[:, m]

        mask = ~np.isnan(y)
        y0 = np.nan_to_num(y, nan=0.0)

        A = beta * penalty.copy()
        A[mask, mask] += 1

        X_hat[:, m] = np.linalg.solve(A, y0)

    return X_hat


# ==========================================================
# OPTIONAL: PSD normalization (as in paper strategy)
# ==========================================================

def normalize_gamma(gamma):
    """
    Normalize PSD so max = 1 (paper strategy)
    """
    return gamma / (np.max(gamma) + 1e-8)