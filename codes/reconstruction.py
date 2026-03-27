import numpy as np
from codes.generation import GSPGraph

def estimate_gamma(graph: GSPGraph, Y, sigma2=0.0):
    """
    Estimates the PSD (gamma) from multiple sampled graph signals.
    Implementation based on empirical covariance correction for missing data.

    Parameters:
        graph: GSPGraph object.
        Y: (N, M) array with NaN or 0.0 on missing entries.
        sigma2: Noise variance.

    Returns:
        gamma: (N,) Estimated PSD values.
    """
    U = graph.eigenvectors
    N, M = Y.shape

    mask = (Y != 0) & (~np.isnan(Y))
    Y0 = np.nan_to_num(Y, nan=0.0)

    # p = empirical probability of observation
    p = mask.sum() / (N * M)

    # Mean estimate for correction
    c = Y0.sum() / mask.sum()

    # Empirical covariance
    C = (Y0 @ Y0.T) / M

    diag = np.diag(C)
    off_diag = C - np.diag(diag)

    # Bias correction for the covariance matrix due to sampling
    diag_corrected = (diag - p*(1-p)*c**2 - p*sigma2) / p
    off_corrected = off_diag / (p**2)

    C_corrected = np.diag(diag_corrected) + off_corrected

    # Project covariance into the spectral domain to find gamma
    gamma = np.sum((U.T @ C_corrected) * U.T, axis=1)
    gamma = normalize_gamma(gamma)

    return np.maximum(gamma, 1e-8)

def reconstruct_smooth(graph: GSPGraph, y, beta=1.0):
    """
    Baseline: Smooth reconstruction (General Model / Tikhonov Regularization).
    Assumes the signal is low-frequency (smooth) on the graph.
    """
    L = graph.L
    mask = ~np.isnan(y)
    y0 = np.nan_to_num(y, nan=0.0)

    # Solve (I_mask + beta * L)x = y_obs
    A = beta * L.copy()
    A[mask, mask] += 1

    return np.linalg.solve(A, y0)

def reconstruct_psd_single(graph: GSPGraph, y, gamma, alpha=10, beta=1.0, scaling=True):
    """
    PSD-informed reconstruction for a single signal.
    Uses an estimated PSD as a prior for the reconstruction penalty.
    """
    U = graph.eigenvectors
    mask = ~np.isnan(y)
    y0 = np.nan_to_num(y, nan=0.0)

    # Weight function: penalize frequencies with low power density
    w = np.exp(-alpha * gamma)
    if scaling:
        w = w / (np.max(w) + 1e-8) * np.max(graph.eigenvalues)

    # Construct the spectral penalty matrix
    penalty = (U * w) @ U.T

    A = beta * penalty.copy()
    A[mask, mask] += 1
    
    return np.linalg.solve(A, y0)

def reconstruct_psd(graph: GSPGraph, Y, gamma=None, alpha=10, beta=1.0):
    """
    PSD-informed reconstruction for multiple signals.
    """
    U = graph.eigenvectors
    N, M = Y.shape

    if gamma is None:
        gamma = estimate_gamma(graph, Y)

    w = np.exp(-alpha * gamma)
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

def normalize_gamma(gamma):
    """
    Normalizes PSD values to a [0, 1] range.
    """
    return gamma / (np.max(gamma) + 1e-8)