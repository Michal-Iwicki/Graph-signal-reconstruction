import numpy as np
from codes.generation import GSPGraph

class SignalReconstructor:
    """Handles signal reconstruction and PSD estimation for a given GSPGraph."""
    def __init__(self, graph: GSPGraph):
        self.graph = graph

    @staticmethod
    def normalize_gamma(gamma):
        return gamma / (np.max(gamma) + 1e-8)

    def estimate_gamma(self, Y, sigma2=0.0):
        U = self.graph.eigenvectors
        N, M = Y.shape

        mask = (Y != 0) & (~np.isnan(Y))
        Y0 = np.nan_to_num(Y, nan=0.0)

        p = mask.sum() / (N * M)
        if p == 0: return np.zeros(N)

        c = Y0.sum() / mask.sum()
        C = (Y0 @ Y0.T) / M

        diag = np.diag(C)
        off_diag = C - np.diag(diag)

        diag_corrected = (diag - p*(1-p)*c**2 - p*sigma2) / p
        off_corrected = off_diag / (p**2)

        C_corrected = np.diag(diag_corrected) + off_corrected

        gamma = np.sum((U.T @ C_corrected) * U.T, axis=1)
        gamma = self.normalize_gamma(gamma)
        return np.maximum(gamma, 1e-8)

    def reconstruct_smooth(self, y, beta=1.0):
        L = self.graph.L
        mask = ~np.isnan(y)
        y0 = np.nan_to_num(y, nan=0.0)

        A = beta * L + np.diag(mask.astype(float))
        return np.linalg.solve(A, y0)

    def reconstruct_psd_single(self, y, gamma, alpha=10, beta=1.0, scaling=True):
        U = self.graph.eigenvectors
        mask = ~np.isnan(y)
        y0 = np.nan_to_num(y, nan=0.0)

        w = np.exp(-alpha * gamma)
        if scaling:
            w = w / (np.max(w) + 1e-8) * np.max(self.graph.eigenvalues)

        penalty = (U * w) @ U.T
        A = beta * penalty.copy()
        A[mask, mask] += 1
        return np.linalg.solve(A, y0)

    def reconstruct_psd(self, Y, gamma=None, alpha=10, beta=1.0):
        if gamma is None:
            gamma = self.estimate_gamma(Y)

        X_hat = np.zeros_like(Y)
        for m in range(Y.shape[1]):
            X_hat[:, m] = self.reconstruct_psd_single(Y[:, m], gamma, alpha, beta)
        return X_hat
    
    def reconstruct_mixed(self, Y, labels, alpha=10, beta=1.0):
        labels = np.asarray(labels)

        X_hat = np.zeros_like(Y)

        for k in range(np.max(labels) + 1):  # bo labels ∈ {0,...,K-1}
            idx = (labels == k)

            if not np.any(idx):
                continue  # pusty klaster – pomijamy

            Yk = Y[:, idx]

            # estymacja PSD dla klastra
            gamma_k = self.estimate_gamma(Yk)

            # rekonstrukcja dla tego klastra
            X_hat[:, idx] = self.reconstruct_psd(
                Yk, gamma=gamma_k, alpha=alpha, beta=beta
            )

        return X_hat