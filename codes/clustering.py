import numpy as np
from scipy.special import logsumexp
from sklearn.metrics import accuracy_score
from scipy.optimize import linear_sum_assignment
from codes.generation import generate_mixed_signals
from codes.reconstruction import reconstruct_smooth
from sklearn.cluster import KMeans


# =========================================
# GMM DIAGONAL EM
# =========================================
class GMM_Diag:
    def __init__(self, K, max_iter=100, tol=1e-4, reg=1e-6):
        self.K = K
        self.max_iter = max_iter
        self.tol = tol
        self.reg = reg

    def _log_gaussian(self, X, mu, var):
        return -0.5 * (
            np.sum(np.log(2 * np.pi * var)) +
            np.sum((X - mu) ** 2 / var, axis=1)
        )

    def fit(self, X):
        """
        X: (M, D)
        """
        M, D = X.shape


        kmeans = KMeans(n_clusters=self.K, n_init=5)
        labels = kmeans.fit_predict(X)

        self.mu = kmeans.cluster_centers_

        # opcjonalnie lepsza inicjalizacja wariancji
        self.var = np.zeros((self.K, D))
        for k in range(self.K):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                self.var[k] = np.var(cluster_points, axis=0) + 1e-6
            else:
                self.var[k] = np.var(X, axis=0) + 1e-6

        self.pi = np.bincount(labels, minlength=self.K) / M
        self.var = np.ones((self.K, D))
        self.pi = np.ones(self.K) / self.K

        prev_ll = -np.inf

        for _ in range(self.max_iter):

            # ===== E-step =====
            log_resp = np.zeros((M, self.K))

            for k in range(self.K):
                log_resp[:, k] = np.log(self.pi[k]) + self._log_gaussian(X, self.mu[k], self.var[k])

            log_norm = logsumexp(log_resp, axis=1, keepdims=True)
            log_resp -= log_norm
            resp = np.exp(log_resp)

            ll = np.sum(log_norm)

            # ===== M-step =====
            Nk = resp.sum(axis=0) + 1e-12

            self.pi = Nk / M
            self.mu = (resp.T @ X) / Nk[:, None]

            for k in range(self.K):
                diff = X - self.mu[k]
                self.var[k] = (resp[:, k][:, None] * diff**2).sum(axis=0) / Nk[k]
            
            self.var += self.reg

            # convergence
            if np.abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll

    def predict(self, X):
        M = X.shape[0]
        log_resp = np.zeros((M, self.K))

        for k in range(self.K):
            log_resp[:, k] = np.log(self.pi[k]) + self._log_gaussian(X, self.mu[k], self.var[k])

        return np.argmax(log_resp, axis=1)
    

def clustering_accuracy(y_true, y_pred, K):
    cost = np.zeros((K, K))

    for i in range(K):
        for j in range(K):
            cost[i, j] = np.sum((y_true == i) & (y_pred == j))

    row_ind, col_ind = linear_sum_assignment(-cost)
    return cost[row_ind, col_ind].sum() / len(y_true)

def graph_fourier_features(graph, X):
    U = graph.eigenvectors
    return (U.T @ X).T  # (M, N)

def gmm_mixed_signal_experiment(
    graph,
    psd_s,
    probs,
    M=500,
    p = 1,
    n_runs=10,
    seed=42
):
    np.random.seed(seed)
    accs = []
    K =len(psd_s)
    for _ in range(n_runs):

        # ===== GENERACJA =====
        X, Y, labels = generate_mixed_signals(graph, M, p = p, psd_s = psd_s, probs=probs)
        if p < 1:
            Y = reconstruct_smooth(graph=graph, y=Y, beta=0.0001)

        X_feat = graph_fourier_features(graph, Y)

        # ===== GMM =====
        gmm = GMM_Diag(K)
        gmm.fit(X_feat)
        pred = gmm.predict(X_feat)
        #print(np.bincount(pred))
        # ===== EVAL =====
        acc = clustering_accuracy(labels, pred, K)
        accs.append(acc)

    return {
        "mean_acc": np.mean(accs),
        "std_acc": np.std(accs),
        "all_accs": accs
    }