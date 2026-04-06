import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import sklearn.neighbors as skn

class GSPGraph(nx.Graph):
    """
    A NetworkX Graph extension with cached Laplacian eigendecomposition.
    Used for spectral filtering and GSP operations.
    """
    def __init__(self, graph: nx.Graph):
        super().__init__()
        self.add_nodes_from(graph.nodes(data=True))
        self.add_edges_from(graph.edges(data=True))
        self.L = nx.laplacian_matrix(self).toarray()
        self.eigenvalues, self.eigenvectors = np.linalg.eigh(self.L)

class GraphFactory:
    @staticmethod
    def generate_nn_graph(N=500, k=40) -> GSPGraph:
        """Generates a K-Nearest Neighbors graph based on random 2D coordinates."""
        coords = np.random.rand(N, 2)
        A = skn.kneighbors_graph(coords, k, mode='connectivity', include_self=False)
        G = nx.from_scipy_sparse_array(A)
        return GSPGraph(G)

class SignalGenerator:
    """Handles generation of signals on a given GSPGraph."""
    def __init__(self, graph: GSPGraph):
        self.graph = graph

    def generate_signals(self, M: int, p: float, psd_fun):
        eigvals = self.graph.eigenvalues
        U = self.graph.eigenvectors
        lmax = eigvals.max()
        gamma = psd_fun(eigvals, lmax)

        z = np.random.randn(len(eigvals), M)
        coeffs = np.sqrt(gamma)[:, None] * z
        X = U @ coeffs

        Mmask = np.random.binomial(1, p, size=X.shape)
        Y = X * Mmask
        return X, Y

    def generate_mixed_signals(self, M: int, p: float, psd_s: list, probs: list):
        indices = np.random.choice(len(psd_s), size=M, p=probs)
        counts = np.bincount(indices, minlength=len(psd_s))

        all_X, all_Y, labels = [], [], []

        for i, psd_fun in enumerate(psd_s):
            sub_M = counts[i]
            if sub_M > 0:
                X_sub, Y_sub = self.generate_signals(sub_M, p, psd_fun)
                all_X.append(X_sub)
                all_Y.append(Y_sub)
                labels.extend([i] * sub_M)

        X = np.hstack(all_X)
        Y = np.hstack(all_Y)
        labels = np.array(labels)
        
        perm = np.random.permutation(X.shape[1])
        return X[:, perm], Y[:, perm], labels[perm]

class GSPVisualizer:
    """Static utility class for GSP visualizations."""
    @staticmethod
    def draw_psd(psd, graph: GSPGraph, l_max: float = 20):
        n_points = 100 * int(l_max)
        x = np.linspace(0, l_max, n_points)
        y = psd(x)

        plt.figure()
        plt.plot(x, y, label="Continuous PSD")
        eigvals = graph.eigenvalues
        plt.scatter(eigvals, psd(eigvals), color="red", s=10, label="Graph Eigenvalues")
        plt.xlabel("Graph Frequency (λ)")
        plt.ylabel("Power Density")
        plt.xlim(0, l_max)
        plt.title("Power Spectral Density Profile")
        plt.legend()
        plt.show()

    @staticmethod
    def draw_signal(graph: GSPGraph, signal: list):
        pos = nx.spring_layout(graph)
        fig, ax = plt.subplots()
        nodes = nx.draw_networkx_nodes(
            graph, pos, node_color=signal, cmap=plt.cm.viridis, node_size=300, ax=ax
        )
        nx.draw_networkx_edges(graph, pos, alpha=0.3, ax=ax)
        cbar = plt.colorbar(nodes, ax=ax)
        cbar.set_label("Signal Value")
        plt.axis("off")
        plt.show()