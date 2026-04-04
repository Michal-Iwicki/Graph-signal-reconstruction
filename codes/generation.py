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

        # Compute the combinatorial Laplacian matrix
        self.L = nx.laplacian_matrix(self).toarray()
        # Compute eigenvalues and eigenvectors (sorted by default for eigh)
        self.eigenvalues, self.eigenvectors = np.linalg.eigh(self.L)

def generate_nn_graph(N=500, k=40):
    """
    Generates a K-Nearest Neighbors graph based on random 2D coordinates.
    """
    coords = np.random.rand(N, 2)
    A = skn.kneighbors_graph(coords, k, mode='connectivity', include_self=False)
    G = nx.from_scipy_sparse_array(A)

    return GSPGraph(G)

def generate_signals(graph: GSPGraph, M: int, p: float, psd_fun):
    """
    Generates M graph signals filtered by a specific PSD function.
    
    Parameters:
        graph: GSPGraph object.
        M: Number of signal realizations.
        p: Sampling probability (0 to 1).
        psd_fun: Function defining the Power Spectral Density.
    """
    eigvals = graph.eigenvalues
    U = graph.eigenvectors

    lmax = eigvals.max()
    gamma = psd_fun(eigvals, lmax)

    # Generate white noise in the spectral domain
    z = np.random.randn(len(eigvals), M)
    # Apply the spectral filter (PSD)
    coeffs = np.sqrt(gamma)[:, None] * z

    # Transform back to vertex domain
    X = U @ coeffs

    # Apply random sampling mask
    Mmask = np.random.binomial(1, p, size=X.shape)
    Y = X * Mmask

    return X, Y

def generate_mixed_signals(graph: GSPGraph, M: int, p: float, psd_s: list, probs: list):
    """
    Generates signals from a mixture of different PSD functions.
    
    Returns:
        X: Full signal matrix (N, M).
        Y: Observed (sampled) signal matrix (N, M).
        labels: Vector (M,) containing the PSD function indices for each signal.
    """
    # Randomly assign each of the M samples to a specific PSD function
    indices = np.random.choice(len(psd_s), size=M, p=probs)
    counts = np.bincount(indices, minlength=len(psd_s))

    all_X = []
    all_Y = []
    labels = []

    for i, psd_fun in enumerate(psd_s):
        sub_M = counts[i]
        if sub_M > 0:
            # Generate sub_M signals for the current PSD function
            X_sub, Y_sub = generate_signals(graph, sub_M, p, psd_fun)
            
            all_X.append(X_sub)
            all_Y.append(Y_sub)
            labels.extend([i] * sub_M)

    # Concatenate along the sample axis (columns)
    X = np.hstack(all_X)
    Y = np.hstack(all_Y)
    labels = np.array(labels)
    
    perm = np.random.permutation(X.shape[1])

    X = X[:, perm]
    Y = Y[:, perm]
    labels = labels[perm]

    return X, Y, labels

def draw_psd(psd, graph: GSPGraph, l_max: float = 20):
    """
    Plots the continuous PSD function and optionally the discrete graph frequencies.
    """
    n_points = 100 * l_max
    x = np.linspace(0, l_max, n_points)
    y = psd(x)

    plt.figure()
    plt.plot(x, y, label="Continuous PSD")
    eigvals = graph.eigenvalues
    y_points = psd(eigvals)
    plt.scatter(eigvals, y_points, color="red", s=10, label="Graph Eigenvalues")

    plt.xlabel("Graph Frequency (λ)")
    plt.ylabel("Power Density")
    plt.xlim(0, l_max)
    plt.title("Power Spectral Density Profile")
    plt.legend()
    plt.show()

def draw_signal(graph: GSPGraph, signal: list):
    """
    Visualizes a single graph signal on the node layout.
    """
    pos = nx.spring_layout(graph)
    fig, ax = plt.subplots()

    nodes = nx.draw_networkx_nodes(
        graph,
        pos,
        node_color=signal,
        cmap=plt.cm.viridis,
        node_size=300,
        ax=ax
    )

    nx.draw_networkx_edges(graph, pos, alpha=0.3, ax=ax)
    cbar = plt.colorbar(nodes, ax=ax)
    cbar.set_label("Signal Value")

    plt.axis("off")
    plt.show()