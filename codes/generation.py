import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import sklearn.neighbors as skn

class GSPGraph(nx.Graph):
    """
    Graph with cached Laplacian eigendecomposition.
    """

    def __init__(self, graph=None):
        super().__init__()

        if graph is not None:
            self.add_nodes_from(graph.nodes(data=True))
            self.add_edges_from(graph.edges(data=True))

        # compute Laplacian and its eigendecomposition
        self.L = nx.laplacian_matrix(self).toarray()
        self.eigenvalues, self.eigenvectors = np.linalg.eigh(self.L)

def generate_nn_graph(N=500, k=40):

    coords = np.random.rand(N, 2)
    A = skn.kneighbors_graph(coords, k, mode='connectivity', include_self=False)
    G = nx.from_scipy_sparse_array(A)

    return GSPGraph(G)

def generate_signals(graph, M, p, psd_fun):
    eigvals = graph.eigenvalues
    U = graph.eigenvectors

    lmax = eigvals.max()
    gamma = psd_fun(eigvals, lmax)

    # normalization (paper strategy)
    gamma = gamma / (np.max(gamma) + 1e-8)

    z = np.random.randn(len(eigvals), M)
    coeffs = np.sqrt(gamma)[:, None] * z

    X = U @ coeffs

    # sampling
    Mmask = np.random.binomial(1, p, size=X.shape)
    Y = X * Mmask

    return X, Y, gamma


def draw_psd(psd, graph=None, l_max=20):
    """
    Plot the PSD function and optionally graph frequencies.
    """

    n_points = 100 * l_max
    x = np.linspace(0, l_max, n_points)
    y = psd(x)

    plt.figure()
    plt.plot(x, y, label="PSD")

    if graph is not None:
        eigvals = graph.eigenvalues
        y_points = psd(eigvals)

        plt.scatter(eigvals, y_points, color="red", label="Graph frequencies")

    plt.xlabel("Graph frequency (eigenvalue)")
    plt.ylabel("PSD")
    plt.xlim(0, l_max)
    plt.title("Power Spectral Density")
    plt.legend()
    plt.show()


def draw_signal(graph, signal):
    """
    Visualize a graph signal on nodes.
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

    nx.draw_networkx_edges(graph, pos, ax=ax)

    cbar = plt.colorbar(nodes, ax=ax)
    cbar.set_label("Signal value")

    plt.axis("off")
    plt.show()