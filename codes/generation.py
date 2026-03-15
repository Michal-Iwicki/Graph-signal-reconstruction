import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


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
        L = nx.laplacian_matrix(self).toarray()
        self.eigenvalues, self.eigenvectors = np.linalg.eigh(L)


def generate_graph(nodes, edges, gen_function=nx.barabasi_albert_graph, draw=True):
    """
    Generate a graph and convert it to GSPGraph.
    """

    G = gen_function(n=nodes, m=edges)
    G = GSPGraph(G)

    if draw:
        nx.draw(G, with_labels=True)

    return G


def generate_signal(graph, psd=lambda x: np.cos(x), n_samples=1, p=1.0):
    """
    Generate graph signals with a given Power Spectral Density.

    Signal model:
        x = U sqrt(gamma) z

    where
        U     - eigenvectors of Laplacian
        gamma - PSD evaluated on eigenvalues
        z     - Gaussian random coefficients
    """

    eigvals = graph.eigenvalues
    eigvecs = graph.eigenvectors

    # Evaluate PSD on graph frequencies
    gamma = psd(eigvals)
    gamma = np.maximum(gamma, 0)

    # Random spectral coefficients
    z = np.random.randn(len(eigvals), n_samples)
    coeffs = np.sqrt(gamma)[:, None] * z

    # Transform to vertex domain
    signals = eigvecs @ coeffs

    # Bernoulli sampling mask
    mask = np.random.binomial(1, p, size=signals.shape)

    if n_samples == 1:
        return signals[:, 0], mask[:, 0]

    return signals, mask


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