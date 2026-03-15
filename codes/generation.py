import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


def generate_graph(nodes, edges, gen_function=nx.barabasi_albert_graph, draw=True):
    G = gen_function(n = nodes, m = edges)

    if draw:
        nx.draw(G, with_labels=True)

    return G


def generate_signal(graph, psd=lambda x: np.cos(x), n_samples=1):
    L = nx.laplacian_matrix(graph).toarray()
    eigvals, eigvecs = np.linalg.eigh(L)

    gamma = psd(eigvals)

    gamma = np.maximum(gamma, 0)

    z = np.random.randn(len(eigvals), n_samples)

    coeffs = np.sqrt(gamma)[:, None] * z

    signals = eigvecs @ coeffs

    if n_samples == 1:
        return signals[:, 0]

    return signals

def draw_psd(psd, graph=None, l_max = 20):
    n_points=100*l_max
    xlim=(0,l_max)
    x = np.linspace(xlim[0], xlim[1], n_points)
    y = psd(x)

    plt.figure()
    plt.plot(x, y, label="PSD")

    if graph is not None:
        L = nx.laplacian_matrix(graph).toarray()
        eigvals = np.linalg.eigvalsh(L)

        y_points = psd(eigvals)

        plt.scatter(eigvals, y_points, color="red", label="Graph frequencies")

    plt.xlabel("Graph frequency (eigenvalue)")
    plt.ylabel("PSD")
    plt.xlim(0,l_max)
    plt.title("Power Spectral Density")
    plt.legend()
    plt.show()

def draw_signal(G, signal):
    
    pos = nx.spring_layout(G)

    fig, ax = plt.subplots()

    nodes = nx.draw_networkx_nodes(
        G,
        pos,
        node_color=signal,
        cmap=plt.cm.viridis,
        node_size=300,
        ax=ax
    )

    nx.draw_networkx_edges(G, pos, ax=ax)

    cbar = plt.colorbar(nodes, ax=ax)
    cbar.set_label("Signal value")

    plt.axis("off")
    plt.show()