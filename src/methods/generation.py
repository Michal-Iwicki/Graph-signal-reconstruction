"""Graph construction, synthetic signal generation, and visualization."""

from collections.abc import Callable, Sequence

import networkx as nx
import numpy as np
import sklearn.neighbors as skn


class GSPGraph(nx.Graph):
    """Undirected graph with a cached Laplacian eigendecomposition.

    Parameters
    ----------
    graph:
        Non-empty NetworkX graph. Directed graphs are converted to undirected
        graphs because the reconstruction methods require a symmetric
        Laplacian.

    Attributes
    ----------
    L:
        Dense combinatorial Laplacian matrix.
    eigenvalues, eigenvectors:
        Eigenpairs of ``L`` in ascending eigenvalue order.
    node_order:
        Node order corresponding to rows of graph signals and ``L``.
    """

    def __init__(self, graph: nx.Graph):
        if not isinstance(graph, nx.Graph):
            raise TypeError("graph must be a NetworkX graph.")
        if graph.number_of_nodes() == 0:
            raise ValueError("graph must contain at least one node.")

        source = graph.to_undirected() if graph.is_directed() else graph
        super().__init__()
        self.graph.update(source.graph)
        self.add_nodes_from(source.nodes(data=True))
        self.add_edges_from(source.edges(data=True))

        self.node_order = tuple(self.nodes)
        self.L = nx.laplacian_matrix(
            self, nodelist=self.node_order, weight="weight"
        ).toarray().astype(float)
        self.eigenvalues, self.eigenvectors = np.linalg.eigh(self.L)


class GraphFactory:
    """Factory for graph topologies used in synthetic experiments."""

    @staticmethod
    def _validate_node_count(N: int, *, minimum: int = 2) -> int:
        """Validate and return a graph node count."""
        if (
            not isinstance(N, (int, np.integer))
            or isinstance(N, bool)
            or N < minimum
        ):
            raise ValueError(
                f"N must be an integer greater than or equal to {minimum}."
            )
        return int(N)

    @staticmethod
    def generate_nn_graph(N: int = 500, k: int = 40) -> GSPGraph:
        """Generate an undirected two-dimensional k-nearest-neighbor graph.

        Parameters
        ----------
        N:
            Number of graph nodes. Must be at least two.
        k:
            Number of nearest neighbors queried for each node. Must satisfy
            ``1 <= k < N``.

        Returns
        -------
        GSPGraph
            Graph whose nodes contain their two-dimensional coordinates in the
            ``"pos"`` attribute.
        """
        N = GraphFactory._validate_node_count(N)
        if not isinstance(k, (int, np.integer)) or isinstance(k, bool) or not 1 <= k < N:
            raise ValueError("k must be an integer satisfying 1 <= k < N.")

        coordinates = np.random.random((N, 2))
        adjacency = skn.kneighbors_graph(
            coordinates,
            k,
            mode="connectivity",
            include_self=False,
        )
        adjacency = adjacency.maximum(adjacency.T)
        graph = nx.from_scipy_sparse_array(adjacency)
        nx.set_node_attributes(
            graph,
            {index: tuple(position) for index, position in enumerate(coordinates)},
            "pos",
        )
        return GSPGraph(graph)

    @staticmethod
    def generate_grid_graph(rows: int = 10, columns: int = 10) -> GSPGraph:
        """Generate a regular two-dimensional grid graph.

        Parameters
        ----------
        rows, columns:
            Positive grid dimensions.
        """
        for name, value in (("rows", rows), ("columns", columns)):
            if (
                not isinstance(value, (int, np.integer))
                or isinstance(value, bool)
                or value < 1
            ):
                raise ValueError(f"{name} must be a positive integer.")
        if rows * columns < 2:
            raise ValueError("The grid must contain at least two nodes.")

        graph = nx.grid_2d_graph(int(rows), int(columns))
        nx.set_node_attributes(
            graph,
            {node: (float(node[1]), float(-node[0])) for node in graph.nodes},
            "pos",
        )
        return GSPGraph(graph)

    @staticmethod
    def generate_erdos_renyi_graph(
        N: int = 100,
        edge_probability: float = 0.08,
        seed: int | None = None,
        max_attempts: int = 100,
    ) -> GSPGraph:
        """Generate a connected Erdős–Rényi graph by rejection sampling."""
        N = GraphFactory._validate_node_count(N)
        if (
            not np.isscalar(edge_probability)
            or not np.isfinite(edge_probability)
            or not 0 < edge_probability <= 1
        ):
            raise ValueError("edge_probability must lie in the interval (0, 1].")
        if (
            not isinstance(max_attempts, (int, np.integer))
            or isinstance(max_attempts, bool)
            or max_attempts < 1
        ):
            raise ValueError("max_attempts must be a positive integer.")

        random_generator = np.random.default_rng(seed)
        for _ in range(max_attempts):
            graph_seed = int(
                random_generator.integers(0, np.iinfo(np.int32).max)
            )
            graph = nx.erdos_renyi_graph(
                N,
                edge_probability,
                seed=graph_seed,
            )
            if nx.is_connected(graph):
                return GSPGraph(graph)
        raise ValueError(
            "Could not generate a connected Erdős–Rényi graph. Increase "
            "edge_probability or max_attempts."
        )

    @staticmethod
    def generate_barabasi_albert_graph(
        N: int = 100,
        m: int = 3,
        seed: int | None = None,
    ) -> GSPGraph:
        """Generate a connected Barabási–Albert preferential-attachment graph."""
        N = GraphFactory._validate_node_count(N)
        if (
            not isinstance(m, (int, np.integer))
            or isinstance(m, bool)
            or not 1 <= m < N
        ):
            raise ValueError("m must be an integer satisfying 1 <= m < N.")
        return GSPGraph(nx.barabasi_albert_graph(N, int(m), seed=seed))

    @staticmethod
    def generate_watts_strogatz_graph(
        N: int = 100,
        k: int = 6,
        rewiring_probability: float = 0.1,
        seed: int | None = None,
        max_attempts: int = 100,
    ) -> GSPGraph:
        """Generate a connected Watts–Strogatz small-world graph."""
        N = GraphFactory._validate_node_count(N, minimum=3)
        if (
            not isinstance(k, (int, np.integer))
            or isinstance(k, bool)
            or not 2 <= k < N
            or k % 2 != 0
        ):
            raise ValueError("k must be an even integer satisfying 2 <= k < N.")
        if (
            not np.isscalar(rewiring_probability)
            or not np.isfinite(rewiring_probability)
            or not 0 <= rewiring_probability <= 1
        ):
            raise ValueError(
                "rewiring_probability must lie in the interval [0, 1]."
            )
        if (
            not isinstance(max_attempts, (int, np.integer))
            or isinstance(max_attempts, bool)
            or max_attempts < 1
        ):
            raise ValueError("max_attempts must be a positive integer.")

        random_generator = np.random.default_rng(seed)
        for _ in range(max_attempts):
            graph_seed = int(
                random_generator.integers(0, np.iinfo(np.int32).max)
            )
            graph = nx.watts_strogatz_graph(
                N,
                int(k),
                rewiring_probability,
                seed=graph_seed,
            )
            if nx.is_connected(graph):
                return GSPGraph(graph)
        raise ValueError(
            "Could not generate a connected Watts–Strogatz graph. Increase "
            "k or max_attempts."
        )


class SignalGenerator:
    """Generate stationary graph signals from PSD profiles."""

    def __init__(self, graph: GSPGraph):
        if not isinstance(graph, GSPGraph):
            raise TypeError("graph must be a GSPGraph instance.")
        self.graph = graph

    def generate_signals(
        self,
        M: int,
        p: float,
        psd_fun: Callable[[np.ndarray, float], np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate complete and partially observed graph signals.

        Parameters
        ----------
        M:
            Number of independent signal realizations.
        p:
            Probability that a vertex value is observed. Must lie in ``[0, 1]``.
        psd_fun:
            Function accepting graph eigenvalues and the largest eigenvalue. It
            must return one finite, non-negative PSD value per eigenvalue.

        Returns
        -------
        X, Y:
            Complete signals ``X`` and partially observed signals ``Y``, both
            with shape ``(N, M)``. Missing entries in ``Y`` are represented by
            ``NaN``; observed zeros remain ordinary observations.
        """
        if not isinstance(M, (int, np.integer)) or isinstance(M, bool) or M < 1:
            raise ValueError("M must be a positive integer.")
        if not np.isscalar(p) or not np.isfinite(p) or not 0 <= p <= 1:
            raise ValueError("p must be a finite number in the interval [0, 1].")
        if not callable(psd_fun):
            raise TypeError("psd_fun must be callable.")

        eigenvalues = self.graph.eigenvalues
        lambda_max = float(eigenvalues[-1])
        gamma = np.asarray(psd_fun(eigenvalues, lambda_max), dtype=float)

        if gamma.shape != eigenvalues.shape:
            raise ValueError("psd_fun must return one value per graph eigenvalue.")
        if not np.all(np.isfinite(gamma)):
            raise ValueError("PSD values must be finite.")
        if np.any(gamma < 0):
            raise ValueError("PSD values must be non-negative.")

        white_noise = np.random.standard_normal((len(eigenvalues), M))
        spectral_coefficients = np.sqrt(gamma)[:, None] * white_noise
        X = self.graph.eigenvectors @ spectral_coefficients

        observation_mask = np.random.binomial(1, p, size=X.shape).astype(bool)
        Y = np.where(observation_mask, X, np.nan)
        return X, Y

    def generate_mixed_signals(
        self,
        M: int,
        p: float,
        psd_s: Sequence[Callable[[np.ndarray, float], np.ndarray]],
        probs: Sequence[float],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate signals from a finite mixture of stationary graph processes.

        Parameters
        ----------
        M:
            Total number of signal realizations.
        p:
            Probability that each vertex value is observed.
        psd_s:
            PSD function for every mixture component.
        probs:
            Component probabilities. Values must be non-negative and sum to one.

        Returns
        -------
        X, Y, labels:
            Complete signals, partially observed signals, and the source label
            of every signal realization.
        """
        if len(psd_s) == 0:
            raise ValueError("psd_s must contain at least one PSD function.")
        if any(not callable(profile) for profile in psd_s):
            raise TypeError("Every item in psd_s must be callable.")
        probabilities = np.asarray(probs, dtype=float)
        if probabilities.ndim != 1 or len(probabilities) != len(psd_s):
            raise ValueError("probs must contain one value per PSD function.")
        if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0):
            raise ValueError("probs must contain finite, non-negative values.")
        if not np.isclose(probabilities.sum(), 1.0):
            raise ValueError("probs must sum to one.")

        # Reuse single-source validation for M, p, and individual PSD profiles.
        if not isinstance(M, (int, np.integer)) or isinstance(M, bool) or M < 1:
            raise ValueError("M must be a positive integer.")
        if not np.isscalar(p) or not np.isfinite(p) or not 0 <= p <= 1:
            raise ValueError("p must be a finite number in the interval [0, 1].")

        source_indices = np.random.choice(len(psd_s), size=M, p=probabilities)
        counts = np.bincount(source_indices, minlength=len(psd_s))
        complete_groups = []
        observed_groups = []
        labels = []

        for component, (psd_fun, count) in enumerate(zip(psd_s, counts)):
            if count == 0:
                continue
            X_component, Y_component = self.generate_signals(count, p, psd_fun)
            complete_groups.append(X_component)
            observed_groups.append(Y_component)
            labels.extend([component] * count)

        X = np.hstack(complete_groups)
        Y = np.hstack(observed_groups)
        labels_array = np.asarray(labels, dtype=int)

        permutation = np.random.permutation(M)
        return X[:, permutation], Y[:, permutation], labels_array[permutation]


