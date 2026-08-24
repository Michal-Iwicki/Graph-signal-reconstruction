"""Build a movie graph and user graph-signals from explicit ratings.

Each movie is a graph vertex and each user is a separate graph signal. The
value of a signal at a vertex is that user's rating of the corresponding
movie. Missing ratings remain ``NaN``; they are replaced by zero only while
computing cosine similarities, where zero means "no contribution".

The default column names match MovieLens ``ratings.csv``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Hashable

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.methods.generation import GSPGraph


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RATINGS_PATH = PROJECT_ROOT / "data" / "movie" / "ml-latest-small" / "ratings.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "movie" / "processed"


@dataclass(frozen=True)
class MovieRatingData:
    """Movie graph and aligned user signals.

    ``movie_ids[i]`` identifies row ``i`` of ``signals`` and ``user_ids[j]``
    identifies column ``j``.  Graph nodes use the original movie identifiers;
    ``graph.node_order`` therefore has the same order as ``movie_ids``.
    """

    graph: GSPGraph
    signals: np.ndarray
    observed: np.ndarray
    movie_ids: tuple[Hashable, ...]
    user_ids: tuple[Hashable, ...]


@dataclass(frozen=True)
class PreparedMovieRatingData:
    """Movie-rating data loaded from the two prepared CSV files."""

    data: MovieRatingData


def save_movie_rating_data(
    data: MovieRatingData,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Save only the aligned signal and adjacency matrices as readable CSV."""
    if not isinstance(data, MovieRatingData):
        raise TypeError("data must be a MovieRatingData instance.")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    if tuple(data.graph.node_order) != tuple(data.movie_ids):
        raise ValueError("graph.node_order must match movie_ids.")
    if data.signals.shape != (len(data.movie_ids), len(data.user_ids)):
        raise ValueError("signals shape does not match movie_ids and user_ids.")
    if data.observed.shape != data.signals.shape:
        raise ValueError("observed shape must match signals.")

    adjacency = nx.to_numpy_array(
        data.graph,
        nodelist=data.graph.node_order,
        weight="weight",
        dtype=float,
    )
    paths = {
        "signals": directory / "signals.csv",
        "adjacency": directory / "adjacency.csv",
    }
    pd.DataFrame(
        data.signals, index=pd.Index(data.movie_ids, name="movieId"), columns=data.user_ids
    ).to_csv(paths["signals"], na_rep="")
    pd.DataFrame(
        adjacency,
        index=pd.Index(data.movie_ids, name="movieId"),
        columns=data.movie_ids,
    ).to_csv(paths["adjacency"])
    return paths


def load_prepared_movie_rating_data(
    input_dir: str | Path,
) -> PreparedMovieRatingData:
    """Load and validate a dataset written by :func:`prepare_movie_rating_data`."""
    directory = Path(input_dir)
    required = ("signals.csv", "adjacency.csv")
    missing = [name for name in required if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing prepared-data files in {directory}: {missing}")
    signal_frame = pd.read_csv(directory / "signals.csv", index_col="movieId")
    adjacency_frame = pd.read_csv(directory / "adjacency.csv", index_col="movieId")
    signals = signal_frame.to_numpy(dtype=float)
    adjacency = adjacency_frame.to_numpy(dtype=float)
    movie_ids = tuple(signal_frame.index.tolist())
    user_ids = tuple(signal_frame.columns.tolist())
    if adjacency.shape != (len(movie_ids), len(movie_ids)):
        raise ValueError("Prepared adjacency dimensions do not match movie IDs.")
    graph = nx.from_numpy_array(adjacency)
    graph = nx.relabel_nodes(graph, dict(enumerate(movie_ids)))
    data = MovieRatingData(
        graph=GSPGraph(graph), signals=signals, observed=np.isfinite(signals),
        movie_ids=movie_ids, user_ids=user_ids,
    )
    return PreparedMovieRatingData(data)


def _filter_ratings(
    ratings: pd.DataFrame,
    user_col: str,
    movie_col: str,
    min_user_ratings: int,
    min_movie_ratings: int,
) -> pd.DataFrame:
    """Iteratively remove sparse users/movies until both limits hold."""
    filtered = ratings
    while True:
        previous_size = len(filtered)
        user_counts = filtered.groupby(user_col)[movie_col].transform("size")
        filtered = filtered[user_counts >= min_user_ratings]
        movie_counts = filtered.groupby(movie_col)[user_col].transform("size")
        filtered = filtered[movie_counts >= min_movie_ratings]
        if len(filtered) == previous_size:
            return filtered


def build_movie_similarity_graph(
    signals: np.ndarray,
    movie_ids: tuple[Hashable, ...] | list[Hashable],
    *,
    n_neighbors: int = 20,
    min_similarity: float = 0.0,
    similarity_shrinkage: float = 10.0,
) -> GSPGraph:
    """Create a weighted, undirected cosine-similarity k-NN movie graph.

    Cosine similarity is computed from zero-filled rating vectors.  A
    significance weight ``n_common / (n_common + similarity_shrinkage)``
    reduces unreliable similarities based on very few co-rated users.
    Finally, each movie retains its strongest ``n_neighbors`` positive links
    and directed choices are symmetrized by taking their maximum.
    """
    values = np.asarray(signals, dtype=float)
    movies = tuple(movie_ids)
    if values.ndim != 2 or values.shape[0] != len(movies):
        raise ValueError("signals rows must match movie_ids.")
    if len(movies) < 2:
        raise ValueError("At least two movies are required to build a graph.")
    if not isinstance(n_neighbors, int) or isinstance(n_neighbors, bool) or n_neighbors < 1:
        raise ValueError("n_neighbors must be a positive integer.")
    if not 0.0 <= min_similarity <= 1.0:
        raise ValueError("min_similarity must lie in [0, 1].")
    if similarity_shrinkage < 0 or not np.isfinite(similarity_shrinkage):
        raise ValueError("similarity_shrinkage must be finite and non-negative.")

    observed = np.isfinite(values)
    if np.any(observed.sum(axis=1) == 0):
        raise ValueError("Every movie must have at least one finite rating.")
    filled = np.where(observed, values, 0.0)
    similarity = cosine_similarity(filled)
    common = observed.astype(np.float64) @ observed.astype(np.float64).T
    if similarity_shrinkage:
        similarity *= common / (common + similarity_shrinkage)
    np.fill_diagonal(similarity, 0.0)
    similarity[similarity < min_similarity] = 0.0

    graph = nx.Graph()
    graph.add_nodes_from(movies)
    k = min(n_neighbors, len(movies) - 1)
    for row, movie_id in enumerate(movies):
        candidates = np.argpartition(similarity[row], -k)[-k:]
        for column in candidates:
            weight = float(similarity[row, column])
            if weight <= 0.0:
                continue
            other_id = movies[int(column)]
            old_weight = graph.get_edge_data(movie_id, other_id, {}).get("weight", 0.0)
            if weight > old_weight:
                graph.add_edge(movie_id, other_id, weight=weight)

    if graph.number_of_edges() == 0:
        raise ValueError(
            "No positive movie similarities found; relax filtering or the "
            "min_similarity threshold."
        )
    return GSPGraph(graph)


def build_movie_rating_data(
    ratings: str | Path | pd.DataFrame,
    *,
    user_col: str = "userId",
    movie_col: str = "movieId",
    rating_col: str = "rating",
    min_user_ratings: int = 5,
    min_movie_ratings: int = 5,
    n_neighbors: int = 20,
    min_similarity: float = 0.0,
    similarity_shrinkage: float = 10.0,
) -> MovieRatingData:
    """Load ratings and return aligned movie vertices and user signals."""
    for name, value in (
        ("min_user_ratings", min_user_ratings),
        ("min_movie_ratings", min_movie_ratings),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")

    frame = pd.read_csv(ratings) if isinstance(ratings, (str, Path)) else ratings.copy()
    required = {user_col, movie_col, rating_col}
    missing_columns = required.difference(frame.columns)
    if missing_columns:
        raise ValueError(f"Missing rating columns: {sorted(missing_columns)}")
    frame = frame.loc[:, [user_col, movie_col, rating_col]].dropna()
    frame[rating_col] = pd.to_numeric(frame[rating_col], errors="coerce")
    frame = frame.dropna(subset=[rating_col])
    if frame.empty or not np.all(np.isfinite(frame[rating_col])):
        raise ValueError("Ratings must contain finite numeric values.")

    frame = _filter_ratings(
        frame, user_col, movie_col, min_user_ratings, min_movie_ratings
    )
    if frame.empty:
        raise ValueError("No ratings remain after minimum-count filtering.")
        
    # UWAGA ZMIANA TUTAJ: Wiersze to filmy, kolumny to użytkownicy
    matrix = frame.pivot_table(
        index=movie_col, columns=user_col, values=rating_col, aggfunc="mean", sort=True
    )
    if matrix.shape[0] < 2:
        raise ValueError("At least two movies must remain after filtering.")

    movie_ids = tuple(matrix.index.tolist())
    user_ids = tuple(matrix.columns.tolist())
    signals = matrix.to_numpy(dtype=float)
    graph = build_movie_similarity_graph(
        signals,
        movie_ids,
        n_neighbors=n_neighbors,
        min_similarity=min_similarity,
        similarity_shrinkage=similarity_shrinkage,
    )
    return MovieRatingData(
        graph=graph,
        signals=signals,
        observed=np.isfinite(signals),
        movie_ids=movie_ids,
        user_ids=user_ids,
    )


def load_rating_submatrix(
    path: str | Path,
    *,
    n_users: int,
    n_movies: int,
    min_movie_ratings: int,
    chunksize: int = 1_000_000,
) -> pd.DataFrame:
    """Stream MovieLens twice and select active users and popular movies."""
    for name, value, minimum in (
        ("n_users", n_users, 2),
        ("n_movies", n_movies, 2),
        ("min_movie_ratings", min_movie_ratings, 1),
        ("chunksize", chunksize, 1),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}.")
    ratings_path = Path(path)
    if not ratings_path.is_file():
        raise FileNotFoundError(f"Ratings CSV not found: {ratings_path}")
    columns = ["userId", "movieId", "rating"]
    counts: pd.Series | None = None
    for chunk in pd.read_csv(ratings_path, usecols=columns, chunksize=chunksize):
        part = chunk.groupby("userId", sort=False).size()
        counts = part if counts is None else counts.add(part, fill_value=0)
    if counts is None or len(counts) < n_users:
        raise ValueError(f"Fewer than {n_users} users are available.")
    selected_users = set(
        counts.rename("count").reset_index()
        .sort_values(["count", "userId"], ascending=[False, True])
        .head(n_users)["userId"]
    )
    parts = []
    for chunk in pd.read_csv(ratings_path, usecols=columns, chunksize=chunksize):
        selected = chunk[chunk["userId"].isin(selected_users)]
        if not selected.empty:
            parts.append(selected)
    ratings = pd.concat(parts, ignore_index=True)
    movie_counts = ratings.groupby("movieId").size()
    eligible = movie_counts[movie_counts >= min_movie_ratings]
    if len(eligible) < n_movies:
        raise ValueError(
            f"Only {len(eligible)} movies meet min_movie_ratings; requested {n_movies}."
        )
    selected_movies = set(
        eligible.rename("count").reset_index()
        .sort_values(["count", "movieId"], ascending=[False, True])
        .head(n_movies)["movieId"]
    )
    matrix = ratings[ratings["movieId"].isin(selected_movies)].pivot_table(
        index="movieId", columns="userId", values="rating", aggfunc="mean", sort=True
    )
    if matrix.shape != (n_movies, n_users):
        raise ValueError(
            f"Selected matrix has shape {matrix.shape}; expected ({n_movies}, {n_users})."
        )
    return matrix


def prepare_movie_rating_data(
    ratings_path: str | Path,
    output_dir: str | Path,
    *,
    n_movies: int = 1000,
    n_users: int = 200,
    min_movie_ratings: int = 20,
    n_neighbors: int = 20,
    min_similarity: float = 0.0,
    similarity_shrinkage: float = 10.0,
    chunksize: int = 1_000_000,
) -> dict[str, Path]:
    """Build and persist the complete input package for the experiment."""
    matrix = load_rating_submatrix(
        ratings_path, n_users=n_users, n_movies=n_movies,
        min_movie_ratings=min_movie_ratings, chunksize=chunksize,
    )
    signals = matrix.to_numpy(dtype=float)
    graph = build_movie_similarity_graph(
        signals, tuple(matrix.index.tolist()),
        n_neighbors=n_neighbors, min_similarity=min_similarity,
        similarity_shrinkage=similarity_shrinkage,
    )
    data = MovieRatingData(
        graph=graph, signals=signals, observed=np.isfinite(signals),
        movie_ids=tuple(matrix.index.tolist()), user_ids=tuple(matrix.columns.tolist()),
    )
    return save_movie_rating_data(data, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratings-path", default=str(DEFAULT_RATINGS_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--n-movies", type=int, default=1000)
    parser.add_argument("--n-users", type=int, default=200)
    parser.add_argument("--min-movie-ratings", type=int, default=20)
    parser.add_argument("--n-neighbors", type=int, default=20)
    parser.add_argument("--min-similarity", type=float, default=0.0)
    parser.add_argument("--similarity-shrinkage", type=float, default=10.0)
    parser.add_argument("--chunksize", type=int, default=1_000_000)
    args = parser.parse_args()
    paths = prepare_movie_rating_data(
        args.ratings_path, args.output_dir, n_movies=args.n_movies,
        n_users=args.n_users, min_movie_ratings=args.min_movie_ratings,
        n_neighbors=args.n_neighbors,
        min_similarity=args.min_similarity,
        similarity_shrinkage=args.similarity_shrinkage,
        chunksize=args.chunksize,
    )
    for path in paths.values():
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
