"""Compare reconstruction methods on explicit movie ratings.

Movies are graph vertices and users are graph signals. The prepared movie
graph uses MFS similarity. Native missing ratings are never used as ground
truth: metrics cover only known test ratings that were deliberately hidden by
the experiment.

Run from the repository root::

    python -m src.real_data_experiments.movie_ratings_experiment
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from src.experiments._helper import evaluate_reconstruction_methods
from src.methods.models import MixedSignalReconstruction
from src.real_data_experiments.movie_ratings import (
    load_prepared_movie_rating_data,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results" / "movie_ratings"
DEFAULT_PREPARED_DATA_DIR = PROJECT_ROOT / "data" / "movie" / "processed"


@dataclass(frozen=True)
class MovieRatingsConfig:
    """Parameters of the movie-rating reconstruction experiment."""

    data_dir: str = str(DEFAULT_PREPARED_DATA_DIR)
    output_dir: str = str(DEFAULT_RESULTS_DIR)
    train_fraction: float = 0.8
    missing_rates: tuple[float, ...] = (0.2, 0.5, 0.8)
    min_components: int = 2
    max_components: int = 15
    n_runs: int = 3
    seed: int = 42
    alpha: float = 10.0
    psd_beta: float = 0.75
    smooth_beta: float = 0.75
    standardize: bool = True


def mask_known_ratings(
    signals: np.ndarray,
    p_observed: float,
    random_scores: np.ndarray,
) -> np.ndarray:
    """Keep an exact rounded fraction of each user's (signal's) known ratings."""
    values = np.asarray(signals, dtype=float)
    scores = np.asarray(random_scores, dtype=float)
    if values.ndim != 2 or scores.shape != values.shape or not np.all(np.isfinite(scores)):
        raise ValueError("signals must be a matrix and random_scores must be finite and aligned.")
    if not 0.0 < p_observed <= 1.0:
        raise ValueError("p_observed must lie in (0, 1].")
    masked = np.full(values.shape, np.nan, dtype=float)
    for column in range(values.shape[1]):
        known = np.flatnonzero(np.isfinite(values[:, column]))
        if known.size == 0:
            raise ValueError(f"User column {column} has no known ratings.")
        n_visible = max(1, int(round(p_observed * known.size)))
        visible = known[np.argsort(scores[known, column])[:n_visible]]
        masked[visible, column] = values[visible, column]
    return masked


def _fit_movie_scaling(train_observed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Estimate movie means/scales from visible training ratings only."""
    observed = np.isfinite(train_observed)
    global_values = train_observed[observed]
    if global_values.size == 0:
        raise ValueError("Training data contain no visible ratings.")
    global_center = float(np.mean(global_values))
    global_scale = float(np.std(global_values))
    counts = observed.sum(axis=1, keepdims=True)
    sums = np.nansum(train_observed, axis=1, keepdims=True)
    center = np.divide(
        sums, counts, out=np.full_like(sums, global_center, dtype=float), where=counts > 0
    )
    squared = np.where(observed, (train_observed - center) ** 2, 0.0).sum(
        axis=1, keepdims=True
    )
    scale = np.sqrt(
        np.divide(
            squared,
            counts,
            out=np.full_like(squared, global_scale**2, dtype=float),
            where=counts > 0,
        )
    )
    scale[~np.isfinite(scale) | (scale < 1e-8)] = 1.0
    return center, scale


def _hidden_errors(
    truth: np.ndarray, estimate: np.ndarray, observed: np.ndarray
) -> np.ndarray:
    artificially_hidden = np.isfinite(truth) & ~np.isfinite(observed)
    if not np.any(artificially_hidden):
        raise ValueError("No known test ratings were hidden.")
    return truth[artificially_hidden] - estimate[artificially_hidden]


def run_experiment(config: MovieRatingsConfig) -> tuple[pd.DataFrame, dict]:
    """Load prepared CSV data and run repeated masks over a seeded user split."""
    if not 0.0 < config.train_fraction < 1.0:
        raise ValueError("train_fraction must lie in (0, 1).")
    if config.min_components < 1 or config.max_components < config.min_components:
        raise ValueError("Require 1 <= min_components <= max_components.")
    missing_rates = tuple(float(rate) for rate in config.missing_rates)
    if not missing_rates or any(
        not np.isfinite(rate) or not 0.0 < rate < 1.0
        for rate in missing_rates
    ):
        raise ValueError("Every missing rate must lie in (0, 1).")

    prepared = load_prepared_movie_rating_data(config.data_dir)
    matrix_data = prepared.data
    signals = matrix_data.signals
    user_order = np.random.default_rng(config.seed).permutation(signals.shape[1])
    split_at = int(config.train_fraction * signals.shape[1])
    train_indices = user_order[:split_at]
    test_indices = user_order[split_at:]
    if len(train_indices) < config.max_components:
        raise ValueError("Prepared training split is too small for max_components.")
    train, test = signals[:, train_indices], signals[:, test_indices]
    graph = matrix_data.graph
    component_candidates = list(range(config.min_components, config.max_components + 1))
    rows: list[dict] = []
    for run in range(config.n_runs):
        run_rng = np.random.default_rng(config.seed + run)
        train_scores = run_rng.random(train.shape)
        test_scores = run_rng.random(test.shape)
        for missing_rate in missing_rates:
            p_observed = 1.0 - missing_rate
            train_observed = mask_known_ratings(train, p_observed, train_scores)
            test_observed = mask_known_ratings(test, p_observed, test_scores)
            if config.standardize:
                center, scale = _fit_movie_scaling(train_observed)
            else:
                center = np.zeros((train.shape[0], 1))
                scale = np.ones((train.shape[0], 1))
            train_scaled = (train_observed - center) / scale
            test_scaled = (test_observed - center) / scale

            selector = MixedSignalReconstruction(graph)
            selector.fit_transform(
                train_scaled,
                method="clustered_reconstruction",
                K_list=component_candidates,
                alpha=config.alpha,
                beta=config.psd_beta,
                init_beta=config.smooth_beta,
                random_state=config.seed + run,
            )
            selected_components = selector.best_K_
            estimates, _, model = evaluate_reconstruction_methods(
                graph,
                train_scaled,
                test_scaled,
                selected_components,
                alpha=config.alpha,
                beta=config.psd_beta,
                smooth_beta=config.smooth_beta,
                random_state=config.seed + run,
            )
            for method, scaled_estimate in estimates.items():
                estimate = scaled_estimate * scale + center
                errors = _hidden_errors(test, estimate, test_observed)
                rows.append(
                    {
                        "run": run,
                        "missing_rate": missing_rate,
                        "method": method,
                        "mae": float(np.mean(np.abs(errors))),
                        "n_hidden_test_ratings": int(errors.size),
                        "n_train": train.shape[1],  # liczba użytkowników treningowych
                        "n_test": test.shape[1],    # liczba użytkowników testowych
                        "n_nodes": train.shape[0],  # liczba filmów
                        "p_observed": p_observed,
                        "selected_n_components": selected_components,
                        "selection_score": selector.best_score_ if method == "proposed" else np.nan,
                        "fallback_clusters": len(model.fallback_clusters) if method == "proposed" else np.nan,
                    }
                )
            print(
                f"[Movie ratings] run {run + 1}/{config.n_runs}, "
                f"missing={missing_rate:.0%}: selected K={selected_components}",
                flush=True,
            )

    train_user_ids = [matrix_data.user_ids[index] for index in train_indices]
    test_user_ids = [matrix_data.user_ids[index] for index in test_indices]
    metadata = {
        **asdict(config),
        "selected_movie_ids": list(matrix_data.movie_ids),
        "train_user_ids": train_user_ids,
        "test_user_ids": test_user_ids,
        "native_rating_density": float(np.isfinite(signals).mean()),
        "graph_edges": graph.number_of_edges(),
        "graph_connected_components": nx.number_connected_components(graph),
        "graph_source": "prepared Multi-factor similarity adjacency",
        "component_candidates": component_candidates,
        "component_selection": "minimum Yang cost on training observations",
        "metric_scope": "known test ratings hidden artificially by the experiment",
        "prepared_data_dir": str(Path(config.data_dir)),
    }
    return pd.DataFrame(rows), metadata


def save_results(
    results: pd.DataFrame, metadata: dict, output_dir: str | Path
) -> tuple[Path, Path, Path]:
    """Save per-run results, aggregate metrics, and configuration."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    raw_path = directory / "movie_ratings_runs.csv"
    summary_path = directory / "movie_ratings_summary.csv"
    config_path = directory / "movie_ratings_config.json"
    results.to_csv(raw_path, index=False, float_format="%.6f")
    summary = (
        results.groupby(["missing_rate", "method"])
        .agg(
            mae=("mae", "mean"),
            std_mae=("mae", "std"),
        )
        .reset_index()
    )
    summary.to_csv(summary_path, index=False, float_format="%.6f")
    config_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return raw_path, summary_path, config_path


def parse_args() -> MovieRatingsConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(DEFAULT_PREPARED_DATA_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--missing-rates", type=float, nargs="+", default=[0.2, 0.5, 0.8])
    parser.add_argument("--min-components", type=int, default=2)
    parser.add_argument("--max-components", type=int, default=10)
    parser.add_argument("--n-runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--psd-beta", type=float, default=0.75)
    parser.add_argument("--smooth-beta", type=float, default=0.75)
    parser.add_argument("--no-standardize", action="store_true")
    args = parser.parse_args()
    return MovieRatingsConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        train_fraction=args.train_fraction,
        missing_rates=tuple(args.missing_rates),
        min_components=args.min_components,
        max_components=args.max_components,
        n_runs=args.n_runs,
        seed=args.seed,
        alpha=args.alpha,
        psd_beta=args.psd_beta,
        smooth_beta=args.smooth_beta,
        standardize=not args.no_standardize,
    )


def main() -> None:
    config = parse_args()
    results, metadata = run_experiment(config)
    paths = save_results(results, metadata, config.output_dir)
    print("\n" + results.groupby(["missing_rate", "method"])[["mae"]].mean().to_string())
    for path in paths:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()