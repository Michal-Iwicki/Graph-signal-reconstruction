"""Compare the proposed, basic-PSD, and smooth methods on METR-LA.

The dataset is interpreted as a sequence of graph signals: sensors are graph
vertices and every timestamp is one signal.  METR-LA uses zero to mark a
missing measurement, so the experiment selects complete timestamps before it
adds a reproducible artificial observation mask.

Run with the repository root as the working directory::

    python -m src.real_data_experiments.metr_la
"""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from src.methods.generation import GSPGraph
from src.methods.models import MixedSignalReconstruction
from src.experiments._helper import (
    evaluate_reconstruction_methods,
    missing_mae,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "metr"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results" / "metr_la"


@dataclass(frozen=True)
class MetrLAConfig:
    """Parameters of the real-data reconstruction experiment."""

    data_dir: str = str(DEFAULT_DATA_DIR)
    output_dir: str = str(DEFAULT_RESULTS_DIR)
    n_observations: int = 1000
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


def _read_hdf_frame(path: Path) -> pd.DataFrame:
    """Read the fixed-format pandas file, including newer PyTables setups."""
    try:
        return pd.read_hdf(path)
    except (TypeError, ValueError):
        # Some recent Python/PyTables combinations do not decode old pandas
        # byte attributes.  METR-LA's fixed frame can be read losslessly from
        # its four arrays without depending on those metadata attributes.
        import tables

        with tables.open_file(path, mode="r") as store:
            group = store.root.df
            values = group.block0_values.read()
            columns = [
                value.decode() if isinstance(value, bytes) else str(value)
                for value in group.axis0.read()
            ]
            index = pd.to_datetime(group.axis1.read())
        return pd.DataFrame(values, index=index, columns=columns)


def load_metr_la(data_dir: str | Path) -> tuple[pd.DataFrame, list[str], np.ndarray]:
    """Load traffic speeds and the supplied sensor-dependency matrix."""
    directory = Path(data_dir)
    frame_path = directory / "METR-LA.h5"
    adjacency_path = directory / "adj_METR-LA.pkl"
    if not frame_path.is_file() or not adjacency_path.is_file():
        raise FileNotFoundError(
            f"Expected METR-LA.h5 and adj_METR-LA.pkl in {directory}."
        )

    frame = _read_hdf_frame(frame_path)
    with adjacency_path.open("rb") as stream:
        sensor_ids, sensor_to_index, adjacency = pickle.load(
            stream, encoding="latin1"
        )

    sensor_ids = [str(sensor_id) for sensor_id in sensor_ids]
    if len(sensor_ids) != len(sensor_to_index):
        raise ValueError("The sensor ID list and index mapping disagree.")
    if set(sensor_ids) != set(map(str, frame.columns)):
        raise ValueError("HDF columns do not match adjacency sensor IDs.")

    # The mapping is authoritative if a pickle happens to store the ID list in
    # a different order.
    ordered_ids = [None] * len(sensor_ids)
    for sensor_id, index in sensor_to_index.items():
        ordered_ids[int(index)] = str(sensor_id)
    matrix = np.asarray(adjacency, dtype=float)
    if matrix.shape != (len(ordered_ids), len(ordered_ids)):
        raise ValueError("Adjacency matrix dimensions do not match sensor IDs.")
    return frame.loc[:, ordered_ids], ordered_ids, matrix


def build_sensor_graph(
    sensor_ids: list[str], adjacency: np.ndarray
) -> tuple[GSPGraph, np.ndarray]:
    """Symmetrize supplied dependencies and retain the largest component."""
    weights = np.maximum(adjacency, adjacency.T)
    np.fill_diagonal(weights, 0.0)
    raw_graph = nx.from_numpy_array(weights)
    component = max(nx.connected_components(raw_graph), key=len)
    retained = np.asarray(sorted(component), dtype=int)
    connected = raw_graph.subgraph(retained).copy()
    connected = nx.relabel_nodes(
        connected, {index: sensor_ids[index] for index in retained}
    )
    return GSPGraph(connected), retained


def select_complete_observations(
    frame: pd.DataFrame,
    retained: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Randomly sample complete timestamps and return them in time order."""
    if count < 2:
        raise ValueError("n_observations must be at least 2.")
    values = frame.iloc[:, retained].to_numpy(dtype=float)
    complete = np.all(np.isfinite(values) & (values > 0.0), axis=1)
    available = np.flatnonzero(complete)
    if len(available) < count:
        raise ValueError(
            f"Only {len(available)} complete timestamps are available; "
            f"requested {count}."
        )
    selected = np.sort(rng.choice(available, size=count, replace=False))
    # Methods use (vertices, signals), while the HDF file uses (time, sensors).
    return values[selected].T, frame.index[selected]


def mask_entries_exactly(
    signals: np.ndarray,
    p_observed: float,
    random_scores: np.ndarray,
) -> np.ndarray:
    """Observe an exact fraction using reusable, nested random rankings."""
    if not 0.0 < p_observed <= 1.0:
        raise ValueError("p_observed must lie in (0, 1].")
    n_nodes, n_signals = signals.shape
    scores = np.asarray(random_scores, dtype=float)
    if scores.shape != signals.shape or not np.all(np.isfinite(scores)):
        raise ValueError("random_scores must be finite and match signals.")
    n_visible = max(1, int(round(p_observed * n_nodes)))
    observed = np.full(signals.shape, np.nan, dtype=float)
    for column in range(n_signals):
        rows = np.argsort(scores[:, column])[:n_visible]
        observed[rows, column] = signals[rows, column]
    return observed


def _fit_scaling(train_observed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Estimate per-sensor location and scale using observed training values."""
    observed = np.isfinite(train_observed)
    counts = observed.sum(axis=1, keepdims=True)
    global_values = train_observed[observed]
    if global_values.size == 0:
        raise ValueError("Training data contain no observed values.")
    global_center = float(np.mean(global_values))
    global_scale = float(np.std(global_values))
    sums = np.nansum(train_observed, axis=1, keepdims=True)
    center = np.divide(
        sums,
        counts,
        out=np.full_like(sums, global_center, dtype=float),
        where=counts > 0,
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


def run_experiment(config: MetrLAConfig) -> tuple[pd.DataFrame, dict]:
    """Run repeated masks over one chronological train/test split."""
    if not 0.0 < config.train_fraction < 1.0:
        raise ValueError("train_fraction must lie in (0, 1).")
    frame, sensor_ids, adjacency = load_metr_la(config.data_dir)
    graph, retained = build_sensor_graph(sensor_ids, adjacency)
    signals, timestamps = select_complete_observations(
        frame,
        retained,
        config.n_observations,
        np.random.default_rng(config.seed),
    )
    split_at = int(config.n_observations * config.train_fraction)
    if split_at < 1 or split_at >= config.n_observations:
        raise ValueError("train_fraction creates an empty train or test split.")
    train, test = signals[:, :split_at], signals[:, split_at:]
    if config.min_components < 1 or config.max_components < config.min_components:
        raise ValueError(
            "Require 1 <= min_components <= max_components."
        )
    component_candidates = list(
        range(config.min_components, config.max_components + 1)
    )
    missing_rates = tuple(float(rate) for rate in config.missing_rates)
    if not missing_rates or any(
        not np.isfinite(rate) or not 0.0 <= rate < 1.0
        for rate in missing_rates
    ):
        raise ValueError("Every missing rate must lie in [0, 1).")

    rows: list[dict] = []
    for run in range(config.n_runs):
        rng = np.random.default_rng(config.seed + run)
        # Reusing rankings makes masks nested across missingness levels.
        train_scores = rng.random(train.shape)
        test_scores = rng.random(test.shape)

        for missing_rate in missing_rates:
            p_observed = 1.0 - missing_rate
            train_observed = mask_entries_exactly(
                train, p_observed, train_scores
            )
            test_observed = mask_entries_exactly(
                test, p_observed, test_scores
            )

            if config.standardize:
                center, scale = _fit_scaling(train_observed)
            else:
                center = np.zeros((train.shape[0], 1))
                scale = np.ones((train.shape[0], 1))
            train_scaled = (train_observed - center) / scale
            test_scaled = (test_observed - center) / scale

            # This is the model-selection stage of the proposed method. Every
            # K is scored on training observations only.
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
                hidden = np.isnan(test_observed)
                error = test[hidden] - estimate[hidden]
                rows.append(
                    {
                        "run": run,
                        "missing_rate": missing_rate,
                        "method": method,
                        "mae": missing_mae(test, estimate, test_observed),
                        "rmse": float(np.sqrt(np.mean(error**2))),
                        "n_train": train.shape[1],
                        "n_test": test.shape[1],
                        "n_nodes": train.shape[0],
                        "p_observed": p_observed,
                        "selected_n_components": selected_components,
                        "selection_score": selector.best_score_
                        if method == "proposed"
                        else np.nan,
                        "fallback_clusters": len(model.fallback_clusters)
                        if method == "proposed"
                        else np.nan,
                    }
                )
            print(
                f"[METR-LA] run {run + 1}/{config.n_runs}, "
                f"missing={missing_rate:.0%}: selected K={selected_components}",
                flush=True,
            )

    results = pd.DataFrame(rows)
    metadata = {
        **asdict(config),
        "retained_sensor_count": len(retained),
        "dropped_sensor_ids": [
            sensor_ids[index]
            for index in sorted(set(range(len(sensor_ids))) - set(retained))
        ],
        "first_timestamp": timestamps[0].isoformat(),
        "last_timestamp": timestamps[-1].isoformat(),
        "split_timestamp": timestamps[split_at].isoformat(),
        "selection": (
            "random sample without replacement from complete positive-speed "
            "timestamps, sorted chronologically"
        ),
        "observation_selection_seed": config.seed,
        "adjacency_symmetrization": "maximum(A, A.T), diagonal removed",
        "component_candidates": component_candidates,
        "component_selection": "minimum Yang cost on training observations",
    }
    return results, metadata


def save_results(
    results: pd.DataFrame, metadata: dict, output_dir: str | Path
) -> tuple[Path, Path, Path]:
    """Save per-run values, aggregates, and a reproducibility config."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    raw_path = directory / "metr_la_runs.csv"
    summary_path = directory / "metr_la_summary.csv"
    config_path = directory / "metr_la_config.json"
    results.to_csv(raw_path, index=False, float_format="%.6f")
    summary = (
        results.groupby(["missing_rate", "method"])
        .agg(
            mae=("mae", "mean"),
            std_mae=("mae", "std"),
            rmse=("rmse", "mean"),
            std_rmse=("rmse", "std"),
        )
        .reset_index()
    )
    summary.to_csv(summary_path, index=False, float_format="%.6f")
    config_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return raw_path, summary_path, config_path


def parse_args() -> MetrLAConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=str)
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--n-observations", type=int)
    parser.add_argument("--train-fraction", type=float)
    parser.add_argument("--missing-rates", type=float, nargs="+", metavar="RATE")
    parser.add_argument("--min-components", type=int)
    parser.add_argument("--max-components", type=int)
    parser.add_argument("--n-runs", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--psd-beta", type=float)
    parser.add_argument("--smooth-beta", type=float)
    
    # Ustawiamy default na None zamiast na standardowe False z store_true
    parser.add_argument("--no-standardize", action="store_true", default=None)
    
    args = parser.parse_args()
    
    # Filtrujemy tylko podane argumenty
    provided_args = {k: v for k, v in vars(args).items() if v is not None}
    
    # Poprawka dla tupli w missing_rates
    if "missing_rates" in provided_args:
        provided_args["missing_rates"] = tuple(provided_args["missing_rates"])
        
    # Odwrócenie logiki dla flagi "no-standardize" i zmiana nazwy pod Dataclass
    if "no_standardize" in provided_args:
        # Jeśli użytkownik podał flagę, no_standardize to True. Wtedy standardize = False.
        provided_args["standardize"] = not provided_args.pop("no_standardize")

    return MetrLAConfig(**provided_args)


def main() -> None:
    config = parse_args()
    results, metadata = run_experiment(config)
    paths = save_results(results, metadata, config.output_dir)
    print(
        "\n"
        + results.groupby(["missing_rate", "method"])[["mae", "rmse"]]
        .mean()
        .to_string()
    )
    for path in paths:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
