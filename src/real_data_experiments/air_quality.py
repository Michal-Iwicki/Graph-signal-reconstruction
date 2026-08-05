"""Run graph-signal reconstruction experiments on Warsaw PM2.5 data.

Stations are vertices of a geographic weighted k-nearest-neighbour graph and
each timestamp is a graph signal.  Exact consecutive snapshots are removed
before the chronological train/test split.  Non-positive PM2.5 readings are
treated as native missing values and are never used to calculate metrics.

Run from the repository root::

    python -m src.real_data_experiments.air_quality
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
from src.methods.generation import GSPGraph
from src.methods.models import MixedSignalReconstruction


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "air_quality"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results" / "air_quality"

# The history endpoint and station dictionary use two different labels for
# these long-running reference stations.
STATION_ID_ALIASES = {
    "Grochowska 244": "ul. Grochowska",
    'al."Solidarności"': "al. Solidarności",
}


@dataclass(frozen=True)
class AirQualityConfig:
    data_dir: str = str(DEFAULT_DATA_DIR)
    output_dir: str = str(DEFAULT_RESULTS_DIR)
    train_fraction: float = 0.8
    missing_rates: tuple[float, ...] = (0.2, 0.5, 0.8)
    min_components: int = 1
    max_components: int = 3
    min_cluster_size: int = 1
    kernel_bandwidth_km: float | None = None
    n_runs: int = 3
    seed: int = 42
    alpha: float = 10.0
    psd_beta: float = 0.75
    smooth_beta: float = 0.75
    standardize: bool = True


@dataclass(frozen=True)
class AirQualityData:
    graph: GSPGraph
    signals: np.ndarray
    timestamps: pd.DatetimeIndex
    stations: pd.DataFrame
    raw_signal_count: int
    consecutive_duplicates_removed: int


def _haversine_distances(coordinates: np.ndarray) -> np.ndarray:
    """Return all pairwise great-circle distances in kilometres."""
    radians = np.radians(np.asarray(coordinates, dtype=float))
    latitude = radians[:, 0, None]
    longitude = radians[:, 1, None]
    delta_latitude = latitude.T - latitude
    delta_longitude = longitude.T - longitude
    value = (
        np.sin(delta_latitude / 2.0) ** 2
        + np.cos(latitude) * np.cos(latitude.T)
        * np.sin(delta_longitude / 2.0) ** 2
    )
    return 2.0 * 6371.0088 * np.arcsin(np.sqrt(np.clip(value, 0.0, 1.0)))


def build_geographic_graph(
    stations: pd.DataFrame, kernel_bandwidth_km: float | None = None
) -> GSPGraph:
    """Build a complete graph with Gaussian geographic-distance weights."""
    required = {"station_id", "name", "station_type", "latitude", "longitude"}
    if missing := required - set(stations.columns):
        raise ValueError(f"Station dictionary is missing columns: {sorted(missing)}")
    if stations["station_id"].duplicated().any():
        raise ValueError("Station IDs must be unique.")
    count = len(stations)
    if count < 2:
        raise ValueError("At least two stations are required.")

    coordinates = stations[["latitude", "longitude"]].to_numpy(dtype=float)
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("Every selected station must have finite coordinates.")
    distances = _haversine_distances(coordinates)
    positive = distances[np.triu_indices(count, k=1)]
    positive = positive[positive > 0]
    if kernel_bandwidth_km is None:
        bandwidth = float(np.median(positive)) if positive.size else 1.0
    else:
        bandwidth = float(kernel_bandwidth_km)
        if not np.isfinite(bandwidth) or bandwidth <= 0:
            raise ValueError("kernel_bandwidth_km must be positive and finite.")

    graph = nx.Graph(
        geographic_distance="haversine_km",
        edge_weight="exp(-distance_km^2 / (2 * bandwidth_km^2))",
        bandwidth_km=bandwidth,
        topology="complete",
    )
    for row in stations.itertuples(index=False):
        graph.add_node(
            row.station_id,
            name=row.name,
            station_type=row.station_type,
            latitude=float(row.latitude),
            longitude=float(row.longitude),
            pos=(float(row.longitude), float(row.latitude)),
        )
    station_ids = stations["station_id"].tolist()
    for source in range(count):
        for target in range(source + 1, count):
            distance = float(distances[source, target])
            weight = float(np.exp(-(distance**2) / (2.0 * bandwidth**2)))
            graph.add_edge(
                station_ids[source], station_ids[target],
                weight=weight, distance_km=distance,
            )
    if not nx.is_connected(graph):
        raise ValueError("The complete geographic graph is unexpectedly disconnected.")
    return GSPGraph(graph)


def _remove_consecutive_duplicates(
    values: np.ndarray, timestamps: pd.DatetimeIndex
) -> tuple[np.ndarray, pd.DatetimeIndex, int]:
    """Collapse runs of exactly equal snapshots, retaining their first item."""
    if values.shape[0] != len(timestamps):
        raise ValueError("One timestamp is required for every signal.")
    keep = np.ones(values.shape[0], dtype=bool)
    if values.shape[0] > 1:
        equal = (values[1:] == values[:-1]) | (
            np.isnan(values[1:]) & np.isnan(values[:-1])
        )
        keep[1:] = ~np.all(equal, axis=1)
    return values[keep], timestamps[keep], int((~keep).sum())


def load_air_quality(
    data_dir: str | Path, kernel_bandwidth_km: float | None = None
) -> AirQualityData:
    """Load history, align station labels, deduplicate it, and build the graph."""
    directory = Path(data_dir)
    history_path = directory / "warszawa_pm25_historia.json"
    stations_path = directory / "warszawa_stacje_slownik.csv"
    if not history_path.is_file() or not stations_path.is_file():
        raise FileNotFoundError(
            f"Expected warszawa_pm25_historia.json and "
            f"warszawa_stacje_slownik.csv in {directory}."
        )
    history = json.loads(history_path.read_text(encoding="utf-8"))
    if not isinstance(history, list) or len(history) < 2:
        raise ValueError("Air-quality history must contain at least two records.")
    stations = pd.read_csv(stations_path, encoding="utf-8-sig")

    records = []
    for item in history:
        if not isinstance(item, dict) or "timestamp" not in item or "pm25" not in item:
            raise ValueError("Every history record needs timestamp and pm25 fields.")
        records.append({STATION_ID_ALIASES.get(key, key): value for key, value in item["pm25"].items()})
    measurement_ids = set().union(*(record.keys() for record in records))
    dictionary_ids = set(stations["station_id"])
    unknown = measurement_ids - dictionary_ids
    if unknown:
        raise ValueError(f"Measurements have no station coordinates: {sorted(unknown)}")

    # Preserve the dictionary order as the graph-signal vertex order.
    selected = stations.loc[stations["station_id"].isin(measurement_ids)].copy()
    station_ids = selected["station_id"].tolist()
    timestamps = pd.DatetimeIndex(pd.to_datetime([item["timestamp"] for item in history]))
    values = np.asarray(
        [[record.get(station_id, np.nan) for station_id in station_ids] for record in records],
        dtype=float,
    )
    order = np.argsort(timestamps.to_numpy(), kind="stable")
    values, timestamps = values[order], timestamps[order]
    values, timestamps, removed = _remove_consecutive_duplicates(values, timestamps)
    # A reported zero represents an unavailable sensor reading in this feed.
    values[~np.isfinite(values) | (values <= 0.0)] = np.nan
    graph = build_geographic_graph(selected, kernel_bandwidth_km)
    return AirQualityData(
        graph=graph,
        signals=values.T,
        timestamps=timestamps,
        stations=selected.reset_index(drop=True),
        raw_signal_count=len(history),
        consecutive_duplicates_removed=removed,
    )


def mask_known_values(
    signals: np.ndarray, p_observed: float, random_scores: np.ndarray
) -> np.ndarray:
    """Hide an exact rounded fraction of the native known values per signal."""
    values = np.asarray(signals, dtype=float)
    scores = np.asarray(random_scores, dtype=float)
    if values.ndim != 2 or scores.shape != values.shape or not np.all(np.isfinite(scores)):
        raise ValueError("signals must be a matrix and random_scores finite and aligned.")
    if not 0.0 < p_observed <= 1.0:
        raise ValueError("p_observed must lie in (0, 1].")
    masked = np.full(values.shape, np.nan, dtype=float)
    for column in range(values.shape[1]):
        known = np.flatnonzero(np.isfinite(values[:, column]))
        if known.size == 0:
            raise ValueError(f"Signal column {column} contains no known values.")
        n_visible = max(1, int(round(p_observed * known.size)))
        visible = known[np.argsort(scores[known, column])[:n_visible]]
        masked[visible, column] = values[visible, column]
    return masked


def _fit_scaling(train_observed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    observed = np.isfinite(train_observed)
    global_values = train_observed[observed]
    if global_values.size == 0:
        raise ValueError("Training data contain no observed values.")
    global_center, global_scale = float(np.mean(global_values)), float(np.std(global_values))
    counts = observed.sum(axis=1, keepdims=True)
    sums = np.nansum(train_observed, axis=1, keepdims=True)
    center = np.divide(sums, counts, out=np.full_like(sums, global_center), where=counts > 0)
    squared = np.where(observed, (train_observed - center) ** 2, 0.0).sum(axis=1, keepdims=True)
    scale = np.sqrt(np.divide(squared, counts, out=np.full_like(squared, global_scale**2), where=counts > 0))
    scale[~np.isfinite(scale) | (scale < 1e-8)] = 1.0
    return center, scale


def _hidden_errors(truth: np.ndarray, estimate: np.ndarray, observed: np.ndarray) -> np.ndarray:
    hidden = np.isfinite(truth) & ~np.isfinite(observed)
    if not np.any(hidden):
        raise ValueError("No known test measurements were artificially hidden.")
    return truth[hidden] - estimate[hidden]


def run_experiment(config: AirQualityConfig) -> tuple[pd.DataFrame, dict, AirQualityData]:
    """Run nested missingness masks over a chronological deduplicated split."""
    if not 0.0 < config.train_fraction < 1.0:
        raise ValueError("train_fraction must lie in (0, 1).")
    if config.min_components < 1 or config.max_components < config.min_components:
        raise ValueError("Require 1 <= min_components <= max_components.")
    if config.min_cluster_size < 1:
        raise ValueError("min_cluster_size must be positive.")
    missing_rates = tuple(float(rate) for rate in config.missing_rates)
    if not missing_rates or any(not np.isfinite(rate) or not 0.0 < rate < 1.0 for rate in missing_rates):
        raise ValueError("Every missing rate must lie in (0, 1).")

    data = load_air_quality(config.data_dir, config.kernel_bandwidth_km)
    split_at = int(config.train_fraction * data.signals.shape[1])
    if split_at < 1 or split_at >= data.signals.shape[1]:
        raise ValueError("train_fraction creates an empty train or test split.")
    train, test = data.signals[:, :split_at], data.signals[:, split_at:]
    candidates = list(range(config.min_components, min(config.max_components, train.shape[1]) + 1))
    if not candidates:
        raise ValueError("No component candidate fits the training split.")

    rows: list[dict] = []
    for run in range(config.n_runs):
        rng = np.random.default_rng(config.seed + run)
        train_scores, test_scores = rng.random(train.shape), rng.random(test.shape)
        for missing_rate in missing_rates:
            p_observed = 1.0 - missing_rate
            train_observed = mask_known_values(train, p_observed, train_scores)
            test_observed = mask_known_values(test, p_observed, test_scores)
            if config.standardize:
                center, scale = _fit_scaling(train_observed)
            else:
                center, scale = np.zeros((train.shape[0], 1)), np.ones((train.shape[0], 1))
            train_scaled = (train_observed - center) / scale
            test_scaled = (test_observed - center) / scale

            selector = MixedSignalReconstruction(data.graph)
            selector.fit_transform(
                train_scaled,
                method="clustered_reconstruction",
                K_list=candidates,
                min_cluster_size=config.min_cluster_size,
                alpha=config.alpha,
                beta=config.psd_beta,
                init_beta=config.smooth_beta,
                random_state=config.seed + run,
            )
            selected_components = selector.best_K_
            estimates, _, model = evaluate_reconstruction_methods(
                data.graph, train_scaled, test_scaled, selected_components,
                alpha=config.alpha, beta=config.psd_beta,
                smooth_beta=config.smooth_beta, random_state=config.seed + run,
                min_cluster_size=config.min_cluster_size,
            )
            for method, scaled_estimate in estimates.items():
                estimate = scaled_estimate * scale + center
                errors = _hidden_errors(test, estimate, test_observed)
                rows.append({
                    "run": run, "missing_rate": missing_rate, "method": method,
                    "mae": float(np.mean(np.abs(errors))),
                    "rmse": float(np.sqrt(np.mean(errors**2))),
                    "n_hidden_test_values": int(errors.size),
                    "n_train": train.shape[1], "n_test": test.shape[1],
                    "n_nodes": train.shape[0], "p_observed": p_observed,
                    "selected_n_components": selected_components,
                    "selection_score": selector.best_score_ if method == "proposed" else np.nan,
                    "fallback_clusters": len(model.fallback_clusters) if method == "proposed" else np.nan,
                })
            print(
                f"[Air quality] run {run + 1}/{config.n_runs}, "
                f"missing={missing_rate:.0%}: selected K={selected_components}", flush=True,
            )

    metadata = {
        **asdict(config),
        "raw_signal_count": data.raw_signal_count,
        "consecutive_duplicates_removed": data.consecutive_duplicates_removed,
        "deduplicated_signal_count": data.signals.shape[1],
        "native_missing_count": int(np.isnan(data.signals).sum()),
        "station_count": data.graph.number_of_nodes(),
        "graph_edges": data.graph.number_of_edges(),
        "graph_connected": nx.is_connected(data.graph),
        "graph_bandwidth_km": data.graph.graph["bandwidth_km"],
        "graph_construction": "complete graph with Gaussian haversine-distance weights",
        "station_id_aliases": STATION_ID_ALIASES,
        "component_candidates": candidates,
        "component_selection": "minimum Yang cost on training observations",
        "split_timestamp": data.timestamps[split_at].isoformat(),
        "metric_scope": "positive finite test PM2.5 values hidden artificially by the experiment",
    }
    return pd.DataFrame(rows), metadata, data


def save_results(results: pd.DataFrame, metadata: dict, data: AirQualityData, output_dir: str | Path) -> tuple[Path, ...]:
    """Save runs, aggregates, graph nodes/edges, and reproducibility metadata."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    raw_path = directory / "air_quality_runs.csv"
    summary_path = directory / "air_quality_summary.csv"
    config_path = directory / "air_quality_config.json"
    stations_path = directory / "air_quality_graph_nodes.csv"
    edges_path = directory / "air_quality_graph_edges.csv"
    results.to_csv(raw_path, index=False, float_format="%.6f")
    summary = results.groupby(["missing_rate", "method"]).agg(
        mae=("mae", "mean"), std_mae=("mae", "std"),
        rmse=("rmse", "mean"), std_rmse=("rmse", "std"),
    ).reset_index()
    summary.to_csv(summary_path, index=False, float_format="%.6f")
    config_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    data.stations.to_csv(stations_path, index=False)
    pd.DataFrame([
        {"source": source, "target": target, "distance_km": attrs["distance_km"], "weight": attrs["weight"]}
        for source, target, attrs in data.graph.edges(data=True)
    ]).to_csv(edges_path, index=False, float_format="%.8f")
    return raw_path, summary_path, config_path, stations_path, edges_path


def parse_args() -> AirQualityConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--missing-rates", type=float, nargs="+", default=[0.2, 0.5, 0.8])
    parser.add_argument("--min-components", type=int, default=1)
    parser.add_argument("--max-components", type=int, default=3)
    parser.add_argument("--min-cluster-size", type=int, default=1)
    parser.add_argument(
        "--kernel-bandwidth-km", type=float, default=None,
        help="Gaussian bandwidth; default is the median pairwise distance.",
    )
    parser.add_argument("--n-runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--psd-beta", type=float, default=0.75)
    parser.add_argument("--smooth-beta", type=float, default=0.75)
    parser.add_argument("--no-standardize", action="store_true")
    args = parser.parse_args()
    return AirQualityConfig(
        data_dir=args.data_dir, output_dir=args.output_dir,
        train_fraction=args.train_fraction, missing_rates=tuple(args.missing_rates),
        min_components=args.min_components, max_components=args.max_components,
        min_cluster_size=args.min_cluster_size,
        kernel_bandwidth_km=args.kernel_bandwidth_km,
        n_runs=args.n_runs, seed=args.seed, alpha=args.alpha,
        psd_beta=args.psd_beta, smooth_beta=args.smooth_beta,
        standardize=not args.no_standardize,
    )


def main() -> None:
    config = parse_args()
    results, metadata, data = run_experiment(config)
    paths = save_results(results, metadata, data, config.output_dir)
    print("\n" + results.groupby(["missing_rate", "method"])[["mae", "rmse"]].mean().to_string())
    for path in paths:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
