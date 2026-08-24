"""Run graph-signal reconstruction experiments on Warsaw PM2.5 data.

Run from the repository root::
    python -m src.real_data_experiments.air_quality
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass

import networkx as nx
import numpy as np
import pandas as pd

from src.experiments._helper import evaluate_reconstruction_methods
from src.methods.generation import GSPGraph
from src.methods.models import MixedSignalReconstruction

# Uniwersalne budowanie ścieżek z wykorzystaniem os.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_HISTORY_PATH = os.path.join(PROJECT_ROOT, "data", "air_quality", "pm25_Warsaw_air_quality.csv")
DEFAULT_STATIONS_PATH = os.path.join(PROJECT_ROOT, "data", "air_quality", "Warszawa_stations_data.csv")
DEFAULT_RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "air_quality")


@dataclass(frozen=True)
class AirQualityConfig:
    history_path: str = DEFAULT_HISTORY_PATH
    stations_path: str = DEFAULT_STATIONS_PATH
    output_dir: str = DEFAULT_RESULTS_DIR
    train_fraction: float = 0.8
    missing_rates: tuple[float, ...] = (0.2, 0.5, 0.8)
    min_components: int = 1
    max_components: int = 15
    min_cluster_size: int = 1
    kernel_bandwidth_km: float | None = None
    k_neighbors: int = -1
    n_runs: int = 10
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
    stations: pd.DataFrame, 
    kernel_bandwidth_km: float | None = None,
    k_neighbors: int = -1
) -> GSPGraph:
    """Build a complete or k-NN graph with Gaussian geographic-distance weights."""
    # Ograniczenie wymagań do absolutnego minimum niezbędnego do stworzenia grafu
    required = {"station_id", "latitude", "longitude"}
    if missing := required - set(stations.columns):
        raise ValueError(f"Station dictionary is missing columns: {sorted(missing)}")
    if stations["station_id"].duplicated().any():
        raise ValueError("Station IDs must be unique.")
    count = len(stations)
    if count < 2:
        raise ValueError("At least two stations are required.")

    coordinates = stations[["latitude", "longitude"]].to_numpy(dtype=float)
    distances = _haversine_distances(coordinates)
    
    positive = distances[np.triu_indices(count, k=1)]
    positive = positive[positive > 0]
    
    if kernel_bandwidth_km is None:
        bandwidth = float(np.median(positive)) if positive.size else 1.0
    else:
        bandwidth = float(kernel_bandwidth_km)

    graph = nx.Graph(bandwidth_km=bandwidth)
    
    # Dodawanie węzłów bez zbędnych atrybutów
    for row in stations.itertuples(index=False):
        graph.add_node(
            row.station_id,
            latitude=float(row.latitude),
            longitude=float(row.longitude),
            pos=(float(row.longitude), float(row.latitude)),
        )
        
    station_ids = stations["station_id"].tolist()
    
    if k_neighbors == -1 or k_neighbors >= count - 1:
        # Graf pełny
        for source in range(count):
            for target in range(source + 1, count):
                distance = float(distances[source, target])
                weight = float(np.exp(-(distance**2) / (2.0 * bandwidth**2)))
                graph.add_edge(station_ids[source], station_ids[target], weight=weight, distance_km=distance)
    else:
        # Graf k-NN
        sorted_indices = np.argsort(distances, axis=1)
        knn_indices = sorted_indices[:, 1:k_neighbors + 1]
        
        for source in range(count):
            for target in knn_indices[source]:
                distance = float(distances[source, target])
                weight = float(np.exp(-(distance**2) / (2.0 * bandwidth**2)))
                graph.add_edge(station_ids[source], station_ids[target], weight=weight, distance_km=distance)

    return GSPGraph(graph)


def _remove_consecutive_duplicates(
    values: np.ndarray, timestamps: pd.DatetimeIndex
) -> tuple[np.ndarray, pd.DatetimeIndex]:
    keep = np.ones(values.shape[0], dtype=bool)
    if values.shape[0] > 1:
        equal = (values[1:] == values[:-1]) | (np.isnan(values[1:]) & np.isnan(values[:-1]))
        keep[1:] = ~np.all(equal, axis=1)
    return values[keep], timestamps[keep]


def load_air_quality(config: AirQualityConfig) -> AirQualityData:
    history = pd.read_csv(config.history_path)
    stations = pd.read_csv(config.stations_path, encoding="utf-8-sig")

    measurement_ids = [col for col in history.columns if col != "data"]
    selected = stations.loc[stations["station_id"].isin(measurement_ids)].copy()
    station_ids = selected["station_id"].tolist()
    
    timestamps = pd.DatetimeIndex(pd.to_datetime(history["data"]))
    values = history[station_ids].to_numpy(dtype=float)
    
    order = np.argsort(timestamps.to_numpy(), kind="stable")
    values, timestamps = values[order], timestamps[order]
    values, timestamps = _remove_consecutive_duplicates(values, timestamps)
    
    # Brakujące dane
    values[~np.isfinite(values) | (values <= 0.0)] = np.nan
    graph = build_geographic_graph(selected, config.kernel_bandwidth_km, config.k_neighbors)
    
    return AirQualityData(
        graph=graph,
        signals=values.T,
        timestamps=timestamps,
        stations=selected.reset_index(drop=True)
    )


def mask_known_values(signals: np.ndarray, p_observed: float, random_scores: np.ndarray) -> np.ndarray:
    values = np.asarray(signals, dtype=float)
    masked = np.full(values.shape, np.nan, dtype=float)
    for column in range(values.shape[1]):
        known = np.flatnonzero(np.isfinite(values[:, column]))
        n_visible = max(1, int(round(p_observed * known.size)))
        visible = known[np.argsort(random_scores[known, column])[:n_visible]]
        masked[visible, column] = values[visible, column]
    return masked


def _fit_scaling(train_observed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    observed = np.isfinite(train_observed)
    global_values = train_observed[observed]
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
    return truth[hidden] - estimate[hidden]


def run_experiment(config: AirQualityConfig) -> tuple[pd.DataFrame, dict, AirQualityData]:
    data = load_air_quality(config)
    split_at = int(config.train_fraction * data.signals.shape[1])
    train, test = data.signals[:, :split_at], data.signals[:, split_at:]
    candidates = list(range(config.min_components, min(config.max_components, train.shape[1]) + 1))

    rows = []
    for run in range(config.n_runs):
        rng = np.random.default_rng(config.seed + run)
        train_scores, test_scores = rng.random(train.shape), rng.random(test.shape)
        
        for missing_rate in config.missing_rates:
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
                train_scaled, method="clustered_reconstruction", K_list=candidates,
                min_cluster_size=config.min_cluster_size, alpha=config.alpha,
                beta=config.psd_beta, init_beta=config.smooth_beta, random_state=config.seed + run,
            )
            selected_components = selector.best_K_
            estimates, _, model = evaluate_reconstruction_methods(
                data.graph, train_scaled, test_scaled, selected_components,
                alpha=config.alpha, beta=config.psd_beta, smooth_beta=config.smooth_beta, 
                random_state=config.seed + run, min_cluster_size=config.min_cluster_size,
            )
            
            for method, scaled_estimate in estimates.items():
                estimate = scaled_estimate * scale + center
                errors = _hidden_errors(test, estimate, test_observed)
                rows.append({
                    "run": run, "missing_rate": missing_rate, "method": method,
                    "mae": float(np.mean(np.abs(errors))),
                    "rmse": float(np.sqrt(np.mean(errors**2))),
                    "selected_n_components": selected_components,
                })
            print(f"Run {run + 1}/{config.n_runs}, missing={missing_rate:.0%}: K={selected_components}", flush=True)

    metadata = asdict(config)
    return pd.DataFrame(rows), metadata, data


def save_results(results: pd.DataFrame, metadata: dict, data: AirQualityData, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    results.to_csv(os.path.join(output_dir, "runs.csv"), index=False, float_format="%.6f")
    
    summary = results.groupby(["missing_rate", "method"]).agg(
        mae=("mae", "mean"), std_mae=("mae", "std"),
        rmse=("rmse", "mean"), std_rmse=("rmse", "std"),
    ).reset_index()
    summary.to_csv(os.path.join(output_dir, "summary.csv"), index=False, float_format="%.6f")
    
    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def parse_args() -> AirQualityConfig:
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--history-path", type=str)
    parser.add_argument("--stations-path", type=str)
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--train-fraction", type=float)
    parser.add_argument("--missing-rates", type=float, nargs="+")
    parser.add_argument("--min-components", type=int)
    parser.add_argument("--max-components", type=int)
    parser.add_argument("--k-neighbors", type=int)
    parser.add_argument("--n-runs", type=int)
    
    args = parser.parse_args()
    
    # Przekształcamy argumenty na słownik i ZATRZYMUJEMY tylko te, które nie są None (czyli zostały podane w konsoli)
    provided_args = {k: v for k, v in vars(args).items() if v is not None}
    
    # Dataclass oczekuje krotki (tuple), a argparse dla nargs="+" zwraca listę, więc poprawiamy typ, jeśli użytkownik to podał
    if "missing_rates" in provided_args:
        provided_args["missing_rates"] = tuple(provided_args["missing_rates"])
    
    # Rozpakowujemy słownik prosto do konstruktora (**provided_args).
    # Wszystkie pominięte parametry wezmą swoje wartości z ustawień w @dataclass!
    return AirQualityConfig(**provided_args)


def main() -> None:
    config = parse_args()
    results, metadata, data = run_experiment(config)
    save_results(results, metadata, data, config.output_dir)
    print("\n" + results.groupby(["missing_rate", "method"])[["mae", "rmse"]].mean().to_string())


if __name__ == "__main__":
    main()