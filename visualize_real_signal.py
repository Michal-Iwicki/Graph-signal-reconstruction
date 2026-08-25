import numpy as np
import matplotlib.pyplot as plt

# Importy rdzenia i ewaluacji
from src.methods.models import MixedSignalReconstruction
from src.experiments._helper import evaluate_reconstruction_methods

# Importy METR-LA
from src.real_data_experiments.metr_la import (
    MetrLAConfig, load_metr_la, build_sensor_graph, 
    select_complete_observations, mask_entries_exactly as mask_metr, _fit_scaling as fit_metr
)
# Importy Air Quality
from src.real_data_experiments.air_quality import (
    AirQualityConfig, load_air_quality, mask_known_values as mask_air, _fit_scaling as fit_air
)
# Importy MovieLens
from src.real_data_experiments.movie_ratings_experiment import (
    MovieRatingsConfig, load_prepared_movie_rating_data, mask_known_ratings as mask_movie, _fit_movie_scaling as fit_movie
)

def plot_beta_impact(truth, estimates_by_beta, title, missing_rate):
    """
    Uniwersalna funkcja do sortowania i rysowania wpływu parametru Beta na rekonstrukcję.
    Ignoruje natywne wartości NaN w Ground Truth.
    """
    valid_mask = np.isfinite(truth)
    truth_valid = truth[valid_mask]
    sort_idx = np.argsort(truth_valid)
    
    plt.figure(figsize=(14, 8))
    plt.plot(truth_valid[sort_idx], label='Ground Truth', color='black', linewidth=3)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for idx, (beta_val, est) in enumerate(estimates_by_beta.items()):
        est_valid = est[valid_mask]
        plt.plot(
            est_valid[sort_idx], 
            label=f'Proponowana (beta = {beta_val})', 
            color=colors[idx % len(colors)], 
            linestyle='--', 
            alpha=0.85
        )

    plt.title(f"Wpływ parametru Beta na rekonstrukcję PSD\n{title} (Brakujące ukryte obserwacje: {missing_rate*100:.0%})")
    plt.xlabel("Węzły grafu (posortowane wg prawdziwej wartości sygnału)")
    plt.ylabel("Wartość sygnału")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def visualize_metr_la_beta():
    print("--- Testowanie różnych wartości Beta (METR-LA) ---")
    betas_to_test = [0.1, 0.01, 0.75]
    config = MetrLAConfig(n_observations=100)
    rng = np.random.default_rng(config.seed)

    frame, sensor_ids, adjacency = load_metr_la(config.data_dir)
    graph, retained = build_sensor_graph(sensor_ids, adjacency)
    signals, timestamps = select_complete_observations(frame, retained, config.n_observations, rng)
    
    p_observed = 0.5  # 50% braków
    signals_observed = mask_metr(signals, p_observed, rng.random(signals.shape))
    center, scale = fit_metr(signals_observed)
    signals_scaled = (signals_observed - center) / scale

    signal_idx = 0
    truth = signals[:, signal_idx]
    estimates_rescaled = {}

    for test_beta in betas_to_test:
        print(f"  > METR-LA rekonstrukcja dla beta = {test_beta}...")
        selector = MixedSignalReconstruction(graph)
        selector.fit_transform(
            signals_scaled, method="clustered_reconstruction", 
            K_list=[3], alpha=config.alpha, beta=test_beta, init_beta=config.smooth_beta, random_state=config.seed
        )
        
        estimates_scaled, _, _ = evaluate_reconstruction_methods(
            graph, signals_scaled, signals_scaled, selector.best_K_,
            alpha=config.alpha, beta=test_beta, smooth_beta=config.smooth_beta, random_state=config.seed,
        )
        estimates_rescaled[test_beta] = estimates_scaled["proposed"][:, signal_idx] * scale[:, 0] + center[:, 0]
        
    plot_beta_impact(truth, estimates_rescaled, f"METR-LA (Timestamp: {timestamps[signal_idx]})", 1.0 - p_observed)

def visualize_air_quality_beta():
    print("\n--- Testowanie różnych wartości Beta (Air Quality Warszawa) ---")
    betas_to_test = [0.1, 0.01, 0.75]
    config = AirQualityConfig()
    rng = np.random.default_rng(config.seed)
    
    data = load_air_quality(config)
    signals = data.signals[:, :100] 
    timestamps = data.timestamps[:100]
    
    p_observed = 0.5  # 50% braków
    signals_observed = mask_air(signals, p_observed, rng.random(signals.shape))
    center, scale = fit_air(signals_observed)
    signals_scaled = (signals_observed - center) / scale

    signal_idx = 0
    truth = signals[:, signal_idx]
    estimates_rescaled = {}

    for test_beta in betas_to_test:
        print(f"  > Air Quality rekonstrukcja dla beta = {test_beta}...")
        selector = MixedSignalReconstruction(data.graph)
        selector.fit_transform(
            signals_scaled, method="clustered_reconstruction", 
            K_list=[3], alpha=config.alpha, beta=test_beta, init_beta=config.smooth_beta, random_state=config.seed
        )
        
        estimates_scaled, _, _ = evaluate_reconstruction_methods(
            data.graph, signals_scaled, signals_scaled, selector.best_K_,
            alpha=config.alpha, beta=test_beta, smooth_beta=config.smooth_beta, random_state=config.seed,
        )
        estimates_rescaled[test_beta] = estimates_scaled["proposed"][:, signal_idx] * scale[:, 0] + center[:, 0]
        
    plot_beta_impact(truth, estimates_rescaled, f"PM2.5 Warszawa (Timestamp: {timestamps[signal_idx]})", 1.0 - p_observed)

def visualize_movie_ratings_beta():
    print("\n--- Testowanie różnych wartości Beta (MovieLens) ---")
    betas_to_test = [0.1, 0.01, 0.75]
    config = MovieRatingsConfig()
    rng = np.random.default_rng(config.seed)
    
    prepared = load_prepared_movie_rating_data(config.data_dir)
    signals = prepared.data.signals[:, :100] 
    
    p_observed = 0.5  # 50% braków
    signals_observed = mask_movie(signals, p_observed, rng.random(signals.shape))
    center, scale = fit_movie(signals_observed)
    signals_scaled = (signals_observed - center) / scale

    signal_idx = 0
    truth = signals[:, signal_idx]
    estimates_rescaled = {}

    for test_beta in betas_to_test:
        print(f"  > MovieLens rekonstrukcja dla beta = {test_beta}...")
        selector = MixedSignalReconstruction(prepared.data.graph)
        selector.fit_transform(
            signals_scaled, method="clustered_reconstruction", 
            K_list=[3], alpha=config.alpha, beta=test_beta, init_beta=config.smooth_beta, random_state=config.seed
        )
        
        estimates_scaled, _, _ = evaluate_reconstruction_methods(
            prepared.data.graph, signals_scaled, signals_scaled, selector.best_K_,
            alpha=config.alpha, beta=test_beta, smooth_beta=config.smooth_beta, random_state=config.seed,
        )
        estimates_rescaled[test_beta] = estimates_scaled["proposed"][:, signal_idx] * scale[:, 0] + center[:, 0]
        
    user_id = prepared.data.user_ids[signal_idx]
    plot_beta_impact(truth, estimates_rescaled, f"Oceny MovieLens (Użytkownik ID: {user_id})", 1.0 - p_observed)

if __name__ == "__main__":
    visualize_metr_la_beta()
    visualize_air_quality_beta()
    visualize_movie_ratings_beta()