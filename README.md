# Graph Signal Reconstruction — Master's Thesis Code

This repository contains the research code and experiments accompanying
[*Reconstruction of Graph Signals Based on the Power Spectral Density Estimation*],
a Master's thesis by Michał Iwicki at the Warsaw University of Technology,
Faculty of Mathematics and Information Science.

It is a thesis companion repository, not a released or supported Python
package. The scripts and notebooks are intended to document the methods,
explore the experiments, and reproduce results from the thesis.

## Research overview

The main contribution is an extension of single-PSD graph signal
reconstruction to heterogeneous multi-PSD data. Instead of estimating one
global power spectral density (PSD) for signals from different latent sources,
the method clusters the signals, estimates a separate PSD for each cluster,
and reconstructs signals using their cluster-specific PSD. The framework uses
graph smoothness for initialization and performs probabilistic clustering in
the graph Fourier domain under the graph wide-sense stationary model. The code
also includes single-PSD and smoothness-based reconstruction baselines and
experiments on synthetic and real-world data.

## Repository map

| Path | Contents |
| --- | --- |
| `src/methods/` | Graph and signal generation, clustering, reconstruction, and combined models |
| `src/experiments/` | Synthetic experiments and shared evaluation helpers |
| `src/real_data_experiments/` | Experiments for Warsaw air quality, METR-LA, and MovieLens ratings |
| `src/additional/preliminary_tests.py` | Earlier and exploratory experiments used by notebooks |
| `src/additional/plotting.py` | Plot helpers used by the experiment notebooks |
| `data/air_quality/` | Warsaw station metadata and PM2.5 observations included in the repository |
| `*.ipynb` | Interactive analyses, figures, and optional dataset-download steps |

## Environment

Use Python 3.14 or newer and install the dependencies from the repository root:

```bash
python -m venv .venv
```

Activate the virtual environment, then run:

```bash
python -m pip install -r requirements.txt
```

The dependency list includes packages for the experiments and notebooks. There
is no package-install step for this repository; run commands from the repository
root so imports such as `src.methods` resolve correctly.

## Notebooks

The notebooks provide interactive views of the work:

- [`preliminary_experiments.ipynb`](./preliminary_experiments.ipynb) introduces
  PSD estimation, reconstruction, and mixed-signal experiments, plus reference
  and aggregated-result plots.
- [`clustering_test.ipynb`](./clustering_test.ipynb) compares spectral
  clustering for mixtures with different PSD profiles and observation rates.
- [`plot_generation.ipynb`](./plot_generation.ipynb) loads experiment summaries
  from `results/` and produces thesis-style reconstruction and clustering plots
  using functions in `src/additional/plotting.py`.
- [`real_data_downloading.ipynb`](./real_data_downloading.ipynb) contains
  optional dataset-download examples; external datasets are not included here.

Notebooks may contain saved cell outputs from previous runs. Re-running an
experiment can take considerable time and may require data or result files not
stored in the repository.

## Running experiments

From the repository root, run the synthetic experiment suite with:

```bash
python -m src.experiments.run_all_experiments
```

It runs studies of graph size and observation visibility, model variants, graph
topologies, PSD transfer between graphs, and mixtures with multiple PSD
profiles. Results are written under `results/reconstruction`,
`results/clustering`, and `results/configs`.

Real-data experiments can be run individually:

```bash
python -m src.real_data_experiments.air_quality
python -m src.real_data_experiments.metr_la
python -m src.real_data_experiments.movie_ratings
python -m src.real_data_experiments.movie_ratings_experiment
```

The Warsaw air-quality input files are included. METR-LA requires
`METR-LA.h5` and `adj_METR-LA.pkl` under `data/metr/`. The MovieLens experiment
expects prepared `signals.csv` and `adjacency.csv` under
`data/movie/processed/`; use `src.real_data_experiments.movie_ratings` to
prepare those files from a compatible ratings CSV. Dataset downloads and
generated results are not part of this repository.


## Thesis reference

Iwicki, M. (2026). *Reconstruction of Graph Signals Based on Power Spectral
Density Estimation*. Master's thesis, Warsaw University of Technology.
