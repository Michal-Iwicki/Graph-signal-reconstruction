# Graph Signal Reconstruction

Research code for graph-signal reconstruction based on power spectral density
(PSD) estimation. The project includes synthetic graph-signal generation,
sampled-covariance PSD estimation, smooth and PSD-adaptive reconstruction,
spectral clustering, mixture models, and transfer of learned PSD profiles to
new graphs.

## Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e .
```

The project uses a standard `src` package layout. After the editable install,
the notebooks and test suite import `graph_reco` and `experiments` directly:

```bash
python3 -m unittest discover -s tests -v
```

## Missing-observation contract

Missing graph-signal values are represented by `NaN` throughout generation and
experiment pipelines. PSD estimation and reconstruction create zero-filled
working copies internally. Consequently, an observed value equal to zero
remains a valid observation and is never confused with missing data.

Signal matrices use shape `(N, M)`, where `N` is the number of graph nodes and
each of the `M` columns is one signal realization.

## Source layout

- `src/graph_reco/generation.py` — graphs, PSD profiles, synthetic signals, and
  visualization.
- `src/graph_reco/reconstruction.py` — PSD estimation and graph-signal
  reconstruction.
- `src/graph_reco/clustering.py` — diagonal-covariance GMM and clustering
  evaluation.
- `src/graph_reco/models.py` — global, clustered, iterative, EM-PSD, and
  transfer workflows.
- `src/experiments/testing.py` — experiment procedures used by the notebooks.
- `src/experiments/synthetic.py` — thesis-planned synthetic experiments using
  a baseline configuration and one-factor-at-a-time parameter studies.
- `src/experiments/real_data.py` — extraction and initial reconstruction
  experiment for the NYC PM2.5 dataset.
- `tests/` — unit and integration tests.

The preferred high-level method names are:

- `smooth`
- `global_psd`
- `clustered_reconstruction`
- `iterative_clustered_reconstruction`
- `simultaneous_method`

The earlier misspelling `simultaneus_method` remains available as a
backward-compatible alias.

## Synthetic experiments

The initial synthetic study is intentionally not a parameter grid. It runs the
three thesis sections at one baseline configuration:

```python
from experiments.synthetic import (
    SyntheticExperimentConfig,
    plot_synthetic_results,
    run_default_synthetic_experiments,
    run_planned_parameter_study,
    summarize_synthetic_results,
)

config = SyntheticExperimentConfig()
baseline = run_default_synthetic_experiments(config)
summary = summarize_synthetic_results(baseline)
figures = plot_synthetic_results(baseline)
```

Each sensitivity study returns to that baseline and changes one factor only:

```python
observation_study = run_planned_parameter_study(
    "observation_probability",
    config=config,
)
```

Supported planned factors are graph `topology`, `profile_name`, `n_train`,
`n_components`, and `observation_probability`. See
`synthetic_experiments.ipynb` for the execution order. Here, `n_train`
corresponds to the thesis parameter \(M\); a separate fixed test set is used to
avoid evaluating reconstruction on the signals used to estimate the PSD.

## NYC PM2.5 real-data experiment

The cleanest supplied real dataset is the complete PM2.5 table. At Community
District level it provides 59 graph nodes and 48 signals: annual, summer, and
winter measurements for 2009–2024. The CSV has no district coordinates or
boundaries, so the initial version constructs a weighted kNN similarity graph
from historical node profiles. In the chronological experiment the graph and
PSD models use training years only.

```python
from experiments.real_data import (
    load_nyc_pm25_signals,
    run_nyc_pm25_experiment,
)

path = "data/NYC EH Data Portal - Fine particles (PM 2.5) (full table).csv"
dataset = load_nyc_pm25_signals(path)
results = run_nyc_pm25_experiment(path)
```

See `real_data_experiments.ipynb` for the baseline run and result plots.
