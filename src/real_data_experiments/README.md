# METR-LA reconstruction experiment

The experiment treats every timestamp as a graph signal over traffic sensors
and compares `proposed`, `basic_psd`, and `smoothing`. By default it:

- loads `data/metr/METR-LA.h5` and `data/metr/adj_METR-LA.pkl`;
- removes self-loops, symmetrizes supplied edge weights, and uses the largest
  connected component (206 of 207 sensors);
- randomly selects 2000 timestamps without replacement from those without
  native missing values (METR-LA encodes those as zero), using `--seed`, and
  then orders the selected timestamps chronologically;
- uses a chronological 1600/400 train/test split;
- tests 20%, 50%, and 80% missing values; masks are random but nested between
  levels and hide an exact number of sensors (up to rounding) in every signal;
- standardizes each sensor using only its visible training values;
- tests every component count from 2 through 10, selects the one with the
  smallest Yang cost using training observations only, and then evaluates that
  selected model on the test split;
- reports MAE and RMSE only on artificially hidden test entries.

Run it from the repository root:

```bash
python -m src.real_data_experiments.metr_la
```

For a quick smoke test:

```bash
python -m src.real_data_experiments.metr_la --n-observations 40 --n-runs 1 --min-components 2 --max-components 3
```

Results are written to `results/metr_la/`. All key choices can be changed from
the command line; use `--help` to list them.

## Movie ratings

`movie_ratings.py` converts an explicit-ratings CSV (including the standard
MovieLens `ratings.csv`) into a user graph and movie signals. Users are graph
vertices, columns are movie signals, and finite entries are observed ratings:

```python
from src.real_data_experiments.movie_ratings import build_movie_rating_data

data = build_movie_rating_data(
    "data/ml-latest-small/ratings.csv",
    min_user_ratings=5,
    min_movie_ratings=5,
    n_neighbors=20,
)

graph = data.graph
signals = data.signals       # (users, movies), missing ratings are np.nan
observed = data.observed     # Boolean observation mask
```

Edge weights are cosine similarities between movie rating vectors, reduced for
pairs with few common users and sparsified to a weighted k-nearest-neighbor
graph. Rows of `signals` follow `data.movie_ids`/`graph.node_order`; columns
follow `data.user_ids`.

First preprocess MovieLens and save two readable CSV matrices:

```bash
python -m src.real_data_experiments.movie_ratings
```

This creates `signals.csv` and `adjacency.csv`. Then run the reconstruction
experiment using only those files:

```bash
python -m src.real_data_experiments.movie_ratings_experiment
```

The preprocessing streams the large MovieLens CSV twice, selects active users
and popular movies, and constructs the movie graph. The experiment splits user
signals reproducibly into train/test sets. Native missing ratings are kept
separate from artificial masks, so MAE/RMSE cover only known ratings.
For a smaller dataset and quicker experiment use:

```bash
python -m src.real_data_experiments.movie_ratings \
  --n-users 30 --n-movies 50 --min-movie-ratings 5 \
  --output-dir data/movie/processed-small

python -m src.real_data_experiments.movie_ratings_experiment \
  --data-dir data/movie/processed-small \
  --n-runs 1 --min-components 2 --max-components 3 \
  --missing-rates 0.5
```

Raw results, aggregates, and the full reproducibility configuration are saved
under `results/movie_ratings/`. Prepared data go to
`data/movie/processed/signals.csv` and `data/movie/processed/adjacency.csv`.
Movie and user IDs are stored directly as CSV row and column labels.

Use `--output-dir PATH` in `movie_ratings` to change the prepared-data
location, and pass the same path as `--data-dir PATH` to the experiment.

## Warsaw air quality

`air_quality.py` maps the PM2.5 stations to a complete geographic graph. Every
pair of stations is connected and its weight uses a Gaussian kernel of the
haversine distance. By default, the kernel bandwidth is the median pairwise
distance; it can be overridden with `--kernel-bandwidth-km`. Exact
snapshots repeated directly after one another are collapsed before a
chronological 80/20 train/test split. Reported zeros are treated as native
missing readings and are excluded from evaluation.

The experiment uses the same `proposed`, `basic_psd`, and `smoothing` methods
and the same 20%, 50%, and 80% artificial missingness levels as METR-LA:

```bash
python -m src.real_data_experiments.air_quality
```

Per-run and aggregate results, configuration, graph nodes, and weighted graph
edges are saved under `results/air_quality/`. The current history is short, so
the defaults allow one through three mixture components and clusters of one
signal. Increase `--min-cluster-size` and `--max-components` after collecting
more timestamps.
