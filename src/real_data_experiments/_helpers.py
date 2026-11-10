"""Shared preprocessing helpers for real-data experiments."""

import numpy as np


def fit_row_scaling(
    train_observed: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate per-row centers and scales using finite training values."""
    values = np.asarray(train_observed, dtype=float)
    if values.ndim != 2:
        raise ValueError("train_observed must be a two-dimensional array.")

    observed = np.isfinite(values)
    global_values = values[observed]
    if global_values.size == 0:
        raise ValueError("Training data contain no observed values.")

    global_center = float(np.mean(global_values))
    global_scale = float(np.std(global_values))
    counts = observed.sum(axis=1, keepdims=True)
    finite_values = np.where(observed, values, 0.0)
    sums = finite_values.sum(axis=1, keepdims=True)
    center = np.divide(
        sums,
        counts,
        out=np.full_like(sums, global_center, dtype=float),
        where=counts > 0,
    )

    differences = np.zeros_like(values)
    np.subtract(values, center, out=differences, where=observed)
    squared = np.square(differences).sum(axis=1, keepdims=True)
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
