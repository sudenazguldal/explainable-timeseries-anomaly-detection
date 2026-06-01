import numpy as np


def paa_transform(series: list[float] | np.ndarray, n_segments: int) -> np.ndarray:
    """
    Applies Piecewise Aggregate Approximation (PAA).

    The input time series is divided into equal segments and each segment
    is represented by its mean value.
    """
    values = np.asarray(series, dtype=float)

    if values.ndim != 1:
        raise ValueError("PAA input must be one-dimensional.")

    if n_segments <= 0:
        raise ValueError("Number of PAA segments must be positive.")

    if n_segments > len(values):
        raise ValueError("Number of PAA segments cannot exceed series length.")

    segments = np.array_split(values, n_segments)

    return np.array([segment.mean() for segment in segments])