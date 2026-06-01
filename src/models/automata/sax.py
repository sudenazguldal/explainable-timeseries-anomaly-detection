import string

import numpy as np
from scipy.stats import norm


def get_sax_breakpoints(alphabet_size: int) -> np.ndarray:
    """
    Returns Gaussian breakpoints for SAX discretization.

    Example:
    alphabet_size = 3 creates 2 breakpoints.
    """
    if alphabet_size < 2:
        raise ValueError("Alphabet size must be at least 2.")

    quantiles = [
        index / alphabet_size
        for index in range(1, alphabet_size)
    ]

    return norm.ppf(quantiles)


def z_normalize(series: list[float] | np.ndarray) -> np.ndarray:
    """
    Z-normalizes a one-dimensional series.
    """
    values = np.asarray(series, dtype=float)

    if values.ndim != 1:
        raise ValueError("SAX input must be one-dimensional.")

    std = values.std()

    if std == 0:
        return np.zeros_like(values)

    return (values - values.mean()) / std


def sax_transform(
    series: list[float] | np.ndarray,
    alphabet_size: int,
) -> str:
    """
    Converts a numeric sequence into a SAX symbol string.
    """
    normalized = z_normalize(series)
    breakpoints = get_sax_breakpoints(alphabet_size)

    alphabet = string.ascii_lowercase[:alphabet_size]

    symbols = []

    for value in normalized:
        symbol_index = np.searchsorted(breakpoints, value, side="right")
        symbols.append(alphabet[symbol_index])

    return "".join(symbols)