import numpy as np


def aggregate_labels_by_segments(
    labels: list[int] | np.ndarray,
    n_segments: int,
    strategy: str = "any",
) -> list[int]:
    """
    Aggregates row-level binary labels into PAA-level segment labels.

   strategies:
    - any: segment is anomalous if any row inside it is anomalous
    - majority: segment label is determined by majority vote
    """
    values = np.asarray(labels, dtype=int)

    if values.ndim != 1:
        raise ValueError("Labels must be one-dimensional.")

    if n_segments <= 0:
        raise ValueError("Number of segments must be positive.")

    if n_segments > len(values):
        raise ValueError("Number of segments cannot exceed label length.")

    invalid_values = set(values.tolist()) - {0, 1}

    if invalid_values:
        raise ValueError(f"Labels must be binary. Invalid values: {invalid_values}")

    segments = np.array_split(values, n_segments)

    aggregated_labels = []

    for segment in segments:
        if strategy == "any":
            aggregated_labels.append(int(segment.max()))
        elif strategy == "majority":
            aggregated_labels.append(int(segment.mean() >= 0.5))
        else:
            raise ValueError(f"Unsupported aggregation strategy: {strategy}")

    return aggregated_labels


def create_pattern_labels(
    segment_labels: list[int] | np.ndarray,
    window_size: int,
    strategy: str = "any",
) -> list[int]:
    """
    Creates one binary label for each sliding-window pattern.

    Example:
    segment_labels = [0, 0, 1, 0]
    window_size = 2

    sliding windows:
        [0, 0] -> 0
        [0, 1] -> 1
        [1, 0] -> 1

    pattern_labels = [0, 1, 1]
    """
    values = np.asarray(segment_labels, dtype=int)

    if values.ndim != 1:
        raise ValueError("Segment labels must be one-dimensional.")

    if window_size <= 0:
        raise ValueError("Window size must be positive.")

    if window_size > len(values):
        raise ValueError("Window size cannot exceed segment label length.")

    invalid_values = set(values.tolist()) - {0, 1}

    if invalid_values:
        raise ValueError(f"Segment labels must be binary. Invalid values: {invalid_values}")

    pattern_labels = []

    for index in range(len(values) - window_size + 1):
        window = values[index:index + window_size]

        if strategy == "any":
            pattern_labels.append(int(window.max()))
        elif strategy == "majority":
            pattern_labels.append(int(window.mean() >= 0.5))
        else:
            raise ValueError(f"Unsupported pattern label strategy: {strategy}")

    return pattern_labels