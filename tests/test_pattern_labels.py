import pytest

from src.models.automata.pattern_labels import (
    aggregate_labels_by_segments,
    create_pattern_labels,
)


def test_aggregate_labels_by_segments_with_any_strategy():
    labels = [0, 0, 1, 0, 0, 0]

    segment_labels = aggregate_labels_by_segments(
        labels=labels,
        n_segments=3,
        strategy="any",
    )

    assert segment_labels == [0, 1, 0]


def test_aggregate_labels_by_segments_with_majority_strategy():
    labels = [0, 1, 1, 0, 0, 0]

    segment_labels = aggregate_labels_by_segments(
        labels=labels,
        n_segments=2,
        strategy="majority",
    )

    assert segment_labels == [1, 0]


def test_create_pattern_labels_with_any_strategy():
    segment_labels = [0, 0, 1, 0]

    pattern_labels = create_pattern_labels(
        segment_labels=segment_labels,
        window_size=2,
        strategy="any",
    )

    assert pattern_labels == [0, 1, 1]


def test_create_pattern_labels_with_majority_strategy():
    segment_labels = [0, 1, 1, 0]

    pattern_labels = create_pattern_labels(
        segment_labels=segment_labels,
        window_size=2,
        strategy="majority",
    )

    assert pattern_labels == [1, 1, 1]


def test_aggregate_labels_rejects_non_binary_values():
    with pytest.raises(ValueError):
        aggregate_labels_by_segments(
            labels=[0, 2, 1],
            n_segments=3,
        )


def test_create_pattern_labels_rejects_too_large_window():
    with pytest.raises(ValueError):
        create_pattern_labels(
            segment_labels=[0, 1],
            window_size=3,
        )