import numpy as np
import pytest

from src.models.automata.paa import paa_transform


def test_paa_transform_returns_segment_means():
    series = [1, 2, 3, 4, 5, 6]

    result = paa_transform(series, n_segments=3)

    assert np.allclose(result, [1.5, 3.5, 5.5])


def test_paa_rejects_more_segments_than_values():
    with pytest.raises(ValueError):
        paa_transform([1, 2], n_segments=3)


def test_paa_rejects_non_positive_segment_count():
    with pytest.raises(ValueError):
        paa_transform([1, 2, 3], n_segments=0)