import pytest

from src.models.automata.sax import (
    get_sax_breakpoints,
    sax_transform,
    z_normalize,
)


def test_z_normalize_returns_zero_mean_series():
    result = z_normalize([1, 2, 3])

    assert round(result.mean(), 7) == 0.0


def test_z_normalize_handles_constant_series():
    result = z_normalize([5, 5, 5])

    assert result.tolist() == [0.0, 0.0, 0.0]


def test_sax_transform_returns_symbol_string_with_expected_length():
    result = sax_transform([1, 2, 3, 4], alphabet_size=3)

    assert isinstance(result, str)
    assert len(result) == 4
    assert set(result).issubset({"a", "b", "c"})


def test_sax_rejects_invalid_alphabet_size():
    with pytest.raises(ValueError):
        get_sax_breakpoints(alphabet_size=1)