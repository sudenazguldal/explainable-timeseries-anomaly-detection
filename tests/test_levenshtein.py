import pytest

from src.models.automata.levenshtein import (
    levenshtein_distance,
    find_nearest_pattern,
)


def test_levenshtein_distance_returns_zero_for_same_pattern():
    assert levenshtein_distance("abc", "abc") == 0


def test_levenshtein_distance_counts_substitution():
    assert levenshtein_distance("abc", "adc") == 1


def test_levenshtein_distance_counts_insertion():
    assert levenshtein_distance("abc", "abcc") == 1


def test_find_nearest_pattern_returns_closest_known_pattern():
    known_patterns = {"abc", "bbb", "ccc"}

    nearest_pattern, distance = find_nearest_pattern(
        pattern="adc",
        known_patterns=known_patterns,
    )

    assert nearest_pattern == "abc"
    assert distance == 1


def test_find_nearest_pattern_rejects_empty_known_set():
    with pytest.raises(ValueError):
        find_nearest_pattern("abc", set())