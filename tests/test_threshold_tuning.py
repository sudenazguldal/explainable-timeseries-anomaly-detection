import pytest

from src.evaluation.threshold_tuning import select_best_threshold


def test_select_best_threshold_returns_best_f1_threshold():
    y_true = [0, 0, 1, 1]
    probabilities = [0.90, 0.80, 0.02, 0.03]

    result = select_best_threshold(
        y_true=y_true,
        probabilities=probabilities,
        threshold_candidates=[0.01, 0.05, 0.10],
    )

    assert result["method"] == "validation_f1_maximization"
    assert result["selected_threshold"] == 0.05
    assert result["selected_metrics"]["f1_score"] == 1.0


def test_select_best_threshold_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        select_best_threshold(
            y_true=[0, 1],
            probabilities=[0.5],
            threshold_candidates=[0.05],
        )


def test_select_best_threshold_rejects_empty_candidates():
    with pytest.raises(ValueError):
        select_best_threshold(
            y_true=[0, 1],
            probabilities=[0.5, 0.01],
            threshold_candidates=[],
        )

def test_select_best_threshold_rejects_unsupported_primary_metric():
    with pytest.raises(ValueError):
        select_best_threshold(
            y_true=[0, 1],
            probabilities=[0.90, 0.01],
            threshold_candidates=[0.05],
            primary_metric="auc",
        )


def test_select_best_threshold_uses_smaller_threshold_when_scores_tie():
    y_true = [0, 0, 1, 1]
    probabilities = [0.90, 0.80, 0.02, 0.03]

    result = select_best_threshold(
        y_true=y_true,
        probabilities=probabilities,
        threshold_candidates=[0.05, 0.10],
    )

    assert result["selected_threshold"] == 0.05


def test_select_best_threshold_returns_reportable_validation_results():
    result = select_best_threshold(
        y_true=[0, 0, 1, 1],
        probabilities=[0.90, 0.80, 0.02, 0.03],
        threshold_candidates=[0.01, 0.05, 0.10],
    )

    assert result["primary_metric"] == "f1_score"
    assert result["threshold_candidates"] == [0.01, 0.05, 0.10]
    assert "selected_threshold" in result
    assert "selected_metrics" in result
    assert "selected_confusion_matrix" in result
    assert "validation_results" in result
    assert len(result["validation_results"]) == 3

    for row in result["validation_results"]:
        assert "threshold" in row
        assert "accuracy" in row
        assert "precision" in row
        assert "recall" in row
        assert "f1_score" in row
        assert "confusion_matrix" in row

def test_select_best_threshold_rejects_empty_validation_data():
    with pytest.raises(ValueError):
        select_best_threshold(
            y_true=[],
            probabilities=[],
            threshold_candidates=[0.05],
        )