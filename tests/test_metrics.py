import pytest

from src.evaluation.metrics import (
    calculate_classification_metrics,
    calculate_confusion_matrix,
    probability_to_binary_prediction,
    probabilities_to_binary_predictions,
)


def test_calculate_classification_metrics_returns_required_metrics():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]

    metrics = calculate_classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
    )

    assert set(metrics.keys()) == {
        "accuracy",
        "precision",
        "recall",
        "f1_score",
    }

    assert metrics["accuracy"] == 0.75
    assert round(metrics["precision"], 4) == 0.6667
    assert metrics["recall"] == 1.0
    assert round(metrics["f1_score"], 4) == 0.8


def test_calculate_confusion_matrix_returns_json_serializable_matrix():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]

    matrix = calculate_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
    )

    assert matrix == [
        [1, 1],
        [0, 2],
    ]


def test_metrics_reject_mismatched_lengths():
    with pytest.raises(ValueError):
        calculate_classification_metrics(
            y_true=[0, 1],
            y_pred=[0],
        )


def test_probability_to_binary_prediction_marks_low_probability_as_anomaly():
    prediction = probability_to_binary_prediction(
        probability=0.001,
        anomaly_threshold=0.05,
    )

    assert prediction == 1


def test_probability_to_binary_prediction_marks_high_probability_as_normal():
    prediction = probability_to_binary_prediction(
        probability=0.80,
        anomaly_threshold=0.05,
    )

    assert prediction == 0


def test_probabilities_to_binary_predictions_converts_list():
    predictions = probabilities_to_binary_predictions(
        probabilities=[0.001, 0.2, 0.03],
        anomaly_threshold=0.05,
    )

    assert predictions == [1, 0, 1]