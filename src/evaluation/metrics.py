import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)


def calculate_classification_metrics(
    y_true: list[int],
    y_pred: list[int],
) -> dict:
    """
    Calculates standard binary classification metrics for anomaly detection.
    """
    true_labels, predicted_labels = _validate_binary_inputs(y_true, y_pred)

    return {
        "accuracy": calculate_accuracy(true_labels, predicted_labels),
        "precision": calculate_precision(true_labels, predicted_labels),
        "recall": calculate_recall(true_labels, predicted_labels),
        "f1_score": calculate_f1_score(true_labels, predicted_labels),
    }


def calculate_confusion_matrix(
    y_true: list[int],
    y_pred: list[int],
) -> list[list[int]]:
    """
    Returns confusion matrix as a JSON-serializable nested list.

    Format:
       [TN, FP]
       [FN, TP]
    """
    return calculate_binary_confusion_matrix(y_true, y_pred).astype(int).tolist()


def calculate_accuracy(y_true: list[int] | np.ndarray, y_pred: list[int] | np.ndarray) -> float:
    """
    Calculates binary classification accuracy.
    """
    true_labels, predicted_labels = _validate_binary_inputs(y_true, y_pred)

    return float(accuracy_score(true_labels, predicted_labels))


def calculate_precision(y_true: list[int] | np.ndarray, y_pred: list[int] | np.ndarray) -> float:
    """
    Calculates binary precision with anomaly as the positive class.
    """
    true_labels, predicted_labels = _validate_binary_inputs(y_true, y_pred)

    return float(precision_score(true_labels, predicted_labels, pos_label=1, zero_division=0))


def calculate_recall(y_true: list[int] | np.ndarray, y_pred: list[int] | np.ndarray) -> float:
    """
    Calculates binary recall with anomaly as the positive class.
    """
    true_labels, predicted_labels = _validate_binary_inputs(y_true, y_pred)

    return float(recall_score(true_labels, predicted_labels, pos_label=1, zero_division=0))


def calculate_f1_score(y_true: list[int] | np.ndarray, y_pred: list[int] | np.ndarray) -> float:
    """
    Calculates binary F1-score with anomaly as the positive class.
    """
    true_labels, predicted_labels = _validate_binary_inputs(y_true, y_pred)

    return float(f1_score(true_labels, predicted_labels, pos_label=1, zero_division=0))


def calculate_binary_confusion_matrix(y_true: list[int] | np.ndarray, y_pred: list[int] | np.ndarray) -> np.ndarray:
    """
    Calculates a binary confusion matrix with labels ordered as normal then anomaly.
    """
    true_labels, predicted_labels = _validate_binary_inputs(y_true, y_pred)

    return confusion_matrix(true_labels, predicted_labels, labels=[0, 1])


def probability_to_binary_prediction(
    probability: float,
    anomaly_threshold: float,
    higher_is_anomaly: bool = False,
) -> int:
    """
    Converts a probability/score into a binary prediction.

    Two scoring conventions are supported:
    - Automata transition probabilities (default, higher_is_anomaly=False):
      low probability path => anomaly, high probability path => normal.
    - Deep learning anomaly scores (higher_is_anomaly=True):
      high score => anomaly, low score => normal.
    """
    probability = float(probability)
    anomaly_threshold = float(anomaly_threshold)

    if higher_is_anomaly:
        return 1 if probability >= anomaly_threshold else 0

    if probability < anomaly_threshold:
        return 1

    return 0


def probabilities_to_binary_predictions(
    probabilities: list[float],
    anomaly_threshold: float,
    higher_is_anomaly: bool = False,
) -> list[int]:
    """
    Converts a list of probabilities/scores into binary predictions.

    See probability_to_binary_prediction for the higher_is_anomaly convention.
    """
    return [
        probability_to_binary_prediction(
            probability=probability,
            anomaly_threshold=anomaly_threshold,
            higher_is_anomaly=higher_is_anomaly,
        )
        for probability in probabilities
    ]


def _validate_binary_inputs(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    true_labels = np.asarray(y_true).astype(int).reshape(-1)
    predicted_labels = np.asarray(y_pred).astype(int).reshape(-1)

    if true_labels.shape[0] != predicted_labels.shape[0]:
        raise ValueError("y_true and y_pred must have the same length.")
    if true_labels.shape[0] == 0:
        raise ValueError("Metric inputs cannot be empty.")

    _validate_binary_label_values(true_labels, "y_true")
    _validate_binary_label_values(predicted_labels, "y_pred")

    return true_labels, predicted_labels


def _validate_binary_label_values(labels: np.ndarray, input_name: str) -> None:
    invalid_values = sorted(set(labels.tolist()) - {0, 1})
    if invalid_values:
        raise ValueError(f"{input_name} must contain only binary labels 0 and 1: {invalid_values}")
