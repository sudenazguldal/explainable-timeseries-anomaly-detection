from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def calculate_classification_metrics(
    y_true: list[int],
    y_pred: list[int],
) -> dict:
    """
    Calculates standard binary classification metrics.

    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def calculate_confusion_matrix(
    y_true: list[int],
    y_pred: list[int],
) -> list[list[int]]:
    """
    Returns confusion matrix as a JSON-serializable nested list.

    Format:
       [TN, FP],
       [FN, TP]    
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    return matrix.astype(int).tolist()


def probability_to_binary_prediction(
    probability: float,
    anomaly_threshold: float,
) -> int:
    """
    Converts automata probability into binary prediction.

    Low probability path => anomaly
    High probability path => normal
    """
    probability = float(probability)
    anomaly_threshold = float(anomaly_threshold)

    if probability < anomaly_threshold:
        return 1

    return 0


def probabilities_to_binary_predictions(
    probabilities: list[float],
    anomaly_threshold: float,
) -> list[int]:
    """
    Converts a list of automata probabilities into binary predictions.
    """
    return [
        probability_to_binary_prediction(
            probability=probability,
            anomaly_threshold=anomaly_threshold,
        )
        for probability in probabilities
    ]