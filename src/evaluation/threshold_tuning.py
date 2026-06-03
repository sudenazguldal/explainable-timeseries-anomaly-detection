from src.evaluation.metrics import (
    calculate_classification_metrics,
    calculate_confusion_matrix,
    probabilities_to_binary_predictions,
)


def select_best_threshold(
    y_true: list[int],
    probabilities: list[float],
    threshold_candidates: list[float],
    primary_metric: str = "f1_score",
) -> dict:
    """
    Selects anomaly threshold on validation data.

    Low transition probability means higher anomaly likelihood.
    Therefore:  probability < threshold => anomaly
    """
    if len(y_true) != len(probabilities):
        raise ValueError("y_true and probabilities must have the same length.")

    if not threshold_candidates:
        raise ValueError("threshold_candidates cannot be empty.")
    

    supported_metrics = {"accuracy", "precision", "recall", "f1_score"}

    if primary_metric not in supported_metrics:
        raise ValueError(f"Unsupported primary metric: {primary_metric}")

    validation_results = []

    for threshold in threshold_candidates:
        y_pred = probabilities_to_binary_predictions(
            probabilities=probabilities,
            anomaly_threshold=float(threshold),
        )

        metrics = calculate_classification_metrics(
            y_true=y_true,
            y_pred=y_pred,
        )

        confusion_matrix = calculate_confusion_matrix(
            y_true=y_true,
            y_pred=y_pred,
        )

        validation_results.append({
            "threshold": float(threshold),
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"],
            "confusion_matrix": confusion_matrix,
        })



    best_result = max(
        validation_results,
        key=lambda row: (
            row[primary_metric],
            row["recall"],
            row["precision"],
            -row["threshold"],
        ),
    )

    return {
        "method": "validation_f1_maximization",
        "primary_metric": primary_metric,
        "threshold_candidates": [float(value) for value in threshold_candidates],
        "selected_threshold": best_result["threshold"],
        "selected_metrics": {
            "accuracy": best_result["accuracy"],
            "precision": best_result["precision"],
            "recall": best_result["recall"],
            "f1_score": best_result["f1_score"],
        },
        "selected_confusion_matrix": best_result["confusion_matrix"],
        "validation_results": validation_results,
    }