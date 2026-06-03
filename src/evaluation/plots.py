from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    auc,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from src.evaluation.metrics import calculate_binary_confusion_matrix


def save_confusion_matrix_heatmap(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    save_path: str | Path,
    class_names: tuple[str, str] = ("Normal", "Anomaly"),
    title: str = "Confusion Matrix",
) -> Path:
    """
    Generates and saves a binary confusion matrix heatmap.
    """
    matrix = calculate_binary_confusion_matrix(y_true, y_pred)
    output_path = _prepare_output_path(save_path)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=False,
    )
    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    return output_path


def save_precision_recall_curve(
    y_true: Sequence[int] | np.ndarray,
    y_scores: Sequence[float] | np.ndarray,
    save_path: str | Path,
    title: str = "Precision-Recall Curve",
) -> Path:
    """
    Generates and saves a Precision-Recall curve for anomaly scores.
    """
    true_labels, score_values = _validate_curve_inputs(y_true, y_scores)
    precision, recall, _ = precision_recall_curve(true_labels, score_values, pos_label=1)
    average_precision = average_precision_score(true_labels, score_values)
    output_path = _prepare_output_path(save_path)

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f"AP = {average_precision:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    return output_path


def save_roc_curve(
    y_true: Sequence[int] | np.ndarray,
    y_scores: Sequence[float] | np.ndarray,
    save_path: str | Path,
    title: str = "ROC Curve",
) -> Path:
    """
    Generates and saves a ROC curve for anomaly scores.
    """
    true_labels, score_values = _validate_curve_inputs(y_true, y_scores)
    false_positive_rate, true_positive_rate, _ = roc_curve(true_labels, score_values, pos_label=1)
    roc_auc = roc_auc_score(true_labels, score_values)
    output_path = _prepare_output_path(save_path)

    plt.figure(figsize=(6, 5))
    plt.plot(false_positive_rate, true_positive_rate, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    return output_path


def calculate_pr_auc(y_true: Sequence[int] | np.ndarray, y_scores: Sequence[float] | np.ndarray) -> float:
    """
    Calculates the area under the Precision-Recall curve.
    """
    true_labels, score_values = _validate_curve_inputs(y_true, y_scores)
    precision, recall, _ = precision_recall_curve(true_labels, score_values, pos_label=1)

    return float(auc(recall, precision))


def _validate_curve_inputs(
    y_true: Sequence[int] | np.ndarray,
    y_scores: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    true_labels = np.asarray(y_true).astype(int).reshape(-1)
    score_values = np.asarray(y_scores).astype(float).reshape(-1)

    if true_labels.shape[0] != score_values.shape[0]:
        raise ValueError("y_true and y_scores must have the same length.")
    if true_labels.shape[0] == 0:
        raise ValueError("Curve inputs cannot be empty.")
    if len(set(true_labels.tolist())) < 2:
        raise ValueError("Curve inputs must contain both normal and anomaly labels.")

    invalid_values = sorted(set(true_labels.tolist()) - {0, 1})
    if invalid_values:
        raise ValueError(f"y_true must contain only binary labels 0 and 1: {invalid_values}")

    return true_labels, score_values


def _prepare_output_path(save_path: str | Path) -> Path:
    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    return output_path
