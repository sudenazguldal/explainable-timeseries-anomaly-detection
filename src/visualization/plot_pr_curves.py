from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

from src.config.load_config import load_config
from src.experiments.run_batadal_automata_metrics import run_batadal_automata_metrics
from src.experiments.run_skab_automata_metrics import run_skab_automata_metrics



CHERRY_RED = "#73070E"
DILL_GREEN = "#4E6813"

DATASET_COLORS = {
    "BATADAL": CHERRY_RED,
    "SKAB": DILL_GREEN,
}

def ensure_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def collect_single_report_scores(report: dict) -> tuple[list[int], list[float], list[float]]:
    prediction_outputs = report["prediction_outputs"]

    return (
        [int(value) for value in prediction_outputs["y_true"]],
        [float(value) for value in prediction_outputs["probabilities"]],
        [float(value) for value in prediction_outputs["anomaly_scores"]],
    )


def collect_folded_report_scores(report: dict) -> tuple[list[int], list[float], list[float]]:
    y_true = []
    probabilities = []
    anomaly_scores = []

    for fold in report["folds"]:
        prediction_outputs = fold["prediction_outputs"]

        y_true.extend([
            int(value)
            for value in prediction_outputs["y_true"]
        ])

        probabilities.extend([
            float(value)
            for value in prediction_outputs["probabilities"]
        ])

        anomaly_scores.extend([
            float(value)
            for value in prediction_outputs["anomaly_scores"]
        ])

    return y_true, probabilities, anomaly_scores


def calculate_threshold_point(
    y_true: list[int],
    probabilities: list[float],
    anomaly_threshold: float,
) -> tuple[float, float]:
    """
    Calculates precision/recall at the selected automata threshold.

    Automata decision rule:
    probability < anomaly_threshold => anomaly
    """
    y_pred = [
        1 if probability < anomaly_threshold else 0
        for probability in probabilities
    ]

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    return float(precision), float(recall)


def plot_precision_recall_curve(
    y_true: list[int],
    probabilities: list[float],
    anomaly_scores: list[float],
    anomaly_threshold: float,
    title: str,
    output_path: Path,
    curve_color: str,
) -> None:
    precision, recall, _ = precision_recall_curve(
        y_true,
        anomaly_scores,
    )

    average_precision = average_precision_score(
        y_true,
        anomaly_scores,
    )

    positive_rate = sum(y_true) / len(y_true) if y_true else 0.0

    threshold_precision, threshold_recall = calculate_threshold_point(
        y_true=y_true,
        probabilities=probabilities,
        anomaly_threshold=anomaly_threshold,
    )

    plt.figure(figsize=(8, 6))

    plt.step(
        recall,
        precision,
        where="post",
        color=curve_color,
        label=f"PR curve, AP={average_precision:.3f}",
    )

    plt.axhline(
        y=positive_rate,
        linestyle="--",
        linewidth=1,
        color="gray",
        label=f"No-skill baseline={positive_rate:.3f}",
    )

    plt.scatter(
        threshold_recall,
        threshold_precision,
        s=70,
        color=curve_color,
        edgecolors="black",
        label=f"Selected threshold={anomaly_threshold:.6f}",
        zorder=3,
    )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{title}\nAverage Precision={average_precision:.3f}")
    plt.legend(loc="upper right")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    config = load_config("config.yaml")

    output_dir = Path("reports/figures")
    ensure_output_dir(output_dir)

    batadal_report = run_batadal_automata_metrics(config)
    skab_report = run_skab_automata_metrics(config)

    batadal_y_true, batadal_probabilities, batadal_scores = collect_single_report_scores(
        batadal_report,
    )

    skab_y_true, skab_probabilities, skab_scores = collect_folded_report_scores(
        skab_report,
    )

    batadal_threshold = batadal_report["automata_parameters"]["anomaly_threshold"]
    skab_threshold = skab_report["automata_parameters"]["anomaly_threshold"]

    plot_precision_recall_curve(
        y_true=batadal_y_true,
        probabilities=batadal_probabilities,
        anomaly_scores=batadal_scores,
        anomaly_threshold=batadal_threshold,
        title="BATADAL Original Automata Precision-Recall Curve",
        output_path=output_dir / "pr_curve_batadal_original.png",
        curve_color=DATASET_COLORS["BATADAL"],
    )

    plot_precision_recall_curve(
        y_true=skab_y_true,
        probabilities=skab_probabilities,
        anomaly_scores=skab_scores,
        anomaly_threshold=skab_threshold,
        title="SKAB Original Automata Precision-Recall Curve",
        output_path=output_dir / "pr_curve_skab_original.png",
        curve_color=DATASET_COLORS["SKAB"],
    )

    print("Precision-Recall curves created.")
    print("Created files:")
    print("- reports/figures/pr_curve_batadal_original.png")
    print("- reports/figures/pr_curve_skab_original.png")


if __name__ == "__main__":
    main()