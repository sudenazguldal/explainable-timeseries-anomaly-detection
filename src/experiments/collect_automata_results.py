import csv
import json
from pathlib import Path
from typing import Any


RESULT_FILES = [
    "automata_metrics_batadal.json",
    "automata_noise_metrics_batadal.json",
    "automata_unseen_metrics_batadal.json",
    "automata_metrics_skab.json",
    "automata_noise_metrics_skab.json",
    "automata_unseen_metrics_skab.json",
]


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def get_nested_value(data: dict, keys: list[str], default: Any = None) -> Any:
    current = data

    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default

        current = current[key]

    return current


def calculate_unseen_ratio(summary: dict) -> float:
    if "unseen_ratio" in summary:
        return float(summary["unseen_ratio"])

    unseen_count = float(summary.get("unseen_test_pattern_count", 0))
    test_count = float(summary.get("test_pattern_count", 0))

    if test_count == 0:
        return 0.0

    return unseen_count / test_count


def flatten_single_dataset_report(report: dict) -> dict:
    """
    Flattens BATADAL-style reports.

    These reports contain one top-level metrics object and one top-level
    automata_summary object.
    """
    params = report["automata_parameters"]
    metrics = report["metrics"]
    summary = report["automata_summary"]

    return {
        "dataset": report["dataset"],
        "scenario": report.get("scenario", "original"),
        "fold_count": 1,

        "window_size": params["window_size"],
        "alphabet_size": params["alphabet_size"],
        "paa_segments": params.get("paa_segments"),
        "fallback_probability": params.get(
        "fallback_probability",
        params.get("smoothing"),
    ),
        "anomaly_threshold": params["anomaly_threshold"],

        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1_score": metrics["f1_score"],

        "accuracy_std": 0.0,
        "precision_std": 0.0,
        "recall_std": 0.0,
        "f1_score_std": 0.0,

        "confusion_matrix": json.dumps(report.get("confusion_matrix", [])),

        "state_count": summary.get("state_count", 0),
        "transition_density": summary.get("transition_density", 0.0),
        "train_pattern_count": summary.get("train_pattern_count", 0),
        "test_pattern_count": summary.get("test_pattern_count", 0),
        "unseen_test_pattern_count": summary.get("unseen_test_pattern_count", 0),
        "unseen_ratio": calculate_unseen_ratio(summary),
        "average_edit_distance": summary.get("average_edit_distance", 0.0),
    }


def flatten_folded_dataset_report(report: dict) -> dict:
    """
    Flattens SKAB-style reports.

    These reports contain fold-level metrics and fold-level automata summaries.
    """
    params = report["automata_parameters"]
    metric_summary = report["metric_summary"]
    folds = report["folds"]

    state_counts = [
        fold["automata_summary"].get("state_count", 0)
        for fold in folds
    ]

    transition_densities = [
        fold["automata_summary"].get("transition_density", 0.0)
        for fold in folds
    ]

    train_pattern_counts = [
        fold["automata_summary"].get("train_pattern_count", 0)
        for fold in folds
    ]

    test_pattern_counts = [
        fold["automata_summary"].get("test_pattern_count", 0)
        for fold in folds
    ]

    unseen_counts = [
        fold["automata_summary"].get("unseen_test_pattern_count", 0)
        for fold in folds
    ]

    unseen_ratios = [
        calculate_unseen_ratio(fold["automata_summary"])
        for fold in folds
    ]

    edit_distances = [
        fold["automata_summary"].get("average_edit_distance", 0.0)
        for fold in folds
    ]

    confusion_matrices = [
        fold.get("confusion_matrix", [])
        for fold in folds
    ]

    return {
        "dataset": report["dataset"],
        "scenario": report.get("scenario", "original"),
        "fold_count": report.get("fold_count", len(folds)),

        "window_size": params["window_size"],
        "alphabet_size": params["alphabet_size"],
        "paa_segments": params.get("paa_segments"),
        "fallback_probability": params.get(
            "fallback_probability",
            params.get("smoothing"),
        ),
        "anomaly_threshold": params["anomaly_threshold"],

        "accuracy": metric_summary["accuracy"]["mean"],
        "precision": metric_summary["precision"]["mean"],
        "recall": metric_summary["recall"]["mean"],
        "f1_score": metric_summary["f1_score"]["mean"],

        "accuracy_std": metric_summary["accuracy"]["std"],
        "precision_std": metric_summary["precision"]["std"],
        "recall_std": metric_summary["recall"]["std"],
        "f1_score_std": metric_summary["f1_score"]["std"],

        "confusion_matrix": json.dumps(confusion_matrices),

        "state_count": mean(state_counts),
        "transition_density": mean(transition_densities),
        "train_pattern_count": mean(train_pattern_counts),
        "test_pattern_count": mean(test_pattern_counts),
        "unseen_test_pattern_count": mean(unseen_counts),
        "unseen_ratio": mean(unseen_ratios),
        "average_edit_distance": mean(edit_distances),
    }


def flatten_report(report: dict) -> dict:
    if "folds" in report:
        return flatten_folded_dataset_report(report)

    return flatten_single_dataset_report(report)


def collect_reports(results_dir: Path) -> list[dict]:
    rows = []
    missing_files = []

    for filename in RESULT_FILES:
        file_path = results_dir / filename

        if not file_path.exists():
            missing_files.append(filename)
            continue

        with file_path.open("r", encoding="utf-8") as file:
            report = json.load(file)

        rows.append(flatten_report(report))

    if missing_files:
        raise FileNotFoundError(
            "Some automata result files are missing. "
            "Run the corresponding experiment scripts first: "
            + ", ".join(missing_files)
        )

    return rows


def write_json(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(rows, file, indent=2, ensure_ascii=False)


def write_csv(rows: list[dict], output_path: Path) -> None:
    if not rows:
        raise ValueError("Cannot write empty automata result summary.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    results_dir = Path("reports/results")

    rows = collect_reports(results_dir)

    json_output_path = results_dir / "automata_summary_results.json"
    csv_output_path = results_dir / "automata_summary_results.csv"

    write_json(rows, json_output_path)
    write_csv(rows, csv_output_path)

    print("Automata result collection completed.")
    print(f"JSON summary written to: {json_output_path}")
    print(f"CSV summary written to: {csv_output_path}")

    print("\nSummary rows")
    for row in rows:
        print(
            f"{row['dataset']} | {row['scenario']} | "
            f"accuracy={row['accuracy']:.4f}, "
            f"precision={row['precision']:.4f}, "
            f"recall={row['recall']:.4f}, "
            f"f1={row['f1_score']:.4f}, "
            f"unseen_ratio={row['unseen_ratio']:.4f}"
        )


if __name__ == "__main__":
    main()