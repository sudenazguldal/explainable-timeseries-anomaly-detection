import copy
import csv
import json
from pathlib import Path

from src.config.load_config import load_config
from src.experiments.run_skab_automata_metrics import run_skab_automata_metrics


def flatten_skab_parameter_result(report: dict) -> dict:
    """
    Converts one SKAB automata report into a flat row for comparison.

    SKAB reports contain fold-based mean and standard deviation values.
    """
    parameters = report["automata_parameters"]
    metric_summary = report["metric_summary"]

    state_counts = [
        fold["automata_summary"]["state_count"]
        for fold in report["folds"]
    ]

    transition_densities = [
        fold["automata_summary"]["transition_density"]
        for fold in report["folds"]
    ]

    unseen_counts = [
        fold["automata_summary"]["unseen_test_pattern_count"]
        for fold in report["folds"]
    ]

    train_pattern_counts = [
        fold["automata_summary"]["train_pattern_count"]
        for fold in report["folds"]
    ]

    test_pattern_counts = [
        fold["automata_summary"]["test_pattern_count"]
        for fold in report["folds"]
    ]

    return {
        "dataset": report["dataset"],
        "window_size": parameters["window_size"],
        "alphabet_size": parameters["alphabet_size"],
        "smoothing": parameters["smoothing"],
        "anomaly_threshold": parameters["anomaly_threshold"],

        "accuracy_mean": metric_summary["accuracy"]["mean"],
        "accuracy_std": metric_summary["accuracy"]["std"],
        "precision_mean": metric_summary["precision"]["mean"],
        "precision_std": metric_summary["precision"]["std"],
        "recall_mean": metric_summary["recall"]["mean"],
        "recall_std": metric_summary["recall"]["std"],
        "f1_score_mean": metric_summary["f1_score"]["mean"],
        "f1_score_std": metric_summary["f1_score"]["std"],

        "state_count_mean": sum(state_counts) / len(state_counts),
        "transition_density_mean": sum(transition_densities) / len(transition_densities),
        "unseen_test_pattern_count_mean": sum(unseen_counts) / len(unseen_counts),
        "train_pattern_count_mean": sum(train_pattern_counts) / len(train_pattern_counts),
        "test_pattern_count_mean": sum(test_pattern_counts) / len(test_pattern_counts),
    }


def run_parameter_sweep(config: dict) -> dict:
    """
    Runs SKAB automata experiments for all required parameter combinations.


    """
    window_sizes = config["automata"]["parameter_grid"]["window_size"]
    alphabet_sizes = config["automata"]["parameter_grid"]["alphabet_size"]

    detailed_reports = []
    flat_results = []

    for window_size in window_sizes:
        for alphabet_size in alphabet_sizes:
            experiment_config = copy.deepcopy(config)

            experiment_config["automata"]["fixed"]["window_size"] = window_size
            experiment_config["automata"]["fixed"]["alphabet_size"] = alphabet_size

            report = run_skab_automata_metrics(experiment_config)
            flat_row = flatten_skab_parameter_result(report)

            detailed_reports.append(report)
            flat_results.append(flat_row)

            print(
                f"SKAB | window={window_size}, alphabet={alphabet_size} | "
                f"f1_mean={flat_row['f1_score_mean']:.4f}, "
                f"f1_std={flat_row['f1_score_std']:.4f}, "
                f"state_mean={flat_row['state_count_mean']:.2f}, "
                f"density_mean={flat_row['transition_density_mean']:.6f}"
            )

    return {
        "dataset": "SKAB",
        "experiment_type": "automata_parameter_sweep",
        "window_sizes": window_sizes,
        "alphabet_sizes": alphabet_sizes,
        "results": flat_results,
        "detailed_reports": detailed_reports,
    }


def write_csv(rows: list[dict], output_path: Path) -> None:
    if not rows:
        raise ValueError("Cannot write empty parameter sweep result.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    config = load_config("config.yaml")

    sweep_report = run_parameter_sweep(config)

    output_dir = Path("reports/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    json_output_path = output_dir / "automata_parameter_sweep_skab.json"
    csv_output_path = output_dir / "automata_parameter_sweep_skab.csv"

    with json_output_path.open("w", encoding="utf-8") as file:
        json.dump(sweep_report, file, indent=2, ensure_ascii=False)

    write_csv(
        rows=sweep_report["results"],
        output_path=csv_output_path,
    )

    best_by_f1 = max(
        sweep_report["results"],
        key=lambda row: row["f1_score_mean"],
    )

    print("\nSKAB automata parameter sweep completed.")
    print(f"JSON report written to: {json_output_path}")
    print(f"CSV report written to: {csv_output_path}")

    print("\nBest configuration by mean F1-score")
    print(f"window_size: {best_by_f1['window_size']}")
    print(f"alphabet_size: {best_by_f1['alphabet_size']}")
    print(f"f1_score_mean: {best_by_f1['f1_score_mean']}")
    print(f"f1_score_std: {best_by_f1['f1_score_std']}")
    print(f"accuracy_mean: {best_by_f1['accuracy_mean']}")
    print(f"precision_mean: {best_by_f1['precision_mean']}")
    print(f"recall_mean: {best_by_f1['recall_mean']}")
    print(f"state_count_mean: {best_by_f1['state_count_mean']}")
    print(f"transition_density_mean: {best_by_f1['transition_density_mean']}")


if __name__ == "__main__":
    main()