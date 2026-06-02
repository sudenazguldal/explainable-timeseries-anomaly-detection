import copy
import csv
import json
from pathlib import Path

from src.config.load_config import load_config
from src.experiments.run_batadal_automata_metrics import run_batadal_automata_metrics


def flatten_batadal_parameter_result(report: dict) -> dict:
    """
    Converts one BATADAL automata report into a flat row for comparison.
    """
    parameters = report["automata_parameters"]
    summary = report["automata_summary"]
    metrics = report["metrics"]

    return {
        "dataset": report["dataset"],
        "scenario": report.get("scenario", "original"),
        "window_size": parameters["window_size"],
        "alphabet_size": parameters["alphabet_size"],
        "smoothing": parameters["smoothing"],
        "anomaly_threshold": parameters["anomaly_threshold"],
        "state_count": summary["state_count"],
        "transition_density": summary["transition_density"],
        "train_pattern_count": summary["train_pattern_count"],
        "test_pattern_count": summary["test_pattern_count"],
        "unseen_test_pattern_count": summary["unseen_test_pattern_count"],
        "unseen_ratio": summary["unseen_ratio"],
        "train_anomaly_pattern_count": summary["train_anomaly_pattern_count"],
        "test_anomaly_pattern_count": summary["test_anomaly_pattern_count"],
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1_score": metrics["f1_score"],
    }


def run_parameter_sweep(config: dict) -> dict:
    """
    Runs BATADAL automata experiments for all required parameter combinations.

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

            report = run_batadal_automata_metrics(experiment_config)
            flat_row = flatten_batadal_parameter_result(report)

            detailed_reports.append(report)
            flat_results.append(flat_row)

            print(
                f"BATADAL | window={window_size}, alphabet={alphabet_size} | "
                f"f1={flat_row['f1_score']:.4f}, "
                f"states={flat_row['state_count']}, "
                f"density={flat_row['transition_density']:.6f}"
            )

    return {
        "dataset": "BATADAL",
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

    json_output_path = output_dir / "automata_parameter_sweep_batadal.json"
    csv_output_path = output_dir / "automata_parameter_sweep_batadal.csv"

    with json_output_path.open("w", encoding="utf-8") as file:
        json.dump(sweep_report, file, indent=2, ensure_ascii=False)

    write_csv(
        rows=sweep_report["results"],
        output_path=csv_output_path,
    )

    best_by_f1 = max(
        sweep_report["results"],
        key=lambda row: row["f1_score"],
    )

    print("\nBATADAL automata parameter sweep completed.")
    print(f"JSON report written to: {json_output_path}")
    print(f"CSV report written to: {csv_output_path}")

    print("\nBest configuration by F1-score")
    print(f"window_size: {best_by_f1['window_size']}")
    print(f"alphabet_size: {best_by_f1['alphabet_size']}")
    print(f"f1_score: {best_by_f1['f1_score']}")
    print(f"accuracy: {best_by_f1['accuracy']}")
    print(f"precision: {best_by_f1['precision']}")
    print(f"recall: {best_by_f1['recall']}")
    print(f"state_count: {best_by_f1['state_count']}")
    print(f"transition_density: {best_by_f1['transition_density']}")


if __name__ == "__main__":
    main()