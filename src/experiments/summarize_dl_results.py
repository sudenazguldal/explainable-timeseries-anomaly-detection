from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


EVALUATION_METRICS_PATH = Path("reports/results/deep_learning/dl_evaluation_metrics.json")
METRIC_NAMES = ("accuracy", "precision", "recall", "f1_score")


def load_evaluation_records(input_path: Path = EVALUATION_METRICS_PATH) -> list[dict[str, object]]:
    """
    Loads deep learning evaluation records exported by the full experiment runner.
    """
    with input_path.open("r", encoding="utf-8") as input_file:
        payload = json.load(input_file)

    if not isinstance(payload, list):
        raise ValueError("Deep learning evaluation metrics must be a JSON list.")

    return payload


def group_records_by_dataset_and_model(
    records: list[dict[str, object]],
) -> dict[tuple[str, str], list[dict[str, object]]]:
    """
    Groups evaluation records by dataset and model name.
    """
    grouped_records: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for record in records:
        dataset_name = str(record["dataset_name"])
        model_name = str(record["model_name"])
        grouped_records[(dataset_name, model_name)].append(record)

    return dict(grouped_records)


def summarize_evaluation_records(records: list[dict[str, object]]) -> list[dict[str, object]]:
    """
    Builds one summary row per dataset/model pair.
    """
    summary_rows = []
    for (dataset_name, model_name), group_records in sorted(group_records_by_dataset_and_model(records).items()):
        metric_summary = aggregate_group_metrics(group_records)
        row: dict[str, object] = {
            "dataset_name": dataset_name,
            "model_name": model_name,
            "run_count": len(group_records),
        }
        for metric_name, summary in metric_summary.items():
            row[f"{metric_name}_mean"] = summary["mean"]
            row[f"{metric_name}_std"] = summary["std"]

        summary_rows.append(row)

    return summary_rows


def aggregate_group_metrics(records: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    """
    Aggregates configured classification metrics for one dataset/model group.
    """
    return {
        metric_name: aggregate_metric_values(
            [
                float(record["metrics"][metric_name])
                for record in records
                if isinstance(record.get("metrics"), dict) and metric_name in record["metrics"]
            ]
        )
        for metric_name in METRIC_NAMES
    }


def aggregate_metric_values(values: list[float]) -> dict[str, float]:
    """
    Calculates mean and sample standard deviation for metric values.
    """
    if not values:
        return {"mean": 0.0, "std": 0.0}

    mean_value = sum(values) / len(values)
    if len(values) < 2:
        return {"mean": mean_value, "std": 0.0}

    variance = sum((value - mean_value) ** 2 for value in values) / (len(values) - 1)
    return {"mean": mean_value, "std": variance**0.5}
