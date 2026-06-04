from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


EVALUATION_METRICS_PATH = Path("reports/results/deep_learning/dl_evaluation_metrics.json")


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
        summary_rows.append(
            {
                "dataset_name": dataset_name,
                "model_name": model_name,
                "run_count": len(group_records),
            }
        )

    return summary_rows
