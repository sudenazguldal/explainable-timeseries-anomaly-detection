from pathlib import Path
import csv
import json
from collections import defaultdict

from src.evaluation.statistical_tests import run_wilcoxon_test


AUTOMATA_SKAB_PATH = Path("reports/results/automata_metrics_skab.json")
AUTOMATA_BATADAL_PATH = Path("reports/results/automata_metrics_batadal.json")
DL_METRICS_PATH = Path("reports/results/deep_learning/dl_evaluation_metrics.json")

OUTPUT_JSON_PATH = Path("reports/results/statistical_analysis_results.json")
OUTPUT_CSV_PATH = Path("reports/tables/statistical_analysis_summary.csv")


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def get_skab_automata_fold_f1() -> list[float]:
    payload = load_json(AUTOMATA_SKAB_PATH)
    return [
        float(fold["metrics"]["f1_score"])
        for fold in payload["folds"]
    ]


def get_skab_dl_split_mean_f1(records: list[dict], model_name: str) -> list[float]:
    split_scores = defaultdict(list)

    for record in records:
        if record["dataset_name"] == "skab" and record["model_name"] == model_name:
            split_scores[record["split_name"]].append(
                float(record["metrics"]["f1_score"])
            )

    return [
        sum(split_scores[f"split_{index}"]) / len(split_scores[f"split_{index}"])
        for index in range(1, 6)
    ]


def get_paired_dl_f1(records: list[dict], dataset_name: str) -> tuple[list[float], list[float]]:
    paired = defaultdict(dict)

    for record in records:
        if record["dataset_name"] != dataset_name:
            continue

        key = (
            record.get("split_name", "single_split"),
            int(record["seed"]),
        )

        paired[key][record["model_name"]] = float(record["metrics"]["f1_score"])

    lstm_scores = []
    cnn_scores = []

    for key in sorted(paired.keys()):
        if "lstm" in paired[key] and "cnn1d" in paired[key]:
            lstm_scores.append(paired[key]["lstm"])
            cnn_scores.append(paired[key]["cnn1d"])

    return lstm_scores, cnn_scores


def add_result(results: list[dict], comparison: str, dataset: str, test_result: dict) -> None:
    results.append({
        "dataset": dataset,
        "comparison": comparison,
        "test": test_result["test"],
        "statistic": test_result["statistic"],
        "p_value": test_result["p_value"],
        "n_pairs": test_result["n_pairs"],
        "significant_at_0_05": test_result["p_value"] < 0.05,
    })


def export_results(results: list[dict]) -> None:
    OUTPUT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_JSON_PATH.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    fieldnames = [
        "dataset",
        "comparison",
        "test",
        "statistic",
        "p_value",
        "n_pairs",
        "significant_at_0_05",
    ]

    with OUTPUT_CSV_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def main() -> None:
    results = []

    dl_records = load_json(DL_METRICS_PATH)

    # SKAB Automata vs DL models
    skab_automata_f1 = get_skab_automata_fold_f1()

    skab_lstm_split_mean_f1 = get_skab_dl_split_mean_f1(
        records=dl_records,
        model_name="lstm",
    )

    skab_cnn_split_mean_f1 = get_skab_dl_split_mean_f1(
        records=dl_records,
        model_name="cnn1d",
    )

    add_result(
        results=results,
        comparison="automata_vs_lstm_f1",
        dataset="skab",
        test_result=run_wilcoxon_test(
            model_a_scores=skab_automata_f1,
            model_b_scores=skab_lstm_split_mean_f1,
        ),
    )

    add_result(
        results=results,
        comparison="automata_vs_cnn1d_f1",
        dataset="skab",
        test_result=run_wilcoxon_test(
            model_a_scores=skab_automata_f1,
            model_b_scores=skab_cnn_split_mean_f1,
        ),
    )

    # SKAB LSTM vs CNN1D
    skab_lstm_f1, skab_cnn_f1 = get_paired_dl_f1(
        records=dl_records,
        dataset_name="skab",
    )

    add_result(
        results=results,
        comparison="lstm_vs_cnn1d_f1",
        dataset="skab",
        test_result=run_wilcoxon_test(
            model_a_scores=skab_lstm_f1,
            model_b_scores=skab_cnn_f1,
        ),
    )

    # BATADAL LSTM vs CNN1D
    batadal_lstm_f1, batadal_cnn_f1 = get_paired_dl_f1(
        records=dl_records,
        dataset_name="batadal",
    )

    add_result(
        results=results,
        comparison="lstm_vs_cnn1d_f1",
        dataset="batadal",
        test_result=run_wilcoxon_test(
            model_a_scores=batadal_lstm_f1,
            model_b_scores=batadal_cnn_f1,
        ),
    )

    export_results(results)

    print("Statistical analysis completed.")
    print(f"Wrote JSON: {OUTPUT_JSON_PATH}")
    print(f"Wrote CSV: {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    main()