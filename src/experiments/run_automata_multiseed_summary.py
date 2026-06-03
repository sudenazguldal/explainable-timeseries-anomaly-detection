import copy
import csv
import json
from pathlib import Path
from typing import Any

from src.config.load_config import load_config
from src.experiments.run_batadal_automata_metrics import run_batadal_automata_metrics
from src.experiments.run_batadal_automata_noise_metrics import run_batadal_automata_noise_metrics
from src.experiments.run_batadal_automata_unseen_metrics import run_batadal_automata_unseen_metrics
from src.experiments.run_skab_automata_metrics import run_skab_automata_metrics
from src.experiments.run_skab_automata_noise_metrics import run_skab_automata_noise_metrics
from src.experiments.run_skab_automata_unseen_metrics import run_skab_automata_unseen_metrics


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0

    avg = mean(values)
    variance = sum((value - avg) ** 2 for value in values) / (len(values) - 1)

    return variance ** 0.5


def with_single_seed(config: dict, seed: int) -> dict:
    experiment_config = copy.deepcopy(config)
    experiment_config["project"]["random_seeds"] = [seed]

    return experiment_config


def get_nested_value(data: dict, path: list[str], default: Any = None) -> Any:
    current = data

    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default

        current = current[key]

    return current


def mean_from_folds(report: dict, summary_key: str, default: float = 0.0) -> float:
    folds = report.get("folds", [])

    values = [
        fold["automata_summary"].get(summary_key, default)
        for fold in folds
    ]

    return mean(values)


def flatten_report_for_raw_row(
    report: dict,
    seed: int | str,
    seed_policy: str,
) -> dict:
    """
    Converts a scenario report into one raw row.

    For BATADAL reports:
    - metrics are top-level values.

    For SKAB reports:
    - metric_summary contains fold mean/std.
    - automata summaries are averaged across folds.
    """
    dataset = report["dataset"]
    scenario = report.get("scenario", "original")
    parameters = report["automata_parameters"]

    if "metric_summary" in report:
        metrics = {
            "accuracy": report["metric_summary"]["accuracy"]["mean"],
            "precision": report["metric_summary"]["precision"]["mean"],
            "recall": report["metric_summary"]["recall"]["mean"],
            "f1_score": report["metric_summary"]["f1_score"]["mean"],
        }

        automata_summary = {
            "state_count": mean_from_folds(report, "state_count"),
            "transition_density": mean_from_folds(report, "transition_density"),
            "train_pattern_count": mean_from_folds(report, "train_pattern_count"),
            "test_pattern_count": mean_from_folds(report, "test_pattern_count"),
            "unseen_test_pattern_count": mean_from_folds(report, "unseen_test_pattern_count"),
            "unseen_ratio": mean_from_folds(report, "unseen_ratio"),
            "average_edit_distance": mean_from_folds(report, "average_edit_distance"),
        }

        fold_count = report.get("fold_count", len(report.get("folds", [])))
    else:
        metrics = report["metrics"]
        automata_summary = report["automata_summary"]
        fold_count = 1

    return {
        "dataset": dataset,
        "scenario": scenario,
        "seed": seed,
        "seed_policy": seed_policy,
        "fold_count": fold_count,

        "window_size": parameters["window_size"],
        "alphabet_size": parameters["alphabet_size"],
        "paa_segments": parameters.get("paa_segments"),
        "fallback_probability": parameters.get("fallback_probability", 0.000001),
        "anomaly_threshold": parameters["anomaly_threshold"],

        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1_score": metrics["f1_score"],

        "state_count": automata_summary.get("state_count", 0.0),
        "transition_density": automata_summary.get("transition_density", 0.0),
        "train_pattern_count": automata_summary.get("train_pattern_count", 0.0),
        "test_pattern_count": automata_summary.get("test_pattern_count", 0.0),
        "unseen_test_pattern_count": automata_summary.get("unseen_test_pattern_count", 0.0),
        "unseen_ratio": automata_summary.get("unseen_ratio", 0.0),
        "average_edit_distance": automata_summary.get("average_edit_distance", 0.0),
    }


def summarize_raw_rows(raw_rows: list[dict]) -> list[dict]:
    group_keys = ["dataset", "scenario", "seed_policy"]

    metric_columns = [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "state_count",
        "transition_density",
        "train_pattern_count",
        "test_pattern_count",
        "unseen_test_pattern_count",
        "unseen_ratio",
        "average_edit_distance",
    ]

    grouped: dict[tuple, list[dict]] = {}

    for row in raw_rows:
        key = tuple(row[group_key] for group_key in group_keys)
        grouped.setdefault(key, []).append(row)

    summary_rows = []

    for key, rows in grouped.items():
        base = {
            "dataset": key[0],
            "scenario": key[1],
            "seed_policy": key[2],
            "n_runs": len(rows),
            "seeds": ",".join(str(row["seed"]) for row in rows),
            "window_size": rows[0]["window_size"],
            "alphabet_size": rows[0]["alphabet_size"],
            "paa_segments": rows[0]["paa_segments"],
            "fallback_probability": rows[0]["fallback_probability"],
            "anomaly_threshold": rows[0]["anomaly_threshold"],
        }

        for column in metric_columns:
            values = [float(row[column]) for row in rows]

            base[f"{column}_mean"] = mean(values)
            base[f"{column}_std"] = sample_std(values)

        summary_rows.append(base)

    return sorted(
        summary_rows,
        key=lambda row: (row["dataset"], row["scenario"], row["seed_policy"]),
    )


def run_multiseed_summary(config: dict) -> tuple[list[dict], list[dict]]:
    seeds = config["project"]["random_seeds"]

    raw_rows = []

    # SKAB: seed affects StratifiedGroupKFold shuffle/random_state.
    skab_seeded_experiments = [
        ("original", run_skab_automata_metrics),
        ("gaussian_noise", run_skab_automata_noise_metrics),
        ("unseen_only", run_skab_automata_unseen_metrics),
    ]

    for scenario_name, runner in skab_seeded_experiments:
        for seed in seeds:
            seed_config = with_single_seed(config, seed)
            report = runner(seed_config)

            raw_rows.append(
                flatten_report_for_raw_row(
                    report=report,
                    seed=seed,
                    seed_policy="five_seed_repeated",
                )
            )

            print(
                f"SKAB | {scenario_name} | seed={seed} | "
                f"f1={raw_rows[-1]['f1_score']:.4f}"
            )

    # BATADAL gaussian noise: seed affects injected Gaussian noise.
    for seed in seeds:
        seed_config = with_single_seed(config, seed)
        report = run_batadal_automata_noise_metrics(seed_config)

        raw_rows.append(
            flatten_report_for_raw_row(
                report=report,
                seed=seed,
                seed_policy="five_seed_repeated",
            )
        )

        print(
            f"BATADAL | gaussian_noise | seed={seed} | "
            f"f1={raw_rows[-1]['f1_score']:.4f}"
        )

    # BATADAL original and unseen are deterministic because the split is time-ordered.
    deterministic_batadal_experiments = [
        ("original", run_batadal_automata_metrics),
        ("unseen_only", run_batadal_automata_unseen_metrics),
    ]

    for scenario_name, runner in deterministic_batadal_experiments:
        report = runner(config)

        raw_rows.append(
            flatten_report_for_raw_row(
                report=report,
                seed="N/A",
                seed_policy="deterministic_time_split",
            )
        )

        print(
            f"BATADAL | {scenario_name} | deterministic_time_split | "
            f"f1={raw_rows[-1]['f1_score']:.4f}"
        )

    summary_rows = summarize_raw_rows(raw_rows)

    return raw_rows, summary_rows


def write_csv(rows: list[dict], output_path: Path) -> None:
    if not rows:
        raise ValueError("Cannot write empty CSV.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(data, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def main() -> None:
    config = load_config("config.yaml")

    raw_rows, summary_rows = run_multiseed_summary(config)

    output_dir = Path("reports/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_csv_path = output_dir / "automata_multiseed_raw.csv"
    summary_csv_path = output_dir / "automata_multiseed_summary.csv"
    summary_json_path = output_dir / "automata_multiseed_summary.json"

    write_csv(raw_rows, raw_csv_path)
    write_csv(summary_rows, summary_csv_path)
    write_json(summary_rows, summary_json_path)

    print("\nAutomata multi-seed summary completed.")
    print(f"Raw CSV written to: {raw_csv_path}")
    print(f"Summary CSV written to: {summary_csv_path}")
    print(f"Summary JSON written to: {summary_json_path}")

    print("\nSummary")
    for row in summary_rows:
        print(
            f"{row['dataset']} | {row['scenario']} | {row['seed_policy']} | "
            f"n={row['n_runs']} | "
            f"f1_mean={row['f1_score_mean']:.4f}, "
            f"f1_std={row['f1_score_std']:.4f}, "
            f"unseen_ratio_mean={row['unseen_ratio_mean']:.4f}"
        )


if __name__ == "__main__":
    main()