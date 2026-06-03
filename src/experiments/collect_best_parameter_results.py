import json
from pathlib import Path

import pandas as pd


def normalize_parameter_sweep_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized_df = df.copy()

    if "f1_score" not in normalized_df.columns and "f1_score_mean" in normalized_df.columns:
        normalized_df["f1_score"] = normalized_df["f1_score_mean"]

    if "accuracy" not in normalized_df.columns and "accuracy_mean" in normalized_df.columns:
        normalized_df["accuracy"] = normalized_df["accuracy_mean"]

    if "precision" not in normalized_df.columns and "precision_mean" in normalized_df.columns:
        normalized_df["precision"] = normalized_df["precision_mean"]

    if "recall" not in normalized_df.columns and "recall_mean" in normalized_df.columns:
        normalized_df["recall"] = normalized_df["recall_mean"]

    if "state_count" not in normalized_df.columns and "state_count_mean" in normalized_df.columns:
        normalized_df["state_count"] = normalized_df["state_count_mean"]

    if (
        "transition_density" not in normalized_df.columns
        and "transition_density_mean" in normalized_df.columns
    ):
        normalized_df["transition_density"] = normalized_df["transition_density_mean"]

    required_columns = [
        "window_size",
        "alphabet_size",
        "f1_score",
        "state_count",
        "transition_density",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in normalized_df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    return normalized_df


def select_best_parameter_row(
    df: pd.DataFrame,
    dataset_name: str,
    source_file: str,
) -> dict:
    normalized_df = normalize_parameter_sweep_columns(df)

    best_row = normalized_df.sort_values(
        by=["f1_score", "recall", "precision", "state_count"],
        ascending=[False, False, False, True],
    ).iloc[0]

    result = {
        "dataset": dataset_name,
        "best_window_size": int(best_row["window_size"]),
        "best_alphabet_size": int(best_row["alphabet_size"]),
        "best_f1_score": float(best_row["f1_score"]),
        "state_count": float(best_row["state_count"]),
        "transition_density": float(best_row["transition_density"]),
        "source_file": source_file,
    }

    optional_columns = [
        "accuracy",
        "precision",
        "recall",
        "f1_score_std",
        "accuracy_std",
        "precision_std",
        "recall_std",
        "state_count_std",
        "transition_density_std",
    ]

    for column in optional_columns:
        if column in normalized_df.columns:
            value = best_row[column]

            if pd.notna(value):
                result[column] = float(value)

    return result


def main() -> None:
    results_dir = Path("reports/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    batadal_path = results_dir / "automata_parameter_sweep_batadal.csv"
    skab_path = results_dir / "automata_parameter_sweep_skab.csv"

    if not batadal_path.exists():
        raise FileNotFoundError(
            f"Missing file: {batadal_path}. "
            "Run src.experiments.run_batadal_automata_parameter_sweep first."
        )

    if not skab_path.exists():
        raise FileNotFoundError(
            f"Missing file: {skab_path}. "
            "Run src.experiments.run_skab_automata_parameter_sweep first."
        )

    batadal_df = pd.read_csv(batadal_path)
    skab_df = pd.read_csv(skab_path)

    best_results = [
        select_best_parameter_row(
            df=batadal_df,
            dataset_name="BATADAL",
            source_file=str(batadal_path),
        ),
        select_best_parameter_row(
            df=skab_df,
            dataset_name="SKAB",
            source_file=str(skab_path),
        ),
    ]

    summary_df = pd.DataFrame(best_results)

    csv_output_path = results_dir / "automata_best_parameter_summary.csv"
    json_output_path = results_dir / "automata_best_parameter_summary.json"

    summary_df.to_csv(csv_output_path, index=False)

    with json_output_path.open("w", encoding="utf-8") as file:
        json.dump(best_results, file, indent=2)

    print("Best automata parameter summary created.")
    print(f"CSV written to: {csv_output_path}")
    print(f"JSON written to: {json_output_path}")

    for result in best_results:
        print(
            f"{result['dataset']}: "
            f"window_size={result['best_window_size']}, "
            f"alphabet_size={result['best_alphabet_size']}, "
            f"F1={result['best_f1_score']:.3f}, "
            f"state_count={result['state_count']:.0f}, "
            f"transition_density={result['transition_density']:.4f}"
        )


if __name__ == "__main__":
    main()