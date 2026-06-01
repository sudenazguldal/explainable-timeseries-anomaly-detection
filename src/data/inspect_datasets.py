import json
from pathlib import Path

from src.config.load_config import load_config
from src.data.skab_loader import load_skab_dataset, get_skab_feature_columns
from src.data.batadal_loader import (
    load_batadal_dataset,
    detect_batadal_target_column,
    detect_batadal_time_column,
    get_batadal_feature_columns,
    split_batadal_time_ordered,
)


def value_counts_as_dict(series):
    counts = series.value_counts(dropna=False).to_dict()
    return {str(key): int(value) for key, value in counts.items()}


def inspect_skab(config: dict) -> dict:
    skab_config = config["datasets"]["skab"]

    df = load_skab_dataset(
        raw_path=skab_config["raw_path"],
        use_groups=skab_config["use_groups"],
    )

    target_column = skab_config["target_column"]

    if target_column not in df.columns:
        raise ValueError(
            f"SKAB target column '{target_column}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    feature_columns = get_skab_feature_columns(
        df=df,
        target_column=target_column,
        excluded_columns=skab_config["excluded_columns"],
    )

    summary = {
        "dataset": "SKAB",
        "rows": int(len(df)),
        "columns": list(df.columns),
        "feature_column_count": int(len(feature_columns)),
        "feature_columns": feature_columns,
        "target_column": target_column,
        "target_distribution": value_counts_as_dict(df[target_column]),
        "source_group_distribution": value_counts_as_dict(df["source_group"]),
        "source_file_count": int(df["source_file"].nunique()),
        "source_files": sorted(df["source_file"].unique().tolist()),
    }

    return summary


def inspect_batadal(config: dict) -> dict:
    batadal_config = config["datasets"]["batadal"]

    df = load_batadal_dataset(
        raw_path=batadal_config["raw_path"],
    )

    target_column = detect_batadal_target_column(
        df=df,
        target_column_candidates=batadal_config["target_column_candidates"],
    )

    time_column = detect_batadal_time_column(
        df=df,
        time_column_candidates=batadal_config["time_column_candidates"],
    )

    feature_columns = get_batadal_feature_columns(
        df=df,
        target_column=target_column,
        time_column=time_column,
    )

    split_config = batadal_config["split"]

    train_df, validation_df, test_df = split_batadal_time_ordered(
        df=df,
        train_ratio=split_config["train"],
        validation_ratio=split_config["validation"],
        test_ratio=split_config["test"],
        time_column=time_column,
    )

    summary = {
        "dataset": "BATADAL",
        "rows": int(len(df)),
        "columns": list(df.columns),
        "feature_column_count": int(len(feature_columns)),
        "feature_columns": feature_columns,
        "target_column": target_column,
        "time_column": time_column,
        "target_distribution": value_counts_as_dict(df[target_column]),
        "split": {
            "train_rows": int(len(train_df)),
            "validation_rows": int(len(validation_df)),
            "test_rows": int(len(test_df)),
        },
        "split_target_distribution": {
            "train": value_counts_as_dict(train_df[target_column]),
            "validation": value_counts_as_dict(validation_df[target_column]),
            "test": value_counts_as_dict(test_df[target_column]),
        },
    }

    return summary


def main() -> None:
    config = load_config("config.yaml")

    skab_summary = inspect_skab(config)
    batadal_summary = inspect_batadal(config)

    report = {
        "skab": skab_summary,
        "batadal": batadal_summary,
    }

    output_path = Path("reports/dataset_schema.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print("Dataset inspection completed.")
    print(f"Report written to: {output_path}")

    print("\nSKAB")
    print(f"Rows: {skab_summary['rows']}")
    print(f"Feature columns: {skab_summary['feature_column_count']}")
    print(f"Target distribution: {skab_summary['target_distribution']}")
    print(f"Source groups: {skab_summary['source_group_distribution']}")
    print(f"Source file count: {skab_summary['source_file_count']}")

    print("\nBATADAL")
    print(f"Rows: {batadal_summary['rows']}")
    print(f"Feature columns: {batadal_summary['feature_column_count']}")
    print(f"Target column: {batadal_summary['target_column']}")
    print(f"Time column: {batadal_summary['time_column']}")
    print(f"Target distribution: {batadal_summary['target_distribution']}")
    print(f"Split: {batadal_summary['split']}")


if __name__ == "__main__":
    main()