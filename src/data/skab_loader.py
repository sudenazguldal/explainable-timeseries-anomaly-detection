from pathlib import Path
import pandas as pd


def load_skab_dataset(raw_path: str, use_groups: list[str]) -> pd.DataFrame:
    """
    Loads SKAB CSV files from selected groups and concatenates them.

    Adds:
    - source_group: valve1 or valve2
    - source_file: original CSV filename

    These metadata columns are used for traceability and group-based splitting, not as model input.
    """
    base_path = Path(raw_path)

    if not base_path.exists():
        raise FileNotFoundError(f"SKAB raw path not found: {raw_path}")

    frames = []

    for group in use_groups:
        group_path = base_path / group

        if not group_path.exists():
            raise FileNotFoundError(f"SKAB group path not found: {group_path}")

        csv_files = sorted(group_path.glob("*.csv"))

        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in: {group_path}")

        for csv_file in csv_files:
            df = pd.read_csv(csv_file, sep=None, engine="python")
            df["source_group"] = group
            df["source_file"] = csv_file.name
            frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    return combined


def get_skab_feature_columns(
    df: pd.DataFrame,
    target_column: str,
    excluded_columns: list[str],
) -> list[str]:
    """
    Returns model input columns by excluding target and metadata/time columns.
    """
    excluded = set(excluded_columns + [target_column])

    feature_columns = [
        column for column in df.columns
        if column not in excluded
    ]

    if not feature_columns:
        raise ValueError("No feature columns found for SKAB dataset.")

    return feature_columns