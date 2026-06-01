from pathlib import Path
import pandas as pd


def load_batadal_dataset(raw_path: str) -> pd.DataFrame:
    """
    Loads BATADAL Training Dataset 2 CSV file

    The project uses only Training Dataset 2. If multiple CSV files are placed under the directory, they are concatenated in sorted filename order.
    """
    base_path = Path(raw_path)

    if not base_path.exists():
        raise FileNotFoundError(f"BATADAL raw path not found: {raw_path}")

    csv_files = sorted(base_path.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {base_path}")

    frames = []

    for csv_file in csv_files:
        df = pd.read_csv(csv_file, sep=None, engine="python")
        df.columns = [column.strip() for column in df.columns]
        df["source_file"] = csv_file.name
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    return combined


def detect_batadal_target_column(
    df: pd.DataFrame,
    target_column_candidates: list[str],
) -> str:
    """
    Detects the target/label column from candidate names.

    
    """
    normalized_columns = {
        column.lower(): column
        for column in df.columns
    }

    for candidate in target_column_candidates:
        candidate_lower = candidate.lower()

        if candidate_lower in normalized_columns:
            return normalized_columns[candidate_lower]

    available_columns = ", ".join(df.columns)

    raise ValueError(
        "BATADAL target column could not be detected. "
        f"Candidates: {target_column_candidates}. "
        f"Available columns: {available_columns}"
    )


def detect_batadal_time_column(
    df: pd.DataFrame,
    time_column_candidates: list[str],
) -> str | None:
    """
    Detects the time column if it exists.

    Time columns are not used as model input. They are used only for preserving time order and interpreting results.
    """
    normalized_columns = {
        column.lower(): column
        for column in df.columns
    }

    for candidate in time_column_candidates:
        candidate_lower = candidate.lower()

        if candidate_lower in normalized_columns:
            return normalized_columns[candidate_lower]

    return None


def get_batadal_feature_columns(
    df: pd.DataFrame,
    target_column: str,
    time_column: str | None = None,
) -> list[str]:
    """
    Returns model input columns for BATADAL by excluding:
    - target/label column
    - time column
    - source_file metadata column
    """
    excluded = {target_column, "source_file"}

    if time_column is not None:
        excluded.add(time_column)

    feature_columns = [
        column for column in df.columns
        if column not in excluded
    ]

    if not feature_columns:
        raise ValueError("No feature columns found for BATADAL dataset.")

    return feature_columns


def split_batadal_time_ordered(
    df: pd.DataFrame,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
    time_column: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits BATADAL data in time order.

    Required protocol:
    - 60% train
    - 20% validation
    - 20% test
    - no random row-based split
    """
    total_ratio = train_ratio + validation_ratio + test_ratio

    if abs(total_ratio - 1.0) > 1e-9:
        raise ValueError("Train, validation and test ratios must sum to 1.0.")

    ordered_df = df.copy()

    if time_column is not None:
        ordered_df = ordered_df.sort_values(by=time_column).reset_index(drop=True)

    n_rows = len(ordered_df)

    train_end = int(n_rows * train_ratio)
    validation_end = train_end + int(n_rows * validation_ratio)

    train_df = ordered_df.iloc[:train_end].copy()
    validation_df = ordered_df.iloc[train_end:validation_end].copy()
    test_df = ordered_df.iloc[validation_end:].copy()

    return train_df, validation_df, test_df