import pandas as pd


def normalize_skab_labels(
    df: pd.DataFrame,
    target_column: str = "anomaly",
    output_column: str = "label",
) -> pd.DataFrame:
    """
    Converts SKAB anomaly labels to binary integer labels.

    Expected:
    - 0 / 0.0 -> 0
    - 1 / 1.0 -> 1
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column not found: {target_column}")

    normalized = df.copy()
    normalized[output_column] = normalized[target_column].astype(float).astype(int)

    invalid_values = set(normalized[output_column].unique()) - {0, 1}

    if invalid_values:
        raise ValueError(f"Invalid SKAB label values found: {invalid_values}")

    return normalized


def normalize_batadal_labels(
    df: pd.DataFrame,
    target_column: str = "ATT_FLAG",
    output_column: str = "label",
) -> pd.DataFrame:
    """
    Converts BATADAL ATT_FLAG labels to binary labels.

    Expected:
    - -999 -> 0 normal
    - 1    -> 1 attack/anomaly
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column not found: {target_column}")

    label_mapping = {
        -999: 0,
        "-999": 0,
        1: 1,
        "1": 1,
    }

    normalized = df.copy()
    normalized[output_column] = normalized[target_column].map(label_mapping)

    if normalized[output_column].isna().any():
        invalid_values = normalized.loc[
            normalized[output_column].isna(),
            target_column,
        ].unique()

        raise ValueError(
            f"Invalid BATADAL label values found in {target_column}: "
            f"{list(invalid_values)}"
        )

    normalized[output_column] = normalized[output_column].astype(int)

    return normalized