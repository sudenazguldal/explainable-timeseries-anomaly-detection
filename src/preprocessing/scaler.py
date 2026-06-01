import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def create_scaler(method: str):
    """
    Creates a scaler instance.

    Supported methods:
    - standard
    - minmax
    """
    if method == "standard":
        return StandardScaler()

    if method == "minmax":
        return MinMaxScaler()

    raise ValueError(f"Unsupported normalization method: {method}")


def fit_transform_train(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    method: str = "standard",
) -> tuple[pd.DataFrame, object]:
    """
    Fits scaler only on train data and transforms train data.

    This prevents data leakage.
    """
    scaler = create_scaler(method)

    transformed = train_df.copy()
    transformed[feature_columns] = scaler.fit_transform(train_df[feature_columns])

    return transformed, scaler


def transform_with_fitted_scaler(
    df: pd.DataFrame,
    feature_columns: list[str],
    scaler,
) -> pd.DataFrame:
    """
    Transforms validation/test data using an already fitted scaler.
    """
    transformed = df.copy()
    transformed[feature_columns] = scaler.transform(df[feature_columns])

    return transformed