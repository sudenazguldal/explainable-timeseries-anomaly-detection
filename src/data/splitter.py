import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold


def create_skab_group_folds(
    df: pd.DataFrame,
    target_column: str,
    group_column: str,
    n_splits: int = 5,
    stratified: bool = True,
    random_seed: int = 42,
) -> list[tuple[list[int], list[int]]]:
    """
    Creates group-based folds for SKAB.

    Required protocol:
    - source_file is used as group variable
    - rows from the same CSV file must not appear in both train and test
    - StratifiedGroupKFold is preferred when possible
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column not found: {target_column}")

    if group_column not in df.columns:
        raise ValueError(f"Group column not found: {group_column}")

    unique_group_count = df[group_column].nunique()

    if unique_group_count < n_splits:
        raise ValueError(
            f"Number of unique groups ({unique_group_count}) "
            f"must be at least n_splits ({n_splits})."
        )

    x_placeholder = df.drop(columns=[target_column])
    y = df[target_column].astype(int)
    groups = df[group_column]

    if stratified:
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_seed,
        )

        folds = splitter.split(
            X=x_placeholder,
            y=y,
            groups=groups,
        )
    else:
        splitter = GroupKFold(n_splits=n_splits)

        folds = splitter.split(
            X=x_placeholder,
            y=y,
            groups=groups,
        )

    return [
        (train_indices.tolist(), test_indices.tolist())
        for train_indices, test_indices in folds
    ]


def validate_no_group_leakage(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    group_column: str,
) -> bool:
    """
    Checks that no group appears in both train and test sets.
    
    """
    train_groups = set(train_df[group_column].unique())
    test_groups = set(test_df[group_column].unique())

    overlap = train_groups.intersection(test_groups)

    return len(overlap) == 0