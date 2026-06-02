import pandas as pd
import pytest

from src.data.splitter import (
    create_skab_group_folds,
    validate_no_group_leakage,
)


def test_create_skab_group_folds_prevents_source_file_leakage():
    df = pd.DataFrame({
        "sensor_1": list(range(12)),
        "label": [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
        "source_file": [
            "valve1/0.csv", "valve1/0.csv",
            "valve1/1.csv", "valve1/1.csv",
            "valve1/2.csv", "valve1/2.csv",
            "valve2/0.csv", "valve2/0.csv",
            "valve2/1.csv", "valve2/1.csv",
            "valve2/2.csv", "valve2/2.csv",
        ],
    })

    folds = create_skab_group_folds(
        df=df,
        target_column="label",
        group_column="source_file",
        n_splits=3,
        stratified=False,
    )

    assert len(folds) == 3

    for train_indices, test_indices in folds:
        train_df = df.iloc[train_indices]
        test_df = df.iloc[test_indices]

        assert validate_no_group_leakage(
            train_df=train_df,
            test_df=test_df,
            group_column="source_file",
        )


def test_create_skab_group_folds_rejects_missing_target_column():
    df = pd.DataFrame({
        "sensor_1": [1, 2, 3],
        "source_file": ["a.csv", "b.csv", "c.csv"],
    })

    with pytest.raises(ValueError):
        create_skab_group_folds(
            df=df,
            target_column="label",
            group_column="source_file",
            n_splits=2,
        )


def test_create_skab_group_folds_rejects_too_few_groups():
    df = pd.DataFrame({
        "sensor_1": [1, 2, 3, 4],
        "label": [0, 1, 0, 1],
        "source_file": ["a.csv", "a.csv", "b.csv", "b.csv"],
    })

    with pytest.raises(ValueError):
        create_skab_group_folds(
            df=df,
            target_column="label",
            group_column="source_file",
            n_splits=3,
        )