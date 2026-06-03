import pandas as pd

from src.data.batadal_loader import split_batadal_time_ordered
from src.experiments.run_dl_experiments import (
    DL_BINARY_TARGET_COLUMN,
    _split_skab_fold,
    normalize_deep_learning_labels,
)


def test_batadal_labels_are_prepared_as_binary_without_changing_time_order():
    dataframe = pd.DataFrame(
        {
            "DATETIME": [
                "03/01/24 00",
                "01/01/24 00",
                "05/01/24 00",
                "02/01/24 00",
                "04/01/24 00",
            ],
            "SENSOR_1": [3, 1, 5, 2, 4],
            "ATT_FLAG": [1, -999, 1, -999, -999],
        }
    )

    prepared_dataframe = normalize_deep_learning_labels(dataframe, target_column="ATT_FLAG")
    train_df, validation_df, test_df = split_batadal_time_ordered(
        df=prepared_dataframe,
        train_ratio=0.60,
        validation_ratio=0.20,
        test_ratio=0.20,
        time_column="DATETIME",
    )

    ordered_sensor_values = (
        train_df["SENSOR_1"].tolist()
        + validation_df["SENSOR_1"].tolist()
        + test_df["SENSOR_1"].tolist()
    )
    ordered_binary_labels = (
        train_df[DL_BINARY_TARGET_COLUMN].tolist()
        + validation_df[DL_BINARY_TARGET_COLUMN].tolist()
        + test_df[DL_BINARY_TARGET_COLUMN].tolist()
    )

    assert prepared_dataframe["ATT_FLAG"].tolist() == [1, -999, 1, -999, -999]
    assert prepared_dataframe[DL_BINARY_TARGET_COLUMN].tolist() == [1, 0, 1, 0, 0]
    assert ordered_sensor_values == [1, 2, 3, 4, 5]
    assert ordered_binary_labels == [0, 0, 1, 0, 1]


def test_skab_deep_learning_split_keeps_source_files_disjoint():
    dataframe = pd.DataFrame(
        {
            "sensor_1": list(range(12)),
            "anomaly": [0, 1] * 6,
            "source_file": [
                "valve1/a.csv",
                "valve1/a.csv",
                "valve1/b.csv",
                "valve1/b.csv",
                "valve1/c.csv",
                "valve1/c.csv",
                "valve1/d.csv",
                "valve1/d.csv",
                "valve2/e.csv",
                "valve2/e.csv",
                "valve2/f.csv",
                "valve2/f.csv",
            ],
        }
    )
    train_indices = list(range(8))
    test_indices = list(range(8, 12))

    train_df, validation_df, test_df = _split_skab_fold(
        dataframe=dataframe,
        train_indices=train_indices,
        test_indices=test_indices,
        group_column="source_file",
        validation_ratio=0.25,
    )

    train_groups = set(train_df["source_file"])
    validation_groups = set(validation_df["source_file"])
    test_groups = set(test_df["source_file"])

    assert train_groups.isdisjoint(validation_groups)
    assert train_groups.isdisjoint(test_groups)
    assert validation_groups.isdisjoint(test_groups)
    assert validation_groups == {"valve1/d.csv"}
