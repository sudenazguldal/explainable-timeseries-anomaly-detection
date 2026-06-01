import pandas as pd

from src.data.batadal_loader import (
    load_batadal_dataset,
    detect_batadal_target_column,
    detect_batadal_time_column,
    get_batadal_feature_columns,
    split_batadal_time_ordered,
)


def test_load_batadal_dataset_reads_csv_and_adds_source_file(tmp_path):
    base_path = tmp_path / "batadal"
    base_path.mkdir(parents=True)

    sample = pd.DataFrame({
        "DATETIME": ["2024-01-01 00:00:00"],
        "SENSOR_1": [1.0],
        "SENSOR_2": [2.0],
        "ATT_FLAG": [0],
    })

    sample.to_csv(base_path / "training_dataset_2.csv", index=False)

    df = load_batadal_dataset(str(base_path))

    assert len(df) == 1
    assert "source_file" in df.columns
    assert df.loc[0, "source_file"] == "training_dataset_2.csv"


def test_detect_batadal_target_column_from_candidates():
    df = pd.DataFrame({
        "DATETIME": ["2024-01-01 00:00:00"],
        "SENSOR_1": [1.0],
        "ATT_FLAG": [0],
    })

    target_column = detect_batadal_target_column(
        df=df,
        target_column_candidates=["attack", "anomaly", "ATT_FLAG", "label"],
    )

    assert target_column == "ATT_FLAG"


def test_detect_batadal_time_column_from_candidates():
    df = pd.DataFrame({
        "DATETIME": ["2024-01-01 00:00:00"],
        "SENSOR_1": [1.0],
        "ATT_FLAG": [0],
    })

    time_column = detect_batadal_time_column(
        df=df,
        time_column_candidates=["datetime", "DATETIME", "DateTime"],
    )

    assert time_column == "DATETIME"


def test_get_batadal_feature_columns_excludes_target_time_and_source_file():
    df = pd.DataFrame({
        "DATETIME": ["2024-01-01 00:00:00"],
        "SENSOR_1": [1.0],
        "SENSOR_2": [2.0],
        "ATT_FLAG": [0],
        "source_file": ["training_dataset_2.csv"],
    })

    feature_columns = get_batadal_feature_columns(
        df=df,
        target_column="ATT_FLAG",
        time_column="DATETIME",
    )

    assert feature_columns == ["SENSOR_1", "SENSOR_2"]


def test_split_batadal_time_ordered_uses_60_20_20_split():
    df = pd.DataFrame({
        "DATETIME": pd.date_range("2024-01-01", periods=10, freq="h"),
        "SENSOR_1": range(10),
        "ATT_FLAG": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
    })

    train_df, validation_df, test_df = split_batadal_time_ordered(
        df=df,
        train_ratio=0.60,
        validation_ratio=0.20,
        test_ratio=0.20,
        time_column="DATETIME",
    )

    assert len(train_df) == 6
    assert len(validation_df) == 2
    assert len(test_df) == 2

    assert train_df["SENSOR_1"].tolist() == [0, 1, 2, 3, 4, 5]
    assert validation_df["SENSOR_1"].tolist() == [6, 7]
    assert test_df["SENSOR_1"].tolist() == [8, 9]