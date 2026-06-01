from pathlib import Path
import pandas as pd

from src.data.skab_loader import load_skab_dataset, get_skab_feature_columns


def test_load_skab_dataset_adds_source_metadata(tmp_path):
    base_path = tmp_path / "skab"
    valve1_path = base_path / "valve1"
    valve2_path = base_path / "valve2"

    valve1_path.mkdir(parents=True)
    valve2_path.mkdir(parents=True)

    sample_1 = pd.DataFrame({
        "datetime": ["2024-01-01 00:00:00"],
        "sensor_1": [1.0],
        "sensor_2": [2.0],
        "anomaly": [0],
        "changepoint": [0],
    })

    sample_2 = pd.DataFrame({
        "datetime": ["2024-01-01 00:01:00"],
        "sensor_1": [3.0],
        "sensor_2": [4.0],
        "anomaly": [1],
        "changepoint": [1],
    })

    sample_1.to_csv(valve1_path / "1.csv", index=False)
    sample_2.to_csv(valve2_path / "1.csv", index=False)

    df = load_skab_dataset(
        raw_path=str(base_path),
        use_groups=["valve1", "valve2"],
    )

    assert len(df) == 2
    assert "source_group" in df.columns
    assert "source_file" in df.columns
    assert set(df["source_group"]) == {"valve1", "valve2"}


def test_get_skab_feature_columns_excludes_metadata_and_target():
    df = pd.DataFrame({
        "datetime": ["2024-01-01"],
        "sensor_1": [1.0],
        "sensor_2": [2.0],
        "anomaly": [0],
        "changepoint": [0],
        "source_group": ["valve1"],
        "source_file": ["1.csv"],
    })

    feature_columns = get_skab_feature_columns(
        df=df,
        target_column="anomaly",
        excluded_columns=["datetime", "changepoint", "source_group", "source_file"],
    )

    assert feature_columns == ["sensor_1", "sensor_2"]