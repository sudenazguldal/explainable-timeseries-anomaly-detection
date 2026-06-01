import pandas as pd

from src.data.label_utils import (
    normalize_skab_labels,
    normalize_batadal_labels,
)


def test_normalize_skab_labels_creates_binary_label_column():
    df = pd.DataFrame({
        "sensor_1": [1.0, 2.0, 3.0],
        "anomaly": [0.0, 1.0, 0.0],
    })

    normalized = normalize_skab_labels(
        df=df,
        target_column="anomaly",
        output_column="label",
    )

    assert normalized["label"].tolist() == [0, 1, 0]


def test_normalize_batadal_labels_maps_minus_999_to_normal_and_1_to_anomaly():
    df = pd.DataFrame({
        "SENSOR_1": [1.0, 2.0, 3.0],
        "ATT_FLAG": [-999, 1, -999],
    })

    normalized = normalize_batadal_labels(
        df=df,
        target_column="ATT_FLAG",
        output_column="label",
    )

    assert normalized["label"].tolist() == [0, 1, 0]


def test_normalize_batadal_labels_rejects_unknown_values():
    df = pd.DataFrame({
        "SENSOR_1": [1.0, 2.0],
        "ATT_FLAG": [-999, 999],
    })

    try:
        normalize_batadal_labels(
            df=df,
            target_column="ATT_FLAG",
            output_column="label",
        )
    except ValueError as error:
        assert "Invalid BATADAL label values" in str(error)
    else:
        raise AssertionError("Expected ValueError for invalid BATADAL label.")