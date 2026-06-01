import pandas as pd

from src.preprocessing.noise import add_gaussian_noise


def test_add_gaussian_noise_changes_feature_columns_only():
    df = pd.DataFrame({
        "sensor_1": [1.0, 2.0, 3.0],
        "sensor_2": [10.0, 20.0, 30.0],
        "anomaly": [0, 1, 0],
    })

    feature_columns = ["sensor_1", "sensor_2"]

    noisy_df = add_gaussian_noise(
        df=df,
        feature_columns=feature_columns,
        mean=0.0,
        std=0.05,
        random_seed=42,
    )

    assert not noisy_df[feature_columns].equals(df[feature_columns])
    assert noisy_df["anomaly"].equals(df["anomaly"])