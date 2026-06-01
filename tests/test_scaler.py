import pandas as pd

from src.preprocessing.scaler import (
    fit_transform_train,
    transform_with_fitted_scaler,
)


def test_scaler_fits_only_on_train_data():
    train_df = pd.DataFrame({
        "sensor_1": [1.0, 2.0, 3.0],
        "sensor_2": [10.0, 20.0, 30.0],
        "anomaly": [0, 0, 1],
    })

    test_df = pd.DataFrame({
        "sensor_1": [4.0],
        "sensor_2": [40.0],
        "anomaly": [1],
    })

    feature_columns = ["sensor_1", "sensor_2"]

    scaled_train, scaler = fit_transform_train(
        train_df=train_df,
        feature_columns=feature_columns,
        method="standard",
    )

    scaled_test = transform_with_fitted_scaler(
        df=test_df,
        feature_columns=feature_columns,
        scaler=scaler,
    )

    assert round(scaled_train["sensor_1"].mean(), 7) == 0.0
    assert "anomaly" in scaled_train.columns
    assert "anomaly" in scaled_test.columns