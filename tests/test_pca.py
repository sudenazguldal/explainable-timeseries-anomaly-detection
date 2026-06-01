import pandas as pd

from src.preprocessing.pca import (
    fit_transform_train_pca,
    transform_with_fitted_pca,
)


def test_pca_creates_one_dimensional_pc1_column():
    train_df = pd.DataFrame({
        "sensor_1": [1.0, 2.0, 3.0, 4.0],
        "sensor_2": [2.0, 4.0, 6.0, 8.0],
        "anomaly": [0, 0, 1, 1],
    })

    test_df = pd.DataFrame({
        "sensor_1": [5.0, 6.0],
        "sensor_2": [10.0, 12.0],
        "anomaly": [0, 1],
    })

    feature_columns = ["sensor_1", "sensor_2"]

    transformed_train, pca = fit_transform_train_pca(
        train_df=train_df,
        feature_columns=feature_columns,
        n_components=1,
        output_column="pc1",
    )

    transformed_test = transform_with_fitted_pca(
        df=test_df,
        feature_columns=feature_columns,
        pca=pca,
        output_column="pc1",
    )

    assert "pc1" in transformed_train.columns
    assert "pc1" in transformed_test.columns
    assert len(transformed_train["pc1"]) == 4
    assert len(transformed_test["pc1"]) == 2