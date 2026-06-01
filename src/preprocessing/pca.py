import pandas as pd
from sklearn.decomposition import PCA


def fit_transform_train_pca(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    n_components: int = 1,
    output_column: str = "pc1",
) -> tuple[pd.DataFrame, PCA]:
    """
    Fits PCA only on train data and transforms train data.  Automata model requires one-dimensional input, so PC1 is used.
    """
    if n_components != 1:
        raise ValueError("This project requires PCA output to be one-dimensional.")

    pca = PCA(n_components=n_components)

    transformed_values = pca.fit_transform(train_df[feature_columns])

    transformed = train_df.copy()
    transformed[output_column] = transformed_values[:, 0]

    return transformed, pca


def transform_with_fitted_pca(
    df: pd.DataFrame,
    feature_columns: list[str],
    pca: PCA,
    output_column: str = "pc1",
) -> pd.DataFrame:
    """
    Transforms validation/test data using PCA fitted only on train data.
    """
    transformed_values = pca.transform(df[feature_columns])

    transformed = df.copy()
    transformed[output_column] = transformed_values[:, 0]

    return transformed